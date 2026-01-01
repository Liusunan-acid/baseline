
"""
DRL-COE Runner v3 (对齐 COE 原框架版)

你反馈：当 action 恒等于 0（即 mask=111 全开）时，效果仍明显差于“不用 PPO”的原 COE。
根因：v1/v2 的 evolve 循环没有 1:1 复用你 COE 的 evolve_gpu 主流程（父代池/精英拼接/coevolution 触发等都不同）。

本版本 v3 做到：
- evolve 主循环严格拷贝 coe-位置级-子种群对齐.py::MultiRunOptimizer.evolve_gpu 的逻辑
- 唯一插入点：children_flat = _mutate_batch_gpu_masked(..., op_mask) （按策略选择算子组合）
- 因此：当强制 action=0 (mask=111) 时，应与原 evolve_gpu 完全一致（除了额外的 state 统计开销，不影响搜索路径）

动作(action)：
  8 个离散动作 = 3-bit mask (violations/base_swap/greedy_cluster)
    0 -> 111  (全开，等价原始 _mutate_batch_gpu)
    1 -> 110
    2 -> 101
    3 -> 011
    4 -> 100
    5 -> 010
    6 -> 001
    7 -> 000  (不做变异)

使用：
  训练：python drl_coe_runner_v3.py --mode train ...
  部署：python drl_coe_runner_v3.py --mode deploy --resume ...
  对齐测试：加 --force_action 0 ，保证每代用 mask=111
"""

from __future__ import annotations

import os
import time
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# 0) 动态加载用户 COE 代码
# -------------------------
def load_user_module(py_path: str):
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location("user_coe_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -------------------------
# 1) PPO 组件（actor-critic）
# -------------------------
@dataclass
class Rollout:
    s: List[np.ndarray]
    a: List[int]
    logp: List[float]
    r: List[float]
    v: List[float]
    done: List[bool]


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.logits = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor):
        x = self.body(s)
        return self.logits(x), self.value(x).squeeze(-1)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    T = rewards.shape[0]
    adv = np.zeros((T,), dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        next_v = values[t + 1] if (t + 1) < T else 0.0
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + values
    return adv, ret


def ppo_update(
    policy: PolicyNet,
    optimizer: optim.Optimizer,
    rollout: Rollout,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 256,
    device: str = "cuda",
):
    s = np.stack(rollout.s, axis=0).astype(np.float32)
    a = np.array(rollout.a, dtype=np.int64)
    old_logp = np.array(rollout.logp, dtype=np.float32)
    r = np.array(rollout.r, dtype=np.float32)
    v = np.array(rollout.v, dtype=np.float32)
    done = np.array(rollout.done, dtype=np.float32)

    adv, ret = compute_gae(r, v, done, gamma=gamma, lam=lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    s_t = torch.from_numpy(s).to(device)
    a_t = torch.from_numpy(a).to(device)
    old_logp_t = torch.from_numpy(old_logp).to(device)
    adv_t = torch.from_numpy(adv).to(device)
    ret_t = torch.from_numpy(ret).to(device)

    N = s_t.size(0)
    idx = torch.arange(N, device=device)

    for _ in range(epochs):
        perm = idx[torch.randperm(N)]
        for start in range(0, N, batch_size):
            mb = perm[start:start + batch_size]
            logits, vpred = policy(s_t[mb])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(a_t[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp_t[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (ret_t[mb] - vpred).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


# -------------------------
# 2) 动作空间：3-bit mask
# -------------------------
ACTION_MASKS = [
    0b111,  # 0: 全开（=原 _mutate_batch_gpu）
    0b110,  # 1: viol + base
    0b101,  # 2: viol + greedy
    0b011,  # 3: base + greedy
    0b100,  # 4: viol only
    0b010,  # 5: base only
    0b001,  # 6: greedy only
    0b000,  # 7: no mutation
]


def make_state_vec(
    best_improved: float,
    improve_rate: float,
    mean_improved: float,
    #is_run0_best: float,
    #run0_gap: float,
    stagnation: float,
    budget: float,
    mean_viols: float,
    feasible_rate: float,
) -> np.ndarray:
    return np.array([
        best_improved, improve_rate, mean_improved,
        stagnation, budget,
        mean_viols, feasible_rate
    ], dtype=np.float32)


class MultiRunOptimizerDRL:
    """
    与 COE 原 MultiRunOptimizer 对齐：
    - evolve_gpu_drl 复制你的 evolve_gpu 主循环
    - 仅在变异处按 mask 选择 step1/2/3
    """
    def __init__(self, user_mod, base_optimizer):
        self.user_mod = user_mod
        self.base = base_optimizer
        self.base._ensure_gpu_engine()

    def _mutate_batch_gpu_masked(
        self,
        X: torch.Tensor,
        parent_violate_mask: torch.Tensor,
        current_gen: int,
        op_mask: int,
        base_swap_prob: float = 0.95,
        greedy_prob: float = 0.5,
    ) -> torch.Tensor:
        if op_mask & 0b100:
            X = self.base._mutate_step1_violations(X, parent_violate_mask)
        if op_mask & 0b010:
            X = self.base._mutate_step2_base_swap(X, current_gen, base_swap_prob)
        if op_mask & 0b001:
            X = self.base._mutate_step3_greedy_cluster(X, greedy_prob)
        return X

    @torch.no_grad()
    def evolve_gpu_drl(
        self,
        generations: int,
        elite_size: int,
        policy: PolicyNet,
        train: bool,
        rollout: Optional[Rollout],
        deterministic: bool = False,
        force_action: Optional[int] = None,
        base_swap_prob: float = 0.95,
        greedy_prob: float = 0.5,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Rollout]]:

        base = self.base
        base._ensure_gpu_engine()
        if base.population_tensor is None:
            raise RuntimeError("种群为空，请先 initialize_population")

        pop = base.population_tensor
        N = base.N
        K, B = base.K, base.B

        global_best = -float("inf")
        mean_fit = 0.0
        prev_fitness = None
        stagnation = 0

        ep_best_fit = -float("inf")
        ep_best_idx = None

        # -------------------------
        # C1 fix: 延迟一拍写 reward
        # pending 存“上一代 action 的 transition”，reward 在下一代评估时才能计算
        # -------------------------
        pending = None  # dict: {"s": np.ndarray, "a": int, "logp": float, "v": float, "baseline_best": float}

        for gen_idx in range(generations):
            # 完全对齐：每 50 代触发 coevolution
            if gen_idx > 0 and gen_idx % 50 == 0:
                base.run_coevolution_phase(co_gens=50)
                pop = base.population_tensor

            pop_flat = pop.view(base.total_pop_size, N)

            # 对齐：return_assignment=False
            out = base._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            fitness = out["fitness"].view(K, B)

            viol_mask_flat = out.get("any_violate_mask_b_n", None)
            if viol_mask_flat is None:
                raise KeyError("fitness_batch 未返回 any_violate_mask_b_n（请检查 COE 引擎输出）")
            violate_mask = viol_mask_flat.view(K, B, N)

            # 统计：best per run
            topk_vals, topk_idx = torch.topk(fitness, k=B, largest=True, dim=1)
            best_fitness_per_run = topk_vals[:, 0]
            gbest_fit = float(best_fitness_per_run.max().item())
            gbest_run = int(torch.argmax(best_fitness_per_run).item())
            run0_best = float(best_fitness_per_run[0].item())
            is_run0_best = 1.0 if (gbest_run == 0) else 0.0
            run0_gap = (run0_best - gbest_fit) / (abs(gbest_fit) + 1e-6)

            # 全局统计初始化
            if gen_idx == 0:
                global_best = gbest_fit
                mean_fit = float(fitness.mean().item())
                prev_fitness = fitness.clone()

            # -------------------------
            # C1 fix: 用“当前评估的 gbest_fit”给上一代 action 回填 reward
            # reward 语义保持不变：仍是 (best - baseline_best)/|baseline_best|
            # 只是把 reward 归因到正确的 action 上
            # -------------------------
            if train and rollout is not None and force_action is None and pending is not None:
                baseline_best = float(pending["baseline_best"])
                reward_prev = (gbest_fit - baseline_best) / (abs(baseline_best) + 1e-6)

                rollout.s.append(pending["s"])
                rollout.a.append(int(pending["a"]))
                rollout.logp.append(float(pending["logp"]))
                rollout.r.append(float(reward_prev))
                rollout.v.append(float(pending["v"]))
                rollout.done.append(False)

            best_improved = 1.0 if (gbest_fit > global_best + 1e-9) else 0.0
            mean_improved = 1.0 if (float(fitness.mean().item()) > mean_fit + 1e-9) else 0.0
            improve_rate = float((fitness > prev_fitness).to(torch.float32).mean().item())

            # violations 统计（对齐你原打印逻辑，来自 out 的各项 cnt）
            if all(k in out for k in ["heart_cnt", "angio_cnt", "weekend_cnt", "device_cnt"]):
                flat_viols = (out["heart_cnt"] + out["angio_cnt"] + out["weekend_cnt"] + out["device_cnt"]).view(K, B)
                best_viols = torch.gather(flat_viols, 1, topk_idx[:, :1]).float()  # [K,1]
                mean_viols = float(best_viols.mean().item())
            else:
                # fallback：用“best 个体的 violate positions 数”近似
                best_mask = torch.gather(
                    violate_mask, 1, topk_idx[:, :1].unsqueeze(2).expand(-1, -1, N)
                ).squeeze(1)
                mean_viols = float(best_mask.sum(dim=1).float().mean().item())

            # feasible rate：best 个体完全可行比例
            best_mask = torch.gather(
                violate_mask, 1, topk_idx[:, :1].unsqueeze(2).expand(-1, -1, N)
            ).squeeze(1)
            feasible_rate = float((best_mask.sum(dim=1) == 0).float().mean().item())

            budget = float(gen_idx) / float(max(1, generations - 1))
            s_vec = make_state_vec(
                best_improved=best_improved,
                improve_rate=improve_rate,
                mean_improved=mean_improved,
                #is_run0_best=is_run0_best,
                #run0_gap=float(run0_gap),
                stagnation=float(stagnation),
                budget=budget,
                mean_viols=mean_viols,
                feasible_rate=feasible_rate,
            )

            # 选动作
            if force_action is not None:
                a = int(force_action)
                logp = 0.0
                v = 0.0
            else:
                dev = next(policy.parameters()).device
                s_t = torch.from_numpy(s_vec).float().unsqueeze(0).to(dev)
                logits, vpred = policy(s_t)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    a = int(torch.argmax(logits, dim=1).item())
                else:
                    a = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor([a], device=dev)).item())
                v = float(vpred.item())

            op_mask = ACTION_MASKS[a]

            # -------------------------
            # 以下块：严格对齐原 evolve_gpu
            # -------------------------
            elite_size_eff = min(elite_size, B)
            elite_idx = topk_idx[:, :elite_size_eff]
            idx_expanded = elite_idx.unsqueeze(2).expand(K, elite_size_eff, N)
            elites = torch.gather(pop, 1, idx_expanded)

            parent_count = max(1, int(0.2 * B))
            parent_idx = topk_idx[:, :parent_count]
            idx_expanded = parent_idx.unsqueeze(2).expand(K, parent_count, N)
            parents = torch.gather(pop, 1, idx_expanded)
            parent_viol = torch.gather(violate_mask, 1, idx_expanded)

            num_children = B - elite_size_eff
            if num_children > 0:
                p_idx1 = torch.randint(0, parent_count, (K, num_children), device=pop.device)
                p_idx2 = torch.randint(0, parent_count, (K, num_children), device=pop.device)

                P1 = torch.gather(parents, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))
                P2 = torch.gather(parents, 1, p_idx2.unsqueeze(2).expand(-1, -1, N))
                Vmask_choice = torch.gather(parent_viol, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))

                P1_flat = P1.view(K * num_children, N)
                P2_flat = P2.view(K * num_children, N)
                children_flat = base._ordered_crossover_batch_gpu(P1_flat, P2_flat)

                Vmask_flat = Vmask_choice.view(K * num_children, N)
                # 唯一差异点：按 mask 选择变异组合
                children_flat = self._mutate_batch_gpu_masked(
                    children_flat, Vmask_flat, base.current_generation,
                    op_mask=op_mask,
                    base_swap_prob=base_swap_prob,
                    greedy_prob=greedy_prob,
                )
                children = children_flat.view(K, num_children, N)
                pop = torch.cat([elites, children], dim=1)
            else:
                pop = elites.clone()

            base.population_tensor = pop
            base.current_generation += 1

            # 更新 best/stagnation（保持你原逻辑不变）
            if gbest_fit > global_best + 1e-9:
                global_best = gbest_fit
                stagnation = 0
            else:
                stagnation += 1

            mean_fit = float(fitness.mean().item())
            prev_fitness = fitness.clone()

            if gbest_fit > ep_best_fit + 1e-9:
                ep_best_fit = gbest_fit
                ep_best_idx = elite_idx[:, :1].clone()

            # -------------------------
            # C1 fix: 本代不写 rollout；只把 transition 暂存，等下一代评估时回填 reward
            # baseline_best 必须是在“选 action 时刻”的 global_best（保持你原 reward 语义）
            # -------------------------
            if train and rollout is not None and force_action is None:
                pending = {
                    "s": s_vec,
                    "a": a,
                    "logp": logp,
                    "v": v,
                    "baseline_best": float(global_best),
                }

            if verbose and ((gen_idx + 1) % 50 == 0 or gen_idx == 0):
                avg_best_fit = float(best_fitness_per_run.mean().item())
                print(
                    f"[gen ({(gen_idx+1)*2:05d}/{generations}] "
                    f"action={a} mask={op_mask:03b} "
                    f"avg_best_fit={avg_best_fit:.2f} gbest_fit={global_best:.2f} "
                    f"mean_viols={mean_viols:.2f} feas_rate={feasible_rate:.2f} stg={stagnation}",
                    flush=True
                )

        # 结束：提取 K 个最佳个体（对齐你原 evolve_gpu 结尾）
        pop_flat = base.population_tensor.view(base.total_pop_size, N)
        final_out = base._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
        final_fitness = final_out["fitness"].view(K, B)

        final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1)
        final_best_vals = final_best_vals.flatten()

        idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(K, 1, N)
        best_individuals_tensor = torch.gather(base.population_tensor, 1, idx_expanded).squeeze(1)

        best_individuals_cpu = best_individuals_tensor.cpu()
        best_fitnesses_cpu = final_best_vals.cpu().tolist()

        results = []
        for k in range(K):
            cids = base._tensor_row_to_cids(best_individuals_cpu[k])
            results.append({"run_id": k, "fitness": float(best_fitnesses_cpu[k]), "individual_cids": cids})

        # -------------------------
        # C1 fix: flush 最后一代 action 的 reward（用 final_fitness 的 gbest）
        # -------------------------
        if train and rollout is not None and force_action is None and pending is not None:
            final_gbest = float(final_best_vals.max().item())
            baseline_best = float(pending["baseline_best"])
            reward_last = (final_gbest - baseline_best) / (abs(baseline_best) + 1e-6)

            rollout.s.append(pending["s"])
            rollout.a.append(int(pending["a"]))
            rollout.logp.append(float(pending["logp"]))
            rollout.r.append(float(reward_last))
            rollout.v.append(float(pending["v"]))
            rollout.done.append(True)

        info = {"episode_best_fit": float(ep_best_fit), "final_results": results}
        return info, rollout

# -------------------------
# 3) main：train/deploy
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coe_path", type=str, required=True, help="你的 COE 程序路径（py）")
    parser.add_argument("--patient_file", type=str, required=True)
    parser.add_argument("--duration_file", type=str, required=True)
    parser.add_argument("--device_file", type=str, required=True)

    parser.add_argument("--mode", type=str, default="train", choices=["train", "deploy"])
    parser.add_argument("--seed", type=int, default=0)

    # COE
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--B", type=int, default=50)
    parser.add_argument("--generations", type=int, default=5000)
    parser.add_argument("--elite_size", type=int, default=5)

    # PPO
    parser.add_argument("--train_episodes", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)

    # mutation params (仍用你原来的；RL 只做“组合选择”)
    parser.add_argument("--base_swap_prob", type=float, default=0.95)
    parser.add_argument("--greedy_prob", type=float, default=0.5)

    # ckpt
    parser.add_argument("--ckpt_path", type=str, default="drl_coe_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)

    # deploy
    parser.add_argument("--deploy_runs", type=int, default=3)
    parser.add_argument("--deploy_deterministic", action="store_true")

    # debug: force action
    parser.add_argument("--force_action", type=int, default=-1, help=">=0 则强制每代 action 固定（用于对齐测试）")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    user_mod = load_user_module(args.coe_path)

    patients = user_mod.import_data(args.patient_file, args.duration_file)
    machine_exam_map = user_mod.import_device_constraints(args.device_file)

    base_opt = user_mod.MultiRunOptimizer(
        patients, machine_exam_map,
        num_parallel_runs=args.K, pop_size_per_run=args.B
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = 7
    action_dim = 8
    policy = PolicyNet(state_dim, action_dim).to(device)
    optim_pol = optim.Adam(policy.parameters(), lr=args.lr)

    start_ep = 0
    if args.resume and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        if "policy_state" in ckpt:
            policy.load_state_dict(ckpt["policy_state"])
        if "optim_state" in ckpt:
            optim_pol.load_state_dict(ckpt["optim_state"])
        start_ep = int(ckpt.get("episode", 0))
        print(f"✓ resumed from {args.ckpt_path} (episode={start_ep})", flush=True)

    drl_opt = None  # after init population

    def _make_drl_opt():
        nonlocal drl_opt
        drl_opt = MultiRunOptimizerDRL(user_mod, base_opt)
        return drl_opt

    # TRAIN
    if args.mode == "train":
        for ep in range(start_ep, start_ep + args.train_episodes):
            base_opt.initialize_population()  # 每个 episode 重新初始化
            _make_drl_opt()

            rollout = Rollout(s=[], a=[], logp=[], r=[], v=[], done=[])

            force_action = args.force_action if args.force_action >= 0 else None
            info, rollout = drl_opt.evolve_gpu_drl(
                generations=args.generations,
                elite_size=args.elite_size,
                policy=policy,
                train=True,
                rollout=rollout,
                deterministic=False,
                force_action=force_action,
                base_swap_prob=args.base_swap_prob,
                greedy_prob=args.greedy_prob,
                verbose=True,
            )

            # 只有不强制 action 时才更新 PPO（否则是对齐测试）
            if force_action is None:
                ppo_update(
                    policy=policy,
                    optimizer=optim_pol,
                    rollout=rollout,
                    gamma=args.gamma,
                    lam=args.lam,
                    clip_eps=args.clip_eps,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    epochs=args.ppo_epochs,
                    batch_size=args.ppo_batch_size,
                    device=device,
                )

            ep_best = float(info["episode_best_fit"])
            print(f"[ep {ep+1:03d}] episode_best_fit={ep_best:.2f}", flush=True)

            if args.save_every > 0 and ((ep + 1) % args.save_every == 0):
                ckpt = {
                    "episode": ep + 1,
                    "policy_state": policy.state_dict(),
                    "optim_state": optim_pol.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, args.ckpt_path)
                print(f"✓ saved ckpt: {args.ckpt_path}", flush=True)

        args.mode = "deploy"

    # DEPLOY
    if args.mode == "deploy":
        best_fit = -float("inf")
        best_results = None

        for i in range(max(1, args.deploy_runs)):
            base_opt.initialize_population()
            _make_drl_opt()

            force_action = args.force_action if args.force_action >= 0 else None
            info, _ = drl_opt.evolve_gpu_drl(
                generations=args.generations,
                elite_size=args.elite_size,
                policy=policy,
                train=False,
                rollout=None,
                deterministic=bool(args.deploy_deterministic),
                force_action=force_action,
                base_swap_prob=args.base_swap_prob,
                greedy_prob=args.greedy_prob,
                verbose=True,
            )
            final_results = info["final_results"]
            run_best = max(final_results, key=lambda d: d["fitness"])
            if run_best["fitness"] > best_fit:
                best_fit = run_best["fitness"]
                best_results = final_results
            print(f"[deploy {i+1}/{args.deploy_runs}] run_best_fit={run_best['fitness']:.2f} | best_fit={best_fit:.2f}", flush=True)

        print("\n=== DRL-COE v3 finished ===")
        print(f"best_fit={best_fit:.2f}", flush=True)

        os.makedirs("output_schedules", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        for res in best_results:
            rid = res["run_id"]
            system = base_opt.generate_schedule(res["individual_cids"])
            out_xlsx = os.path.join("output_schedules", f"DRL_COE_RUN{rid}_{ts}_fit_{res['fitness']:.0f}.xlsx")
            user_mod.export_schedule(system, patients, out_xlsx)
            print(f"✓ exported: {out_xlsx}", flush=True)


if __name__ == "__main__":
    main()


'''
# 训练并保存 ckpt
CUDA_VISIBLE_DEVICES=1 python /home/preprocess/_funsearch/baseline/0EMBC/drl-实验.py \
  --mode train \
  --coe_path "/home/preprocess/_funsearch/baseline/coe-位置级-子种群对齐.py" \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 --generations 5000 --elite_size 5 \
  --train_episodes 50 \
  --ckpt_path drl_coe_ckpt.pt --save_every 1

# 从 ckpt 继续训练
CUDA_VISIBLE_DEVICES=0 python drl_coe_runner.py \
  --mode train --resume \
  --coe_path /path/to/coe-位置级-子种群对齐.py \
  --patient_file ... --duration_file ... --device_file ... \
  --ckpt_path drl_coe_ckpt.pt

# 部署（不训练），会导出 output_schedules/*.xlsx
CUDA_VISIBLE_DEVICES=0 python drl_coe_runner.py \
  --mode deploy --resume \
  --coe_path /path/to/coe-位置级-子种群对齐.py \
  --patient_file ... --duration_file ... --device_file ... \
  --ckpt_path drl_coe_ckpt.pt \
  --deploy_runs 3 --deploy_deterministic
'''
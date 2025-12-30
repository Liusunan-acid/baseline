
"""
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

# =========================
# ACTION_MASKS -> ACTION_SPECS
# action_id -> (n_viol_times, n_base_times, use_greedy)
# - step1 violations: 重复 nv 次
# - step2 base_swap : 重复 nb 次
# - step3 greedy    : 是否启用（启用则只调用 1 次；次数不受控）
# =========================

MAX_VIOL_TIMES = 3   # step1 最大重复次数（可调）
MAX_BASE_TIMES = 3   # step2 最大重复次数（可调）

ACTION_SPECS = [
    (nv, nb, use_greedy)
    for nv in range(MAX_VIOL_TIMES + 1)
    for nb in range(MAX_BASE_TIMES + 1)
    for use_greedy in (0, 1)
]


def make_state_vec(
    best_improved: float,
    improve_rate: float,
    mean_improved: float,
    stagnation: float,
    budget: float,
    mean_viols: float,
    mean_wait: float,
    mean_switch: float,
) -> np.ndarray:
    return np.array([
        best_improved, improve_rate, mean_improved,
        stagnation, budget,
        mean_viols, mean_wait, mean_switch
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
        n_viol_times: int,
        n_base_times: int,
        use_greedy: bool,
        base_swap_prob: float = 0.95,
        greedy_prob: float = 0.5,
    ) -> torch.Tensor:
        nv = int(n_viol_times)
        nb = int(n_base_times)
        for _ in range(nv):
            X = self.base._mutate_step1_violations(X, parent_violate_mask)
        for _ in range(nb):
            X = self.base._mutate_step2_base_swap(X, current_gen, base_swap_prob)
        if bool(use_greedy):
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

        # C1 fix：reward 延迟一拍归因（保留你之前修复）
        pending = None  # {"s": np.ndarray, "a": int, "logp": float, "v": float, "baseline_best": float}

        for gen_idx in range(generations):
            # 对齐：每 50 代触发 coevolution
            if gen_idx > 0 and gen_idx % 50 == 0:
                base.run_coevolution_phase(co_gens=50)
                pop = base.population_tensor

            pop_flat = pop.view(base.total_pop_size, N)

            out = base._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            fitness = out["fitness"].view(K, B)

            viol_mask_flat = out.get("any_violate_mask_b_n", None)
            if viol_mask_flat is None:
                raise KeyError("fitness_batch 未返回 any_violate_mask_b_n（请检查 COE 引擎输出）")
            violate_mask = viol_mask_flat.view(K, B, N)

            # best per run
            topk_vals, topk_idx = torch.topk(fitness, k=B, largest=True, dim=1)
            best_fitness_per_run = topk_vals[:, 0]
            gbest_fit = float(best_fitness_per_run.max().item())

            # init stats
            if gen_idx == 0:
                global_best = gbest_fit
                mean_fit = float(fitness.mean().item())
                prev_fitness = fitness.clone()

            # --- C1 fix：回填上一代 action 的 reward ---
            if train and rollout is not None and force_action is None and pending is not None:
                baseline_best = float(pending["baseline_best"])
                reward_prev = (gbest_fit - baseline_best) / (abs(baseline_best) + 1e-6)

                rollout.s.append(pending["s"])
                rollout.a.append(int(pending["a"]))
                rollout.logp.append(float(pending["logp"]))
                rollout.r.append(float(reward_prev))
                rollout.v.append(float(pending["v"]))
                rollout.done.append(False)

            # -------- state 统计（沿用你之前版本：全体统计 + wait/switch + 去掉 run0/feasible）--------
            best_improved = 1.0 if (gbest_fit > global_best + 1e-9) else 0.0
            mean_improved = 1.0 if (float(fitness.mean().item()) > mean_fit + 1e-9) else 0.0
            improve_rate = float((fitness > prev_fitness).to(torch.float32).mean().item())

            # mean_viols：全体个体均值
            if all(k in out for k in ["heart_cnt", "angio_cnt", "weekend_cnt", "device_cnt"]):
                flat_viols = (out["heart_cnt"] + out["angio_cnt"] + out["weekend_cnt"] + out["device_cnt"]).view(K, B)
                mean_viols = float(flat_viols.float().mean().item())
            else:
                mean_viols = float(violate_mask.sum(dim=2).float().mean().item())  # [K,B,N] -> [K,B]

            # 等待时间：全体个体均值（如果 key 不匹配，请改成你引擎真实 key）
            mean_wait = 0.0
            wt = out["wait_days_sum"]              # [K*B]
            wt = wt.view(K, B)
            mean_wait = float(wt.float().mean().item())
            mean_wait = mean_wait / float(N)


            # 换模个数：全体个体均值（如果 key 不匹配，请改成你引擎真实 key）
            mean_switch = 0.0
            for key in ["switch_cnt", "switch_count", "changeover_cnt", "setup_cnt", "model_switch_cnt"]:
                if key in out and torch.is_tensor(out[key]):
                    sc = out[key]
                    sc = sc.view(K, B) if sc.dim() >= 1 else sc
                    mean_switch = float(sc.float().mean().item())
                    break

            budget = float(gen_idx) / float(max(1, generations - 1))

            # 你已经改过的 make_state_vec（8维）：
            s_vec = make_state_vec(
                best_improved=best_improved,
                improve_rate=improve_rate,
                mean_improved=mean_improved,
                stagnation=float(stagnation),
                budget=budget,
                mean_viols=mean_viols,
                mean_wait=mean_wait,
                mean_switch=mean_switch,
            )

            # -------- 选动作 --------
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

            # action -> (step1次数, step2次数, greedy是否启用)
            n_viol_times, n_base_times, use_greedy = ACTION_SPECS[a]

            # -------- 遗传操作：精英 -> 父代池 -> 交叉 -> 变异 -> 拼接 --------
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

                # 关键修改：step1/step2 次数由策略控制；step3 仍按“是否启用”逻辑（启用则只调用 1 次）
                children_flat = self._mutate_batch_gpu_masked(
                    children_flat,
                    Vmask_flat,
                    base.current_generation,
                    n_viol_times=int(n_viol_times),
                    n_base_times=int(n_base_times),
                    use_greedy=bool(use_greedy),
                    base_swap_prob=base_swap_prob,
                    greedy_prob=greedy_prob,
                )

                children = children_flat.view(K, num_children, N)
                pop = torch.cat([elites, children], dim=1)
            else:
                pop = elites.clone()

            base.population_tensor = pop
            base.current_generation += 1

            # 更新 best/stagnation（保持你原逻辑）
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

            # 暂存 transition（供下一代回填 reward）
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
                    f"action={a} (viol×{int(n_viol_times)}, base×{int(n_base_times)}, greedy={'1' if use_greedy else '0'}) "
                    f"avg_best_fit={avg_best_fit:.2f} gbest_fit={global_best:.2f} "
                    f"mean_viols={mean_viols:.2f} mean_wait={mean_wait:.2f} mean_switch={mean_switch:.2f} "
                    f"stg={stagnation}",
                    flush=True
                )

        # -------- 结束：提取 K 个最佳个体 --------
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

        # flush 最后一代 action 的 reward
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
    state_dim = 8
    action_dim = len(ACTION_SPECS)
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
CUDA_VISIBLE_DEVICES=1 python /home/preprocess/_funsearch/baseline/0EMBC/drl.py \
  --mode train \
  --coe_path "/home/preprocess/_funsearch/baseline/0EMBC/coe-位置级-子种群对齐.py" \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 4 --B 50 --generations 5000 --elite_size 5 \
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
  --deploy_runs 3 --deploy_deterministic\

'''
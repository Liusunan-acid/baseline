# # dr_alns_runner.py
# # DR-ALNS-style controller for your ALNS permutation scheduling
# # - Reuses PermState + destroy/repair + GPU engine from your ALNS.py
# # - Implements PPO with multi-discrete heads (destroy, repair, severity, temperature)

# import argparse
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataclasses import dataclass

# # ---- import from your existing ALNS.py ----
# from ALNS import (
#     load_user_module,
#     build_engine_via_optimizer,
#     PermState,
#     destroy_random_k,
#     destroy_wait_violate_guided,
#     repair_random_insert,
#     repair_type_cluster_insert,
#     repair_wait_violate_guided_insert,
# )

# # -------------------------
# # 1) DR-ALNS state features (7-dim, problem-agnostic)
# # -------------------------
# @dataclass
# class DRALNSStats:
#     it: int = 0
#     stagnation: int = 0
#     last_accepted: int = 0
#     last_improved: int = 0
#     last_best_improved: int = 0

# def make_state_vec(
#     best_improved: int,
#     current_accepted: int,
#     current_improved: int,
#     is_current_best: int,
#     cost_diff_best: float,
#     stagnation: int,
#     search_budget: float,
# ) -> np.ndarray:
#     # 7 features as described in DR-ALNS paper (Table 1), order aligned to their description
#     return np.array(
#         [
#             float(best_improved),
#             float(current_accepted),
#             float(current_improved),
#             float(is_current_best),
#             float(cost_diff_best),
#             float(stagnation),
#             float(search_budget),
#         ],
#         dtype=np.float32,
#     )

# # -------------------------
# # 2) Action mapping (multi-discrete)
# #    - destroy op id
# #    - repair op id
# #    - severity in {1..10}
# #    - temperature in {1..50} -> T in [0.1, 5.0]
# # -------------------------
# def severity_to_ratio(sev_1_10: int, min_ratio=0.02, max_ratio=0.20) -> float:
#     # map 1..10 to [min_ratio, max_ratio]
#     sev = int(sev_1_10)
#     sev = max(1, min(10, sev))
#     t = (sev - 1) / 9.0
#     return float(min_ratio + t * (max_ratio - min_ratio))

# def temp_action_to_T(a_1_50: int) -> float:
#     # 1..50 -> 0.1..5.0 (same mapping concept as DR-ALNS paper)
#     import numpy as np
#     a = int(a_1_50)
#     a = max(1, min(50, a))
#     Tmin = 1e4
#     Tmax = 1e7
#     t = (a - 1) / 49.0  # 0..1
#     logT = np.log10(Tmin) + t * (np.log10(Tmax) - np.log10(Tmin))
#     return float(10 ** logT)

# def sa_accept(delta: float, T: float, rng: np.random.Generator) -> bool:
#     # delta = cand_cost - cur_cost (minimization)
#     if delta <= 0:
#         return True
#     if T <= 1e-12:
#         return False
#     p = np.exp(-delta / T)
#     return bool(rng.random() < p)

# # -------------------------
# # 3) Destroy operators with severity
# # -------------------------
# def destroy_segment_with_ratio(state: PermState, rng: np.random.Generator, ratio: float) -> PermState:
#     s = state.copy()
#     N = len(s.perm)
#     if N <= 1:
#         s.removed = np.array([], dtype=np.int64)
#         return s
#     k = max(1, int(N * ratio))
#     k = min(k, N)
#     start = int(rng.integers(0, N))
#     end = min(N, start + k)
#     idx = np.arange(start, end, dtype=np.int64)
#     s.removed = s.perm[idx].copy()
#     mask = np.ones(N, dtype=bool)
#     mask[idx] = False
#     s.perm = s.perm[mask]
#     return s

# # We reuse your destroy_random_k / destroy_wait_violate_guided which already support ratio
# # - destroy_random_k(state, rng, ratio=...)
# # - destroy_wait_violate_guided(state, rng, ratio=...)

# DESTROY_OPS = [
#     ("segment", destroy_segment_with_ratio),
#     ("random_k", destroy_random_k),
#     ("wait_violate", destroy_wait_violate_guided),
# ]

# REPAIR_OPS = [
#     ("rand_insert", repair_random_insert),
#     ("type_cluster", repair_type_cluster_insert),
#     ("wait_violate_insert", repair_wait_violate_guided_insert),
# ]

# # -------------------------
# # 4) PPO policy (multi-head categorical) + value head
# # -------------------------
# class PolicyNet(nn.Module):
#     def __init__(self, state_dim: int, n_destroy: int, n_repair: int):
#         super().__init__()
#         hid = 128
#         self.body = nn.Sequential(
#             nn.Linear(state_dim, hid),
#             nn.Tanh(),
#             nn.Linear(hid, hid),
#             nn.Tanh(),
#         )
#         self.logits_destroy = nn.Linear(hid, n_destroy)
#         self.logits_repair = nn.Linear(hid, n_repair)
#         self.logits_sev = nn.Linear(hid, 10)      # 1..10
#         self.logits_temp = nn.Linear(hid, 50)     # 1..50
#         self.value = nn.Linear(hid, 1)

#     def forward(self, s: torch.Tensor):
#         x = self.body(s)
#         return (
#             self.logits_destroy(x),
#             self.logits_repair(x),
#             self.logits_sev(x),
#             self.logits_temp(x),
#             self.value(x).squeeze(-1),
#         )

# def sample_actions(logits_tuple, rng: np.random.Generator):
#     # sample from categorical distributions, return (a_destroy, a_repair, a_sev(1..10), a_temp(1..50), logp_sum)
#     logits_destroy, logits_repair, logits_sev, logits_temp = logits_tuple
#     d_dist = torch.distributions.Categorical(logits=logits_destroy)
#     r_dist = torch.distributions.Categorical(logits=logits_repair)
#     s_dist = torch.distributions.Categorical(logits=logits_sev)
#     t_dist = torch.distributions.Categorical(logits=logits_temp)

#     a_d = d_dist.sample()
#     a_r = r_dist.sample()
#     a_s0 = s_dist.sample()  # 0..9
#     a_t0 = t_dist.sample()  # 0..49

#     logp = d_dist.log_prob(a_d) + r_dist.log_prob(a_r) + s_dist.log_prob(a_s0) + t_dist.log_prob(a_t0)
#     # convert to python ints with the intended 1-based mapping for sev/temp
#     return int(a_d.item()), int(a_r.item()), int(a_s0.item()) + 1, int(a_t0.item()) + 1, logp

# # -------------------------
# # 5) Rollout buffer + GAE
# # -------------------------
# @dataclass
# class Rollout:
#     s: list
#     a: list
#     logp: list
#     r: list
#     v: list
#     done: list

# def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
#     adv = np.zeros_like(rewards, dtype=np.float32)
#     lastgaelam = 0.0
#     for t in reversed(range(len(rewards))):
#         nextnonterminal = 1.0 - float(dones[t])
#         nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
#         delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
#         lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
#         adv[t] = lastgaelam
#     ret = adv + values[:len(adv)]
#     return adv, ret

# # -------------------------
# # 6) One episode environment loop (your scheduling instance)
# # -------------------------
# def dr_alns_episode(engine, init_perm, policy: PolicyNet, rng: np.random.Generator, max_iters: int,
#                    min_ratio=0.02, max_ratio=0.20):
#     # init
#     cur = PermState(engine, init_perm.copy())
#     cur_cost = float(cur.objective())
#     best = cur
#     best_cost = cur_cost

#     stats = DRALNSStats(it=0, stagnation=0)

#     # initial state vector (all zeros except budget)
#     s_vec = make_state_vec(
#         best_improved=0,
#         current_accepted=0,
#         current_improved=0,
#         is_current_best=1,
#         cost_diff_best=0.0,
#         stagnation=0,
#         search_budget=0.0,
#     )

#     rollout = Rollout(s=[], a=[], logp=[], r=[], v=[], done=[])

#     for it in range(max_iters):
#         stats.it = it
#         s_t = torch.from_numpy(s_vec).float().unsqueeze(0)  # (1,7)

#         with torch.no_grad():
#             ld, lr, ls, lt, v = policy(s_t)
#             a_d, a_r, a_sev, a_temp, logp = sample_actions((ld, lr, ls, lt), rng)

#         ratio = severity_to_ratio(a_sev, min_ratio=min_ratio, max_ratio=max_ratio)
#         T = temp_action_to_T(a_temp)

#         # apply destroy+repair
#         d_name, d_op = DESTROY_OPS[a_d]
#         r_name, r_op = REPAIR_OPS[a_r]

#         destroyed = d_op(cur, rng, ratio=ratio) if d_name != "segment" else d_op(cur, rng, ratio)
#         cand = r_op(destroyed, rng)

#         cand_cost = float(cand.objective())
#         delta = cand_cost - cur_cost
#         accepted = int(sa_accept(delta, T, rng))

#         prev_cost = cur_cost
#         prev_best_cost = best_cost

#         if accepted:
#             cur = cand
#             cur_cost = cand_cost

#         best_improved = int(cand_cost < best_cost)
#         if best_improved:
#             best = cand
#             best_cost = cand_cost
#             stats.stagnation = 0
#         else:
#             stats.stagnation += 1

#         current_improved = int(accepted and (cur_cost < prev_cost))
#         is_current_best = int(abs(cur_cost - best_cost) <= 1e-9)

#         denom = abs(best_cost) if abs(best_cost) > 1e-9 else 1.0
#         cost_diff_best = float((cur_cost - best_cost) / denom)
#         search_budget = float((it + 1) / max_iters)

#         reward = 5.0 if best_improved else 0.0  # same shaped reward idea as DR-ALNS paper

#         # record transition
#         rollout.s.append(s_vec.copy())
#         rollout.a.append((a_d, a_r, a_sev, a_temp))
#         rollout.logp.append(float(logp.item()))
#         rollout.r.append(float(reward))
#         rollout.v.append(float(v.item()))
#         rollout.done.append(0.0)

#         # next state
#         s_vec = make_state_vec(
#             best_improved=best_improved,
#             current_accepted=accepted,
#             current_improved=current_improved,
#             is_current_best=is_current_best,
#             cost_diff_best=cost_diff_best,
#             stagnation=stats.stagnation,
#             search_budget=search_budget,
#         )

#     # mark done
#     rollout.done[-1] = 1.0
#     return best, best_cost, rollout

# # -------------------------
# # 7) PPO update
# # -------------------------
# def ppo_update(policy: PolicyNet, optimizer, rollout: Rollout,
#                gamma=0.99, lam=0.95, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
#                epochs=4, batch_size=64):
#     s = torch.from_numpy(np.stack(rollout.s)).float()
#     old_logp = torch.from_numpy(np.array(rollout.logp, dtype=np.float32))
#     rewards = np.array(rollout.r, dtype=np.float32)
#     values = np.array(rollout.v, dtype=np.float32)
#     dones = np.array(rollout.done, dtype=np.float32)

#     adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
#     adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#     adv_t = torch.from_numpy(adv).float()
#     ret_t = torch.from_numpy(ret).float()

#     # actions
#     a = np.array(rollout.a, dtype=np.int64)
#     a_d = torch.from_numpy(a[:, 0])
#     a_r = torch.from_numpy(a[:, 1])
#     a_s = torch.from_numpy(a[:, 2] - 1)  # back to 0..9
#     a_t = torch.from_numpy(a[:, 3] - 1)  # back to 0..49

#     n = s.shape[0]
#     idx = np.arange(n)

#     for _ in range(epochs):
#         np.random.shuffle(idx)
#         for st in range(0, n, batch_size):
#             mb = idx[st:st + batch_size]
#             mb_s = s[mb]
#             mb_oldlogp = old_logp[mb]
#             mb_adv = adv_t[mb]
#             mb_ret = ret_t[mb]

#             ld, lr, ls, lt, v = policy(mb_s)
#             d_dist = torch.distributions.Categorical(logits=ld)
#             r_dist = torch.distributions.Categorical(logits=lr)
#             s_dist = torch.distributions.Categorical(logits=ls)
#             t_dist = torch.distributions.Categorical(logits=lt)

#             mb_logp = (
#                 d_dist.log_prob(a_d[mb])
#                 + r_dist.log_prob(a_r[mb])
#                 + s_dist.log_prob(a_s[mb])
#                 + t_dist.log_prob(a_t[mb])
#             )

#             ratio = torch.exp(mb_logp - mb_oldlogp)
#             surr1 = ratio * mb_adv
#             surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
#             policy_loss = -torch.min(surr1, surr2).mean()

#             value_loss = ((v - mb_ret) ** 2).mean()

#             entropy = (d_dist.entropy() + r_dist.entropy() + s_dist.entropy() + t_dist.entropy()).mean()

#             loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

# # -------------------------
# # 8) Main (train + deploy)
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--user_code", type=str, default="测量时间full-GPU实验-Multi.py")
#     parser.add_argument("--device_file", type=str, default="/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx")
#     parser.add_argument("--patient_file", type=str, default="/home/preprocess/_funsearch/baseline/data/实验数据6.1small - 副本.xlsx")
#     parser.add_argument("--duration_file", type=str, default="/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx")

#     parser.add_argument("--train_episodes", type=int, default=30)
#     parser.add_argument("--iters_per_episode", type=int, default=1000)   # DR-ALNS论文在routing里常用100~200; 你可自行加
#     parser.add_argument("--seed", type=int, default=0)

#     parser.add_argument("--min_ratio", type=float, default=0.02)
#     parser.add_argument("--max_ratio", type=float, default=0.20)

#     args = parser.parse_args()
#     rng = np.random.default_rng(args.seed)

#     user_mod = load_user_module(args.user_code)
#     patients = user_mod.import_data(args.patient_file, args.duration_file)
#     machine_exam_map = user_mod.import_device_constraints(args.device_file)
#     engine, opt = build_engine_via_optimizer(user_mod, patients, machine_exam_map)

#     N = opt.N
#     init_perm = np.arange(N, dtype=np.int64)

#     policy = PolicyNet(state_dim=7, n_destroy=len(DESTROY_OPS), n_repair=len(REPAIR_OPS))
#     optimizer = optim.Adam(policy.parameters(), lr=3e-4)

#     best_global_cost = float("inf")
#     best_global_perm = None

#     # ---- train ----
#     for ep in range(args.train_episodes):
#         best_state, best_cost, rollout = dr_alns_episode(
#             engine,
#             init_perm,
#             policy,
#             rng,
#             max_iters=args.iters_per_episode,
#             min_ratio=args.min_ratio,
#             max_ratio=args.max_ratio,
#         )
#         ppo_update(policy, optimizer, rollout)

#         if best_cost < best_global_cost:
#             best_global_cost = best_cost
#             best_global_perm = best_state.perm.copy()

#         print(f"[ep {ep+1:03d}] episode_best_cost={best_cost:.3f} | global_best_cost={best_global_cost:.3f}")

#     # ---- deploy (use trained policy, one longer run) ----
#     deploy_iters = max(3 * args.iters_per_episode, 1000)
#     best_state, best_cost, _ = dr_alns_episode(
#         engine, init_perm, policy, rng, max_iters=deploy_iters,
#         min_ratio=args.min_ratio, max_ratio=args.max_ratio
#     )
#     best_perm = best_state.perm
#     best_fitness = -best_cost

#     print("\n=== DR-ALNS deploy finished ===")
#     print(f"N={N} | best_fitness={best_fitness:.4f}")

#     # map back to cid and export schedule
#     idx_to_cid = opt._idx_to_cid
#     best_patient_order = [idx_to_cid[i] for i in best_perm.tolist()]
#     final_system = opt.generate_schedule(best_patient_order)

#     os.makedirs("output_schedules", exist_ok=True)
#     filename = os.path.join("output_schedules", f"DR_ALNS_fit_{best_fitness:.0f}.xlsx")
#     user_mod.export_schedule(final_system, patients, filename)
#     print(f"✓ exported: {filename}")

# if __name__ == "__main__":
#     main()


# dr_alns_runner.py
# ------------------------------------------------------------
# DR-ALNS-style controller for your permutation scheduling ALNS
#
# Implements features (1)-(6):
# 1) Save PPO policy checkpoints (policy + optimizer + best perm/cost + args)
# 2) Resume training from checkpoint
# 3) Deploy-only mode (no training) using a saved checkpoint
# 4) Export best_global_perm found during training (more reliable than a single deploy run)
# 5) Compare best_global vs deploy best; export the best among them
# 6) Keep run args inside checkpoint for easy reproducibility
#
# ALNS source file path is configurable via --alns_path
# Default: /home/preprocess/_funsearch/baseline/0ALNS/ALNS.py
# ------------------------------------------------------------

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import importlib.util
from typing import Optional, Tuple, Dict, Any


# -------------------------
# 0) Load your ALNS.py by absolute path
# -------------------------
def load_alns_module(alns_path: str):
    if not os.path.exists(alns_path):
        raise FileNotFoundError(f"ALNS.py not found: {alns_path}")
    spec = importlib.util.spec_from_file_location("ALNS_mod", alns_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec from: {alns_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -------------------------
# 1) DR-ALNS state features (7-dim)
# -------------------------
@dataclass
class DRALNSStats:
    it: int = 0
    stagnation: int = 0


def make_state_vec(
    best_improved: int,
    current_accepted: int,
    current_improved: int,
    is_current_best: int,
    cost_diff_best: float,
    stagnation: int,
    search_budget: float,
) -> np.ndarray:
    return np.array(
        [
            float(best_improved),
            float(current_accepted),
            float(current_improved),
            float(is_current_best),
            float(cost_diff_best),
            float(stagnation),
            float(search_budget),
        ],
        dtype=np.float32,
    )


# -------------------------
# 2) Action mapping (multi-discrete)
# -------------------------
def severity_to_ratio(sev_1_10: int, min_ratio=0.02, max_ratio=0.20) -> float:
    sev = int(sev_1_10)
    sev = max(1, min(10, sev))
    t = (sev - 1) / 9.0
    return float(min_ratio + t * (max_ratio - min_ratio))


def temp_action_to_T(a_1_50: int, Tmin: float = 1e4, Tmax: float = 1e7) -> float:
    """
    Map discrete temperature action (1..50) to a real SA temperature T.
    Use log-spaced mapping for large-magnitude deltas.
    """
    a = int(a_1_50)
    a = max(1, min(50, a))
    t = (a - 1) / 49.0  # 0..1
    logT = np.log10(Tmin) + t * (np.log10(Tmax) - np.log10(Tmin))
    return float(10 ** logT)


def sa_accept(delta: float, T: float, rng: np.random.Generator) -> bool:
    # delta = cand_cost - cur_cost (minimization)
    if delta <= 0:
        return True
    if T <= 1e-12:
        return False
    p = float(np.exp(-delta / T))
    return bool(rng.random() < p)


# -------------------------
# 3) PPO policy (multi-head categorical) + value head
# -------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, n_destroy: int, n_repair: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.logits_destroy = nn.Linear(hidden, n_destroy)
        self.logits_repair = nn.Linear(hidden, n_repair)
        self.logits_sev = nn.Linear(hidden, 10)   # 1..10
        self.logits_temp = nn.Linear(hidden, 50)  # 1..50
        self.value = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor):
        x = self.body(s)
        return (
            self.logits_destroy(x),
            self.logits_repair(x),
            self.logits_sev(x),
            self.logits_temp(x),
            self.value(x).squeeze(-1),
        )


def sample_actions(logits_tuple):
    logits_destroy, logits_repair, logits_sev, logits_temp = logits_tuple
    d_dist = torch.distributions.Categorical(logits=logits_destroy)
    r_dist = torch.distributions.Categorical(logits=logits_repair)
    s_dist = torch.distributions.Categorical(logits=logits_sev)
    t_dist = torch.distributions.Categorical(logits=logits_temp)

    a_d = d_dist.sample()
    a_r = r_dist.sample()
    a_s0 = s_dist.sample()  # 0..9
    a_t0 = t_dist.sample()  # 0..49
    logp = d_dist.log_prob(a_d) + r_dist.log_prob(a_r) + s_dist.log_prob(a_s0) + t_dist.log_prob(a_t0)

    return int(a_d.item()), int(a_r.item()), int(a_s0.item()) + 1, int(a_t0.item()) + 1, logp


def argmax_actions(logits_tuple):
    logits_destroy, logits_repair, logits_sev, logits_temp = logits_tuple
    a_d = int(torch.argmax(logits_destroy, dim=-1).item())
    a_r = int(torch.argmax(logits_repair, dim=-1).item())
    a_s = int(torch.argmax(logits_sev, dim=-1).item()) + 1
    a_t = int(torch.argmax(logits_temp, dim=-1).item()) + 1

    d_dist = torch.distributions.Categorical(logits=logits_destroy)
    r_dist = torch.distributions.Categorical(logits=logits_repair)
    s_dist = torch.distributions.Categorical(logits=logits_sev)
    t_dist = torch.distributions.Categorical(logits=logits_temp)
    logp = (
        d_dist.log_prob(torch.tensor(a_d))
        + r_dist.log_prob(torch.tensor(a_r))
        + s_dist.log_prob(torch.tensor(a_s - 1))
        + t_dist.log_prob(torch.tensor(a_t - 1))
    )
    return a_d, a_r, a_s, a_t, logp


# -------------------------
# 4) Rollout buffer + GAE
# -------------------------
@dataclass
class Rollout:
    s: list
    a: list
    logp: list
    r: list
    v: list
    done: list


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values[: len(adv)]
    return adv, ret


# -------------------------
# 5) Environment loop (episode)
# -------------------------
def destroy_segment_with_ratio(state, rng: np.random.Generator, ratio: float):
    s = state.copy()
    N = len(s.perm)
    if N <= 1:
        s.removed = np.array([], dtype=np.int64)
        return s
    k = max(1, int(N * ratio))
    k = min(k, N)
    start = int(rng.integers(0, N))
    end = min(N, start + k)
    idx = np.arange(start, end, dtype=np.int64)
    s.removed = s.perm[idx].copy()
    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    s.perm = s.perm[mask]
    return s


def dr_alns_episode(
    engine,
    init_perm: np.ndarray,
    policy: PolicyNet,
    rng: np.random.Generator,
    max_iters: int,
    destroy_ops,
    repair_ops,
    min_ratio: float,
    max_ratio: float,
    Tmin: float,
    Tmax: float,
    reward_best: float = 5.0,
    greedy: bool = False,
) -> Tuple[Any, float, Rollout]:
    cur = PermState(engine, init_perm.copy())
    cur_cost = float(cur.objective())
    best = cur
    best_cost = cur_cost

    stats = DRALNSStats(it=0, stagnation=0)

    s_vec = make_state_vec(
        best_improved=0,
        current_accepted=0,
        current_improved=0,
        is_current_best=1,
        cost_diff_best=0.0,
        stagnation=0,
        search_budget=0.0,
    )

    rollout = Rollout(s=[], a=[], logp=[], r=[], v=[], done=[])

    for it in range(max_iters):
        stats.it = it
        s_t = torch.from_numpy(s_vec).float().unsqueeze(0)  # (1,7)

        with torch.no_grad():
            ld, lr, ls, lt, v = policy(s_t)
            if greedy:
                a_d, a_r, a_sev, a_temp, logp = argmax_actions((ld, lr, ls, lt))
            else:
                a_d, a_r, a_sev, a_temp, logp = sample_actions((ld, lr, ls, lt))

        ratio = severity_to_ratio(a_sev, min_ratio=min_ratio, max_ratio=max_ratio)
        T = temp_action_to_T(a_temp, Tmin=Tmin, Tmax=Tmax)

        d_name, d_op = destroy_ops[a_d]
        r_name, r_op = repair_ops[a_r]

        if d_name == "segment":
            destroyed = d_op(cur, rng, ratio)
        else:
            destroyed = d_op(cur, rng, ratio=ratio)

        cand = r_op(destroyed, rng)
        cand_cost = float(cand.objective())
        delta = cand_cost - cur_cost

        accepted = int(sa_accept(delta, T, rng))
        prev_cost = cur_cost

        if accepted:
            cur = cand
            cur_cost = cand_cost

        best_improved = int(cand_cost < best_cost)
        if best_improved:
            best = cand
            best_cost = cand_cost
            stats.stagnation = 0
        else:
            stats.stagnation += 1

        current_improved = int(accepted and (cur_cost < prev_cost))
        is_current_best = int(abs(cur_cost - best_cost) <= 1e-9)

        denom = abs(best_cost) if abs(best_cost) > 1e-9 else 1.0
        cost_diff_best = float((cur_cost - best_cost) / denom)
        search_budget = float((it + 1) / max_iters)

        # reward = float(reward_best) if best_improved else 0.0
        denom = abs(best_cost) if abs(best_cost) > 1e-9 else 1.0  # 你后面本来就这么算 cost_diff_best 用 :contentReference[oaicite:6]{index=6}
        step_improve = max(0.0, (prev_cost - cur_cost) / denom)   # 只奖励“本步接受后带来的改进”（密集、可归因）
        reward = float(step_improve) + (float(reward_best) if best_improved else 0.0)

        rollout.s.append(s_vec.copy())
        rollout.a.append((a_d, a_r, a_sev, a_temp))
        rollout.logp.append(float(logp.item()))
        rollout.r.append(float(reward))
        rollout.v.append(float(v.item()))
        rollout.done.append(0.0)

        s_vec = make_state_vec(
            best_improved=best_improved,
            current_accepted=accepted,
            current_improved=current_improved,
            is_current_best=is_current_best,
            cost_diff_best=cost_diff_best,
            stagnation=stats.stagnation,
            search_budget=search_budget,
        )

    if rollout.done:
        rollout.done[-1] = 1.0
    return best, best_cost, rollout


# -------------------------
# 6) PPO update
# -------------------------
def ppo_update(
    policy: PolicyNet,
    optimizer,
    rollout: Rollout,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    epochs=4,
    batch_size=64,
):
    s = torch.from_numpy(np.stack(rollout.s)).float()
    old_logp = torch.from_numpy(np.array(rollout.logp, dtype=np.float32))
    rewards = np.array(rollout.r, dtype=np.float32)
    values = np.array(rollout.v, dtype=np.float32)
    dones = np.array(rollout.done, dtype=np.float32)

    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    adv_t = torch.from_numpy(adv).float()
    ret_t = torch.from_numpy(ret).float()

    a = np.array(rollout.a, dtype=np.int64)
    a_d = torch.from_numpy(a[:, 0])
    a_r = torch.from_numpy(a[:, 1])
    a_s = torch.from_numpy(a[:, 2] - 1)  # 0..9
    a_t = torch.from_numpy(a[:, 3] - 1)  # 0..49

    n = s.shape[0]
    idx = np.arange(n)

    for _ in range(epochs):
        np.random.shuffle(idx)
        for st in range(0, n, batch_size):
            mb = idx[st : st + batch_size]
            mb_s = s[mb]
            mb_oldlogp = old_logp[mb]
            mb_adv = adv_t[mb]
            mb_ret = ret_t[mb]

            ld, lr, ls, lt, v = policy(mb_s)
            d_dist = torch.distributions.Categorical(logits=ld)
            r_dist = torch.distributions.Categorical(logits=lr)
            s_dist = torch.distributions.Categorical(logits=ls)
            t_dist = torch.distributions.Categorical(logits=lt)

            mb_logp = (
                d_dist.log_prob(a_d[mb])
                + r_dist.log_prob(a_r[mb])
                + s_dist.log_prob(a_s[mb])
                + t_dist.log_prob(a_t[mb])
            )

            ratio = torch.exp(mb_logp - mb_oldlogp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((v - mb_ret) ** 2).mean()
            entropy = (d_dist.entropy() + r_dist.entropy() + s_dist.entropy() + t_dist.entropy()).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # ALNS.py absolute path
    parser.add_argument(
        "--alns_path",
        type=str,
        default="/home/preprocess/_funsearch/baseline/0ALNS/ALNS.py",
        help="Absolute path to your ALNS.py",
    )

    # Your user code & data files (same as before)
    parser.add_argument("--user_code", type=str, default="测量时间full-GPU实验-Multi.py")
    parser.add_argument("--device_file", type=str, default="/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx")
    parser.add_argument("--patient_file", type=str, default="/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx")
    parser.add_argument("--duration_file", type=str, default="/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx")

    # Mode
    parser.add_argument("--mode", type=str, default="train", choices=["train", "deploy"])

    # Train config
    parser.add_argument("--train_episodes", type=int, default=30)
    parser.add_argument("--iters_per_episode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)

    # Destroy / repair ranges
    parser.add_argument("--min_ratio", type=float, default=0.02)
    parser.add_argument("--max_ratio", type=float, default=0.20)

    # Temperature mapping
    parser.add_argument("--Tmin", type=float, default=1e4)
    parser.add_argument("--Tmax", type=float, default=1e7)

    # PPO hyperparams (keep defaults reasonable)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch_size", type=int, default=64)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)

    # Checkpointing (1)(2)(6)
    parser.add_argument("--ckpt_path", type=str, default="dr_alns_ckpt.pt")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")

    # Deploy control (3)(4)(5)
    parser.add_argument("--deploy_iters", type=int, default=0, help="0 means 3*iters_per_episode or >=1000")
    parser.add_argument("--deploy_runs", type=int, default=1, help="run deploy multiple times and take the best")
    parser.add_argument("--deploy_greedy", action="store_true", help="use argmax actions (less randomness)")

    args = parser.parse_args()

    # Load ALNS module from absolute path
    alns_mod = load_alns_module(args.alns_path)

    # Import needed symbols from ALNS.py
    load_user_module = alns_mod.load_user_module
    build_engine_via_optimizer = alns_mod.build_engine_via_optimizer
    global PermState
    PermState = alns_mod.PermState

    destroy_random_k = alns_mod.destroy_random_k
    destroy_wait_violate_guided = alns_mod.destroy_wait_violate_guided
    repair_random_insert = alns_mod.repair_random_insert
    repair_type_cluster_insert = alns_mod.repair_type_cluster_insert
    repair_wait_violate_guided_insert = alns_mod.repair_wait_violate_guided_insert

    # RNG / seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load user module and build engine
    user_mod = load_user_module(args.user_code)
    patients = user_mod.import_data(args.patient_file, args.duration_file)
    machine_exam_map = user_mod.import_device_constraints(args.device_file)
    engine, opt = build_engine_via_optimizer(user_mod, patients, machine_exam_map)

    N = int(opt.N)
    init_perm = np.arange(N, dtype=np.int64)

    # Define operators
    destroy_ops = [
        ("segment", destroy_segment_with_ratio),
        ("random_k", destroy_random_k),
        ("wait_violate", destroy_wait_violate_guided),
    ]
    repair_ops = [
        ("rand_insert", repair_random_insert),
        ("type_cluster", repair_type_cluster_insert),
        ("wait_violate_insert", repair_wait_violate_guided_insert),
    ]

    # Create policy & optimizer
    policy = PolicyNet(state_dim=7, n_destroy=len(destroy_ops), n_repair=len(repair_ops))
    optimizer = optim.Adam(policy.parameters(), lr=args.ppo_lr)

    # (2) Resume logic
    start_ep = 0
    best_global_cost = float("inf")
    best_global_perm: Optional[np.ndarray] = None

    if args.resume and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state"])
        if "optim_state" in ckpt and args.mode == "train":
            optimizer.load_state_dict(ckpt["optim_state"])
        start_ep = int(ckpt.get("episode", 0))
        best_global_cost = float(ckpt.get("best_global_cost", float("inf")))
        bgp = ckpt.get("best_global_perm", None)
        if bgp is not None:
            best_global_perm = np.array(bgp, dtype=np.int64)
        print(f"✓ Resumed from {args.ckpt_path} @ episode={start_ep}, best_global_cost={best_global_cost}")

    # -------------------------
    # TRAIN
    # -------------------------
    if args.mode == "train":
        for ep in range(start_ep, start_ep + args.train_episodes):
            best_state, best_cost, rollout = dr_alns_episode(
                engine=engine,
                init_perm=init_perm,
                policy=policy,
                rng=rng,
                max_iters=args.iters_per_episode,
                destroy_ops=destroy_ops,
                repair_ops=repair_ops,
                min_ratio=args.min_ratio,
                max_ratio=args.max_ratio,
                Tmin=args.Tmin,
                Tmax=args.Tmax,
                reward_best=5.0,
                greedy=False,
            )

            ppo_update(
                policy=policy,
                optimizer=optimizer,
                rollout=rollout,
                gamma=args.gamma,
                lam=args.lam,
                clip_eps=args.clip_eps,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                epochs=args.ppo_epochs,
                batch_size=args.ppo_batch_size,
            )

            if best_cost < best_global_cost:
                best_global_cost = float(best_cost)
                best_global_perm = best_state.perm.copy()

            print(f"[ep {ep+1:03d}] episode_best_cost={best_cost:.3f} | global_best_cost={best_global_cost:.3f}", flush=True)

            # (1)(6) save checkpoint
            if args.save_every > 0 and ((ep + 1) % args.save_every == 0):
                ckpt = {
                    "episode": ep + 1,
                    "policy_state": policy.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_global_cost": best_global_cost,
                    "best_global_perm": None if best_global_perm is None else best_global_perm.tolist(),
                    "args": vars(args),
                }
                torch.save(ckpt, args.ckpt_path)
                print(f"✓ saved ckpt: {args.ckpt_path}", flush=True)

    # -------------------------
    # DEPLOY (3)(4)(5)
    # -------------------------
    # determine deploy iters
    deploy_iters = args.deploy_iters
    if deploy_iters <= 0:
        deploy_iters = max(3 * args.iters_per_episode, 10000)

    best_deploy_cost = float("inf")
    best_deploy_perm: Optional[np.ndarray] = None

    for run_i in range(max(1, args.deploy_runs)):
        best_state, best_cost, _ = dr_alns_episode(
            engine=engine,
            init_perm=init_perm,
            policy=policy,
            rng=rng,
            max_iters=deploy_iters,
            destroy_ops=destroy_ops,
            repair_ops=repair_ops,
            min_ratio=args.min_ratio,
            max_ratio=args.max_ratio,
            Tmin=args.Tmin,
            Tmax=args.Tmax,
            reward_best=5.0,
            greedy=bool(args.deploy_greedy),
        )
        if best_cost < best_deploy_cost:
            best_deploy_cost = float(best_cost)
            best_deploy_perm = best_state.perm.copy()
        print(f"[deploy {run_i+1}/{args.deploy_runs}] best_cost={best_cost:.3f} | best_deploy_cost={best_deploy_cost:.3f}", flush=True)

    # Choose final perm: min(best_global, best_deploy)
    final_perm = best_deploy_perm
    final_cost = best_deploy_cost

    if best_global_perm is not None and best_global_cost < final_cost:
        final_perm = best_global_perm
        final_cost = best_global_cost

    if final_perm is None:
        raise RuntimeError("No final_perm found. (Did you run train or deploy correctly?)")

    final_fitness = -float(final_cost)  # because cost = -fitness (assuming your PermState.objective uses -fit)
    print("\n=== DR-ALNS finished ===")
    print(f"N={N} | final_cost={final_cost:.4f} | final_fitness={final_fitness:.4f}", flush=True)

    # Export schedule
    idx_to_cid = opt._idx_to_cid
    best_patient_order = [idx_to_cid[i] for i in final_perm.tolist()]
    final_system = opt.generate_schedule(best_patient_order)

    os.makedirs("output_schedules", exist_ok=True)
    filename = os.path.join("output_schedules", f"DR_ALNS_fit_{final_fitness:.0f}.xlsx")
    user_mod.export_schedule(final_system, patients, filename)
    print(f"✓ exported: {filename}", flush=True)


if __name__ == "__main__":
    main()



# #CUDA_VISIBLE_DEVICES=3 /home/preprocess/.conda/envs/fastsurfer_gpu/bin/python /home/preprocess/_funsearch/baseline/0DR-ALNS/dr_alns_runner.py --train_episodes 3 --iters_per_episode 1000

#训练并保存
# CUDA_VISIBLE_DEVICES=3 \
# nohup /home/preprocess/.conda/envs/fastsurfer_gpu/bin/python /home/preprocess/_funsearch/baseline/0DR-ALNS/dr_alns_runner.py > DR-ALNS.log 2>&1 \
#   --mode train \
#   --alns_path /home/preprocess/_funsearch/baseline/0ALNS/ALNS.py \
#   --train_episodes 300 --iters_per_episode 10000 \
#   --ckpt_path dr_alns_ckpt.pt --save_every 1
#从 checkpoint 接着训练
# CUDA_VISIBLE_DEVICES=3 /home/preprocess/.conda/envs/fastsurfer_gpu/bin/python /home/preprocess/_funsearch/baseline/0DR-ALNS/dr_alns_runner.py \
#   --mode train --resume \
#   --alns_path /home/preprocess/_funsearch/baseline/0ALNS/ALNS.py \
#   --ckpt_path dr_alns_ckpt.pt \
#   --train_episodes 30 --iters_per_episode 300
#只部署（不训练），直接导出结果
# CUDA_VISIBLE_DEVICES=3 /home/preprocess/.conda/envs/fastsurfer_gpu/bin/python /home/preprocess/_funsearch/baseline/0DR-ALNS/dr_alns_runner.py \
#   --mode deploy --resume \
#   --alns_path /home/preprocess/_funsearch/baseline/0ALNS/ALNS.py \
#   --ckpt_path dr_alns_ckpt.pt \
#   --deploy_runs 5 --deploy_greedy




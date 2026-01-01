
# -*- coding: utf-8 -*-
"""
DRL-COE Runner v4 (GNN 编码解图 -> PPO 选择 3-bit 算子 mask)
-------------------------------------------------------
目标：
- 尽量对齐你现有 coe-位置级-子种群对齐.py 的 MultiRunOptimizer.evolve_gpu 主流程
- 仅在“每代 children 生成前”插入：用 GNN 编码当前最优解（藕状图 + 稀疏跨边），再让 policy 选 3-bit mask
- 尽量复用 COE 中已有的 GPU 引擎字段与函数：fitness_batch(return_assignment=True)、_device_violate、_special_violates 等

参考范式：GRLOS (Graph Reinforcement Learning for Operator Selection) 思路
- Johnn et al., Computers & Operations Research, 2024
- 官方仓库：SYU-NING/grl-alns-framework

使用：
  python drl_coe_runner_v4_gnn.py --coe_path coe-位置级-子种群对齐.py --mode train ...
  python drl_coe_runner_v4_gnn.py --mode deploy --resume ...

重要说明：
- 为了不让构图开销压垮搜索，默认每 decision_every 代才“重算一次图 + 选一次 action”，中间沿用上次 action
- 图构造基于“解码后的排班结构”：每一天每台机器一条链（藕节），再加少量跨边（同 main_exam_type / 同规则标签）
- 不依赖 torch_geometric；GNN 使用 torch.sparse.mm 的轻量 GCN

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
import torch.nn.functional as F


# -------------------------
# 0) 动态加载你的 COE 文件
# -------------------------
def load_user_module(py_path: str):
    spec = importlib.util.spec_from_file_location("user_coe_module", "/home/preprocess/_funsearch/baseline/0EMBC/coe-位置级-子种群对齐.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 coe 模块: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -------------------------
# 1) 动作空间：3-bit mask
#    [violations, base_swap, greedy_cluster]
# -------------------------
ACTION_MASKS = [
    (1, 1, 1),  # 0 -> 111
    (1, 1, 0),  # 1 -> 110
    (1, 0, 1),  # 2 -> 101
    (1, 0, 0),  # 3 -> 100
    (0, 1, 1),  # 4 -> 011
    (0, 1, 0),  # 5 -> 010
    (0, 0, 1),  # 6 -> 001
    (0, 0, 0),  # 7 -> 000
]


# -------------------------
# 2) PPO rollout buffer
# -------------------------
@dataclass
class Rollout:
    s: List[np.ndarray]
    a: List[int]
    logp: List[float]
    r: List[float]
    v: List[float]
    done: List[bool]


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    epochs: int = 4,
    batch_size: int = 64,
):
    """
    标准 PPO(GAE) 更新。rollout.s 是 numpy 的 state 向量（已包含 GNN pooling 输出）。
    """
    device = next(policy.parameters()).device
    S = torch.from_numpy(np.asarray(rollout.s, dtype=np.float32)).to(device)
    A = torch.tensor(rollout.a, dtype=torch.long, device=device)
    OLD_LOGP = torch.tensor(rollout.logp, dtype=torch.float32, device=device)
    R = torch.tensor(rollout.r, dtype=torch.float32, device=device)
    V = torch.tensor(rollout.v, dtype=torch.float32, device=device)
    DONE = torch.tensor(rollout.done, dtype=torch.float32, device=device)

    # GAE
    T = len(rollout.r)
    adv = torch.zeros(T, device=device)
    lastgaelam = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - DONE[t]
        delta = R[t] + gamma * next_value * nonterminal - V[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
        next_value = V[t]
    ret = adv + V
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    idx = torch.arange(T, device=device)
    for _ in range(epochs):
        perm = idx[torch.randperm(T)]
        for start in range(0, T, batch_size):
            mb = perm[start:start + batch_size]
            logits, vpred = policy(S[mb])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(A[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - OLD_LOGP[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (ret[mb] - vpred).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()


# -------------------------
# 3) 轻量 GNN：稀疏 GCN + pooling
# -------------------------
class SparseGCNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        A_norm: sparse [N, N] (torch.sparse_coo_tensor) 归一化邻接
        """
        neigh = torch.sparse.mm(A_norm, x)  # [N,D]
        out = self.lin_self(x) + self.lin_neigh(neigh)
        return F.relu(out)


class GraphEncoder(nn.Module):
    """
    输入：节点的 categorical + continuous 特征 + 稀疏邻接
    输出：
      - g: pooled graph embedding [Dg]
      - h: node embeddings [N, Dg]
    """
    def __init__(
        self,
        num_machines: int = 8,
        num_weekdays: int = 7,
        num_types: int = 512,
        emb_dim: int = 16,
        cont_dim: int = 12,
        hidden_dim: int = 128,
        gnn_layers: int = 2,
    ):
        super().__init__()
        self.machine_emb = nn.Embedding(num_machines, emb_dim)
        self.weekday_emb = nn.Embedding(num_weekdays, emb_dim)
        self.type_emb = nn.Embedding(num_types, emb_dim)

        in_dim = emb_dim * 3 + cont_dim
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([SparseGCNLayer(hidden_dim) for _ in range(gnn_layers)])
        self.out_dim = hidden_dim

    def forward(
        self,
        machine_idx: torch.Tensor,   # [N]
        weekday_idx: torch.Tensor,   # [N]
        type_idx: torch.Tensor,      # [N]
        cont_feat: torch.Tensor,     # [N, cont_dim]
        A_norm: torch.Tensor,        # sparse [N,N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([
            self.machine_emb(machine_idx),
            self.weekday_emb(weekday_idx),
            self.type_emb(type_idx),
            cont_feat,
        ], dim=1)
        h = self.in_proj(x)
        for layer in self.layers:
            h = h + layer(h, A_norm)  # residual
        # pooling：mean + max
        g_mean = h.mean(dim=0)
        g_max = h.max(dim=0).values
        g = torch.cat([g_mean, g_max], dim=0)  # [2D]
        return g, h


class GNNPolicyNet(nn.Module):
    """
    policy(s) 输出离散动作 logits + value
    s = [scalar_state || pooled_graph_embedding]
    """
    def __init__(self, scalar_dim: int, action_dim: int = 8, hidden: int = 256,
                 gnn_cfg: Optional[dict] = None):
        super().__init__()
        gnn_cfg = gnn_cfg or {}
        self.encoder = GraphEncoder(**gnn_cfg)
        g_dim = self.encoder.out_dim * 2  # mean+max

        self.body = nn.Sequential(
            nn.Linear(scalar_dim + g_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, state_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(state_vec)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def encode_graph(self, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        g, _ = self.encoder(
            machine_idx=graph["machine_idx"],
            weekday_idx=graph["weekday_idx"],
            type_idx=graph["type_idx"],
            cont_feat=graph["cont_feat"],
            A_norm=graph["A_norm"],
        )
        return g


# -------------------------
# 4) 解图构造：藕状链 + 稀疏跨边
# -------------------------
def _build_sparse_adj(num_nodes: int, edges: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    edges: [E,2] numpy int64，节点索引 0..N-1
    返回 A_norm: torch.sparse_coo_tensor [N,N]，对称 + self-loop + D^{-1/2} A D^{-1/2}
    """
    if edges.size == 0:
        # 退化：仅 self-loop
        idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(num_nodes, device=device)
        A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
        return A

    # 对称化
    src = edges[:, 0]
    dst = edges[:, 1]
    und_src = np.concatenate([src, dst, np.arange(num_nodes)], axis=0)
    und_dst = np.concatenate([dst, src, np.arange(num_nodes)], axis=0)

    indices = torch.tensor(np.stack([und_src, und_dst], axis=0), device=device, dtype=torch.long)
    values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
    A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    # degree
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)  # [N]
    deg_inv_sqrt = deg.pow(-0.5)
    # 归一化：对 sparse values 做 D^{-1/2} A D^{-1/2}
    idx0 = A.indices()[0]
    idx1 = A.indices()[1]
    norm_vals = deg_inv_sqrt[idx0] * A.values() * deg_inv_sqrt[idx1]
    A_norm = torch.sparse_coo_tensor(A.indices(), norm_vals, A.shape).coalesce()
    return A_norm


def build_solution_graph_from_best(
    base_opt,
    best_perm: torch.Tensor,           # [N] on GPU
    k_type: int = 4,
    k_rule: int = 4,
    max_type_edges_per_group: int = 20000,
    max_rule_edges_per_group: int = 20000,
) -> Dict[str, torch.Tensor]:
    """
    复用 COE GPU 引擎：从 best_perm 解码到 assigned_day/machine/weekday/order，
    构造“藕状链 + 稀疏跨边”的图，并输出给 GNN 的张量字典。

    返回 dict 张量都放在 best_perm.device 上（通常是 CUDA）。
    """
    engine = base_opt._gpu_engine
    device = best_perm.device
    N = best_perm.numel()

    # 1) 解码：用 fitness_batch(return_assignment=True) 取结构字段
    out = engine.fitness_batch(best_perm.view(1, N), return_assignment=True)
    day = out["assigned_day"][0]            # [N], int
    mach = out["assigned_machine"][0]       # [N], int
    # weekday 可能不在 out 里（你的 COE fitness_batch 没返回 weekday），做 fallback
    wd_batch = out.get("weekday", None)  # dict.get 可避免 KeyError :contentReference[oaicite:0]{index=0}
    if wd_batch is None:
        # COE 内部实际就是 (start_weekday + assigned_day) % 7
        start_wd = getattr(engine, "start_weekday", 0)
        if not torch.is_tensor(start_wd):
            start_wd = torch.tensor(int(start_wd), device=day.device, dtype=day.dtype)
        wd = (start_wd + day) % 7   # day 是 out["assigned_day"][0]
    else:
        wd = wd_batch[0]
    order = out["order_in_machine"][0]      # [N], int (within bin)

    # 2) 取静态标签：类型/规则
    #    这些张量在 engine 中已经准备好了（与 perms 的 index 对齐）
    type_id = engine.patient_main_type_id.index_select(0, best_perm)  # [N]
    is_self = engine.is_self_selected.index_select(0, best_perm).to(torch.float32)
    has_heart = engine.has_heart.index_select(0, best_perm).to(torch.float32)
    has_angio = engine.has_angio.index_select(0, best_perm).to(torch.float32)
    has_contrast = engine.has_contrast.index_select(0, best_perm).to(torch.float32)

    # 3) 违规/局部结构提示（尽量复用 engine 内部函数）
    # 设备覆盖：是否该机器支持该检查
    dev_v = engine._device_violate(mach.view(1, N), best_perm.view(1, N))[0].to(torch.float32)  # [N]
    heart_v, angio_v, weekend_v = engine._special_violates(wd.view(1, N), mach.view(1, N), best_perm.view(1, N))
    heart_v = heart_v[0].to(torch.float32)
    angio_v = angio_v[0].to(torch.float32)
    weekend_v = weekend_v[0].to(torch.float32)

    # 机内相邻切换（用 day&mach 近似 same_bin）
    prev_type = torch.roll(type_id, 1, dims=0)
    same_bin = (day == torch.roll(day, 1, dims=0)) & (mach == torch.roll(mach, 1, dims=0))
    is_transition = (same_bin & (type_id != prev_type)).to(torch.float32)
    is_transition[0] = 0.0

    # 4) continuous features（归一化到 0..1 左右）
    day_f = day.to(torch.float32)
    mach_f = mach.to(torch.float32)
    order_f = order.to(torch.float32)
    # 防止除零
    day_norm = day_f / (day_f.max().clamp_min(1.0))
    order_norm = order_f / (order_f.max().clamp_min(1.0))
    # weekday 也可作为连续输入
    wd_norm = wd.to(torch.float32) / 6.0

    # cont_feat: 你可以在这里继续追加（例如：等待天数、slot 紧张度、局部 cost 等）
    cont_feat = torch.stack([
        day_norm,
        order_norm,
        wd_norm,
        is_self,
        has_heart,
        has_angio,
        has_contrast,
        dev_v,
        heart_v,
        angio_v,
        weekend_v,
        is_transition,
    ], dim=1)  # [N, 12]

    # 5) 构边（在 CPU 上做分组/排序，避免 GPU 上大量 Python 逻辑）
    day_cpu = day.detach().cpu().numpy().astype(np.int64)
    mach_cpu = mach.detach().cpu().numpy().astype(np.int64)
    order_cpu = order.detach().cpu().numpy().astype(np.int64)
    type_cpu = type_id.detach().cpu().numpy().astype(np.int64)
    # 规则：heart/angio/contrast 三个标签（你也可扩展更多规则）
    heart_cpu = (has_heart.detach().cpu().numpy() > 0.5)
    angio_cpu = (has_angio.detach().cpu().numpy() > 0.5)
    contrast_cpu = (has_contrast.detach().cpu().numpy() > 0.5)

    # 全局“排程顺序 key”：先按 day，再 machine，再 order
    sort_idx = np.lexsort((order_cpu, mach_cpu, day_cpu))  # [N]

    edges = []

    # (a) 藕状主干：同 day&machine 的相邻边
    for t in range(N - 1):
        i = int(sort_idx[t])
        j = int(sort_idx[t + 1])
        if day_cpu[i] == day_cpu[j] and mach_cpu[i] == mach_cpu[j]:
            edges.append((i, j))

    # (b) 同类型稀疏跨边：每个 type 组内按 schedule key 排序，连后续 k_type 个
    if k_type > 0:
        # 为了快速：先得到按 type 分组的 indices（用 dict）
        type_to_indices: Dict[int, List[int]] = {}
        for idx in sort_idx.tolist():  # 已经按 schedule 排好
            t_id = int(type_cpu[idx])
            type_to_indices.setdefault(t_id, []).append(int(idx))

        for t_id, inds in type_to_indices.items():
            if len(inds) <= 1:
                continue
            # 限制边数，防止某个大组爆炸
            max_edges = max_type_edges_per_group
            local_edges = 0
            L = len(inds)
            for u in range(L):
                for step in range(1, k_type + 1):
                    v = u + step
                    if v >= L:
                        break
                    edges.append((inds[u], inds[v]))
                    local_edges += 1
                    if local_edges >= max_edges:
                        break
                if local_edges >= max_edges:
                    break

    # (c) 同规则稀疏跨边：对每个规则标签做同样的“局部连边”
    def add_rule_edges(mask: np.ndarray, k: int, max_edges: int):
        if k <= 0:
            return
        inds = [int(i) for i in sort_idx.tolist() if mask[i]]
        if len(inds) <= 1:
            return
        local_edges = 0
        L = len(inds)
        for u in range(L):
            for step in range(1, k + 1):
                v = u + step
                if v >= L:
                    break
                edges.append((inds[u], inds[v]))
                local_edges += 1
                if local_edges >= max_edges:
                    return

    add_rule_edges(heart_cpu, k_rule, max_rule_edges_per_group)
    add_rule_edges(angio_cpu, k_rule, max_rule_edges_per_group)
    add_rule_edges(contrast_cpu, k_rule, max_rule_edges_per_group)

    edges_np = np.asarray(edges, dtype=np.int64)
    A_norm = _build_sparse_adj(N, edges_np, device=device)

    # categorical indices（embedding 输入）
    # machine 的 embedding size 由 policy 构造时决定；这里先 clamp 到非负
    machine_idx = mach.clamp_min(0).long()
    weekday_idx = wd.clamp_min(0).clamp_max(6).long()
    type_idx = type_id.clamp_min(0).long()

    return {
        "machine_idx": machine_idx,
        "weekday_idx": weekday_idx,
        "type_idx": type_idx,
        "cont_feat": cont_feat,
        "A_norm": A_norm,
    }


# -------------------------
# 5) 与你 v3 对齐的 masked mutation（复用原逻辑）
# -------------------------
@torch.no_grad()
def _mutate_batch_gpu_masked(
    base_opt,
    children_flat: torch.Tensor,   # [K*num_children, N]
    viol_flat: torch.Tensor,       # [K*num_children, N] bool
    op_mask: Tuple[int, int, int], # (violations, base_swap, greedy)
    current_gen: int,
    base_swap_prob: float = 0.95,
    greedy_prob: float = 0.5,
):
    """
    尽量调用 COE 中已有的 GPU 算子实现：
      - _mutate_step1_violations
      - _mutate_step2_base_swap
      - _mutate_step3_greedy_cluster
    并用 op_mask 控制是否启用。
    """
    engine = base_opt._gpu_engine
    # 这里复用你 COE 文件里的实现：MultiRunOptimizer._mutate_step*
    # 注意：这些方法在 coe-位置级-子种群对齐.py 中存在（你的 v3 已经能跑通）
    if op_mask[0] == 1:
        children_flat = base_opt._mutate_step1_violations(children_flat, viol_flat)
    if op_mask[1] == 1:
        children_flat = base_opt._mutate_step2_base_swap(children_flat, current_gen=current_gen, base_swap_prob=base_swap_prob)
    if op_mask[2] == 1:
        children_flat = base_opt._mutate_step3_greedy_cluster(children_flat, greedy_prob=greedy_prob)
    return children_flat


# -------------------------
# 6) 标量 state（保留你 v3 的特征）
# -------------------------
def build_scalar_state(
    budget: float,
    mean_best: float,
    global_best: float,
    mean_improved: float,
    is_run0_best: float,
    run0_gap: float,
    stagnation: float,
    mean_viols: float,
    feasible_rate: float,
) -> np.ndarray:
    """
    这里保持“轻量标量统计”的思路；GNN embedding 负责补充结构信息。
    你可以按需要继续加你 v3 的字段（例如：avg_fit、std、elite_gap 等）。
    """
    # 归一化/截断：避免数值尺度爆炸（你的 fitness 量级很大）
    def clip(x, lo=-1e6, hi=1e6):
        return float(np.clip(x, lo, hi))

    s = np.array([
        clip(budget, 0.0, 1.0),
        clip(mean_best / 1e8),
        clip(global_best / 1e8),
        clip(mean_improved),
        clip(is_run0_best),
        clip(run0_gap / 1e8),
        clip(stagnation / 1000.0, 0.0, 10.0),
        clip(mean_viols / 100.0, 0.0, 10.0),
        clip(feasible_rate, 0.0, 1.0),
    ], dtype=np.float32)
    return s


# -------------------------
# 7) DRL-COE 训练器：严格对齐 evolve_gpu 的骨架，仅插 action
# -------------------------
class DRLCOETrainer:
    def __init__(self, base_opt):
        self.base = base_opt

    @torch.no_grad()
    def evolve_gpu_drl(
        self,
        generations: int,
        elite_size: int,
        policy: GNNPolicyNet,
        train: bool,
        rollout: Optional[Rollout],
        deterministic: bool = False,
        force_action: Optional[int] = None,
        base_swap_prob: float = 0.95,
        greedy_prob: float = 0.5,
        # GNN/decision 控制
        decision_every: int = 10,
        k_type: int = 4,
        k_rule: int = 4,
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
        prev_mean_best = None
        stagnation = 0

        # 复用 action：降低构图开销
        last_action = 0
        last_graph_state: Optional[Dict[str, torch.Tensor]] = None

        for gen in range(generations):
            # -------------------------
            # 0) 评估（对齐：return_assignment=False）
            # -------------------------
            pop_flat = pop.view(base.total_pop_size, N)
            out = base._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            fitness = out["fitness"].view(K, B)

            viol_mask_flat = out.get("any_violate_mask_b_n", None)
            if viol_mask_flat is None:
                raise KeyError("fitness_batch 未返回 any_violate_mask_b_n（请检查 COE 引擎输出）")
            violate_mask = viol_mask_flat.view(K, B, N)

            # -------------------------
            # 1) 统计：best/mean/feasible
            # -------------------------
            topk_vals, topk_idx = torch.topk(fitness, k=B, largest=True, dim=1)
            best_fitness_per_run = topk_vals[:, 0]
            mean_best = float(best_fitness_per_run.mean().item())
            run0_best = float(best_fitness_per_run[0].item())
            best_run = int(torch.argmax(best_fitness_per_run).item())
            global_best = max(global_best, float(best_fitness_per_run[best_run].item()))
            run0_gap = run0_best - float(best_fitness_per_run[best_run].item())
            is_run0_best = 1.0 if best_run == 0 else 0.0

            # mean_viols / feas_rate
            viols_per_ind = violate_mask.to(torch.int32).sum(dim=2)  # [K,B]
            mean_viols = float(viols_per_ind.float().mean().item())
            feasible_rate = float((viols_per_ind == 0).float().mean().item())

            # improved
            if prev_mean_best is None:
                mean_improved = 0.0
            else:
                mean_improved = 1.0 if mean_best > prev_mean_best + 1e-6 else 0.0
            prev_mean_best = mean_best
            stagnation = stagnation + 1 if mean_improved < 0.5 else 0

            budget = (gen + 1) / float(generations)

            # -------------------------
            # 2) 构造状态 & 选动作（每 decision_every 代做一次）
            # -------------------------
            do_decide = (gen % max(1, decision_every) == 0)
            if do_decide:
                scalar = build_scalar_state(
                    budget=budget,
                    mean_best=mean_best,
                    global_best=global_best,
                    mean_improved=mean_improved,
                    is_run0_best=is_run0_best,
                    run0_gap=run0_gap,
                    stagnation=float(stagnation),
                    mean_viols=mean_viols,
                    feasible_rate=feasible_rate,
                )

                # 取“当前全局最优个体”构图（一个就够）
                best_idx_in_B = int(topk_idx[best_run, 0].item())
                best_perm = pop[best_run, best_idx_in_B, :]  # [N] on GPU

                # 构图（图张量放在 GPU 上）
                last_graph_state = build_solution_graph_from_best(
                    base_opt=base,
                    best_perm=best_perm,
                    k_type=k_type,
                    k_rule=k_rule,
                )

                # GNN pooling
                g = policy.encode_graph(last_graph_state)  # [2D]
                s_vec = np.concatenate([scalar, g.detach().cpu().numpy().astype(np.float32)], axis=0)

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

                last_action = a

                # rollout 记录（只在 decision 点记录一次；中间代沿用 action，不写入 rollout）
                if train and rollout is not None:
                    rollout.s.append(s_vec.astype(np.float32))
                    rollout.a.append(a)
                    rollout.logp.append(logp)
                    rollout.v.append(v)
                    # r/done 在末尾填
                    rollout.r.append(0.0)
                    rollout.done.append(False)

            a = last_action
            op_mask = ACTION_MASKS[a]

            # -------------------------
            # 3) 以下块：严格对齐你 v3/原 evolve_gpu 的父代选择 + 交叉 + 变异 + 更新
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

                p1 = parents.gather(1, p_idx1.unsqueeze(2).expand(K, num_children, N))
                p2 = parents.gather(1, p_idx2.unsqueeze(2).expand(K, num_children, N))

                # 交叉（调用你 base_opt 里已有的 GPU 交叉函数）
                P1_flat = p1.reshape(K * num_children, N)
                P2_flat = p2.reshape(K * num_children, N)
                children_flat = base._ordered_crossover_batch_gpu(P1_flat, P2_flat)
                children = children_flat.view(K, num_children, N)

                # 变异：mask 控制算子组合（尽量复用 COE 内函数）
                viol_sel = parent_viol.gather(1, p_idx1.unsqueeze(2).expand(K, num_children, N))
                viol_flat = viol_sel.reshape(K * num_children, N)

                children_flat = _mutate_batch_gpu_masked(
                    base_opt=base,
                    children_flat=children_flat,
                    viol_flat=viol_flat,
                    op_mask=op_mask,
                    current_gen=gen,
                    base_swap_prob=base_swap_prob,
                    greedy_prob=greedy_prob,
                )

                children = children_flat.view(K, num_children, N)
                new_pop = torch.cat([elites, children], dim=1)
            else:
                new_pop = elites

            pop = new_pop
            base.population_tensor = pop

            # -------------------------
            # 4) 奖励（仅在 decision 点把上一条 rollout 的 reward 填上）
            # -------------------------
            if train and rollout is not None and do_decide and len(rollout.r) > 0:
                # 用“mean_best 改进 + 可行性 shaping”作为 reward（你可在这里按需替换）
                # 这里只能用当前统计；上一 decision 的 reward 在下一次 decision 点写入
                # 简化：本次 reward 写到上一条（刚 append 的那条）上
                reward = 0.0
                # mean_best 越大越好
                reward += (mean_improved * 1.0)
                # mean_viols 越小越好
                reward += (0.5 if mean_viols < 1e-6 else 0.0)
                # feas_rate 越大越好
                reward += feasible_rate * 0.5
                rollout.r[-1] = float(reward)

            if verbose and (gen % 50 == 0 or gen == generations - 1):
                print(f"[gen ({gen}/{generations}] action={a} mask={op_mask[0]}{op_mask[1]}{op_mask[2]} "
                      f"avg_best_fit={mean_best:.2f} gbest_fit={global_best:.2f} "
                      f"mean_viols={mean_viols:.2f} feas_rate={feasible_rate:.2f} stg={stagnation}",
                      flush=True)

        # 末尾：标记 done
        if train and rollout is not None and len(rollout.done) > 0:
            rollout.done[-1] = True

        # 返回最终每个 run 的 best 个体（复用你 base_opt 输出格式）
        pop_flat = pop.view(base.total_pop_size, N)
        final_out = base._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
        final_fitness = final_out["fitness"].view(K, B)
        final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1)
        final_best_vals = final_best_vals.flatten()
        idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(K, 1, N)
        best_individuals_tensor = torch.gather(pop, 1, idx_expanded).squeeze(1)

        best_individuals_cpu = best_individuals_tensor.cpu()
        best_fitnesses_cpu = final_best_vals.cpu().tolist()

        final_results = []
        for rid in range(K):
            ind = best_individuals_cpu[rid].numpy().tolist()
            # COE 里 id=0 通常是 padding，请过滤掉
            individual_cids = [base._idx_to_cid[int(x)] for x in ind if int(x) > 0]
            final_results.append({
                "run_id": rid,
                "fitness": float(best_fitnesses_cpu[rid]),
                "individual_cids": individual_cids,
            })

        info = {
            "final_results": final_results,
            "global_best": float(max(best_fitnesses_cpu)),
        }
        return info, rollout


# -------------------------
# 8) 训练 / 部署入口
# -------------------------
def save_ckpt(path: str, policy: nn.Module, optim: torch.optim.Optimizer, step: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
    }, path)


def load_ckpt(path: str, policy: nn.Module, optim: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    optim.load_state_dict(ckpt["optim"])
    return int(ckpt.get("step", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coe_path", type=str, default="coe-位置级-子种群对齐.py")
    parser.add_argument("--mode", type=str, choices=["train", "deploy"], default="train")

    # data
    parser.add_argument("--patient_file", type=str, default="patients.xlsx")
    parser.add_argument("--duration_file", type=str, default="durations.xlsx")
    parser.add_argument("--device_file", type=str, default="devices.xlsx")

    # COE population
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--generations", type=int, default=5000)
    parser.add_argument("--elite_size", type=int, default=5)

    # PPO
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo_every", type=int, default=10)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--ppo_batch", type=int, default=64)
    parser.add_argument("--train_episodes", type=int, default=20)
    parser.add_argument("--start_ep", type=int, default=0)

    # mutation params (仍用你原来的；RL 只做组合选择)
    parser.add_argument("--base_swap_prob", type=float, default=0.95)
    parser.add_argument("--greedy_prob", type=float, default=0.5)

    # GNN/decision
    parser.add_argument("--decision_every", type=int, default=10)
    parser.add_argument("--k_type", type=int, default=4)
    parser.add_argument("--k_rule", type=int, default=4)
    parser.add_argument("--gnn_hidden", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--emb_dim", type=int, default=16)

    # ckpt
    parser.add_argument("--ckpt_path", type=str, default="drl_coe_gnn_ckpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=50)

    # deploy
    parser.add_argument("--deploy_runs", type=int, default=3)
    parser.add_argument("--deploy_deterministic", action="store_true")

    # debug: force action
    parser.add_argument("--force_action", type=int, default=-1, help=">=0 则强制每次决策 action 固定（用于对齐测试）")

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
    base_opt.initialize_population()

    trainer = DRLCOETrainer(base_opt)

    # scalar state dim 固定为 9（build_scalar_state）；graph embedding dim = 2*hidden
    scalar_dim = 9
    # num_types 估计：尽量从引擎里读到 E，否则给个较大上限
    base_opt._ensure_gpu_engine()
    eng = base_opt._gpu_engine
    num_types = int(getattr(eng, "_E", 256)) + 1

    policy = GNNPolicyNet(
        scalar_dim=scalar_dim,
        action_dim=8,
        hidden=256,
        gnn_cfg=dict(
            num_machines=16,      # 保险起见给大点
            num_weekdays=7,
            num_types=max(64, num_types),
            emb_dim=args.emb_dim,
            cont_dim=12,
            hidden_dim=args.gnn_hidden,
            gnn_layers=args.gnn_layers,
        )
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    step0 = 0
    if args.resume and os.path.exists(args.ckpt_path):
        step0 = load_ckpt(args.ckpt_path, policy, optim)
        print(f"[resume] loaded ckpt step={step0}", flush=True)

    if args.mode == "train":
        force_action = None if args.force_action < 0 else args.force_action

        start_ep = int(args.start_ep)
        for ep in range(start_ep, start_ep + int(args.train_episodes)):
            base_opt.initialize_population()

            rollout = Rollout(s=[], a=[], logp=[], r=[], v=[], done=[])
            info, rollout = trainer.evolve_gpu_drl(
                generations=args.generations,
                elite_size=args.elite_size,
                policy=policy,
                train=True,
                rollout=rollout,
                deterministic=False,
                force_action=force_action,
                base_swap_prob=args.base_swap_prob,
                greedy_prob=args.greedy_prob,
                decision_every=args.decision_every,
                k_type=args.k_type,
                k_rule=args.k_rule,
                verbose=True,
            )

            if len(rollout.s) > 4:
                ppo_update(
                    policy, optim, rollout,
                    epochs=args.ppo_epochs,
                    batch_size=args.ppo_batch,
                )

            if (ep + 1) % max(1, args.save_every) == 0:
                save_ckpt(args.ckpt_path, policy, optim, ep + 1)
                print(f"✓ saved ckpt: {args.ckpt_path} (ep={ep+1})", flush=True)

            print(f"[ep {ep+1:03d}] global_best={info['global_best']:.2f}", flush=True)

        save_ckpt(args.ckpt_path, policy, optim, start_ep + int(args.train_episodes))
        print("\n=== DRL-COE v4 (GNN) finished ===")
        print(f"last_ckpt={args.ckpt_path}", flush=True)

    else:
        # deploy：多次运行，取最优
        best_fit = -1e30
        best_results = None
        force_action = None if args.force_action < 0 else args.force_action

        for i in range(args.deploy_runs):
            base_opt.initialize_population()
            info, _ = trainer.evolve_gpu_drl(
                generations=args.generations,
                elite_size=args.elite_size,
                policy=policy,
                train=False,
                rollout=None,
                deterministic=args.deploy_deterministic,
                force_action=force_action,
                base_swap_prob=args.base_swap_prob,
                greedy_prob=args.greedy_prob,
                decision_every=args.decision_every,
                k_type=args.k_type,
                k_rule=args.k_rule,
                verbose=True,
            )
            final_results = info["final_results"]
            run_best = max(final_results, key=lambda d: d["fitness"])
            if run_best["fitness"] > best_fit:
                best_fit = run_best["fitness"]
                best_results = final_results
            print(f"[deploy {i+1}/{args.deploy_runs}] run_best={run_best['fitness']:.2f} | best_fit={best_fit:.2f}", flush=True)

        print("\n=== DRL-COE v4 (GNN) deploy finished ===")
        print(f"best_fit={best_fit:.2f}", flush=True)

        if best_results is not None:
            os.makedirs("output_schedules", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            for res in best_results:
                rid = res["run_id"]
                system = base_opt.generate_schedule(res["individual_cids"])
                out_path = os.path.join("output_schedules", f"schedule_run{rid}_{ts}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    import json
                    json.dump(system, f, ensure_ascii=False, indent=2)
                print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()


'''
# 训练
CUDA_VISIBLE_DEVICES=3 \
python /home/preprocess/_funsearch/baseline/0EMBC/gnn.py \
  --coe_path /home/preprocess/_funsearch/baseline/0EMBC/coe-位置级-子种群对齐.py \
  --mode train \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 --generations 5000 --elite_size 5 \
  --train_episodes 20 \
  --decision_every 1 \
  --k_type 4 --k_rule 4 \
  --ckpt_path drl_coe_gnn_ckpt.pt

# 部署（用训练好的 ckpt，deterministic 可打开）
python drl_coe_runner_v4_gnn.py \
  --coe_path coe-位置级-子种群对齐.py \
  --mode deploy --resume \
  --ckpt_path drl_coe_gnn_ckpt.pt \
  --deploy_runs 3 --deploy_deterministic

'''
# -*- coding: utf-8 -*-
"""
测量时间 full-GPU 实验 - PSO 版本（最小侵入）
- 通过 importlib 动态加载原 Multi 脚本
- 复用其数据导入、约束、GPU 适应度引擎、排程导出
- 使用 Random-Keys 离散 PSO：连续 pos -> argsort -> 患者排列
"""

from __future__ import annotations
from typing import List, Any, Dict
import os
import time
import traceback
import multiprocessing
import importlib.util

# =========================
# 1) 动态加载原 Multi 脚本
# =========================

def load_multi_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    multi_path = os.path.join(current_dir, "测量时间full-GPU实验-Multi.py")

    if not os.path.exists(multi_path):
        raise FileNotFoundError(f"找不到原始 Multi 文件: {multi_path}")

    spec = importlib.util.spec_from_file_location("multi_full_gpu", multi_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法为 Multi 文件创建 import spec")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


multi = load_multi_module()

# 复用 multi 中的关键对象/函数
torch = multi.torch

import_data = multi.import_data
import_device_constraints = multi.import_device_constraints

MachineSchedule = multi.MachineSchedule
SchedulingSystem = multi.SchedulingSystem

MultiRunOptimizer = multi.MultiRunOptimizer


export_schedule = multi.export_schedule

# 使用同一套设备/精度定义，保证完全对齐
DEVICE = multi.DEVICE
DTYPE_LONG = multi.DTYPE_LONG
DTYPE_FLOAT = multi.DTYPE_FLOAT


# ======================================
# 2) PSO Optimizer：继承并复用 MultiRun
# ======================================

class MultiRunPSOOptimizer(MultiRunOptimizer):
    """
    基于你现有 MultiRunOptimizer 的 PSO 版本：
    - 继承 __init__ 保持患者预处理、字段结构不变
    - 复用 _ensure_gpu_engine / _tensor_row_to_cids / generate_schedule
    - 新增 initialize_particles / evolve_pso
    """

    def __init__(self,
                 patients,
                 machine_exam_map,
                 num_parallel_runs: int,
                 pop_size_per_run: int,
                 block_start_date=None):
        super().__init__(patients, machine_exam_map,
                         num_parallel_runs, pop_size_per_run,
                         block_start_date=block_start_date)

        # PSO 状态
        self.pos = None  # [K, B, N], float
        self.vel = None  # [K, B, N], float

        self.pbest_pos = None
        self.pbest_fit = None

        self.gbest_pos = None  # [K, N]
        self.gbest_fit = None  # [K]

    def initialize_particles(self):
        """
        ✅ 新版初始化（按你的要求）：
        1) 直接复用 multi 的 initialize_population 规则生成 [K,B,N] 初始患者序列
        2) 将这些“排列”反解为 Random-Keys 的分数表 pos
        使得 argsort(pos) 还原出同一条序列
        3) vel 置零
        4) 用 GPU fitness 批量评估并初始化 pbest/gbest
        """
        self._ensure_gpu_engine()

        K, B, N = self.K, self.B, self.N

        # ---------- 1) 复用 multi 的块内随机初始化 ----------
        # 这一步会生成 self.population_tensor: [K, B, N]
        super().initialize_population()
        if self.population_tensor is None:
            raise RuntimeError("population_tensor 为空，无法初始化 PSO")

        pop_indices = self.population_tensor  # [K, B, N]，元素是患者 idx

        # ---------- 2) 排列 -> 分数表（均匀 rank 分数） ----------
        # rank_values[..., j] = j / N
        rank = torch.arange(N, device=DEVICE, dtype=DTYPE_FLOAT)
        rank_values = (rank / max(1, N)).view(1, 1, N).expand(K, B, N)  # [K,B,N]

        # pos[k,b, patient_idx] = 该 patient 在排列中的 rank/N
        # 用 scatter 按“患者索引”写入分数
        pos = torch.empty((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
        pos.scatter_(dim=2, index=pop_indices, src=rank_values)

        # 可选：极小扰动（防止数值极端情况下的排序不稳定）
        pos = pos + (torch.rand_like(pos) * (1e-6))

        self.pos = pos
        self.vel = torch.zeros((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)

        # ---------- 3) 用 pos 还原排列（应与 pop_indices 一致） ----------
        perms = torch.argsort(self.pos, dim=2)  # [K,B,N]

        # ---------- 4) 初评估 ----------
        perms_flat = perms.reshape(K * B, N)
        out = self._gpu_engine.fitness_batch(perms_flat, return_assignment=False)
        fit = out["fitness"].reshape(K, B)

        # ---------- 5) 初始化 pbest ----------
        self.pbest_pos = self.pos.clone()
        self.pbest_fit = fit.clone()

        # ---------- 6) 初始化 gbest（每个 run 独立） ----------
        best_vals, best_idx = torch.max(fit, dim=1)  # [K]
        idx_exp = best_idx.view(K, 1, 1).expand(K, 1, N)
        self.gbest_pos = torch.gather(self.pos, 1, idx_exp).squeeze(1)  # [K,N]
        self.gbest_fit = best_vals

        print(f"✓ PSO 初始化完成：已按 multi 规则生成 {K*B} 个初始序列并转换为分数表")
        return fit


    @torch.no_grad()
    def evolve_pso(self,
                   iters: int = 5000,
                   w: float = 0.7,
                   c1: float = 1.4,
                   c2: float = 1.4,
                   vmax: float = 0.2,
                   restart_every: int = 200,
                   restart_frac: float = 0.05,
                   log_every: int = 100):
        """
        完整 PSO 迭代（全 GPU）：
        - 连续 pos -> argsort -> 排列
        - 复用你 multi 的 fitness_batch
        - 更新 pbest/gbest
        - 速度/位置更新 + 限速
        - 周期性重启最差粒子防早熟

        返回：
        - results: List[Dict] (每个 run 的最优个体 cid 序列与 fitness)
        """
        self._ensure_gpu_engine()

        if self.pos is None:
            self.initialize_particles()

        K, B, N = self.K, self.B, self.N

        for t in range(iters):
            # 1) 连续 -> 排列
            perms = torch.argsort(self.pos, dim=2)
            perms_flat = perms.reshape(K * B, N)

            # 2) 评估
            out = self._gpu_engine.fitness_batch(perms_flat, return_assignment=False)
            fit = out["fitness"].reshape(K, B)

            # 3) 更新 pbest
            improve = fit > self.pbest_fit
            self.pbest_fit = torch.where(improve, fit, self.pbest_fit)

            improve_exp = improve.unsqueeze(2).expand(K, B, N)
            self.pbest_pos = torch.where(improve_exp, self.pos, self.pbest_pos)

            # 4) 更新 gbest（每个 run 独立）
            best_vals, best_idx = torch.max(self.pbest_fit, dim=1)  # [K]
            better_g = best_vals > self.gbest_fit

            self.gbest_fit = torch.where(better_g, best_vals, self.gbest_fit)

            idx_exp = best_idx.view(K, 1, 1).expand(K, 1, N)
            cand_gbest_pos = torch.gather(self.pbest_pos, 1, idx_exp).squeeze(1)

            better_g_exp = better_g.view(K, 1).expand(K, N)
            self.gbest_pos = torch.where(better_g_exp, cand_gbest_pos, self.gbest_pos)

            # 5) 速度/位置更新
            r1 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
            r2 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)

            gbest_expand = self.gbest_pos.unsqueeze(1).expand(K, B, N)

            self.vel = (
                w * self.vel
                + c1 * r1 * (self.pbest_pos - self.pos)
                + c2 * r2 * (gbest_expand - self.pos)
            )

            # 限速
            self.vel = torch.clamp(self.vel, -vmax, vmax)

            # 更新位置
            self.pos = self.pos + self.vel

            # 6) 周期性重启最差粒子
            if restart_every > 0 and (t + 1) % restart_every == 0 and restart_frac > 0:
                k_bad = max(1, int(B * restart_frac))
                worst_idx = torch.topk(fit, k=k_bad, largest=False, dim=1).indices  # [K,k_bad]

                worst_mask = torch.zeros((K, B), device=DEVICE, dtype=torch.bool)
                worst_mask.scatter_(1, worst_idx, True)

                worst_mask_exp = worst_mask.unsqueeze(2).expand(K, B, N)

                self.pos = torch.where(worst_mask_exp, torch.rand_like(self.pos), self.pos)
                self.vel = torch.where(worst_mask_exp, torch.zeros_like(self.vel), self.vel)

            # 7) 日志
            if log_every > 0 and (t + 1) % log_every == 0:
                avg_best = float(self.gbest_fit.mean().item())
                print(f"[PSO] Iter {t+1:5d}/{iters} | Avg gbest(K={K}): {avg_best:.4f}")

        # 8) 输出 K 个最优解
        final_perm_idx = torch.argsort(self.gbest_pos, dim=1)  # [K, N]

        results: List[Dict[str, Any]] = []
        for k in range(K):
            row = final_perm_idx[k].detach().cpu()
            cids = self._tensor_row_to_cids(row)

            results.append({
                "run_id": k,
                "fitness": float(self.gbest_fit[k].item()),
                "individual_cids": cids
            })

        return results
    



# =========================
# 3) PSO 主程序入口
# =========================

def main():
    try:
        # ================== 配置（可按 GA 对齐） ==================
        NUM_PARALLEL_RUNS = 10
        POP_SIZE_PER_RUN = 100

        # PSO 参数
        ITERS = 1000000
        W = 0.7
        C1 = 1.4
        C2 = 1.4
        VMAX = 0.2

        RESTART_EVERY = 200
        RESTART_FRAC = 0.05
        LOG_EVERY = 100
        # ==========================================================

        print(f"启动 PSO Megabatch 模式: K={NUM_PARALLEL_RUNS}, B={POP_SIZE_PER_RUN}")
        print(f"总 GPU 批量: {NUM_PARALLEL_RUNS * POP_SIZE_PER_RUN} 粒子")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

        for f in [patient_file, duration_file, device_constraint_file]:
            if not os.path.exists(f):
                print(f"❌ 错误：找不到文件 {f}")
                return
        print("✓ 所有数据文件均已找到。")

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)
        print(f"✓ 导入完成：患者数={len(patients)}")

        # 创建 PSO 优化器
        optimizer = MultiRunPSOOptimizer(
            patients,
            machine_exam_map,
            num_parallel_runs=NUM_PARALLEL_RUNS,
            pop_size_per_run=POP_SIZE_PER_RUN
        )

        # 初始化粒子
        t0_init = time.perf_counter()
        optimizer.initialize_particles()
        t_init = time.perf_counter() - t0_init
        print(f"✓ 已生成 {NUM_PARALLEL_RUNS} 个初始粒子群，耗时: {t_init:.4f}s")

        # PSO 迭代
        print(f"\n开始 PSO 迭代: iters={ITERS} ...")
        t0 = time.perf_counter()

        results = optimizer.evolve_pso(
            iters=ITERS,
            w=W, c1=C1, c2=C2, vmax=VMAX,
            restart_every=RESTART_EVERY,
            restart_frac=RESTART_FRAC,
            log_every=LOG_EVERY
        )

        # results = optimizer.evolve_pso(
        #     iters=ITERS,
        #     w_start=0.95,
        #     w_end=0.55,
        #     c1=C1,          # 你也可以直接写 1.8
        #     c2=C2,          # 你也可以直接写 1.15
        #     vmax=VMAX,

        #     restart_every=RESTART_EVERY,
        #     restart_frac=RESTART_FRAC,
        #     log_every=LOG_EVERY,

        #     use_lbest=True,
        #     repair_every=200,
        #     repair_candidates=4
        # )

        t_total = time.perf_counter() - t0
        print(f"\n✓ PSO 完成，总耗时: {t_total:.2f}s")

        # 统计与导出
        fitness_list = [r["fitness"] for r in results]
        avg_fitness = float(sum(fitness_list) / len(fitness_list))
        std_fitness = float((sum((x - avg_fitness) ** 2 for x in fitness_list) / len(fitness_list)) ** 0.5)
        min_fitness = float(min(fitness_list))
        max_fitness = float(max(fitness_list))

        print("\nPSO 多次并行运行结果统计：")
        print(f"  最佳适应度 (平均): {avg_fitness:.2f}")
        print(f"  最佳适应度 (标准差): {std_fitness:.2f}")
        print(f"  最佳适应度 (范围): {min_fitness:.2f} ... {max_fitness:.2f}")

        # 导出每个 run 的最优排程
        export_dir = os.path.join(current_dir, "PSO_results")
        os.makedirs(export_dir, exist_ok=True)

        for r in results:
            run_id = r["run_id"]
            individual = r["individual_cids"]

            system = optimizer.generate_schedule(individual)

            out_xlsx = os.path.join(export_dir, f"PSO_best_run_{run_id}.xlsx")
            export_schedule(system, patients, out_xlsx)

            print(f"✓ 导出 run {run_id} 最优排程: {out_xlsx}")

        print("\n所有 PSO 运行均已完成。")

    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()
    finally:
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()





# """
# 测量时间 full-GPU 实验 - PSO 版本（改进版：Lbest拓扑 + 动态权重）
# """

# from __future__ import annotations
# from typing import List, Any, Dict
# import os
# import time
# import traceback
# import multiprocessing
# import importlib.util
# import math

# # =========================
# # 1) 动态加载原 Multi 脚本 (保持不变)
# # =========================

# def load_multi_module():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     multi_path = os.path.join(current_dir, "测量时间full-GPU实验-Multi.py")

#     if not os.path.exists(multi_path):
#         raise FileNotFoundError(f"找不到原始 Multi 文件: {multi_path}")

#     spec = importlib.util.spec_from_file_location("multi_full_gpu", multi_path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError("无法为 Multi 文件创建 import spec")

#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# multi = load_multi_module()

# torch = multi.torch
# import_data = multi.import_data
# import_device_constraints = multi.import_device_constraints
# MachineSchedule = multi.MachineSchedule
# SchedulingSystem = multi.SchedulingSystem
# MultiRunOptimizer = multi.MultiRunOptimizer
# export_schedule = multi.export_schedule
# DEVICE = multi.DEVICE
# DTYPE_LONG = multi.DTYPE_LONG
# DTYPE_FLOAT = multi.DTYPE_FLOAT


# # ======================================
# # 2) PSO Optimizer：改进版
# # ======================================

# class MultiRunPSOOptimizer(MultiRunOptimizer):
#     def __init__(self,
#                  patients,
#                  machine_exam_map,
#                  num_parallel_runs: int,
#                  pop_size_per_run: int,
#                  block_start_date=None):
#         super().__init__(patients, machine_exam_map,
#                          num_parallel_runs, pop_size_per_run,
#                          block_start_date=block_start_date)

#         # PSO 状态
#         self.pos = None  # [K, B, N]
#         self.vel = None  # [K, B, N]
#         self.pbest_pos = None
#         self.pbest_fit = None
#         self.gbest_pos = None  # [K, N] (实际上这里存储的是每个Run的全局最优)
#         self.gbest_fit = None  # [K]

#     def initialize_particles(self):
#         """
#         初始化：生成初始解并映射到 Random Keys 连续空间
#         """
#         self._ensure_gpu_engine()
#         K, B, N = self.K, self.B, self.N

#         # 1) 生成初始离散排列 [K, B, N]
#         super().initialize_population()
#         if self.population_tensor is None:
#             raise RuntimeError("population_tensor 为空")
        
#         pop_indices = self.population_tensor

#         # 2) 映射到连续空间 [0, 1]
#         # logic: rank_values[..., j] = j / N
#         rank = torch.arange(N, device=DEVICE, dtype=DTYPE_FLOAT)
#         # 添加微小抖动，避免后续 argsort 不稳定
#         rank_values = (rank / max(1, N)).view(1, 1, N).expand(K, B, N)
        
#         # pos[k, b, index] = rank_score
#         self.pos = torch.empty((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
#         self.pos.scatter_(dim=2, index=pop_indices, src=rank_values)
        
#         # 叠加噪声，增加初始多样性
#         self.pos += (torch.rand_like(self.pos) * 0.05)
#         self.pos.clamp_(0.0, 1.0)

#         self.vel = torch.zeros((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
        
#         # 3) 评估
#         perms = torch.argsort(self.pos, dim=2)
#         perms_flat = perms.reshape(K * B, N)
#         out = self._gpu_engine.fitness_batch(perms_flat, return_assignment=False)
#         fit = out["fitness"].reshape(K, B)

#         # 4) 初始化 Pbest
#         self.pbest_pos = self.pos.clone()
#         self.pbest_fit = fit.clone()

#         # 5) 初始化 Gbest (Run内部最优)
#         best_vals, best_idx = torch.max(fit, dim=1)  # [K]
#         idx_exp = best_idx.view(K, 1, 1).expand(K, 1, N)
#         self.gbest_pos = torch.gather(self.pos, 1, idx_exp).squeeze(1)
#         self.gbest_fit = best_vals

#         print(f"✓ PSO 初始化完成 (K={K}, B={B})")
#         return fit

#     @torch.no_grad()
#     def evolve_pso(self,
#                    iters: int = 5000,
#                    w_start: float = 0.9,
#                    w_end: float = 0.4,
#                    c1: float = 2.0,      # 自我认知权重
#                    c2: float = 2.0,      # 社会认知权重
#                    vmax: float = 0.1,    # 最大速度限制 (Random Keys 空间通常是 0-1，0.1 已经很大了)
#                    use_lbest: bool = True, # 是否启用环形拓扑 (关键改进)
#                    mutation_prob: float = 0.01, # 扰动概率
#                    log_every: int = 100):
#         """
#         改进版 PSO 迭代
#         """
#         self._ensure_gpu_engine()
#         if self.pos is None:
#             self.initialize_particles()

#         K, B, N = self.K, self.B, self.N
        
#         # 预先生成 range 用于 Lbest 索引
#         # 我们构建一个环形索引，每个粒子的邻居是 i-1 和 i+1
#         # 这可以在 Tensor 上通过 roll 操作实现
        
#         print(f"🚀 开始进化: Lbest={use_lbest}, W={w_start}->{w_end}")

#         for t in range(iters):
#             # --- 1. 适应度评估 ---
#             # Random Keys: 连续 pos -> argsort -> 离散排列
#             perms = torch.argsort(self.pos, dim=2)
#             perms_flat = perms.reshape(K * B, N)
            
#             out = self._gpu_engine.fitness_batch(perms_flat, return_assignment=False)
#             fit = out["fitness"].reshape(K, B)

#             # --- 2. 更新 Pbest ---
#             improve = fit > self.pbest_fit
#             self.pbest_fit = torch.where(improve, fit, self.pbest_fit)
#             # 只有变好时才更新位置
#             mask_exp = improve.unsqueeze(2).expand(K, B, N)
#             self.pbest_pos = torch.where(mask_exp, self.pos, self.pbest_pos)

#             # --- 3. 更新 Gbest (记录每个群的历史最优) ---
#             current_best_vals, current_best_idx = torch.max(self.pbest_fit, dim=1) # [K]
#             better_g = current_best_vals > self.gbest_fit
            
#             self.gbest_fit = torch.where(better_g, current_best_vals, self.gbest_fit)
            
#             idx_exp = current_best_idx.view(K, 1, 1).expand(K, 1, N)
#             cand_gbest_pos = torch.gather(self.pbest_pos, 1, idx_exp).squeeze(1)
#             better_g_exp = better_g.view(K, 1).expand(K, N)
#             self.gbest_pos = torch.where(better_g_exp, cand_gbest_pos, self.gbest_pos)

#             # --- 4. 计算社会学习目标 (Social Target) ---
#             if use_lbest:
#                 # == 环形拓扑 (Ring Topology) ==
#                 # 邻居：左边 (roll 1) 和 右边 (roll -1)
#                 # 比较 Self, Left, Right 的 Pbest Fitness，取最好的作为 attractor
                
#                 left_fit = torch.roll(self.pbest_fit, shifts=1, dims=1)
#                 right_fit = torch.roll(self.pbest_fit, shifts=-1, dims=1)
                
#                 # 找到谁是邻域老大
#                 # is_left_best: left > self AND left > right
#                 # ...
#                 # 简单做法：构建一个 [K, B, 3] 的矩阵取 max
                
#                 # Stack fits: [K, B, 3] -> (self, left, right)
#                 fits_stack = torch.stack([self.pbest_fit, left_fit, right_fit], dim=2)
#                 best_neighbor_idx = torch.argmax(fits_stack, dim=2) # [K, B] 0,1,2
                
#                 # 对应的 Pos 也 stack 起来: [K, B, 3, N]
#                 left_pos = torch.roll(self.pbest_pos, shifts=1, dims=1)
#                 right_pos = torch.roll(self.pbest_pos, shifts=-1, dims=1)
#                 pos_stack = torch.stack([self.pbest_pos, left_pos, right_pos], dim=2)
                
#                 # Gather social target
#                 # index 需要扩展成 [K, B, 1, N]
#                 gather_idx = best_neighbor_idx.view(K, B, 1, 1).expand(K, B, 1, N)
#                 social_target = torch.gather(pos_stack, 2, gather_idx).squeeze(2) # [K, B, N]
                
#             else:
#                 # == 全局拓扑 (Global Topology) ==
#                 # 所有粒子都向 Gbest 学习
#                 social_target = self.gbest_pos.unsqueeze(1).expand(K, B, N)

#             # --- 5. 更新速度与位置 ---
#             # 动态权重
#             w = w_start - (w_start - w_end) * (t / iters)
            
#             r1 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
#             r2 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)

#             # 速度更新公式
#             self.vel = (w * self.vel + 
#                         c1 * r1 * (self.pbest_pos - self.pos) + 
#                         c2 * r2 * (social_target - self.pos))
            
#             self.vel.clamp_(-vmax, vmax)
#             self.pos += self.vel
            
#             # Random Keys 边界处理：限制在 [0,1] 之间能保持数值稳定性
#             self.pos.clamp_(0.0, 1.0)
            
#             # --- 6. 扰动/变异 (Mutation) ---
#             # 防止停滞：以小概率随机交换 Pbest 中的某些维度的值，
#             # 或者给 Pos 加一点随机噪声
#             if mutation_prob > 0:
#                 # 生成随机掩码
#                 mut_mask = (torch.rand((K, B, N), device=DEVICE) < mutation_prob)
#                 if mut_mask.any():
#                     # 对选中的维度赋予新的随机值
#                     noise = torch.rand_like(self.pos)
#                     self.pos = torch.where(mut_mask, noise, self.pos)

#             # --- 日志 ---
#             if log_every > 0 and (t + 1) % log_every == 0:
#                 avg_best = self.gbest_fit.mean().item()
#                 max_best = self.gbest_fit.max().item()
#                 print(f"[PSO] Iter {t+1:5d}/{iters} | Avg Best: {avg_best:.4f} | Max Best: {max_best:.4f}")

#         # 结束，返回结果
#         final_perm_idx = torch.argsort(self.gbest_pos, dim=1)  # [K, N]

#         results: List[Dict[str, Any]] = []
#         for k in range(K):
#             row = final_perm_idx[k].detach().cpu()
#             cids = self._tensor_row_to_cids(row)
#             results.append({
#                 "run_id": k,
#                 "fitness": float(self.gbest_fit[k].item()),
#                 "individual_cids": cids
#             })

#         return results


# # =========================
# # 3) PSO 主程序入口
# # =========================

# def main():
#     try:
#         # ================== 改进后的配置 ==================
#         NUM_PARALLEL_RUNS = 8     # 并行跑8个独立的实验
#         POP_SIZE_PER_RUN = 128    # 种群稍微大一点，Lbest需要更多粒子传递信息

#         ITERS = 1              # 迭代次数
        
#         # 改进后的 PSO 参数
#         W_START = 0.9
#         W_END = 0.4
#         C1 = 2.0  # 个人认知
#         C2 = 2.0  # 社会认知 (Lbest)
#         VMAX = 0.15 
        
#         USE_LBEST = True          # 启用环形拓扑，强烈建议开启
#         MUTATION_PROB = 0.005     # 0.5% 的概率发生基因突变(重置位置)

#         print(f"启动 PSO 改进版: K={NUM_PARALLEL_RUNS}, B={POP_SIZE_PER_RUN}")
#         print(f"策略: Lbest={USE_LBEST}, Dynamic W={W_START}->{W_END}")

#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#         duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#         device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

#         for f in [patient_file, duration_file, device_constraint_file]:
#             if not os.path.exists(f):
#                 print(f"❌ 错误：找不到文件 {f}")
#                 return

#         print("正在导入数据...")
#         patients = import_data(patient_file, duration_file)
#         machine_exam_map = import_device_constraints(device_constraint_file)
        
#         optimizer = MultiRunPSOOptimizer(
#             patients,
#             machine_exam_map,
#             num_parallel_runs=NUM_PARALLEL_RUNS,
#             pop_size_per_run=POP_SIZE_PER_RUN
#         )

#         t0_init = time.perf_counter()
#         optimizer.initialize_particles()
#         print(f"初始化耗时: {time.perf_counter() - t0_init:.4f}s")

#         print(f"\n开始 PSO 迭代 (Total {ITERS} iters)...")
#         t0 = time.perf_counter()

#         results = optimizer.evolve_pso(
#             iters=ITERS,
#             w_start=W_START,
#             w_end=W_END,
#             c1=C1,
#             c2=C2,
#             vmax=VMAX,
#             use_lbest=USE_LBEST,
#             mutation_prob=MUTATION_PROB,
#             log_every=100
#         )

#         t_total = time.perf_counter() - t0
#         print(f"\n✓ 优化完成，总耗时: {t_total:.2f}s")

#         # 统计
#         fitness_list = [r["fitness"] for r in results]
#         avg_fit = sum(fitness_list) / len(fitness_list)
#         max_fit = max(fitness_list)
#         print(f"\n结果统计 (K={NUM_PARALLEL_RUNS}):")
#         print(f"  平均 Fitness: {avg_fit:.2f}")
#         print(f"  最佳 Fitness: {max_fit:.2f}")

#         # 导出
#         export_dir = os.path.join(current_dir, "PSO_Improved_Results")
#         os.makedirs(export_dir, exist_ok=True)

#         for r in results:
#             if r["fitness"] >= max_fit - 0.01: # 只导出最好的
#                 run_id = r["run_id"]
#                 individual = r["individual_cids"]
#                 fit_val = r["fitness"]
                
#                 system = optimizer.generate_schedule(individual)
#                 fname = f"PSO_Best_Run{run_id}_Fit{abs(fit_val):.0f}.xlsx"
#                 out_xlsx = os.path.join(export_dir, fname)
#                 export_schedule(system, patients, out_xlsx)
#                 print(f"✓ 已导出最佳排程: {out_xlsx}")

#     except Exception as e:
#         print(f"运行时错误: {e}")
#         traceback.print_exc()

# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()
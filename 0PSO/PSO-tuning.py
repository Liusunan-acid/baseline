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
    current_dir = "/home/preprocess/_funsearch/baseline"
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

        return fit

    @torch.no_grad()
    def evolve_pso(self,
                   iters: int = 5000,
                   w: float = 0.7,
                   c1: float = 1.4,
                   c2: float = 1.4,
                   vmax: float = 0.1,
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
        """
        self._ensure_gpu_engine()

        if self.pos is None:
            self.initialize_particles()

        K, B, N = self.K, self.B, self.N

        # ✅ 仅作为“轻探索”的随机扰动配置（不改函数参数）
        JITTER_EVERY = 0      # 每隔多少代做一次
        JITTER_FRAC  = 0.1    # 做扰动的粒子比例

        # ✅ 强化定点变异：每个违规粒子连续做几步邻域交换（不改函数参数）
        PIN_STEPS = 1
        PIN_WINDOW = 400

        for t in range(iters):

            # ================================================================================
            # 0) 离散扰动：不要每代对所有粒子做（那会把 PSO 变随机游走）
            #    改为：每隔 JITTER_EVERY 代，对 JITTER_FRAC 的粒子做一次随机 swap
            # ================================================================================
            if JITTER_EVERY > 0 and (t + 1) % JITTER_EVERY == 0 and JITTER_FRAC > 0:
                do_jit = (torch.rand((K, B), device=DEVICE) < JITTER_FRAC)  # [K,B]
                if do_jit.any():
                    idx1 = torch.randint(0, N, (K, B), device=DEVICE)
                    offset = torch.randint(1, N, (K, B), device=DEVICE)
                    idx2 = (idx1 + offset) % N

                    # 只对 do_jit 生效：无效处让 idx1==idx2，swap 无影响
                    idx1 = torch.where(do_jit, idx1, idx2)

                    i1 = idx1.unsqueeze(2)  # [K,B,1]
                    i2 = idx2.unsqueeze(2)  # [K,B,1]
                    v1 = self.pos.gather(2, i1)
                    v2 = self.pos.gather(2, i2)
                    self.pos.scatter_(2, i1, v2)
                    self.pos.scatter_(2, i2, v1)

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

            # ================== ✅ 违规定点变异（增强版）：连续 PIN_STEPS 次邻域交换 ==================
            pos_before = self.pos.clone()  # 用于回滚

            viol = out.get("any_violate_mask_b_n", None)
            if viol is not None:
                viol = viol.view(K, B, N)         # [K,B,N]（位置维）
                has_bad = viol.any(dim=2)         # [K,B]

                if has_bad.any():
                    # 连续做几步（单步往往修不动硬约束）
                    for _ in range(PIN_STEPS):
                        # 当前排列（注意：每步变异后排列会变，所以这里要重算 perms）
                        perms_cur = torch.argsort(self.pos, dim=2)

                        # 选一个违规位置（取第一个 True；简单但有效）
                        bad_pos = viol.float().argmax(dim=2).to(torch.long)  # [K,B]

                        # 在 ±PIN_WINDOW 里选邻居位置
                        W = min(PIN_WINDOW, N - 1)
                        off = torch.randint(1, W + 1, (K, B), device=DEVICE, dtype=torch.long)
                        sign = torch.where(torch.rand((K, B), device=DEVICE) < 0.5,
                                           torch.full((K, B), -1, device=DEVICE, dtype=torch.long),
                                           torch.full((K, B),  1, device=DEVICE, dtype=torch.long))
                        off = off * sign

                        nbr_pos = (bad_pos + off).clamp(0, N - 1)
                        nbr_pos = torch.where(nbr_pos == bad_pos,
                                              (bad_pos + 1).clamp(0, N - 1),
                                              nbr_pos)
                        nbr_pos = torch.where(has_bad, nbr_pos, bad_pos)

                        # 取出两个位置对应的患者 id
                        pid1 = perms_cur.gather(2, bad_pos.unsqueeze(2)).squeeze(2)  # [K,B]
                        pid2 = perms_cur.gather(2, nbr_pos.unsqueeze(2)).squeeze(2)  # [K,B]

                        # 交换这两个患者的 key
                        i1 = pid1.unsqueeze(2)
                        i2 = pid2.unsqueeze(2)
                        v1 = self.pos.gather(2, i1)
                        v2 = self.pos.gather(2, i2)
                        self.pos.scatter_(2, i1, v2)
                        self.pos.scatter_(2, i2, v1)

            # ===== ✅ 变异后重新评估，并且只接受更优（否则回滚）=====
            perms2 = torch.argsort(self.pos, dim=2)
            out2 = self._gpu_engine.fitness_batch(perms2.reshape(K * B, N), return_assignment=False)
            fit2 = out2["fitness"].reshape(K, B)

            better = fit2 > fit
            better_exp = better.unsqueeze(2).expand(K, B, N)
            self.pos = torch.where(better_exp, self.pos, pos_before)
            fit = torch.where(better, fit2, fit)

            # 补更新 pbest/gbest（让变异真正能留下来）
            improve2 = fit > self.pbest_fit
            self.pbest_fit = torch.where(improve2, fit, self.pbest_fit)
            improve2_exp = improve2.unsqueeze(2).expand(K, B, N)
            self.pbest_pos = torch.where(improve2_exp, self.pos, self.pbest_pos)

            best_vals2, best_idx2 = torch.max(self.pbest_fit, dim=1)
            better_g2 = best_vals2 > self.gbest_fit
            self.gbest_fit = torch.where(better_g2, best_vals2, self.gbest_fit)

            idx_exp2 = best_idx2.view(K, 1, 1).expand(K, 1, N)
            cand_gbest_pos2 = torch.gather(self.pbest_pos, 1, idx_exp2).squeeze(1)
            better_g2_exp = better_g2.view(K, 1).expand(K, N)
            self.gbest_pos = torch.where(better_g2_exp, cand_gbest_pos2, self.gbest_pos)

            # 5) 速度/位置更新
            r1 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
            r2 = torch.rand((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
            gbest_expand = self.gbest_pos.unsqueeze(1).expand(K, B, N)

            self.vel = (
                w * self.vel
                + c1 * r1 * (self.pbest_pos - self.pos)
                + c2 * r2 * (gbest_expand - self.pos)
            )
            self.vel = torch.clamp(self.vel, -vmax, vmax)
            self.pos = self.pos + self.vel

            # 6) 周期性重启最差粒子
            if restart_every > 0 and (t + 1) % restart_every == 0 and restart_frac > 0:
                k_bad = max(1, int(B * restart_frac))
                worst_idx = torch.topk(fit, k=k_bad, largest=False, dim=1).indices  # [K,k_bad]
                worst_mask = torch.zeros((K, B), device=DEVICE, dtype=torch.bool)
                worst_mask.scatter_(1, worst_idx, True)
                worst_mask_exp = worst_mask.unsqueeze(2).expand(K, B, N)

                super().initialize_population()
                new_pop_indices = self.population_tensor  # [K,B,N]

                rank = torch.arange(N, device=DEVICE, dtype=DTYPE_FLOAT)
                rank_values = (rank / max(1, N)).view(1, 1, N).expand(K, B, N)

                new_pos_full = torch.empty((K, B, N), device=DEVICE, dtype=DTYPE_FLOAT)
                new_pos_full.scatter_(dim=2, index=new_pop_indices, src=rank_values)

                self.pos = torch.where(worst_mask_exp, new_pos_full, self.pos)
                self.vel = torch.where(worst_mask_exp, torch.zeros_like(self.vel), self.vel)

            # 7) 日志
            if log_every > 0 and (t + 1) % log_every == 0:
                avg_best = float(self.gbest_fit.mean().item())
                # 额外打印：变异接受率（帮助你确认“算子有没有在起作用”）
                acc = float(better.float().mean().item())
                print(f"[PSO] Iter {t+1:5d}/{iters} | Avg gbest(K={K}): {avg_best:.4f} | mut_accept={acc:.3f}")

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
        C1 = 1
        C2 = 2
        VMAX = 0.07

        RESTART_EVERY = 2000
        RESTART_FRAC = 0.02
        LOG_EVERY = 100
        # ==========================================================

        print(f"启动 PSO Megabatch 模式: K={NUM_PARALLEL_RUNS}, B={POP_SIZE_PER_RUN}")
        print(f"总 GPU 批量: {NUM_PARALLEL_RUNS * POP_SIZE_PER_RUN} 粒子")

        current_dir = "/home/preprocess/_funsearch/baseline/data"
        patient_file = os.path.join(current_dir, '实验数据6.1 - 副本.xlsx')
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
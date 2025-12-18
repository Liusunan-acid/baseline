# # alns_baseline_runner.py
# # 目标：用 N-Wouda 的 alns 库做一个“纯 ALNS” baseline，
# #      并尽量复用你现有的 GPU fitness 与数据导入逻辑。

# import argparse
# import importlib.util
# import numpy as np
# import torch
# import os
# from datetime import datetime

# from alns import ALNS
# from alns.accept import SimulatedAnnealing
# from alns.select import RouletteWheel
# from alns.stop import MaxIterations


# # -------------------------
# # 1) 动态加载你的原脚本（避免触发 __main__）
# # -------------------------
# def load_user_module(py_path: str):
#     spec = importlib.util.spec_from_file_location("user_ga_module", py_path)
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)  # 不会触发 if __name__ == "__main__"
#     return mod


# # -------------------------
# # 2) ALNS State：只存“患者索引排列”
# # -------------------------
# class PermState:
#     def __init__(self, engine, perm: np.ndarray):
#         self.engine = engine
#         self.perm = perm.astype(np.int64)

#     def copy(self):
#         # 注意：copy只复制核心状态(perm)，不复制临时的 removed 属性
#         # destroy 算子会给生成的副本添加 removed 属性
#         return PermState(self.engine, self.perm.copy())

#     def objective(self):
#         """
#         alns 默认是最小化。你 GPU 引擎返回的是 fitness（越大越好），
#         所以我们用 cost = -fitness。
#         """
#         # 简单检查长度，防止 GPU 报错 cryptic error
#         expected_N = self.engine.patient_durations.shape[0]
#         if len(self.perm) != expected_N:
#              # 如果发生这种情况，说明 repair 算子没能把所有被移除的元素加回来
#              # 或者是 destroy 算子有问题
#              raise RuntimeError(f"State size mismatch! Expected {expected_N}, got {len(self.perm)}. "
#                                 f"This usually means the Repair operator failed to restore all removed elements.")

#         p = torch.from_numpy(self.perm).to(self.engine.patient_durations.device)
#         p = p.view(1, -1)
#         out = self.engine.fitness_batch(p, return_assignment=False)
#         fit = out["fitness"].detach().float().cpu().item()
#         return -fit


# # -------------------------
# # 3) Destroy operators
# # -------------------------
# def destroy_random_segment(state: PermState, rng: np.random.Generator):
#     s = state.copy()
#     N = len(s.perm)
#     a = int(rng.integers(0, N))
#     b = int(rng.integers(0, N))
#     i, j = (a, b) if a < b else (b, a)

#     # 记录被移除的元素
#     s.removed = s.perm[i:j+1].copy()
#     # 移除片段
#     s.perm = np.concatenate([s.perm[:i], s.perm[j+1:]])
#     return s


# def destroy_random_k(state: PermState, rng: np.random.Generator, ratio=0.05):
#     s = state.copy()
#     N = len(s.perm)
#     k = max(1, int(N * ratio))
#     idx = rng.choice(N, size=k, replace=False)

#     mask = np.ones(N, dtype=bool)
#     mask[idx] = False

#     # 记录被移除的元素
#     s.removed = s.perm[~mask].copy()
#     # 仅保留未被移除的
#     s.perm = s.perm[mask]
#     return s


# # -------------------------
# # 4) Repair operators
# # -------------------------
# def repair_random_insert(state: PermState, rng: np.random.Generator):
#     # 注意：这里的 state 是 destroy 算子返回的状态（已经包含 .removed 属性）
#     s = state.copy()
    
#     # [修复] 应该从输入的 state 中读取 removed，而不是从刚 copy 出来的 s 中读
#     # 因为 s = state.copy() 不会复制动态添加的属性
#     removed = getattr(state, "removed", np.array([], dtype=np.int64))

#     perm_list = s.perm.tolist()
#     for x in removed:
#         pos = int(rng.integers(0, len(perm_list) + 1))
#         perm_list.insert(pos, int(x))

#     s.perm = np.array(perm_list, dtype=np.int64)
#     # 修复完成后，通常不需要保留 removed 属性
#     return s


# def repair_type_cluster_insert(state: PermState, rng: np.random.Generator):
#     """
#     利用你 GPU 引擎里已有的 patient_main_type_id 做一个
#     “按类型聚合插回”的修复算子，面向换模惩罚。
#     """
#     s = state.copy()
    
#     # [修复] 从输入的 state 读取 removed
#     removed = getattr(state, "removed", np.array([], dtype=np.int64))

#     # 从 GPU 引擎拿类型 id（Tensor）
#     # 注意：需要确保 removed 里的索引在 type_ids 范围内（通常没问题）
#     type_ids = s.engine.patient_main_type_id.detach().cpu().numpy()

#     removed_list = removed.tolist()
#     # 按类型排序，尝试让相同类型的检查聚在一起
#     removed_list.sort(key=lambda pid: int(type_ids[pid]))

#     perm_list = s.perm.tolist()
#     base = int(rng.integers(0, len(perm_list) + 1))
    
#     # 简单策略：选个基准点，依次往后插
#     for t, x in enumerate(removed_list):
#         pos = min(base + t, len(perm_list))
#         perm_list.insert(pos, int(x))

#     s.perm = np.array(perm_list, dtype=np.int64)
#     return s


# # -------------------------
# # 5) 构建 ALNS baseline
# # -------------------------
# def build_engine_via_optimizer(user_mod, patients, machine_exam_map, block_start_date=None):
#     # 借用你的 MultiRunOptimizer 来构建 GPU fitness 引擎
#     opt = user_mod.MultiRunOptimizer(
#         patients=patients,
#         machine_exam_map=machine_exam_map,
#         num_parallel_runs=1,
#         pop_size_per_run=1,
#         block_start_date=block_start_date
#     )
#     opt._ensure_gpu_engine()
#     return opt._gpu_engine, opt


# def run_alns(engine, init_perm: np.ndarray, seed=0, iters=2000):
#     rng = np.random.default_rng(seed)
#     alns = ALNS(rng)

#     # destroy
#     alns.add_destroy_operator(destroy_random_segment)
#     # [修复] 给 lambda 算子指定唯一的 name，防止因名称冲突(都是<lambda>)被覆盖或计数错误
#     alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.03), name="destroy_k_03")
#     alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.08), name="destroy_k_08")

#     # repair
#     alns.add_repair_operator(repair_random_insert)
#     alns.add_repair_operator(repair_type_cluster_insert)

#     # 选择算子
#     selector = RouletteWheel(
#         scores=[5, 2, 1, 0.5],
#         decay=0.8,
#         num_destroy=3,
#         num_repair=2
#     )
    
#     # 模拟退火
#     # 适应度通常很大（e.g. -200,000），所以温度也要相应大一些
#     accept = SimulatedAnnealing(
#         start_temperature=5000, 
#         end_temperature=100,
#         step=0.995
#     )
#     stop = MaxIterations(iters)

#     init_state = PermState(engine, init_perm)
#     result = alns.iterate(init_state, selector, accept, stop)
#     best = result.best_state
#     return best.perm, best.objective()


# # -------------------------
# # 6) CLI
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--user_code", type=str, default="测量时间full-GPU实验-Multi.py")

#     # ✅ 直接写死默认路径（你给的三份）
#     parser.add_argument(
#         "--device_file",
#         type=str,
#         default="/home/preprocess/_funsearch/baseline/设备限制4.xlsx"
#     )
#     parser.add_argument(
#         "--patient_file",
#         type=str,
#         default="/home/preprocess/_funsearch/baseline/实验数据6.1small - 副本.xlsx"
#     )
#     parser.add_argument(
#         "--duration_file",
#         type=str,
#         default="/home/preprocess/_funsearch/baseline/程序使用实际平均耗时3 - 副本.xlsx"
#     )

#     parser.add_argument("--iters", type=int, default=2000)
#     parser.add_argument("--seed", type=int, default=0)
#     args = parser.parse_args()

#     user_mod = load_user_module(args.user_code)

#     # 复用你原代码的数据导入
#     patients = user_mod.import_data(args.patient_file, args.duration_file)
#     machine_exam_map = user_mod.import_device_constraints(args.device_file)

#     engine, opt = build_engine_via_optimizer(user_mod, patients, machine_exam_map)

#     # 初始解：登记日期排序对应的索引顺序（与你现有逻辑一致）
#     N = opt.N
#     init_perm = np.arange(N, dtype=np.int64)

#     best_perm, best_cost = run_alns(engine, init_perm, seed=args.seed, iters=args.iters)

#     # 输出结果（可按你原格式进一步保存）
#     best_fitness = -best_cost
#     print("=== ALNS baseline finished ===")
#     print(f"N = {N}")
#     print(f"best_fitness = {best_fitness:.4f}")

#     # 如需把 perm 映射回 patient_id：
#     idx_to_cid = opt._idx_to_cid
#     best_patient_order = [idx_to_cid[i] for i in best_perm.tolist()]
#     print("best_patient_order_head20 =", best_patient_order[:20])


#     print("\n正在生成排程 Excel...")
    
#     # 1. 复用 opt 里的 generate_schedule 方法，传入最佳顺序(CID列表)生成排程对象
#     final_system = opt.generate_schedule(best_patient_order)
    
#     # 2. 构造文件名
#     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out_dir = 'output_schedules'
#     os.makedirs(out_dir, exist_ok=True)
#     filename = os.path.join(out_dir, f'ALNS_schedule_{ts}_fit_{best_fitness:.0f}.xlsx')

#     # 3. 复用 user_mod 里的 export_schedule 函数导出
#     user_mod.export_schedule(final_system, patients, filename)
#     print(f"✓ 已导出排程文件至: {filename}")

# if __name__ == "__main__":
#     main()
# alns_baseline_runner.py
# 纯 ALNS baseline（复用你的 full-GPU 适应度引擎）
# 改动点：
# 1) 使用 SegmentedRouletteWheel
# 2) 新增等待/违规导向 destroy + repair
# 3) 默认直接使用你给的三个 xlsx 路径

import argparse
import importlib.util
import numpy as np
import torch
import os
from datetime import datetime

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations

# 选择器：优先 SegmentedRouletteWheel
try:
    from alns.select import SegmentedRouletteWheel
    _HAS_SEG = True
except Exception:
    _HAS_SEG = False
    from alns.select import RouletteWheel


# -------------------------
# 1) 动态加载你的原脚本
# -------------------------
def load_user_module(py_path: str):
    spec = importlib.util.spec_from_file_location("user_ga_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 不触发 __main__
    return mod


# -------------------------
# 2) ALNS State：只存 permutation
# -------------------------
class PermState:
    def __init__(self, engine, perm: np.ndarray):
        self.engine = engine
        self.perm = perm.astype(np.int64)

    def copy(self):
        # 拷贝时保留 engine 引用
        s = PermState(self.engine, self.perm.copy())
        # 可能存在的临时字段
        if hasattr(self, "removed"):
            s.removed = self.removed.copy()
        return s

    def objective(self):
        """
        alns 默认最小化；
        你的引擎 fitness 越大越好；
        所以 cost = -fitness。
        """
        device = self.engine.patient_durations.device
        perms_t = torch.from_numpy(self.perm).to(device).view(1, -1)

        out = self.engine.fitness_batch(perms_t, return_assignment=False)
        fit = out["fitness"].detach().float().cpu().item()
        return -float(fit)

# -------------------------
# 3) Destroy operators（基础随机）
# -------------------------
def destroy_random_segment(state: PermState, rng: np.random.Generator):
    s = state.copy()
    N = len(s.perm)
    if N <= 1:
        s.removed = np.array([], dtype=np.int64)
        return s

    a = int(rng.integers(0, N))
    b = int(rng.integers(0, N))
    i, j = (a, b) if a < b else (b, a)

    s.removed = s.perm[i:j + 1].copy()
    s.perm = np.concatenate([s.perm[:i], s.perm[j + 1:]])
    return s


def destroy_random_k(state: PermState, rng: np.random.Generator, ratio=0.05):
    s = state.copy()
    N = len(s.perm)
    if N <= 1:
        s.removed = np.array([], dtype=np.int64)
        return s

    k = max(1, int(N * ratio))
    idx = rng.choice(N, size=k, replace=False)

    mask = np.ones(N, dtype=bool)
    mask[idx] = False

    s.removed = s.perm[~mask].copy()
    s.perm = s.perm[mask]
    return s


# -------------------------
# 4) Destroy operator（新增：等待/违规导向）
# -------------------------
def destroy_wait_violate_guided(state: PermState, rng: np.random.Generator, ratio=0.05):
    """
    面向你的 fitness 的 destroy：
    - 优先移除“违规位置”的患者（按排列位置）
    - 其次移除“等待贡献大”的患者
    """
    s = state.copy()
    N = len(s.perm)
    if N <= 1:
        s.removed = np.array([], dtype=np.int64)
        return s

    k = max(1, int(N * ratio))

    device = s.engine.patient_durations.device
    perms_t = torch.from_numpy(s.perm).to(device).view(1, -1)

    # 需要 return_assignment=True 才能拿到 assigned_day/violate mask
    out = s.engine.fitness_batch(perms_t, return_assignment=True)

    # 违规位置 mask（按排列位置 B×N）
    viol_pos = out.get("any_violate_mask_b_n", None)
    if viol_pos is not None:
        viol_pos = viol_pos[0].detach().cpu().numpy().astype(bool)
    else:
        viol_pos = np.zeros(N, dtype=bool)

    # 等待天数（按排列位置）
    assigned_day = out.get("assigned_day", None)
    if assigned_day is not None:
        assigned_day = assigned_day[0].detach().cpu().numpy()
        reg = s.engine.reg_day_offsets.detach().cpu().numpy()
        # reg 是按 patient_id 编号，需要用当前 perm 映射
        wait = assigned_day - reg[s.perm]
        wait = np.clip(wait, 0, None)
    else:
        wait = np.zeros(N, dtype=np.int64)

    # 综合评分：违规位置强优先
    max_wait = int(wait.max()) if wait.size else 0
    big = max_wait + 1_000_000
    score = wait.astype(np.int64) + viol_pos.astype(np.int64) * big

    # 选 score 最大的 k 个“位置”
    idx = np.argsort(-score)[:k]
    idx = np.sort(idx)

    s.removed = s.perm[idx].copy()

    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    s.perm = s.perm[mask]
    return s


# -------------------------
# 5) Repair operators（基础）
# -------------------------
def repair_random_insert(state: PermState, rng: np.random.Generator):
    s = state.copy()
    removed = getattr(s, "removed", np.array([], dtype=np.int64))
    if removed.size == 0:
        return s

    perm_list = s.perm.tolist()
    for x in removed:
        pos = int(rng.integers(0, len(perm_list) + 1))
        perm_list.insert(pos, int(x))

    s.perm = np.array(perm_list, dtype=np.int64)
    return s

def repair_type_cluster_insert(state: PermState, rng: np.random.Generator):
    """
    新版本思路：
    1) 将 removed 按 patient_main_type_id 分组（每种类型形成一块）
    2) 每一块选择一个随机位置，整块一次性插回
    这样“有几个同类型就生成几块”，每块插一个位置。
    """
    s = state.copy()
    removed = getattr(s, "removed", np.array([], dtype=np.int64))
    if removed.size == 0:
        return s

    type_ids = s.engine.patient_main_type_id.detach().cpu().numpy()

    # --- 1) 分组：type_id -> list[pids]
    groups = {}
    for pid in removed.tolist():
        t = int(type_ids[pid])
        groups.setdefault(t, []).append(int(pid))

    # 可选：块内顺序固定为原 removed 中的出现顺序（上面已自然保持）
    # 也可按 pid 排序：groups[t].sort()

    # --- 2) 形成“块列表”
    # 为了可复现/稳定一点，按 type_id 排序块的处理顺序
    blocks = [groups[t] for t in sorted(groups.keys())]

    perm_list = s.perm.tolist()

    # --- 3) 每一块随机选一个位置，整块插回
    # 使用切片赋值一次性插入一段
    for block in blocks:
        if not block:
            continue
        pos = int(rng.integers(0, len(perm_list) + 1))
        perm_list[pos:pos] = block  # 整块插回（连续）

    s.perm = np.array(perm_list, dtype=np.int64)
    return s



# -------------------------
# 6) Repair operator（新增：等待/违规导向）
# -------------------------
def repair_wait_violate_guided_insert(state: PermState, rng: np.random.Generator):
    """
    面向等待/违规的轻量 repair：
    - 登记更早的优先靠前插回
    - 心脏/造影（若引擎提供）优先靠前插回
    - 再用 type_id 做弱聚合倾向
    """
    s = state.copy()
    removed = getattr(s, "removed", np.array([], dtype=np.int64))
    if removed.size == 0:
        return s

    reg = s.engine.reg_day_offsets.detach().cpu().numpy()
    type_ids = s.engine.patient_main_type_id.detach().cpu().numpy()

    has_heart = getattr(s.engine, "has_heart", None)
    has_angio = getattr(s.engine, "has_angio", None)
    has_heart = has_heart.detach().cpu().numpy() if has_heart is not None else None
    has_angio = has_angio.detach().cpu().numpy() if has_angio is not None else None

    def risk_key(pid: int):
        # 心脏/造影优先
        if has_heart is None or has_angio is None:
            return 1
        return 0 if (bool(has_heart[pid]) or bool(has_angio[pid])) else 1

    removed_list = removed.tolist()
    removed_list.sort(key=lambda pid: (int(reg[pid]), risk_key(pid), int(type_ids[pid])))

    perm_list = s.perm.tolist()

    for x in removed_list:
        L = len(perm_list)
        early_window = max(1, int(L))
        pos = int(rng.integers(0, early_window + 1))
        perm_list.insert(pos, int(x))

    s.perm = np.array(perm_list, dtype=np.int64)
    return s


# -------------------------
# 7) 用你的 MultiRunOptimizer 构建 GPU engine
# -------------------------
def build_engine_via_optimizer(user_mod, patients, machine_exam_map, block_start_date=None):
    # 只用来拿 engine，不跑 GA
    opt = user_mod.MultiRunOptimizer(
        patients=patients,
        machine_exam_map=machine_exam_map,
        num_parallel_runs=1,
        pop_size_per_run=1,
        block_start_date=block_start_date
    )
    opt._ensure_gpu_engine()
    return opt._gpu_engine, opt


# -------------------------
# 8) ALNS 主流程
# -------------------------
def run_alns(engine, init_perm: np.ndarray, seed=0, iters=10000):
    rng = np.random.default_rng()
    alns = ALNS(rng)

    # ---- destroy ----
    alns.add_destroy_operator(destroy_random_segment)
    alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.03), name="destroy_k_03")
    alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.08), name="destroy_k_08")

    # 新增：等待/违规导向 destroy
    alns.add_destroy_operator(lambda st, r: destroy_wait_violate_guided(st, r, ratio=0.05),
                              name="destroy_wait_violate_05")

    # ---- repair ----
    alns.add_repair_operator(repair_random_insert)
    alns.add_repair_operator(repair_type_cluster_insert)

    # 新增：等待/违规导向 repair
    alns.add_repair_operator(repair_wait_violate_guided_insert)

    # ---- selector ----
    num_destroy = 4
    num_repair = 3
    scores = [5, 2, 1, 0.5]
    decay = 0.8

    if _HAS_SEG:
        selector = SegmentedRouletteWheel(
            scores=scores,
            decay=decay,
            num_destroy=num_destroy,
            num_repair=num_repair,
            seg_length=50
        )
    else:
        # 兼容你环境里若没有 Segmented 版本
        selector = RouletteWheel(
            scores=scores,
            decay=decay,
            num_destroy=num_destroy,
            num_repair=num_repair
        )

    # ---- acceptance ----
    accept = SimulatedAnnealing(
        start_temperature=5000,
        end_temperature=100,
        step=0.995
    )

    # ---- stop ----
    stop = MaxIterations(iters)

    init_state = PermState(engine, init_perm.copy())
    result = alns.iterate(init_state, selector, accept, stop)

    print("\n=== Final Operator Weights (Scores) ===")
    d_ops = ["destroy_random_segment", "destroy_k_03", "destroy_k_08", "destroy_wait_violate_05"]
    r_ops = ["repair_random_insert", "repair_type_cluster_insert", "repair_wait_violate_guided"]

    print("Destroy Operators:")
    for name, w in zip(d_ops, selector.destroy_weights):
        print(f"  {name:<25} : {w:.4f}")

    print("\nRepair Operators:")
    for name, w in zip(r_ops, selector.repair_weights):
        print(f"  {name:<25} : {w:.4f}")
    # <<<<< 插入结束 <<<<<

    best = result.best_state
    return best.perm, best.objective()

# -------------------------
# 9) CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--user_code", type=str, default="测量时间full-GPU实验-Multi.py")

    # 你给的默认路径
    parser.add_argument(
        "--device_file",
        type=str,
        default="/home/preprocess/_funsearch/baseline/设备限制4.xlsx"
    )
    parser.add_argument(
        "--patient_file",
        type=str,
        default="/home/preprocess/_funsearch/baseline/实验数据6.1small - 副本.xlsx"
    )
    parser.add_argument(
        "--duration_file",
        type=str,
        default="/home/preprocess/_funsearch/baseline/程序使用实际平均耗时3 - 副本.xlsx"
    )

    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    print("=== Using ALNS baseline files ===")
    print("user_code    :", args.user_code)
    print("device_file  :", args.device_file)
    print("patient_file :", args.patient_file)
    print("duration_file:", args.duration_file)
    print("iters        :", args.iters)
    print("seed         :", args.seed)
    print("selector     :", "SegmentedRouletteWheel" if _HAS_SEG else "RouletteWheel")

    user_mod = load_user_module(args.user_code)

    # 复用你原来的数据导入
    patients = user_mod.import_data(args.patient_file, args.duration_file)
    machine_exam_map = user_mod.import_device_constraints(args.device_file)

    engine, opt = build_engine_via_optimizer(user_mod, patients, machine_exam_map)

    N = opt.N

    # 初始解：最简单的 0..N-1（你也可以换成“按登记时间排序的 index”）
    init_perm = np.arange(N, dtype=np.int64)

    best_perm, best_cost = run_alns(engine, init_perm, seed=args.seed, iters=args.iters)

    best_fitness = -best_cost
    print("\n=== ALNS baseline finished ===")
    print(f"N = {N}")
    print(f"best_fitness = {best_fitness:.4f}")

    # 映射回 patient_id（如果你后续想保存）
    idx_to_cid = opt._idx_to_cid
    best_patient_order = [idx_to_cid[i] for i in best_perm.tolist()]
    print("best_patient_order_head20 =", best_patient_order[:20])

    print("\n正在生成排程 Excel...")
    
    # 1. 复用 opt 里的 generate_schedule 方法，传入最佳顺序(CID列表)生成排程对象
    final_system = opt.generate_schedule(best_patient_order)
    
    # 2. 构造文件名
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = 'output_schedules'
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f'ALNS_schedule_{ts}_fit_{best_fitness:.0f}.xlsx')

    # 3. 复用 user_mod 里的 export_schedule 函数导出
    user_mod.export_schedule(final_system, patients, filename)
    print(f"✓ 已导出排程文件至: {filename}")

if __name__ == "__main__":
    main()
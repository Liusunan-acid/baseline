# alns_baseline_runner.py
# 目标：用 N-Wouda 的 alns 库做一个“纯 ALNS” baseline，
#      并尽量复用你现有的 GPU fitness 与数据导入逻辑。

import argparse
import importlib.util
import numpy as np
import torch

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations


# -------------------------
# 1) 动态加载你的原脚本（避免触发 __main__）
# -------------------------
def load_user_module(py_path: str):
    spec = importlib.util.spec_from_file_location("user_ga_module", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 不会触发 if __name__ == "__main__"
    return mod


# -------------------------
# 2) ALNS State：只存“患者索引排列”
# -------------------------
class PermState:
    def __init__(self, engine, perm: np.ndarray):
        self.engine = engine
        self.perm = perm.astype(np.int64)

    def copy(self):
        # 注意：copy只复制核心状态(perm)，不复制临时的 removed 属性
        # destroy 算子会给生成的副本添加 removed 属性
        return PermState(self.engine, self.perm.copy())

    def objective(self):
        """
        alns 默认是最小化。你 GPU 引擎返回的是 fitness（越大越好），
        所以我们用 cost = -fitness。
        """
        # 简单检查长度，防止 GPU 报错 cryptic error
        expected_N = self.engine.patient_durations.shape[0]
        if len(self.perm) != expected_N:
             # 如果发生这种情况，说明 repair 算子没能把所有被移除的元素加回来
             # 或者是 destroy 算子有问题
             raise RuntimeError(f"State size mismatch! Expected {expected_N}, got {len(self.perm)}. "
                                f"This usually means the Repair operator failed to restore all removed elements.")

        p = torch.from_numpy(self.perm).to(self.engine.patient_durations.device)
        p = p.view(1, -1)
        out = self.engine.fitness_batch(p, return_assignment=False)
        fit = out["fitness"].detach().float().cpu().item()
        return -fit


# -------------------------
# 3) Destroy operators
# -------------------------
def destroy_random_segment(state: PermState, rng: np.random.Generator):
    s = state.copy()
    N = len(s.perm)
    a = int(rng.integers(0, N))
    b = int(rng.integers(0, N))
    i, j = (a, b) if a < b else (b, a)

    # 记录被移除的元素
    s.removed = s.perm[i:j+1].copy()
    # 移除片段
    s.perm = np.concatenate([s.perm[:i], s.perm[j+1:]])
    return s


def destroy_random_k(state: PermState, rng: np.random.Generator, ratio=0.05):
    s = state.copy()
    N = len(s.perm)
    k = max(1, int(N * ratio))
    idx = rng.choice(N, size=k, replace=False)

    mask = np.ones(N, dtype=bool)
    mask[idx] = False

    # 记录被移除的元素
    s.removed = s.perm[~mask].copy()
    # 仅保留未被移除的
    s.perm = s.perm[mask]
    return s


# -------------------------
# 4) Repair operators
# -------------------------
def repair_random_insert(state: PermState, rng: np.random.Generator):
    # 注意：这里的 state 是 destroy 算子返回的状态（已经包含 .removed 属性）
    s = state.copy()
    
    # [修复] 应该从输入的 state 中读取 removed，而不是从刚 copy 出来的 s 中读
    # 因为 s = state.copy() 不会复制动态添加的属性
    removed = getattr(state, "removed", np.array([], dtype=np.int64))

    perm_list = s.perm.tolist()
    for x in removed:
        pos = int(rng.integers(0, len(perm_list) + 1))
        perm_list.insert(pos, int(x))

    s.perm = np.array(perm_list, dtype=np.int64)
    # 修复完成后，通常不需要保留 removed 属性
    return s


def repair_type_cluster_insert(state: PermState, rng: np.random.Generator):
    """
    利用你 GPU 引擎里已有的 patient_main_type_id 做一个
    “按类型聚合插回”的修复算子，面向换模惩罚。
    """
    s = state.copy()
    
    # [修复] 从输入的 state 读取 removed
    removed = getattr(state, "removed", np.array([], dtype=np.int64))

    # 从 GPU 引擎拿类型 id（Tensor）
    # 注意：需要确保 removed 里的索引在 type_ids 范围内（通常没问题）
    type_ids = s.engine.patient_main_type_id.detach().cpu().numpy()

    removed_list = removed.tolist()
    # 按类型排序，尝试让相同类型的检查聚在一起
    removed_list.sort(key=lambda pid: int(type_ids[pid]))

    perm_list = s.perm.tolist()
    base = int(rng.integers(0, len(perm_list) + 1))
    
    # 简单策略：选个基准点，依次往后插
    for t, x in enumerate(removed_list):
        pos = min(base + t, len(perm_list))
        perm_list.insert(pos, int(x))

    s.perm = np.array(perm_list, dtype=np.int64)
    return s


# -------------------------
# 5) 构建 ALNS baseline
# -------------------------
def build_engine_via_optimizer(user_mod, patients, machine_exam_map, block_start_date=None):
    # 借用你的 MultiRunOptimizer 来构建 GPU fitness 引擎
    opt = user_mod.MultiRunOptimizer(
        patients=patients,
        machine_exam_map=machine_exam_map,
        num_parallel_runs=1,
        pop_size_per_run=1,
        block_start_date=block_start_date
    )
    opt._ensure_gpu_engine()
    return opt._gpu_engine, opt


def run_alns(engine, init_perm: np.ndarray, seed=0, iters=2000):
    rng = np.random.default_rng(seed)
    alns = ALNS(rng)

    # destroy
    alns.add_destroy_operator(destroy_random_segment)
    # [修复] 给 lambda 算子指定唯一的 name，防止因名称冲突(都是<lambda>)被覆盖或计数错误
    alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.03), name="destroy_k_03")
    alns.add_destroy_operator(lambda st, r: destroy_random_k(st, r, ratio=0.08), name="destroy_k_08")

    # repair
    alns.add_repair_operator(repair_random_insert)
    alns.add_repair_operator(repair_type_cluster_insert)

    # 选择算子
    selector = RouletteWheel(
        scores=[5, 2, 1, 0.5],
        decay=0.8,
        num_destroy=3,
        num_repair=2
    )
    
    # 模拟退火
    # 适应度通常很大（e.g. -200,000），所以温度也要相应大一些
    accept = SimulatedAnnealing(
        start_temperature=5000, 
        end_temperature=100,
        step=0.995
    )
    stop = MaxIterations(iters)

    init_state = PermState(engine, init_perm)
    result = alns.iterate(init_state, selector, accept, stop)
    best = result.best_state
    return best.perm, best.objective()


# -------------------------
# 6) CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_code", type=str, default="测量时间full-GPU实验-Multi.py")

    # ✅ 直接写死默认路径（你给的三份）
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

    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    user_mod = load_user_module(args.user_code)

    # 复用你原代码的数据导入
    patients = user_mod.import_data(args.patient_file, args.duration_file)
    machine_exam_map = user_mod.import_device_constraints(args.device_file)

    engine, opt = build_engine_via_optimizer(user_mod, patients, machine_exam_map)

    # 初始解：登记日期排序对应的索引顺序（与你现有逻辑一致）
    N = opt.N
    init_perm = np.arange(N, dtype=np.int64)

    best_perm, best_cost = run_alns(engine, init_perm, seed=args.seed, iters=args.iters)

    # 输出结果（可按你原格式进一步保存）
    best_fitness = -best_cost
    print("=== ALNS baseline finished ===")
    print(f"N = {N}")
    print(f"best_fitness = {best_fitness:.4f}")

    # 如需把 perm 映射回 patient_id：
    idx_to_cid = opt._idx_to_cid
    best_patient_order = [idx_to_cid[i] for i in best_perm.tolist()]
    print("best_patient_order_head20 =", best_patient_order[:20])


if __name__ == "__main__":
    main()
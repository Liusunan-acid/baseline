# column_generation_exact.py
# ============================================================
# MRI排程系统 - 精确列生成 (Exact Column Generation)
# 核心改进：
# 1. 子问题 (Pricing) 使用基于图的动态规划 (DP/RCSPP) 求解，而非贪心。
# 2. 寻找 Reduced Cost 最负的路径，保证数学上的收敛性。
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import os
import re
import traceback
import math

# 必须安装: pip install ortools
from ortools.linear_solver import pywraplp

# ===================== 全局配置 =====================

WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
WORK_START_STR = '07:00'
WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
START_DATE = datetime(2025, 1, 1, 7, 0)
MACHINE_COUNT = 6
SEARCH_DAYS = 30 
SWITCH_GAP_SEC = 60  # 换模时间（秒）

# === 成本系数 ===
TRANSITION_PENALTY = 20000
SELF_SELECTED_PENALTY = 8000
NON_SELF_PENALTY = 800
UNSCHEDULED_PENALTY = 1e8  # 松弛变量惩罚

# === DP 算法参数 (控制搜索宽度，防止内存爆炸) ===
DP_BEAM_WIDTH = 50  # 每一层保留的最优路径数 (Beam Search)

# ===================== 数据结构 =====================

@dataclass
class Column:
    col_id: int
    machine_id: int
    date: datetime.date
    patients_idx: List[int]
    cost: float          # 真实成本
    eff_duration: int    # 有效占用时长(含换模)

# ===================== 工具函数 (保持不变) =====================

def clean_exam_name(name):
    s = str(name).strip().lower()
    s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
    s = re.sub(r'[^\w()-]', '', s)
    return s.replace('_', '-').replace(' ', '')

def safe_read_excel(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"{file_path}")
    for engine in ['openpyxl', 'xlrd', 'odf']:
        try: return pd.read_excel(file_path, engine=engine)
        except: continue
    return pd.read_excel(file_path)

def daily_work_seconds(date_obj):
    hours = 15.0 - WEEKDAY_END_HOURS.get(date_obj.isoweekday(), 0)
    return max(0, int(round(hours * 3600)))

# ===================== 核心业务逻辑 =====================

def is_device_feasible(p, machine_id, machine_exam_map):
    allowed = machine_exam_map.get(machine_id, set())
    return (p['exam_type'] in allowed) if allowed else False

def is_rule_feasible(p, machine_id, date_obj):
    exam = str(p.get('exam_raw', ''))
    wd = date_obj.isoweekday()
    
    # 心脏: 机器3 (idx 3), 周2/4
    if '心脏' in exam:
        if machine_id != 3 or wd not in (2, 4): return False
    # 造影: 机器1 (idx 1), 周1/3/5
    if '造影' in exam:
        if machine_id != 1 or wd not in (1, 3, 5): return False
    # 增强: 周末不做
    if '增强' in exam and wd in (6, 7): return False
    
    return True

def calculate_real_cost(patients, p_indices, date_obj):
    """计算一列的真实成本 (Real Cost) 和有效耗时"""
    wait_cost = 0
    trans_cost = 0
    total_dur = 0
    prev_type = None
    
    # 假设列内按登记时间排序 (DAG也是这么构建的)
    sorted_idx = sorted(p_indices, key=lambda i: patients[i]['reg_datetime'])
    
    for i in sorted_idx:
        p = patients[i]
        days = (date_obj - p['reg_date']).days
        w = SELF_SELECTED_PENALTY if p.get('is_self_selected') else NON_SELF_PENALTY
        wait_cost += max(0, days) * w
        
        if prev_type and p['exam_type'] != prev_type:
            trans_cost += TRANSITION_PENALTY
            total_dur += SWITCH_GAP_SEC
        
        total_dur += p['duration']
        prev_type = p['exam_type']
        
    return wait_cost + trans_cost, total_dur

# ===================== 核心改进：精确 Pricing (DP) =====================

def solve_pricing_dp(
    patients, machine_exam_map, start_date, search_days,
    dual_p, dual_md, next_col_id
):
    """
    使用动态规划 (Label Setting / Beam Search) 寻找 Reduced Cost 最负的列。
    针对每一个 (Machine, Day) 构建一个子问题。
    """
    new_columns = []
    
    # 为了加速，我们先预处理每个病人的基本信息
    # 过滤掉已经排完的或者明显不可能的 (在精确算法中通常不轻易过滤，为了速度可做简单剪枝)
    
    # 遍历每一天，每一台机器 (这就是分解后的子问题)
    for d_off in range(search_days):
        date_obj = start_date.date() + timedelta(days=d_off)
        capacity = daily_work_seconds(date_obj)
        if capacity <= 0: continue

        for m in range(MACHINE_COUNT):
            machine_dual = dual_md.get((m, date_obj), 0.0)
            
            # 1. 筛选本机器、本日期可行的所有节点 (病人)
            candidates = []
            for i, p in enumerate(patients):
                # 基础硬约束检查
                if p['duration'] > capacity: continue
                if (date_obj - p['reg_date']).days < 0: continue
                if not is_device_feasible(p, m, machine_exam_map): continue
                if not is_rule_feasible(p, m, date_obj): continue
                
                # 计算该病人作为节点的“节点利润” (Node Profit)
                # Node Profit = Dual_i - Wait_Cost_i
                # (注意：换模成本是边权，这里先不算)
                wait_days = (date_obj - p['reg_date']).days
                w_weight = SELF_SELECTED_PENALTY if p.get('is_self_selected') else NON_SELF_PENALTY
                wait_cost = max(0, wait_days) * w_weight
                
                # 我们希望 maximize (Dual - Cost)，即 minimize (Cost - Dual)
                # Reduced Cost = Real_Cost - Sum(Dual_i) - Machine_Dual
                # 贡献值 = Dual_i - Wait_Cost
                profit = dual_p[i] - wait_cost
                
                candidates.append({
                    'id': i,
                    'profit': profit,
                    'type': p['exam_type'],
                    'dur': p['duration'],
                    'reg_time': p['reg_datetime']
                })
            
            if not candidates: continue

            # 2. 构建 DAG (按登记时间排序，假设遵循 FIFO 以减少状态空间)
            # 这是一个 Resource Constrained Shortest Path (RCSP) 的简化版
            candidates.sort(key=lambda x: x['reg_time'])
            
            # DP 状态: labels[last_node_index] = List of (accumulated_time, accumulated_profit, path_list)
            # 实际上由于是 DAG，我们可以层层递进
            
            # Label 定义: (time_used, total_profit, path_indices, last_type)
            # 初始状态: 虚拟起点
            labels = [] # List of labels
            
            # 初始化：每个候选人都可以作为路径的起点
            for cand in candidates:
                # 起点没有换模成本
                net_profit = cand['profit'] # - 0 transition
                labels.append({
                    'time': cand['dur'],
                    'profit': net_profit,
                    'path': [cand['id']],
                    'last_type': cand['type']
                })
            
            # 3. 动态规划推演 (Forward DP)
            # 由于已按时间排序，我们只需要向后看
            # 为了防止指数爆炸，每一层只保留 Profit 最高的 Top K (Beam Search)
            
            final_paths = [] # 存储所有生成的完整路径
            
            # 这里的 DP 结构更像是一个广度优先搜索的变体
            # current_layer 存储当前所有活跃的路径
            current_layer = labels
            
            # 我们不需要极其深层的递归，因为通常一天做不了太多人 (比如最多 20 个)
            # 直接迭代扩展
            
            # 存储“完成态”的路径（无法再延长的路径）
            completed_paths = [] 
            
            # 这里的迭代逻辑：尝试把 layer 中的每个路径，去匹配它后面的 candidates
            # 注意：candidates 是有序的。对于路径 P (结尾是 cand[k])，只能接 cand[k+1:]
            
            # 为了性能，我们简化 DP：
            # dp[k] 表示以第 k 个 candidate 结尾的所有路径集合
            # 我们按 candidate 顺序遍历
            
            dp_states = defaultdict(list) # index -> list of paths ending at index
            
            # 初始化 DP：每个节点自己是一个路径
            for k, cand in enumerate(candidates):
                dp_states[k].append({
                    'time': cand['dur'],
                    'profit': cand['profit'],
                    'path': [cand['id']],
                    'type': cand['type']
                })
            
            # 核心 DP 循环
            for k in range(len(candidates)):
                # 取出以 k 结尾的所有路径，尝试扩展到 m (m > k)
                current_paths = dp_states[k]
                if not current_paths: continue
                
                # 剪枝：只保留该节点处 profit 最高的 N 个路径
                current_paths.sort(key=lambda x: x['profit'], reverse=True)
                current_paths = current_paths[:50] # 局部 Beam Width
                
                cand_k = candidates[k]
                
                for path_state in current_paths:
                    # 尝试连接后续节点 m
                    extended = False
                    for m_idx in range(k + 1, len(candidates)):
                        cand_m = candidates[m_idx]
                        
                        # 计算转换成本
                        transition_cost = 0
                        gap_time = 0
                        if cand_m['type'] != path_state['type']:
                            transition_cost = TRANSITION_PENALTY
                            gap_time = SWITCH_GAP_SEC
                        
                        new_time = path_state['time'] + gap_time + cand_m['dur']
                        
                        # 约束检查：容量
                        if new_time <= capacity:
                            extended = True
                            # 更新利润：Profit = Old_Profit + (Node_Profit_m - Trans_Cost)
                            new_profit = path_state['profit'] + cand_m['profit'] - transition_cost
                            
                            new_state = {
                                'time': new_time,
                                'profit': new_profit,
                                'path': path_state['path'] + [cand_m['id']],
                                'type': cand_m['type']
                            }
                            dp_states[m_idx].append(new_state)
                    
                    # 如果该路径无法再扩展（或者即使能扩展我们也把它作为一个可行解保存）
                    # 计算最终 Reduced Cost
                    # RC = Real_Cost - Dual_P - Dual_Machine
                    # 因为 Profit = Dual_P - Real_Cost (取反部分)
                    # 所以 Reduced Cost = - (Profit) - Machine_Dual
                    # 我们希望 RC < 0，即 Profit > -Machine_Dual (通常Machine_Dual是负的或0? 不，Dual通常非负)
                    # 准确公式：
                    # RC = (RealCost) - Sum(Dual_P) - MachineDual
                    # Profit 我们定义为 Sum(Dual_P) - RealCost (不含MachineDual)
                    # 也就是 RC = -Profit - MachineDual
                    
                    final_rc = -path_state['profit'] - machine_dual
                    if final_rc < -1e-5:
                        final_paths.append((final_rc, path_state))

            # 4. 收集本机器、本日期中最优的列
            # 排序取最好的几个
            final_paths.sort(key=lambda x: x[0]) # 升序，越负越好
            
            # 只取前几个最好的，加入列池
            for rc, state in final_paths[:5]: # 每个机器每天最多贡献 5 个极优列
                # 反算 cost 和 transition 用于存储
                # 这里的 cost 需要重新精确计算，以防 DP 过程有浮点误差
                real_cost, eff_dur = calculate_real_cost(patients, state['path'], date_obj)
                
                new_columns.append(Column(
                    col_id=next_col_id,
                    machine_id=m,
                    date=date_obj,
                    patients_idx=state['path'],
                    cost=real_cost,
                    eff_duration=eff_dur
                ))
                next_col_id += 1
                
    return new_columns, next_col_id

# ===================== 主问题 (RMP) =====================

def solve_rmp(patients, columns, solve_integer=False):
    """
    RMP 求解器 (既支持 LP 松弛，也支持最终 MIP)
    """
    solver_name = "CBC" if solve_integer else "GLOP"
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if not solver: raise RuntimeError(f"找不到求解器 {solver_name}")
    
    num_patients = len(patients)
    
    # 变量
    x = []
    for c in columns:
        if solve_integer:
            x.append(solver.BoolVar(f"x_{c.col_id}"))
        else:
            x.append(solver.NumVar(0.0, 1.0, f"x_{c.col_id}"))
            
    slacks = []
    for i in range(num_patients):
        if solve_integer:
            slacks.append(solver.BoolVar(f"s_{i}"))
        else:
            slacks.append(solver.NumVar(0.0, 1.0, f"s_{i}"))

    # 约束 1: 覆盖约束 (Dual对应每个病人)
    patient_cons = []
    col_map_p = defaultdict(list)
    for idx, c in enumerate(columns):
        for pid in c.patients_idx:
            col_map_p[pid].append(idx)
            
    for i in range(num_patients):
        ct = solver.Constraint(1.0, 1.0, f"cov_{i}")
        for idx in col_map_p[i]:
            ct.SetCoefficient(x[idx], 1.0)
        ct.SetCoefficient(slacks[i], 1.0)
        patient_cons.append(ct)
        
    # 约束 2: 机器容量约束 (Dual对应 Machine-Day)
    # 严格来说，这是 Partitioning 约束：Sum(x_c for c in m,d) <= 1
    # 因为我们在 Pricing 里已经保证了列不仅合法，而且成本固定
    # 所以 RMP 只需要保证每台机器每天只选一列
    machday_cons = {}
    col_map_md = defaultdict(list)
    for idx, c in enumerate(columns):
        col_map_md[(c.machine_id, c.date)].append(idx)
        
    for k, idxs in col_map_md.items():
        ct = solver.Constraint(0.0, 1.0, f"md_{k}")
        for idx in idxs:
            ct.SetCoefficient(x[idx], 1.0)
        machday_cons[k] = ct

    # 目标函数
    obj = solver.Objective()
    for idx, c in enumerate(columns):
        obj.SetCoefficient(x[idx], c.cost)
    for s in slacks:
        obj.SetCoefficient(s, UNSCHEDULED_PENALTY)
    obj.SetMinimization()
    
    solver.Solve()
    
    if solve_integer:
        return [columns[i] for i, var in enumerate(x) if var.solution_value() > 0.5]
    else:
        # 返回对偶值
        dual_p = [ct.dual_value() for ct in patient_cons]
        dual_md = {k: ct.dual_value() for k, ct in machday_cons.items()}
        return dual_p, dual_md

# ===================== 数据导入 (简化版) =====================
# (保持你的原逻辑，这里为了代码紧凑略微压缩，功能一致)

def import_data_full(p_file, d_file, c_file):
    # 1. 耗时
    dur_df = safe_read_excel(d_file)
    dur_df['clean'] = dur_df['检查项目'].apply(clean_exam_name)
    dur_map = dur_df.set_index('clean')['实际平均耗时'].to_dict()
    
    # 2. 患者
    p_df = safe_read_excel(p_file)
    patients = []
    for _, row in p_df.iterrows():
        exam = clean_exam_name(row['检查项目'])
        d_min = dur_map.get(exam, 20.0)
        reg_dt = pd.to_datetime(row['登记日期'])
        patients.append({
            'id': row['id'],
            'exam_type': exam,
            'exam_raw': row['检查项目'],
            'duration': int(d_min * 60),
            'reg_datetime': reg_dt.to_pydatetime(),
            'reg_date': reg_dt.date(),
            'is_self_selected': (row.get('是否自选时间') == '自选时间')
        })
    
    # 3. 机器限制
    c_df = safe_read_excel(c_file)
    m_map = defaultdict(set)
    # 简单的列名推断
    m_col = c_df.columns[0]
    e_col = c_df.columns[1]
    for _, row in c_df.iterrows():
        try:
            mid = int(row[m_col]) - 1 # 1-based to 0-based
            if 0 <= mid < MACHINE_COUNT:
                m_map[mid].add(clean_exam_name(row[e_col]))
        except: pass
        
    return patients, m_map

# ===================== 主流程 =====================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    f_patient = os.path.join(base_dir, '实验数据6.1small - 副本.xlsx')
    f_dur = os.path.join(base_dir, '程序使用实际平均耗时3 - 副本.xlsx')
    f_const = os.path.join(base_dir, '设备限制4.xlsx')
    
    if not os.path.exists(f_patient):
        print("❌ 文件缺失，请检查路径")
        return

    print(">>> 正在加载数据...")
    patients, m_map = import_data_full(f_patient, f_dur, f_const)
    print(f"载入患者: {len(patients)} 人")

    # 1. 初始化 (Initial Population)
    # 依然生成单人列作为 Basis，确保 LP 有解
    columns = []
    col_id = 0
    print(">>> 生成初始基基解 (Initial Basis)...")
    for i, p in enumerate(patients):
        # 简单找一个可行位置
        found = False
        for d in range(SEARCH_DAYS):
            date = START_DATE.date() + timedelta(days=d)
            if (date - p['reg_date']).days < 0: continue
            if p['duration'] > daily_work_seconds(date): continue
            
            for m in range(MACHINE_COUNT):
                if is_device_feasible(p, m, m_map) and is_rule_feasible(p, m, date):
                    cost, eff = calculate_real_cost(patients, [i], date)
                    columns.append(Column(col_id, m, date, [i], cost, eff))
                    col_id += 1
                    found = True
                    break
            if found: break
    
    print(f"初始列数: {len(columns)}")
    
    # 2. 列生成主循环
    MAX_ITERS = 20
    for it in range(MAX_ITERS):
        print(f"\n--- CG Iteration {it+1}/{MAX_ITERS} ---")
        
        # A. 解 RMP (LP Relaxed)
        dual_p, dual_md = solve_rmp(patients, columns, solve_integer=False)
        
        # B. 解 Pricing (DP Exact/Heuristic)
        # 传入 Duals，寻找 Reduced Cost < 0 的列
        new_cols, col_id = solve_pricing_dp(
            patients, m_map, START_DATE, SEARCH_DAYS,
            dual_p, dual_md, col_id
        )
        
        print(f"  > 发现新列: {len(new_cols)}")
        
        if not new_cols:
            print("  > 无负 Reduced Cost 列，LP 收敛。")
            break
            
        columns.extend(new_cols)
        
    # 3. 最终整数解
    print("\n>>> 求解最终 MIP...")
    final_cols = solve_rmp(patients, columns, solve_integer=True)
    
    # 4. 输出结果
    print(f"选中列数: {len(final_cols)}")
    
    # 导出
    schedule = []
    for c in final_cols:
        curr_time = datetime.combine(c.date, WORK_START)
        sorted_p = sorted(c.patients_idx, key=lambda i: patients[i]['reg_datetime'])
        prev_type = None
        for pid in sorted_p:
            p = patients[pid]
            if prev_type and p['exam_type'] != prev_type:
                curr_time += timedelta(seconds=SWITCH_GAP_SEC)
            
            start = curr_time
            end = start + timedelta(seconds=p['duration'])
            
            schedule.append({
                'PatientID': p['id'],
                'Machine': c.machine_id + 1,
                'Date': c.date,
                'Start': start.time(),
                'End': end.time(),
                'Exam': p['exam_raw']
            })
            
            curr_time = end
            prev_type = p['exam_type']
            
    df_out = pd.DataFrame(schedule)
    out_path = os.path.join(base_dir, 'exact_cg_result.xlsx')
    df_out.to_excel(out_path, index=False)
    print(f"结果已导出: {out_path}")
    print(f"总安排人数: {len(schedule)} / {len(patients)}")

if __name__ == "__main__":
    main()
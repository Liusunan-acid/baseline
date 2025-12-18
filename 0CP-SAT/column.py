# # # column_generation_baseline.py
# # # ============================================================
# # # MRI排程系统 - 列生成算法基线 (Column Generation Baseline)
# # # 核心思想：使用列生成(CG)处理大规模排程，主问题(RMP)负责选择，子问题(Pricing)负责生成
# # # 修复：引入松弛变量(Slack Variables)解决 Infeasible 问题
# # # ============================================================

# # from __future__ import annotations
# # from dataclasses import dataclass
# # from typing import List, Dict, Tuple, Set, Optional
# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta
# # from collections import defaultdict
# # import os
# # import re
# # import traceback

# # # 必须安装: pip install ortools
# # from ortools.linear_solver import pywraplp

# # # ===================== 全局配置与常量 =====================

# # # 定义不同星期的每日工作结束时间（用于计算每日可用工时）
# # # 15.0 - WEEKDAY_END_HOURS[w] = 每日可用小时数
# # WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}

# # WORK_START_STR = '07:00'
# # WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# # START_DATE = datetime(2025, 1, 1, 7, 0) # 排程开始日期
# # MACHINE_COUNT = 6   # 机器总数
# # SEARCH_DAYS = 130    # 向后搜索/排程的天数窗口

# # # === 评分与惩罚权重 (越低越好，用于目标函数最小化) ===
# # TRANSITION_PENALTY = 20000    # 换模惩罚（同一天切换检查类型）
# # SELF_SELECTED_PENALTY = 8000  # 自选时间患者的等待惩罚系数 (元/天)
# # NON_SELF_PENALTY = 800        # 非自选时间患者的等待惩罚系数 (元/天)
# # DEVICE_PENALTY = 500000       # 违反设备/硬规则的巨额惩罚
# # LOGICAL_PENALTY = 10000       # 逻辑错误（如排在登记日前）惩罚

# # # 新增：未排程惩罚 (Slack Variable Cost)
# # # 必须设置得非常大，大于任何可能的正常列成本，确保求解器只在万不得已时才丢弃患者
# # UNSCHEDULED_PENALTY = 1e9     

# # # ===================== 数据清洗与导入工具函数 =====================

# # def clean_exam_name(name):
# #     """标准化检查项目名称，去除特殊符号，统一格式"""
# #     s = str(name).strip().lower()
# #     # 统一括号格式
# #     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
# #     # 去除杂质字符
# #     s = re.sub(r'[^\w()-]', '', s)
# #     return s.replace('_', '-').replace(' ', '')

# # def safe_read_excel(file_path, sheet_name=0):
# #     """尝试使用不同的引擎读取Excel，兼容旧版xls和新版xlsx"""
# #     if file_path.endswith('.xlsx'):
# #         engines = ['openpyxl', 'odf']
# #     elif file_path.endswith('.xls'):
# #         engines = ['xlrd']
# #     else:
# #         engines = ['openpyxl', 'xlrd', 'odf']
    
# #     for engine in engines:
# #         try:
# #             return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
# #         except Exception:
# #             continue
# #     # 最后尝试默认引擎
# #     return pd.read_excel(file_path, sheet_name=sheet_name)

# # def import_data(patient_file, duration_file):
# #     """
# #     导入患者数据和耗时标准
# #     Returns:
# #         patients: List[dict] 患者列表，包含ID、类型、耗时(秒)、登记时间等
# #     """
# #     print(f"正在读取患者数据: {patient_file}")
# #     try:
# #         # 1. 读取耗时标准
# #         duration_df = safe_read_excel(duration_file)
# #         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
# #         # 建立 检查项目 -> 平均耗时(分钟) 的映射
# #         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

# #         # 2. 读取患者列表
# #         patient_df = safe_read_excel(patient_file)
# #         patients = []

# #         for _, row in patient_df.iterrows():
# #             if pd.isnull(row.get('id')) or pd.isnull(row.get('登记日期')):
# #                 continue

# #             raw_id = str(row['id']).strip()
# #             reg_dt = pd.to_datetime(row['登记日期'])
# #             cid = (raw_id, reg_dt.strftime('%Y%m%d')) # 复合ID

# #             exam_type = clean_exam_name(row['检查项目'])

# #             # 获取耗时，默认为15分钟
# #             val = exam_durations.get(exam_type, 15.0)
# #             try:
# #                 duration_raw_min = float(val)
# #             except Exception:
# #                 duration_raw_min = 15.0
            
# #             # 转换为秒
# #             duration_sec = int(round(duration_raw_min * 60))
# #             duration_sec = max(1, duration_sec)

# #             is_self_selected = (row.get('是否自选时间') == '自选时间')

# #             p = {
# #                 'id': raw_id,
# #                 'cid': cid,
# #                 'exam_type': exam_type,
# #                 'duration': duration_sec,  # 秒
# #                 'reg_date': reg_dt.date(),
# #                 'reg_datetime': reg_dt,
# #                 'is_self_selected': is_self_selected,
# #                 'original_row': row
# #             }
# #             patients.append(p)

# #         # 按登记时间排序，这对后续贪心算法有帮助（优先处理先来的）
# #         patients.sort(key=lambda x: x['reg_datetime'])
# #         print(f"成功导入 {len(patients)} 名患者。")
# #         return patients
# #     except Exception as e:
# #         print(f"数据导入错误: {e}")
# #         traceback.print_exc()
# #         raise

# # def import_device_constraints(file_path):
# #     """读取设备限制：某台机器只能做哪些项目"""
# #     print(f"正在读取设备限制: {file_path}")
# #     try:
# #         df = safe_read_excel(file_path)
# #         machine_exam_map = defaultdict(set)
# #         for _, row in df.iterrows():
# #             mid = int(row['设备']) - 1 # 转为0-based索引
# #             exam = clean_exam_name(row['检查项目'])
# #             machine_exam_map[mid].add(exam)
# #         return machine_exam_map
# #     except Exception as e:
# #         print(f"导入设备限制数据错误: {e}")
# #         traceback.print_exc()
# #         raise


# # # ===================== 业务规则与逻辑校验 =====================

# # def daily_work_seconds(date_obj):
# #     """计算某一天该机器的总可用秒数"""
# #     weekday = date_obj.isoweekday()
# #     # 假设每日标准结束时间为 15:00 (即15.0)，减去特定星期的缩减时间
# #     hours_avail = 15.0 - WEEKDAY_END_HOURS.get(weekday, 0)
# #     return int(round(hours_avail * 3600))

# # def is_rule_feasible(p, machine_id: int, date_obj):
# #     """
# #     硬规则检查：
# #     1. 心脏检查：必须周二/周四，且必须在机器4 (index 3)
# #     2. 造影检查：必须周一/三/五，且必须在机器2 (index 1)
# #     3. 增强检查：周末禁止做
# #     """
# #     exam_name = str(p['exam_type'])
# #     weekday = date_obj.isoweekday()
# #     m_idx = machine_id

# #     is_heart = '心脏' in exam_name
# #     is_angio = '造影' in exam_name
# #     is_contrast = '增强' in exam_name

# #     # 规则1：心脏
# #     if is_heart:
# #         ok_wd = (weekday == 2 or weekday == 4)
# #         ok_mc = (m_idx == 3)
# #         if not (ok_wd and ok_mc):
# #             return False

# #     # 规则2：造影
# #     if is_angio:
# #         ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)
# #         ok_mc = (m_idx == 1)
# #         if not (ok_wd and ok_mc):
# #             return False

# #     # 规则3：周末无增强
# #     is_weekend = (weekday == 6 or weekday == 7)
# #     if is_contrast and is_weekend:
# #         return False

# #     return True

# # def is_device_feasible(p, machine_id: int, machine_exam_map):
# #     """检查设备能力限制"""
# #     allowed = machine_exam_map.get(machine_id, set())
# #     return (p['exam_type'] in allowed) if allowed else False

# # def patient_wait_weight(p):
# #     """获取患者的等待权重"""
# #     return SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY


# # # ===================== 列生成核心数据结构 =====================

# # @dataclass
# # class Column:
# #     """
# #     列（Column）代表一个具体的排班方案片段：
# #     即“某台机器(machine_id)在某一天(date)服务了一组患者(patients_idx)”
# #     """
# #     col_id: int
# #     machine_id: int
# #     date: datetime.date
# #     patients_idx: List[int]         # 患者在全局列表中的索引
# #     cost: int                       # 该列的计算成本 (reduced cost计算的基础)
# #     transition_count: int           # 该列内部的换模次数


# # # ===================== 成本计算函数 =====================

# # def compute_column_cost(patients: List[dict], col_patients_idx: List[int], date_obj):
# #     """
# #     计算单列的实际成本 (Real Cost)：
# #     Cost = (总等待天数 * 权重) + (换模次数 * 换模惩罚)
# #     """
# #     if not col_patients_idx:
# #         return 0, 0

# #     # 为了计算换模，假设列内患者按登记时间排序执行
# #     sorted_idx = sorted(col_patients_idx, key=lambda i: patients[i]['reg_datetime'])

# #     wait_cost = 0
# #     transition_cnt = 0
# #     prev_type = None

# #     for i in sorted_idx:
# #         p = patients[i]
# #         wait_days = (date_obj - p['reg_date']).days
        
# #         if wait_days < 0:
# #             # 逻辑防御：排在登记日之前的非法情况
# #             wait_cost += LOGICAL_PENALTY
# #         else:
# #             wait_cost += wait_days * patient_wait_weight(p)

# #         # 换模检测
# #         if prev_type is not None and p['exam_type'] != prev_type:
# #             transition_cnt += 1
# #         prev_type = p['exam_type']

# #     cost = int(wait_cost + transition_cnt * TRANSITION_PENALTY)
# #     return cost, transition_cnt


# # # ===================== 第一步：初始化 (Initialization) =====================

# # def build_initial_columns(patients, machine_exam_map, start_date, search_days):
# #     """
# #     生成初始列集合。
# #     策略：为确保有解，尝试为每个患者分配一个“最早可行的单人列”。
# #     """
# #     print("正在生成初始列...")
# #     columns: List[Column] = []
# #     col_id = 0

# #     for i, p in enumerate(patients):
# #         assigned = False
# #         earliest_date = max(p['reg_date'], start_date.date())
# #         # 从最早可行日期开始向后找几天
# #         start_offset = (earliest_date - start_date.date()).days

# #         for d in range(start_offset, start_offset + search_days):
# #             date_obj = start_date.date() + timedelta(days=d)
# #             # 跳过休息日/无工时日
# #             if daily_work_seconds(date_obj) <= 0:
# #                 continue

# #             for m in range(MACHINE_COUNT):
# #                 # 检查设备能力
# #                 if not is_device_feasible(p, m, machine_exam_map):
# #                     continue
# #                 # 检查业务规则
# #                 if not is_rule_feasible(p, m, date_obj):
# #                     continue

# #                 # 只要当天容量够一个人用
# #                 if p['duration'] <= daily_work_seconds(date_obj):
# #                     cost, tcnt = compute_column_cost(patients, [i], date_obj)
# #                     columns.append(Column(col_id, m, date_obj, [i], cost, tcnt))
# #                     col_id += 1
# #                     assigned = True
# #                     break 
# #             if assigned:
# #                 break
        
# #         # 修复：不再强制生成可能冲突的“兜底列”。
# #         # 如果这里找不到列，后面的RMP会使用松弛变量（Slack）来处理该患者，
# #         # 并报告该患者“未排程”，而不是让程序崩溃。

# #     return columns, col_id


# # # ===================== 第二步：主问题 (RMP LP) =====================

# # def solve_rmp_lp(columns: List[Column], num_patients: int):
# #     """
# #     求解限制主问题 (Restricted Master Problem) 的线性规划松弛。
# #     目标：min sum(cost_c * x_c) + sum(UNSCHEDULED_PENALTY * slack_i)
# #     约束1 (覆盖): sum(x_c) + slack_i == 1  (允许 slack_i=1 代表未被覆盖)
# #     约束2 (机器): sum(x_c) <= 1
# #     """
# #     solver = pywraplp.Solver.CreateSolver("GLOP")
# #     if solver is None:
# #         raise RuntimeError("无法创建 GLOP 求解器，请检查 ortools 是否安装正确。")

# #     # 定义变量 x_c (0 <= x_c <= 1, 连续变量)
# #     x = []
# #     for c in columns:
# #         x.append(solver.NumVar(0.0, 1.0, f"x_{c.col_id}"))

# #     # 定义松弛变量 (Slack Variables)，用于处理无法覆盖的患者
# #     slacks = []
# #     for i in range(num_patients):
# #         slacks.append(solver.NumVar(0.0, 1.0, f"slack_{i}"))

# #     # 1. 患者覆盖约束
# #     patient_cons = []
# #     cols_by_patient = [[] for _ in range(num_patients)]
# #     for idx_c, c in enumerate(columns):
# #         for i in c.patients_idx:
# #             cols_by_patient[i].append(idx_c)

# #     for i in range(num_patients):
# #         # sum(x_c) + slack_i = 1
# #         # 如果所有 x_c 都是 0，那么 slack_i 必须是 1，这会触发巨大的惩罚
# #         ct = solver.Constraint(1.0, 1.0, f"cover_p_{i}")
# #         for idx_c in cols_by_patient[i]:
# #             ct.SetCoefficient(x[idx_c], 1.0)
# #         # 加上松弛变量
# #         ct.SetCoefficient(slacks[i], 1.0)
# #         patient_cons.append(ct)

# #     # 2. 机器容量约束
# #     machday_cons = {}
# #     cols_by_machday = defaultdict(list)
# #     for idx_c, c in enumerate(columns):
# #         cols_by_machday[(c.machine_id, c.date)].append(idx_c)

# #     for (m, d), idx_list in cols_by_machday.items():
# #         ct = solver.Constraint(0.0, 1.0, f"machday_{m}_{d}")
# #         for idx_c in idx_list:
# #             ct.SetCoefficient(x[idx_c], 1.0)
# #         machday_cons[(m, d)] = ct

# #     # 3. 目标函数
# #     obj = solver.Objective()
# #     # 正常列的成本
# #     for idx_c, c in enumerate(columns):
# #         obj.SetCoefficient(x[idx_c], float(c.cost))
# #     # 松弛变量的成本（巨额惩罚）
# #     for i in range(num_patients):
# #         obj.SetCoefficient(slacks[i], UNSCHEDULED_PENALTY)
        
# #     obj.SetMinimization()

# #     status = solver.Solve()
# #     if status != pywraplp.Solver.OPTIMAL:
# #         print(f"⚠️ RMP LP 状态: {status} (可能使用松弛变量)")

# #     return solver, x, patient_cons, machday_cons


# # # ===================== 第三步：子问题 (Pricing) =====================

# # def heuristic_pricing(
# #     patients: List[dict],
# #     machine_exam_map,
# #     start_date,
# #     search_days,
# #     dual_p: List[float],
# #     dual_md: Dict[Tuple[int, datetime.date], float],
# #     next_col_id: int,
# #     max_new_cols: int = 80,
# #     candidate_patients_topk: int = 200
# # ):
# #     """
# #     启发式 Pricing 算法
# #     """
# #     num_patients = len(patients)

# #     # 1. 筛选高价值患者：按对偶值降序排列
# #     ranked = sorted(range(num_patients), key=lambda i: dual_p[i], reverse=True)
# #     ranked = ranked[:min(candidate_patients_topk, num_patients)]

# #     new_columns: List[Column] = []
# #     col_id = next_col_id

# #     for d_off in range(search_days):
# #         date_obj = start_date.date() + timedelta(days=d_off)
# #         cap = daily_work_seconds(date_obj)
# #         if cap <= 0:
# #             continue

# #         for m in range(MACHINE_COUNT):
# #             sigma = dual_md.get((m, date_obj), 0.0)

# #             feasible = []
# #             for i in ranked:
# #                 p = patients[i]
# #                 if p['duration'] > cap:
# #                     continue
# #                 if not is_device_feasible(p, m, machine_exam_map):
# #                     continue
# #                 if not is_rule_feasible(p, m, date_obj):
# #                     continue
# #                 if (date_obj - p['reg_date']).days < 0:
# #                     continue
# #                 feasible.append(i)

# #             if not feasible:
# #                 continue

# #             feasible.sort(
# #                 key=lambda i: (dual_p[i] / max(1, patients[i]['duration'])),
# #                 reverse=True
# #             )

# #             packed = []
# #             used = 0
            
# #             for i in feasible:
# #                 dur = patients[i]['duration']
# #                 if used + dur > cap:
# #                     continue 
# #                 packed.append(i)
# #                 used += dur
# #                 if used >= cap * 0.90:
# #                     break

# #             if not packed:
# #                 continue

# #             real_cost, tcnt = compute_column_cost(patients, packed, date_obj)
# #             sum_patient_dual = sum(dual_p[i] for i in packed)
# #             reduced_cost = real_cost - sum_patient_dual - sigma

# #             if reduced_cost < -1e-6:
# #                 new_columns.append(Column(col_id, m, date_obj, packed, real_cost, tcnt))
# #                 col_id += 1
# #                 if len(new_columns) >= max_new_cols:
# #                     return new_columns, col_id

# #     return new_columns, col_id


# # # ===================== 第四步：求解整数解 (MIP) =====================

# # def solve_rmp_mip(columns: List[Column], num_patients: int):
# #     """
# #     求解最终整数规划 (MIP)，同样包含松弛变量以防无解。
# #     """
# #     solver = pywraplp.Solver.CreateSolver("CBC")
# #     if solver is None:
# #         raise RuntimeError("无法创建 CBC 求解器。")

# #     # 列变量 (Binary)
# #     x = []
# #     for c in columns:
# #         x.append(solver.BoolVar(f"x_{c.col_id}"))
    
# #     # 松弛变量 (Binary: 1代表该患者被放弃)
# #     slacks = []
# #     for i in range(num_patients):
# #         slacks.append(solver.BoolVar(f"slack_{i}"))

# #     # 1. 患者覆盖
# #     cols_by_patient = [[] for _ in range(num_patients)]
# #     for idx_c, c in enumerate(columns):
# #         for i in c.patients_idx:
# #             cols_by_patient[i].append(idx_c)

# #     for i in range(num_patients):
# #         ct = solver.Constraint(1.0, 1.0, f"cover_p_{i}")
# #         for idx_c in cols_by_patient[i]:
# #             ct.SetCoefficient(x[idx_c], 1.0)
# #         # 加上松弛变量
# #         ct.SetCoefficient(slacks[i], 1.0)

# #     # 2. 机器约束
# #     cols_by_machday = defaultdict(list)
# #     for idx_c, c in enumerate(columns):
# #         cols_by_machday[(c.machine_id, c.date)].append(idx_c)

# #     for (m, d), idx_list in cols_by_machday.items():
# #         ct = solver.Constraint(0.0, 1.0, f"machday_{m}_{d}")
# #         for idx_c in idx_list:
# #             ct.SetCoefficient(x[idx_c], 1.0)

# #     # 3. 目标
# #     obj = solver.Objective()
# #     for idx_c, c in enumerate(columns):
# #         obj.SetCoefficient(x[idx_c], float(c.cost))
# #     # 松弛变量成本
# #     for i in range(num_patients):
# #         obj.SetCoefficient(slacks[i], UNSCHEDULED_PENALTY)
    
# #     obj.SetMinimization()

# #     print("开始求解最终整数规划...")
# #     solver.SetTimeLimit(60000) 
# #     status = solver.Solve()
    
# #     chosen = []
# #     unscheduled_count = 0
    
# #     if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
# #         for i, var in enumerate(x):
# #             if var.solution_value() > 0.5:
# #                 chosen.append(columns[i])
# #         for i, var in enumerate(slacks):
# #             if var.solution_value() > 0.5:
# #                 unscheduled_count += 1
# #     else:
# #         print(f"⚠️ RMP MIP 依然未找到解，status={status}")
        
# #     print(f"MIP 求解完成。放弃治疗的患者数: {unscheduled_count}")
# #     return chosen


# # # ===================== 结果导出与处理 =====================

# # def build_final_schedule_from_columns(patients: List[dict], chosen_cols: List[Column]):
# #     """
# #     将选中的列（抽象的Machine-Day集合）转换为具体的秒级时间表。
# #     """
# #     final = []
# #     SWITCH_GAP_SEC = 60 

# #     for col in chosen_cols:
# #         date_obj = col.date
# #         m_id = col.machine_id

# #         idxs = sorted(col.patients_idx, key=lambda i: patients[i]['reg_datetime'])

# #         cur_sec = 0
# #         prev_type = None

# #         for i in idxs:
# #             p = patients[i]
# #             if prev_type is not None and p['exam_type'] != prev_type:
# #                 cur_sec += SWITCH_GAP_SEC

# #             start_dt = datetime.combine(date_obj, WORK_START) + timedelta(seconds=cur_sec)
# #             end_dt = start_dt + timedelta(seconds=p['duration'])

# #             record = {
# #                 'patient_id': p['id'],
# #                 'exam_type': p['exam_type'],
# #                 'reg_date': p['reg_date'],
# #                 'is_self_selected': p['is_self_selected'],
# #                 'machine_id': m_id + 1,
# #                 'date': date_obj,
# #                 'start_time': start_dt.time(),
# #                 'end_time': end_dt.time(),
# #                 'wait_days': (date_obj - p['reg_date']).days
# #             }
# #             final.append(record)

# #             cur_sec += p['duration']
# #             prev_type = p['exam_type']

# #     final.sort(key=lambda x: (x['machine_id'], x['date'], x['start_time']))
# #     return final

# # def evaluate_score(final_schedule: List[dict], machine_exam_map):
# #     """
# #     对最终结果进行评分统计
# #     """
# #     if not final_schedule:
# #         return 0, {}

# #     total_score = 0
# #     details = defaultdict(int)

# #     prev_machine = -1
# #     prev_exam_type = None
# #     prev_date = None

# #     for item in final_schedule:
# #         wait_days = (item['date'] - item['reg_date']).days
# #         if wait_days < 0:
# #             total_score -= LOGICAL_PENALTY
# #             details['logical_violation'] += 1
# #             wait_cost = 0
# #         else:
# #             weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
# #             wait_cost = wait_days * weight

# #         total_score -= wait_cost
# #         details['wait_cost'] += wait_cost

# #         if (item['machine_id'] == prev_machine and item['date'] == prev_date):
# #             if item['exam_type'] != prev_exam_type:
# #                 total_score -= TRANSITION_PENALTY
# #                 details['transition_cost'] += TRANSITION_PENALTY
# #                 details['transition_count'] += 1

# #         prev_machine = item['machine_id']
# #         prev_exam_type = item['exam_type']
# #         prev_date = item['date']

# #         weekday = item['date'].isoweekday()
# #         m_idx = item['machine_id'] - 1
        
# #         rule_violated = False
# #         allowed = machine_exam_map.get(m_idx, set())
        
# #         if allowed and (item['exam_type'] not in allowed):
# #             rule_violated = True
# #             details['device_violation'] += 1

# #         exam_name = str(item['exam_type'])
# #         if '心脏' in exam_name and not ((weekday == 2 or weekday == 4) and m_idx == 3):
# #             rule_violated = True
# #             details['heart_violation'] += 1
        
# #         if rule_violated:
# #             total_score -= DEVICE_PENALTY

# #     return total_score, details

# # def export_excel(final_schedule: List[dict], filename: str, score_data=None):
# #     if not final_schedule:
# #         print("无数据导出。")
# #         return

# #     df = pd.DataFrame(final_schedule)
# #     cols = [
# #         'patient_id', 'exam_type', 'reg_date', 'is_self_selected',
# #         'machine_id', 'date', 'start_time', 'end_time', 'wait_days'
# #     ]
# #     for c in cols:
# #         if c not in df.columns:
# #             df[c] = ''
# #     df = df[cols]

# #     with pd.ExcelWriter(filename) as writer:
# #         df.to_excel(writer, sheet_name='详细排程', index=False)
        
# #         if 'date' in df.columns:
# #             stats = df.groupby('date').size().reset_index(name='每日检查量')
# #             stats.to_excel(writer, sheet_name='统计', index=False)

# #         if score_data:
# #             score, details = score_data
# #             score_items = [['Total Score', score]] + [[k, v] for k, v in details.items()]
# #             pd.DataFrame(score_items, columns=['Metric', 'Value']).to_excel(
# #                 writer, sheet_name='评分报告', index=False
# #             )

# #     print(f"✅ 排程文件已生成: {filename}")


# # # ===================== 主流程入口 =====================

# # def column_generation_solve(
# #     patients: List[dict],
# #     machine_exam_map,
# #     start_date: datetime,
# #     search_days: int = SEARCH_DAYS,
# #     max_iters: int = 30,
# #     max_new_cols_per_iter: int = 80
# # ):
# #     print(">>> 启动列生成算法 (Column Generation) <<<")
    
# #     # 1. 初始化
# #     columns, next_col_id = build_initial_columns(
# #         patients, machine_exam_map, start_date, search_days
# #     )
# #     print(f"初始列数: {len(columns)}")

# #     # 2. 迭代 CG Loop
# #     for it in range(1, max_iters + 1):
# #         print(f"\n--- Iteration {it}/{max_iters} ---")

# #         # 求解 RMP
# #         solver_lp, x_vars, patient_cons, machday_cons = solve_rmp_lp(
# #             columns, len(patients)
# #         )

# #         # 提取对偶值
# #         dual_p = [ct.dual_value() for ct in patient_cons]
# #         dual_md = {k: ct.dual_value() for k, ct in machday_cons.items()}

# #         # 求解 Pricing 寻找新列
# #         new_cols, next_col_id = heuristic_pricing(
# #             patients,
# #             machine_exam_map,
# #             start_date,
# #             search_days,
# #             dual_p,
# #             dual_md,
# #             next_col_id,
# #             max_new_cols=max_new_cols_per_iter
# #         )

# #         if not new_cols:
# #             print("没有发现更优的列 (Negative Reduced Cost)，迭代提前结束。")
# #             break

# #         columns.extend(new_cols)
# #         print(f"本轮新增有效列: {len(new_cols)}，当前总列池: {len(columns)}")

# #     # 3. 求解最终整数解
# #     print("\n>>> 进入整数规划阶段 (Integer RMP) <<<")
# #     chosen_cols = solve_rmp_mip(columns, len(patients))
# #     print(f"最终选中列数: {len(chosen_cols)}")

# #     return chosen_cols

# # def main():
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
    
# #     # === 输入文件路径配置 (请修改这里) ===
# #     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
# #     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
# #     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

# #     # 检查文件是否存在
# #     missing_files = [f for f in [patient_file, duration_file, device_constraint_file] if not os.path.exists(f)]
# #     if missing_files:
# #         print(f"❌ 错误：找不到以下数据文件，请确认路径:\n{missing_files}")
# #         return

# #     # 1. 导入数据
# #     patients = import_data(patient_file, duration_file)
# #     machine_exam_map = import_device_constraints(device_constraint_file)

# #     # 估算一周内最大单日容量（秒）
# #     caps = [daily_work_seconds(START_DATE.date() + timedelta(days=i)) for i in range(7)]
# #     cap_max = max(caps)

# #     too_long = [p for p in patients if p['duration'] > cap_max]
# #     print("❌ 单次检查耗时超过任意单日容量的患者数:", len(too_long))
# #     print("cap_max(sec)=", cap_max, "示例duration=", [p['duration'] for p in too_long[:10]])


# #     # 2. 运行求解
# #     chosen_cols = column_generation_solve(
# #         patients,
# #         machine_exam_map,
# #         start_date=START_DATE,
# #         search_days=SEARCH_DAYS,
# #         max_iters=25,               # 最大迭代次数
# #         max_new_cols_per_iter=60    # 每次迭代生成的最大列数
# #     )

# #     # 3. 结果处理
# #     final_schedule = build_final_schedule_from_columns(patients, chosen_cols)
# #     score, details = evaluate_score(final_schedule, machine_exam_map)

# #     # 计算未排程人数
# #     scheduled_pids = set(item['patient_id'] for item in final_schedule)
# #     all_pids = set(p['id'] for p in patients)
# #     missing_count = len(all_pids) - len(scheduled_pids)

# #     print("\n" + "=" * 50)
# #     print("📊 最终结果统计")
# #     print("=" * 50)
# #     print(f"总评分 (负分制): {score:,.0f}")
# #     print(f"等待成本: {details.get('wait_cost', 0):,.0f}")
# #     print(f"换模成本: {details.get('transition_cost', 0):,.0f}")
# #     print(f"未排程人数(通过松弛变量丢弃): {missing_count} 人")
    
# #     # 4. 导出Excel
# #     out_dir = os.path.join(current_dir, 'output_schedules')
# #     os.makedirs(out_dir, exist_ok=True)
# #     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# #     out_xlsx = os.path.join(out_dir, f'schedule_result_{ts}.xlsx')

# #     export_excel(final_schedule, out_xlsx, score_data=(score, details))

# # if __name__ == "__main__":
# #     main()

# # column_generation_routeA.py
# # ============================================================
# # Route A: Proper Column Generation for Machine-Day Patterns
# # 语义：
# #   - 一列 = (machine, day) 的“完整可执行日程模式 pattern”
# #   - Master: 每个 machine-day 最多选 1 列
# #   - Pricing/Init: 为每个 machine-day 生成多个风格的可行 pattern
# #   - Slack: 允许丢弃患者（高惩罚）避免无解
# # ============================================================


# #最后一步运行过慢
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Set, Optional
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from collections import defaultdict
# import os
# import re
# import traceback
# import random

# from ortools.linear_solver import pywraplp


# # ===================== 全局配置与常量 =====================

# WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}

# WORK_START_STR = '07:00'
# WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()

# START_DATE = datetime(2025, 1, 1, 7, 0)
# MACHINE_COUNT = 6
# SEARCH_DAYS = 30

# # 换模间隙（秒）
# SWITCH_GAP_SEC = 60

# # 成本权重
# TRANSITION_PENALTY = 20000
# SELF_SELECTED_PENALTY = 8000
# NON_SELF_PENALTY = 800
# DEVICE_PENALTY = 500000
# LOGICAL_PENALTY = 10000

# # Slack 惩罚（必须巨大）
# UNSCHEDULED_PENALTY = 1e9

# # Pricing 相关参数
# MAX_ITERS = 25
# MAX_NEW_COLS_PER_ITER = 80
# CANDIDATE_PATIENTS_TOPK = 400   # 适当放大，增强 pattern 构造质量

# # 初始化 pattern 每个 machine-day 生成多少种风格
# INIT_PATTERNS_PER_MD = 3


# # ===================== 工具函数 =====================

# def clean_exam_name(name):
#     s = str(name).strip().lower()
#     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
#     s = re.sub(r'[^\w()-]', '', s)
#     return s.replace('_', '-').replace(' ', '')

# def safe_read_excel(file_path, sheet_name=0):
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"文件不存在: {file_path}")

#     if file_path.endswith('.xlsx'):
#         engines = ['openpyxl', 'odf']
#     elif file_path.endswith('.xls'):
#         engines = ['xlrd']
#     else:
#         engines = ['openpyxl', 'xlrd', 'odf']

#     for engine in engines:
#         try:
#             return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
#         except Exception:
#             continue

#     return pd.read_excel(file_path, sheet_name=sheet_name)


# # ===================== 数据导入 =====================

# def import_data(patient_file, duration_file):
#     print(f"正在读取患者数据: {patient_file}")
#     duration_df = safe_read_excel(duration_file)
#     duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
#     exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

#     patient_df = safe_read_excel(patient_file)
#     required_cols = ['id', '登记日期', '检查项目']
#     for c in required_cols:
#         if c not in patient_df.columns:
#             raise ValueError(f"患者表缺少必要列: {c}")

#     patients = []
#     for _, row in patient_df.iterrows():
#         raw_id = row['id']
#         reg_dt = pd.to_datetime(row['登记日期'])
#         exam_raw = row['检查项目']
#         exam = clean_exam_name(exam_raw)

#         dur_min = exam_durations.get(exam, None)
#         if dur_min is None:
#             dur_min = 20.0

#         duration_sec = int(round(float(dur_min) * 60))

#         is_self = False
#         if '是否自选时间' in patient_df.columns:
#             try:
#                 is_self = bool(row['是否自选时间'])
#             except Exception:
#                 is_self = False

#         cid = (raw_id, reg_dt.strftime('%Y%m%d'))

#         patients.append({
#             'id': raw_id,
#             'cid': cid,
#             'exam_type': exam,
#             'exam_raw': str(exam_raw),
#             'duration': duration_sec,
#             'reg_datetime': reg_dt.to_pydatetime(),
#             'reg_date': reg_dt.date(),
#             'is_self_selected': is_self,
#         })

#     print(f"✅ 患者导入完成，总人数: {len(patients)}")
#     return patients

# def import_device_constraints(device_file):
#     print(f"正在读取设备限制数据: {device_file}")
#     df = safe_read_excel(device_file)

#     machine_col = None
#     exam_col = None
#     for c in df.columns:
#         cs = str(c).strip()
#         if cs in ['机器', '机器ID', '设备', 'machine', 'machine_id']:
#             machine_col = c
#         if cs in ['检查项目', '项目', 'exam', 'exam_type']:
#             exam_col = c

#     if machine_col is None or exam_col is None:
#         machine_col = df.columns[0]
#         exam_col = df.columns[1]

#     machine_exam_map: Dict[int, Set[str]] = defaultdict(set)

#     for _, row in df.iterrows():
#         mid_raw = row[machine_col]
#         exam_raw = row[exam_col]
#         if pd.isna(mid_raw) or pd.isna(exam_raw):
#             continue

#         try:
#             mid = int(mid_raw) - 1
#         except Exception:
#             try:
#                 mid = int(mid_raw)
#             except Exception:
#                 continue

#         if mid < 0 or mid >= MACHINE_COUNT:
#             continue

#         machine_exam_map[mid].add(clean_exam_name(exam_raw))

#     print("✅ 设备限制导入完成。")
#     return machine_exam_map


# # ===================== 业务规则 =====================

# def daily_work_seconds(date_obj):
#     weekday = date_obj.isoweekday()
#     hours_avail = 15.0 - WEEKDAY_END_HOURS.get(weekday, 0)
#     return int(round(hours_avail * 3600))

# def is_device_feasible(p, machine_id: int, machine_exam_map):
#     allowed = machine_exam_map.get(machine_id, set())
#     return (p['exam_type'] in allowed) if allowed else False

# def is_rule_feasible(p, machine_id: int, date_obj):
#     exam_name = p.get('exam_raw', '') or p.get('exam_type', '')

#     is_heart = ('心脏' in str(exam_name))
#     if is_heart:
#         if machine_id != 3:
#             return False
#         if date_obj.isoweekday() not in (2, 4):
#             return False

#     is_angio = ('造影' in str(exam_name))
#     if is_angio:
#         if machine_id != 1:
#             return False
#         if date_obj.isoweekday() not in (1, 3, 5):
#             return False

#     is_contrast = ('增强' in str(exam_name))
#     if is_contrast:
#         if date_obj.isoweekday() in (6, 7):
#             return False

#     return True


# # ===================== 列结构与成本 =====================

# @dataclass
# class Column:
#     col_id: int
#     machine_id: int
#     date: datetime.date
#     patients_idx: List[int]
#     cost: int
#     transition_count: int

# def compute_column_cost(patients: List[dict], col_patients_idx: List[int], date_obj):
#     if not col_patients_idx:
#         return 0, 0

#     sorted_idx = sorted(col_patients_idx, key=lambda i: patients[i]['reg_datetime'])

#     wait_cost = 0
#     transition_cnt = 0
#     prev_type = None

#     for i in sorted_idx:
#         p = patients[i]
#         wait_days = (date_obj - p['reg_date']).days

#         if p.get('is_self_selected', False):
#             wait_cost += max(0, wait_days) * SELF_SELECTED_PENALTY
#         else:
#             wait_cost += max(0, wait_days) * NON_SELF_PENALTY

#         cur_type = p['exam_type']
#         if prev_type is not None and cur_type != prev_type:
#             transition_cnt += 1
#         prev_type = cur_type

#     total_cost = wait_cost + transition_cnt * TRANSITION_PENALTY
#     return int(total_cost), int(transition_cnt)


# # ===================== Pattern 构造核心 =====================

# def _try_pack_pattern(
#     patients: List[dict],
#     candidate_idxs: List[int],
#     date_obj,
#     cap_sec: int,
#     strategy: str = "wait_first",
# ):
#     """
#     生成一个 machine-day 的可行 pattern（患者子序列）
#     strategy:
#       - wait_first: 等待天数/紧急度优先（按登记时间早者优先）
#       - type_cluster: 同类型聚类优先（减换模）
#       - random_mix: 随机扰动 + 贪心
#     """
#     if not candidate_idxs:
#         return []

#     # 计算一个简单“紧急度”排序键
#     def wait_key(i):
#         p = patients[i]
#         # 登记越早，等待越大
#         return p['reg_datetime']

#     if strategy == "wait_first":
#         ordered = sorted(candidate_idxs, key=wait_key)
#     elif strategy == "type_cluster":
#         # 类型优先，其次登记时间
#         ordered = sorted(candidate_idxs, key=lambda i: (patients[i]['exam_type'], patients[i]['reg_datetime']))
#     elif strategy == "random_mix":
#         ordered = candidate_idxs[:]
#         random.shuffle(ordered)
#         # 让随机序列再轻度按登记时间稳定一下
#         # 避免完全无意义的乱序
#         ordered = sorted(ordered, key=lambda i: (random.randint(0, 3), patients[i]['reg_datetime']))
#     else:
#         ordered = sorted(candidate_idxs, key=wait_key)

#     packed = []
#     used = 0
#     prev_type = None

#     for i in ordered:
#         dur = int(patients[i]['duration'])
#         add_gap = 0
#         cur_type = patients[i]['exam_type']
#         if prev_type is not None and cur_type != prev_type:
#             add_gap = SWITCH_GAP_SEC

#         if used + dur + add_gap <= cap_sec:
#             packed.append(i)
#             used += dur + add_gap
#             prev_type = cur_type

#     return packed


# # ===================== 初始化：按 machine-day 生成 pattern 列 =====================

# def build_initial_columns_patterns(
#     patients: List[dict],
#     machine_exam_map,
#     start_date: datetime,
#     search_days: int,
#     patterns_per_md: int = INIT_PATTERNS_PER_MD,
# ):
#     """
#     真正符合 CG 语义的初始化：
#       对每个 (machine, day) 生成若干种风格的可行 pattern 列
#     """
#     columns: List[Column] = []
#     col_id = 0

#     # 预排序患者索引（供候选筛选）
#     all_idx = list(range(len(patients)))
#     all_idx.sort(key=lambda i: patients[i]['reg_datetime'])

#     strategies = ["wait_first", "type_cluster", "random_mix"]

#     for d_off in range(search_days):
#         date_obj = start_date.date() + timedelta(days=d_off)
#         cap = daily_work_seconds(date_obj)
#         if cap <= 0:
#             continue

#         for m in range(MACHINE_COUNT):
#             # 找可行候选
#             cand = []
#             for i in all_idx:
#                 p = patients[i]
#                 if (date_obj - p['reg_date']).days < 0:
#                     continue
#                 if p['duration'] > cap:
#                     continue
#                 if not is_device_feasible(p, m, machine_exam_map):
#                     continue
#                 if not is_rule_feasible(p, m, date_obj):
#                     continue
#                 cand.append(i)

#             if not cand:
#                 continue

#             # 生成多风格 pattern
#             used_strats = strategies[:patterns_per_md]
#             for st in used_strats:
#                 packed = _try_pack_pattern(patients, cand, date_obj, cap, strategy=st)
#                 if not packed:
#                     continue

#                 cost, tcnt = compute_column_cost(patients, packed, date_obj)
#                 columns.append(Column(col_id, m, date_obj, packed, cost, tcnt))
#                 col_id += 1

#     return columns, col_id


# # ===================== RMP LP: Set Partitioning + Slack =====================

# def solve_rmp_lp(patients: List[dict], columns: List[Column]):
#     """
#     LP Relaxation Master:
#       min Σ cost_c x_c + Σ bigM * slack_i
#     s.t.
#       Σ_{c covering i} x_c + slack_i == 1
#       Σ_{c in (m,d)} x_c <= 1   (pattern语义)
#     """
#     solver = pywraplp.Solver.CreateSolver("GLOP")
#     if solver is None:
#         raise RuntimeError("无法创建 GLOP 求解器。")

#     n_pat = len(patients)
#     n_col = len(columns)

#     x = [solver.NumVar(0.0, 1.0, f"x_{c.col_id}") for c in columns]
#     slack = [solver.NumVar(0.0, 1.0, f"slack_{i}") for i in range(n_pat)]

#     # 1) 覆盖约束
#     cols_by_patient = [[] for _ in range(n_pat)]
#     for idx_c, col in enumerate(columns):
#         for i in col.patients_idx:
#             cols_by_patient[i].append(idx_c)

#     patient_cons = []
#     for i in range(n_pat):
#         ct = solver.Constraint(1.0, 1.0, f"cover_{i}")
#         for idx_c in cols_by_patient[i]:
#             ct.SetCoefficient(x[idx_c], 1.0)
#         ct.SetCoefficient(slack[i], 1.0)
#         patient_cons.append(ct)

#     # 2) machine-day 只能选 1 个 pattern
#     cols_by_md = defaultdict(list)
#     for idx_c, col in enumerate(columns):
#         cols_by_md[(col.machine_id, col.date)].append(idx_c)

#     md_cons = {}
#     for (m, d), idx_list in cols_by_md.items():
#         ct = solver.Constraint(0.0, 1.0, f"md_{m}_{d}")
#         for idx_c in idx_list:
#             ct.SetCoefficient(x[idx_c], 1.0)
#         md_cons[(m, d)] = ct

#     # 目标
#     obj = solver.Objective()
#     for idx_c, col in enumerate(columns):
#         obj.SetCoefficient(x[idx_c], float(col.cost))
#     for i in range(n_pat):
#         obj.SetCoefficient(slack[i], float(UNSCHEDULED_PENALTY))
#     obj.SetMinimization()

#     status = solver.Solve()
#     if status != pywraplp.Solver.OPTIMAL:
#         print(f"⚠️ RMP LP 未达到最优，status={status}")

#     return solver, x, slack, patient_cons, md_cons


# # ===================== RMP MIP =====================

# def solve_rmp_mip(patients: List[dict], columns: List[Column]):
#     """
#     Integer Master:
#       同机同日选 1 个 pattern
#       患者覆盖等式 + slack
#     """
#     solver = pywraplp.Solver.CreateSolver("CBC")
#     if solver is None:
#         raise RuntimeError("无法创建 CBC 求解器。")

#     n_pat = len(patients)
#     n_col = len(columns)

#     x = [solver.BoolVar(f"x_{c.col_id}") for c in columns]
#     slack = [solver.BoolVar(f"slack_{i}") for i in range(n_pat)]

#     # 覆盖
#     cols_by_patient = [[] for _ in range(n_pat)]
#     for idx_c, col in enumerate(columns):
#         for i in col.patients_idx:
#             cols_by_patient[i].append(idx_c)

#     for i in range(n_pat):
#         ct = solver.Constraint(1.0, 1.0, f"cover_{i}")
#         for idx_c in cols_by_patient[i]:
#             ct.SetCoefficient(x[idx_c], 1.0)
#         ct.SetCoefficient(slack[i], 1.0)

#     # machine-day 选 1 个 pattern
#     cols_by_md = defaultdict(list)
#     for idx_c, col in enumerate(columns):
#         cols_by_md[(col.machine_id, col.date)].append(idx_c)

#     for (m, d), idx_list in cols_by_md.items():
#         ct = solver.Constraint(0.0, 1.0, f"md_{m}_{d}")
#         for idx_c in idx_list:
#             ct.SetCoefficient(x[idx_c], 1.0)

#     # 目标
#     obj = solver.Objective()
#     for idx_c, col in enumerate(columns):
#         obj.SetCoefficient(x[idx_c], float(col.cost))
#     for i in range(n_pat):
#         obj.SetCoefficient(slack[i], float(UNSCHEDULED_PENALTY))
#     obj.SetMinimization()

#     print("开始求解最终整数规划...")
#     solver.SetTimeLimit(6000000000000)
#     status = solver.Solve()

#     chosen_cols = []
#     unscheduled_count = 0

#     if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
#         for idx_c, col in enumerate(columns):
#             if x[idx_c].solution_value() > 0.5:
#                 chosen_cols.append(col)
#         for i in range(n_pat):
#             if slack[i].solution_value() > 0.5:
#                 unscheduled_count += 1
#     else:
#         print(f"⚠️ MIP 未找到可行解，status={status}")

#     print(f"MIP 求解完成。放弃治疗的患者数: {unscheduled_count}")
#     print(f"最终选中列数: {len(chosen_cols)}")
#     return chosen_cols, unscheduled_count


# # ===================== Pricing：为 machine-day 生成新 pattern =====================

# def heuristic_pricing(
#     patients: List[dict],
#     machine_exam_map,
#     start_date: datetime,
#     search_days: int,
#     dual_p: List[float],
#     next_col_id: int,
#     max_new_cols: int = MAX_NEW_COLS_PER_ITER,
#     candidate_patients_topk: int = CANDIDATE_PATIENTS_TOPK,
# ):
#     """
#     启发式 pricing:
#       - 选对偶值高的患者作为候选核心
#       - 对每个 machine-day 生成多风格可行 pattern
#       - reduced cost = cost - sum(dual_p[i]) （pattern master 的典型形式）
#     """
#     n_pat = len(patients)
#     ranked = sorted(range(n_pat), key=lambda i: dual_p[i], reverse=True)
#     ranked = ranked[:min(candidate_patients_topk, n_pat)]

#     new_cols = []
#     col_id = next_col_id

#     strategies = ["wait_first", "type_cluster", "random_mix"]

#     for d_off in range(search_days):
#         date_obj = start_date.date() + timedelta(days=d_off)
#         cap = daily_work_seconds(date_obj)
#         if cap <= 0:
#             continue

#         for m in range(MACHINE_COUNT):
#             # 过滤该机该日可行候选（基于高对偶患者集合）
#             cand = []
#             for i in ranked:
#                 p = patients[i]
#                 if (date_obj - p['reg_date']).days < 0:
#                     continue
#                 if p['duration'] > cap:
#                     continue
#                 if not is_device_feasible(p, m, machine_exam_map):
#                     continue
#                 if not is_rule_feasible(p, m, date_obj):
#                     continue
#                 cand.append(i)

#             if not cand:
#                 continue

#             # 为该 machine-day 生成多风格 pattern
#             for st in strategies:
#                 packed = _try_pack_pattern(patients, cand, date_obj, cap, strategy=st)
#                 if len(packed) == 0:
#                     continue

#                 cost, tcnt = compute_column_cost(patients, packed, date_obj)
#                 dual_sum = sum(dual_p[i] for i in packed)
#                 reduced = cost - dual_sum

#                 # 只引入负 reduced cost 的列
#                 if reduced < -1e-6:
#                     new_cols.append(Column(col_id, m, date_obj, packed, cost, tcnt))
#                     col_id += 1

#                     if len(new_cols) >= max_new_cols:
#                         return new_cols, col_id

#     return new_cols, col_id


# # ===================== 导出排程 =====================

# def build_final_schedule_from_columns(patients: List[dict], chosen_cols: List[Column]):
#     """
#     Route A 语义下导出是自然正确的：
#       同机同日最多 1 列，所以每列从 07:00 排一次不会冲突
#     """
#     final = []

#     for col in chosen_cols:
#         date_obj = col.date
#         m_id = col.machine_id

#         idxs = sorted(col.patients_idx, key=lambda i: patients[i]['reg_datetime'])

#         cur_sec = 0
#         prev_type = None

#         for i in idxs:
#             p = patients[i]

#             if prev_type is not None and p['exam_type'] != prev_type:
#                 cur_sec += SWITCH_GAP_SEC

#             start_dt = datetime.combine(date_obj, WORK_START) + timedelta(seconds=cur_sec)
#             cur_sec += p['duration']
#             end_dt = datetime.combine(date_obj, WORK_START) + timedelta(seconds=cur_sec)

#             wait_days = (date_obj - p['reg_date']).days

#             final.append({
#                 'patient_id': p['id'],
#                 'exam_type': p['exam_type'],
#                 'reg_date': p['reg_date'],
#                 'is_self_selected': p.get('is_self_selected', False),
#                 'machine_id': m_id + 1,
#                 'date': date_obj,
#                 'start_time': start_dt.time(),
#                 'end_time': end_dt.time(),
#                 'wait_days': wait_days
#             })

#             prev_type = p['exam_type']

#         # 可选：容量检查
#         cap = daily_work_seconds(date_obj)
#         if cur_sec > cap:
#             print(f"⚠️ 警告：机器{m_id+1} {date_obj} "
#                   f"导出序列用时 {cur_sec}s 超过容量 {cap}s "
#                   f"(pattern 构造策略可再收紧)")

#     final.sort(key=lambda x: (x['date'], x['machine_id'], x['start_time']))
#     return final


# def evaluate_score(final_schedule: List[dict]):
#     if not final_schedule:
#         return 0, {}

#     total_score = 0
#     details = defaultdict(int)

#     prev_machine = None
#     prev_exam_type = None
#     prev_date = None

#     for item in final_schedule:
#         wait_days = (item['date'] - item['reg_date']).days
#         if item.get('is_self_selected', False):
#             wait_cost = max(0, wait_days) * SELF_SELECTED_PENALTY
#         else:
#             wait_cost = max(0, wait_days) * NON_SELF_PENALTY

#         total_score += wait_cost
#         details['等待成本'] += wait_cost

#         if prev_machine == item['machine_id'] and prev_date == item['date']:
#             if prev_exam_type is not None and item['exam_type'] != prev_exam_type:
#                 total_score += TRANSITION_PENALTY
#                 details['换模成本'] += TRANSITION_PENALTY

#         prev_machine = item['machine_id']
#         prev_exam_type = item['exam_type']
#         prev_date = item['date']

#     details['总评分'] = total_score
#     return total_score, dict(details)


# def export_excel(final_schedule: List[dict], filename: str, score_data=None):
#     if not final_schedule:
#         print("无数据导出。")
#         return

#     df = pd.DataFrame(final_schedule)
#     cols = [
#         'patient_id', 'exam_type', 'reg_date', 'is_self_selected',
#         'machine_id', 'date', 'start_time', 'end_time', 'wait_days'
#     ]
#     for c in cols:
#         if c not in df.columns:
#             df[c] = ''
#     df = df[cols]

#     with pd.ExcelWriter(filename) as writer:
#         df.to_excel(writer, sheet_name='详细排程', index=False)

#         stats = df.groupby(['date', 'machine_id']).size().reset_index(name='当日该机检查量')
#         stats.to_excel(writer, sheet_name='机日统计', index=False)

#         if score_data:
#             score, details = score_data
#             score_items = [['Total Score', score]] + [[k, v] for k, v in details.items()]
#             pd.DataFrame(score_items, columns=['Metric', 'Value']).to_excel(
#                 writer, sheet_name='评分报告', index=False
#             )

#     print(f"✅ 排程文件已生成: {filename}")


# # ===================== 列生成主流程 =====================

# def column_generation_solve(
#     patients: List[dict],
#     machine_exam_map,
#     start_date: datetime,
#     search_days: int = SEARCH_DAYS,
#     max_iters: int = MAX_ITERS,
#     max_new_cols_per_iter: int = MAX_NEW_COLS_PER_ITER
# ):
#     print(">>> 启动列生成算法 (Column Generation) - Route A <<<")

#     # 1) 初始化 pattern 列
#     columns, next_col_id = build_initial_columns_patterns(
#         patients, machine_exam_map, start_date, search_days, patterns_per_md=INIT_PATTERNS_PER_MD
#     )
#     print(f"初始列数: {len(columns)}")

#     # 2) CG Loop
#     for it in range(1, max_iters + 1):
#         print(f"\n--- Iteration {it}/{max_iters} ---")

#         solver_lp, x, slack, patient_cons, md_cons = solve_rmp_lp(patients, columns)

#         dual_p = [ct.dual_value() for ct in patient_cons]

#         new_cols, next_col_id = heuristic_pricing(
#             patients,
#             machine_exam_map,
#             start_date,
#             search_days,
#             dual_p,
#             next_col_id,
#             max_new_cols=max_new_cols_per_iter,
#             candidate_patients_topk=CANDIDATE_PATIENTS_TOPK
#         )

#         if not new_cols:
#             print("本轮未找到有效新列，提前终止 CG。")
#             break

#         columns.extend(new_cols)
#         print(f"本轮新增有效列: {len(new_cols)}，当前总列池: {len(columns)}")

#     # 3) 最终整数求解
#     print("\n>>> 进入整数规划阶段 (Integer RMP) <<<")
#     chosen_cols, unscheduled_count = solve_rmp_mip(patients, columns)

#     return chosen_cols, unscheduled_count


# # ===================== main =====================

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     # === 修改为你的真实路径 ===
#     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

#     missing_files = [f for f in [patient_file, duration_file, device_constraint_file] if not os.path.exists(f)]
#     if missing_files:
#         print(f"❌ 错误：找不到以下数据文件，请确认路径:\n{missing_files}")
#         return

#     # 1) 导入
#     patients = import_data(patient_file, duration_file)
#     machine_exam_map = import_device_constraints(device_constraint_file)

#     # 2) 运行 Route A 列生成
#     chosen_cols, unscheduled_count = column_generation_solve(
#         patients,
#         machine_exam_map,
#         start_date=START_DATE,
#         search_days=SEARCH_DAYS,
#         max_iters=MAX_ITERS,
#         max_new_cols_per_iter=MAX_NEW_COLS_PER_ITER
#     )

#     # 3) 导出排程
#     final_schedule = build_final_schedule_from_columns(patients, chosen_cols)
#     score, details = evaluate_score(final_schedule)

#     print("\n" + "=" * 50)
#     print("📊 最终结果统计 (Route A)")
#     print("=" * 50)
#     print(f"总评分 (负分制): -{int(score):,}")
#     print(f"等待成本: {int(details.get('等待成本', 0)):,}")
#     print(f"换模成本: {int(details.get('换模成本', 0)):,}")
#     print(f"未排程人数(基于MIP slack): {unscheduled_count} 人")

#     out_file = os.path.join(current_dir, 'column_generation_schedule_routeA.xlsx')
#     export_excel(final_schedule, out_file, score_data=(score, details))


# if __name__ == "__main__":
#     main()
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import os
import random
import importlib.util
from pathlib import Path

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model


# ===================== 加载 multi 脚本并复用已有函数 =====================

def load_multi_module():
    """
    从当前目录加载 '测量时间full-GPU实验-Multi.py'，并作为模块返回。
    """
    current_dir = Path(__file__).resolve().parent
    multi_path = current_dir / "测量时间full-GPU实验-Multi.py"
    if not multi_path.exists():
        raise FileNotFoundError(f"找不到 multi 脚本: {multi_path}")

    spec = importlib.util.spec_from_file_location("multi_module", multi_path)
    multi = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(multi)
    return multi


_multi = load_multi_module()

# 直接复用 multi 中已有的函数
clean_exam_name = _multi.clean_exam_name
safe_read_excel = _multi.safe_read_excel
import_device_constraints = _multi.import_device_constraints

# 复用部分全局常量
WEEKDAY_END_HOURS = _multi.WEEKDAY_END_HOURS
WORK_START = _multi.WORK_START
SELF_SELECTED_PENALTY = _multi.SELF_SELECTED_PENALTY
NON_SELF_PENALTY = _multi.NON_SELF_PENALTY
START_DATE = _multi.START_DATE
MACHINE_COUNT = _multi.MACHINE_COUNT

# 这几个与 multi 中同名，但在本 CG 版本中只用到 WAIT 部分
SWITCH_GAP_SEC = 60
UNSCHEDULED_PENALTY = 1e9

SEARCH_DAYS = 30
MAX_ITERS = 25
MAX_NEW_COLS_PER_ITER = 80
INIT_PATTERNS_PER_MD = 3


# ===================== 数据导入（CG 专用：每条记录对应一次检查） =====================

def import_data_for_cg(patient_file, duration_file):
    """
    列生成使用的专用 import：
    - 每一行代表一个检查
    - 与 multi.import_data 的结构不同（multi 是以“患者”为单位的复合结构）
    """
    print(f"正在读取患者数据: {patient_file}")

    duration_df = safe_read_excel(duration_file)
    duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
    exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

    patient_df = safe_read_excel(patient_file)
    required_cols = ['id', '登记日期', '检查项目']
    for c in required_cols:
        if c not in patient_df.columns:
            raise ValueError(f"患者表缺少必要列: {c}")

    patients: List[dict] = []
    for _, row in patient_df.iterrows():
        if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
            continue

        raw_id = row['id']
        reg_dt = pd.to_datetime(row['登记日期'])
        exam_raw = row['检查项目']
        exam = clean_exam_name(exam_raw)

        dur_min = exam_durations.get(exam, None)
        duration_sec = int(round(float(dur_min) * 60))

        is_self = False
        if '是否自选时间' in patient_df.columns:
            try:
                val = row['是否自选时间']
                if isinstance(val, str):
                    is_self = (val.strip() == '自选时间')
                else:
                    is_self = bool(val)
            except Exception:
                is_self = False

        cid = (str(raw_id).strip(), reg_dt.strftime('%Y%m%d'))

        patients.append({
            'id': raw_id,
            'cid': cid,
            'exam_type': exam,
            'exam_raw': str(exam_raw),
            'duration': duration_sec,
            'reg_datetime': reg_dt.to_pydatetime(),
            'reg_date': reg_dt.date(),
            'is_self_selected': is_self,
        })

    print(f"✅ 患者导入完成，总人数: {len(patients)}")
    return patients


# ===================== 业务规则（与原 multi 中规则保持一致） =====================

def daily_work_seconds(date_obj):
    weekday = date_obj.isoweekday()
    hours_avail = 15.0 - WEEKDAY_END_HOURS.get(weekday, 0)
    return int(round(hours_avail * 3600))


def is_device_feasible(p: dict, machine_id: int, machine_exam_map):
    allowed = machine_exam_map.get(machine_id, [])
    if not allowed:
        return False
    return p['exam_type'] in allowed


def is_rule_feasible(p: dict, machine_id: int, date_obj):
    exam_name = p.get('exam_raw', '') or p.get('exam_type', '')
    exam_name = str(exam_name)

    is_heart = ('心脏' in exam_name)
    if is_heart:
        if machine_id != 3:
            return False
        if date_obj.isoweekday() not in (2, 4):
            return False

    is_angio = ('造影' in exam_name)
    if is_angio:
        if machine_id != 1:
            return False
        if date_obj.isoweekday() not in (1, 3, 5):
            return False

    is_contrast = ('增强' in exam_name)
    if is_contrast:
        if date_obj.isoweekday() in (6, 7):
            return False

    return True


# ===================== 列与成本 =====================

@dataclass
class Column:
    col_id: int
    machine_id: int
    date: datetime.date
    patients_idx: List[int]
    cost: int
    transition_count: int


def compute_wait_cost_single_patient(p: dict, date_obj: datetime.date) -> int:
    wait_days = (date_obj - p['reg_date']).days
    wait_days = max(0, wait_days)
    if p.get('is_self_selected', False):
        return int(wait_days * SELF_SELECTED_PENALTY)
    else:
        return int(wait_days * NON_SELF_PENALTY)


def compute_column_cost(patients: List[dict], col_patients_idx: List[int], date_obj: datetime.date):
    if not col_patients_idx:
        return 0, 0

    sorted_idx = sorted(col_patients_idx, key=lambda i: patients[i]['reg_datetime'])

    wait_cost = 0
    transition_cnt = 0
    prev_type = None

    for i in sorted_idx:
        p = patients[i]
        wait_cost += compute_wait_cost_single_patient(p, date_obj)

        cur_type = p['exam_type']
        if prev_type is not None and cur_type != prev_type:
            transition_cnt += 1
        prev_type = cur_type

    total_cost = int(wait_cost)
    return total_cost, int(transition_cnt)


# ===================== 初始化列池（pattern） =====================

def _try_pack_pattern(
    patients: List[dict],
    candidate_idxs: List[int],
    date_obj: datetime.date,
    cap_sec: int,
    strategy: str = "wait_first",
) -> List[int]:
    """
    在候选检查中按某种策略打包成一个列（不重叠、容量内）。
    【已去掉换模 60 秒，只考虑纯 duration 累加。】
    """
    if not candidate_idxs:
        return []

    def wait_key(i: int):
        p = patients[i]
        return p['reg_datetime']

    # 不同的排序策略只是影响“先放谁”，不影响容量计算
    if strategy == "wait_first":
        ordered = sorted(candidate_idxs, key=wait_key)
    elif strategy == "type_cluster":
        ordered = sorted(candidate_idxs, key=lambda i: (patients[i]['exam_type'], patients[i]['reg_datetime']))
    elif strategy == "random_mix":
        ordered = candidate_idxs[:]
        random.shuffle(ordered)
        ordered = sorted(ordered, key=lambda i: (random.randint(0, 3), patients[i]['reg_datetime']))
    else:
        ordered = sorted(candidate_idxs, key=wait_key)

    packed: List[int] = []
    used = 0  # 当前已占用秒数（只算 duration）

    for i in ordered:
        dur = int(patients[i]['duration'])
        # 只要纯 duration 不超就塞进去
        if used + dur <= cap_sec:
            packed.append(i)
            used += dur

    return packed



def build_initial_columns_patterns(
    patients: List[dict],
    machine_exam_map,
    start_date: datetime,
    search_days: int,
    patterns_per_md: int = INIT_PATTERNS_PER_MD,
):
    columns: List[Column] = []
    col_id = 0

    all_idx = list(range(len(patients)))
    all_idx.sort(key=lambda i: patients[i]['reg_datetime'])

    strategies = ["wait_first", "type_cluster", "random_mix"]

    for d_off in range(search_days):
        date_obj = (start_date + timedelta(days=d_off)).date()
        cap = daily_work_seconds(date_obj)
        if cap <= 0:
            continue

        for m in range(MACHINE_COUNT):
            cand: List[int] = []
            for i in all_idx:
                p = patients[i]
                if (date_obj - p['reg_date']).days < 0:
                    continue
                if p['duration'] > cap:
                    continue
                if not is_device_feasible(p, m, machine_exam_map):
                    continue
                if not is_rule_feasible(p, m, date_obj):
                    continue
                cand.append(i)

            if not cand:
                continue

            used_strats = strategies[:patterns_per_md]
            for st in used_strats:
                packed = _try_pack_pattern(patients, cand, date_obj, cap, strategy=st)
                if not packed:
                    continue
                cost, tcnt = compute_column_cost(patients, packed, date_obj)
                columns.append(Column(col_id, m, date_obj, packed, cost, tcnt))
                col_id += 1

    return columns, col_id


# ===================== RMP: LP & MIP =====================

def solve_rmp_lp(patients: List[dict], columns: List[Column]):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("无法创建 GLOP 求解器。")

    n_pat = len(patients)
    x = [solver.NumVar(0.0, 1.0, f"x_{c.col_id}") for c in columns]
    slack = [solver.NumVar(0.0, 1.0, f"slack_{i}") for i in range(n_pat)]

    cols_by_patient: List[List[int]] = [[] for _ in range(n_pat)]
    for idx_c, col in enumerate(columns):
        for i in col.patients_idx:
            cols_by_patient[i].append(idx_c)

    patient_cons = []
    for i in range(n_pat):
        ct = solver.Constraint(1.0, 1.0, f"cover_{i}")
        for idx_c in cols_by_patient[i]:
            ct.SetCoefficient(x[idx_c], 1.0)
        ct.SetCoefficient(slack[i], 1.0)
        patient_cons.append(ct)

    cols_by_md: Dict[Tuple[int, datetime.date], List[int]] = defaultdict(list)
    for idx_c, col in enumerate(columns):
        cols_by_md[(col.machine_id, col.date)].append(idx_c)

    md_cons: Dict[Tuple[int, datetime.date], pywraplp.Constraint] = {}
    for (m, d), idx_list in cols_by_md.items():
        ct = solver.Constraint(0.0, 1.0, f"md_{m}_{d}")
        for idx_c in idx_list:
            ct.SetCoefficient(x[idx_c], 1.0)
        md_cons[(m, d)] = ct

    obj = solver.Objective()
    for idx_c, col in enumerate(columns):
        obj.SetCoefficient(x[idx_c], float(col.cost))
    for i in range(n_pat):
        obj.SetCoefficient(slack[i], float(UNSCHEDULED_PENALTY))
    obj.SetMinimization()

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print(f"⚠️ RMP LP 未达到最优，status={status}")

    return solver, x, slack, patient_cons, md_cons


def solve_rmp_mip(patients: List[dict], columns: List[Column]):
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("无法创建 CBC 求解器。")

    n_pat = len(patients)
    x = [solver.BoolVar(f"x_{c.col_id}") for c in columns]
    slack = [solver.BoolVar(f"slack_{i}") for i in range(n_pat)]

    cols_by_patient: List[List[int]] = [[] for _ in range(n_pat)]
    for idx_c, col in enumerate(columns):
        for i in col.patients_idx:
            cols_by_patient[i].append(idx_c)

    for i in range(n_pat):
        ct = solver.Constraint(1.0, 1.0, f"cover_{i}")
        for idx_c in cols_by_patient[i]:
            ct.SetCoefficient(x[idx_c], 1.0)
        ct.SetCoefficient(slack[i], 1.0)

    cols_by_md: Dict[Tuple[int, datetime.date], List[int]] = defaultdict(list)
    for idx_c, col in enumerate(columns):
        cols_by_md[(col.machine_id, col.date)].append(idx_c)

    for (m, d), idx_list in cols_by_md.items():
        ct = solver.Constraint(0.0, 1.0, f"md_{m}_{d}")
        for idx_c in idx_list:
            ct.SetCoefficient(x[idx_c], 1.0)

    obj = solver.Objective()
    for idx_c, col in enumerate(columns):
        obj.SetCoefficient(x[idx_c], float(col.cost))
    for i in range(n_pat):
        obj.SetCoefficient(slack[i], float(UNSCHEDULED_PENALTY))
    obj.SetMinimization()

    print("开始求解最终整数规划...")
    solver.SetTimeLimit(600000)

    status = solver.Solve()
    chosen_cols: List[Column] = []
    unscheduled_count = 0

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for idx_c, col in enumerate(columns):
            if x[idx_c].solution_value() > 0.5:
                chosen_cols.append(col)
        for i in range(n_pat):
            if slack[i].solution_value() > 0.5:
                unscheduled_count += 1
    else:
        print(f"⚠️ MIP 未找到可行解，status={status}")

    print(f"MIP 求解完成。放弃治疗的患者数: {unscheduled_count}")
    print(f"最终选中列数: {len(chosen_cols)}")
    return chosen_cols, unscheduled_count


# ===================== 精确定价（CP-SAT） =====================

def solve_exact_pricing_single_md(
    patients: List[dict],
    machine_exam_map,
    m: int,
    date_obj: datetime.date,
    cap_sec: int,
    dual_p: List[float],
    dual_mu_md: float,
    next_col_id: int,
):
    n_pat = len(patients)

    candidate: List[int] = []
    for i in range(n_pat):
        p = patients[i]
        if (date_obj - p['reg_date']).days < 0:
            continue
        if p['duration'] > cap_sec:
            continue
        if not is_device_feasible(p, m, machine_exam_map):
            continue
        if not is_rule_feasible(p, m, date_obj):
            continue
        candidate.append(i)

    if not candidate:
        return None, 0.0

    model = cp_model.CpModel()
    y = {i: model.NewBoolVar(f"y_{i}") for i in candidate}

    model.Add(
        sum(int(patients[i]['duration']) * y[i] for i in candidate) <= cap_sec
    )

    objective_terms = []
    for i in candidate:
        p = patients[i]
        wait_cost = compute_wait_cost_single_patient(p, date_obj)
        coeff = float(wait_cost) - float(dual_p[i])
        objective_terms.append(coeff * y[i])
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    # 如需多核，可加：
    # solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, 0.0

    selected = [i for i in candidate if solver.Value(y[i]) == 1]
    if not selected:
        return None, 0.0

    cost_c = 0.0
    sum_pi = 0.0
    for i in selected:
        p = patients[i]
        wait_cost = compute_wait_cost_single_patient(p, date_obj)
        cost_c += wait_cost
        sum_pi += dual_p[i]

    rc = cost_c - sum_pi - dual_mu_md
    if rc >= -1e-6:
        return None, rc

    cost_int, tcnt = compute_column_cost(patients, selected, date_obj)
    new_col = Column(
        col_id=next_col_id,
        machine_id=m,
        date=date_obj,
        patients_idx=selected,
        cost=cost_int,
        transition_count=tcnt,
    )
    return new_col, rc


def exact_pricing(
    patients: List[dict],
    machine_exam_map,
    start_date: datetime,
    search_days: int,
    dual_p: List[float],
    dual_mu: Dict[Tuple[int, datetime.date], float],
    next_col_id: int,
    max_new_cols: int = MAX_NEW_COLS_PER_ITER,
):
    new_cols: List[Column] = []
    col_id = next_col_id

    for d_off in range(search_days):
        date_obj = (start_date + timedelta(days=d_off)).date()
        cap = daily_work_seconds(date_obj)
        if cap <= 0:
            continue

        for m in range(MACHINE_COUNT):
            mu_md = dual_mu.get((m, date_obj), 0.0)
            col, rc = solve_exact_pricing_single_md(
                patients,
                machine_exam_map,
                m,
                date_obj,
                cap,
                dual_p,
                mu_md,
                col_id,
            )
            if col is not None:
                new_cols.append(col)
                col_id += 1
                if len(new_cols) >= max_new_cols:
                    return new_cols, col_id

    return new_cols, col_id


# ===================== 列生成主流程 =====================

def column_generation_solve(
    patients: List[dict],
    machine_exam_map,
    start_date: datetime,
    search_days: int = SEARCH_DAYS,
    max_iters: int = MAX_ITERS,
    max_new_cols_per_iter: int = MAX_NEW_COLS_PER_ITER,
):
    print(">>> 启动列生成算法 (Column Generation, exact pricing) <<<")

    columns, next_col_id = build_initial_columns_patterns(
        patients, machine_exam_map, start_date, search_days, patterns_per_md=INIT_PATTERNS_PER_MD
    )
    print(f"初始列数: {len(columns)}")

    for it in range(1, max_iters + 1):
        print(f"\n--- Iteration {it}/{max_iters} ---")

        solver_lp, x, slack, patient_cons, md_cons = solve_rmp_lp(patients, columns)

        dual_p = [ct.dual_value() for ct in patient_cons]
        dual_mu = {key: ct.dual_value() for key, ct in md_cons.items()}

        new_cols, next_col_id = exact_pricing(
            patients,
            machine_exam_map,
            start_date,
            search_days,
            dual_p,
            dual_mu,
            next_col_id,
            max_new_cols=max_new_cols_per_iter,
        )

        if not new_cols:
            print("本轮未找到负 reduced cost 新列，提前终止 CG。")
            break

        columns.extend(new_cols)
        print(f"本轮新增有效列: {len(new_cols)}，当前总列池: {len(columns)}")

    print("\n>>> 进入整数规划阶段 (Integer RMP on generated columns) <<<")
    chosen_cols, unscheduled_count = solve_rmp_mip(patients, columns)
    return chosen_cols, unscheduled_count


# ===================== 导出与评分 =====================

def build_final_schedule_from_columns(patients: List[dict], chosen_cols: List[Column]):
    """
    根据选中的列生成最终排程表：
    - 同一列内患者按登记时间排序
    - 从 WORK_START 开始顺序排，时间只按 duration 累加
    - 不再在类型切换时加入任何 60 秒间隔
    """
    final: List[dict] = []

    for col in chosen_cols:
        date_obj = col.date
        m_id = col.machine_id  # 0-based

        # 列内患者按登记时间排序
        idxs = sorted(col.patients_idx, key=lambda i: patients[i]['reg_datetime'])

        cur_sec = 0  # 本机本日已用秒数

        for i in idxs:
            p = patients[i]

            # 不考虑换模，直接按 duration 排
            start_dt = datetime.combine(date_obj, WORK_START) + timedelta(seconds=cur_sec)
            cur_sec += p['duration']
            end_dt = datetime.combine(date_obj, WORK_START) + timedelta(seconds=cur_sec)

            wait_days = (date_obj - p['reg_date']).days

            final.append({
                'patient_id': p['id'],
                'exam_type': p['exam_type'],
                'reg_date': p['reg_date'],
                'is_self_selected': p.get('is_self_selected', False),
                'machine_id': m_id + 1,          # 导出为 1-based
                'date': date_obj,
                'start_time': start_dt.time(),
                'end_time': end_dt.time(),
                'wait_days': wait_days,
            })

        # 这里的 cur_sec 现在只包含 duration，总是 ≤ daily_work_seconds(date_obj)
        cap = daily_work_seconds(date_obj)
        if cur_sec > cap:
            print(f"⚠️ 警告：机器{m_id+1} {date_obj} 导出序列用时 {cur_sec}s 超过容量 {cap}s")

    # 全部机器排序
    final.sort(key=lambda x: (x['date'], x['machine_id'], x['start_time']))
    return final



def evaluate_score(final_schedule: List[dict]):
    if not final_schedule:
        return 0, {}

    total_score = 0
    details = defaultdict(int)

    for item in final_schedule:
        wait_days = (item['date'] - item['reg_date']).days
        wait_days = max(0, wait_days)
        if item.get('is_self_selected', False):
            wait_cost = wait_days * SELF_SELECTED_PENALTY
        else:
            wait_cost = wait_days * NON_SELF_PENALTY
        total_score += wait_cost
        details['等待成本'] += wait_cost

    details['总评分'] = total_score
    return total_score, dict(details)


def export_excel(final_schedule: List[dict], filename: str, score_data=None):
    if not final_schedule:
        print("无数据导出。")
        return

    df = pd.DataFrame(final_schedule)
    cols = [
        'patient_id', 'exam_type', 'reg_date', 'is_self_selected',
        'machine_id', 'date', 'start_time', 'end_time', 'wait_days',
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ''
    df = df[cols]

    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name='详细排程', index=False)
        stats = df.groupby(['date', 'machine_id']).size().reset_index(name='当日该机检查量')
        stats.to_excel(writer, sheet_name='机日统计', index=False)

        if score_data:
            score, details = score_data
            score_items = [['Total Score', score]] + [[k, v] for k, v in details.items()]
            pd.DataFrame(score_items, columns=['Metric', 'Value']).to_excel(
                writer, sheet_name='评分报告', index=False
            )

    print(f"✅ 排程文件已生成: {filename}")


# ===================== main =====================

def main():
    current_dir = Path(__file__).resolve().parent

    patient_file = current_dir / '实验数据6.1small - 副本.xlsx'
    duration_file = current_dir / '程序使用实际平均耗时3 - 副本.xlsx'
    device_constraint_file = current_dir / '设备限制4.xlsx'

    missing_files = [f for f in [patient_file, duration_file, device_constraint_file] if not f.exists()]
    if missing_files:
        print(f"❌ 错误：找不到以下数据文件，请确认路径:\n{missing_files}")
        return

    patients = import_data_for_cg(str(patient_file), str(duration_file))
    machine_exam_map = import_device_constraints(str(device_constraint_file))

    chosen_cols, unscheduled_count = column_generation_solve(
        patients,
        machine_exam_map,
        start_date=START_DATE,
        search_days=SEARCH_DAYS,
        max_iters=MAX_ITERS,
        max_new_cols_per_iter=MAX_NEW_COLS_PER_ITER,
    )

    final_schedule = build_final_schedule_from_columns(patients, chosen_cols)
    score, details = evaluate_score(final_schedule)

    print("\n" + "=" * 50)
    print("📊 最终结果统计 (Column Generation + exact pricing)")
    print("=" * 50)
    print(f"总评分: {int(score):,}")
    print(f"等待成本: {int(details.get('等待成本', 0)):,}")
    print(f"未排程人数(基于MIP slack): {unscheduled_count} 人")

    out_file = current_dir / 'column_generation_schedule_exactCG.xlsx'
    export_excel(final_schedule, str(out_file), score_data=(score, details))


if __name__ == "__main__":
    main()

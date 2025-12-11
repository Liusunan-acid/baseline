# from __future__ import annotations
# from typing import List, Dict, Set, Tuple, Any
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, time
# import os
# from collections import defaultdict
# import traceback
# import re
# import copy

# # ==============================================================================
# # PART 1: 复用代码 (来自附件 '测量时间full-GPU实验-Multi.py')
# # ==============================================================================

# # 全局常量
# WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# WORK_START_STR = '07:00'
# WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# START_DATE = datetime(2025, 1, 1, 7, 0)
# MACHINE_COUNT = 6

# # 成本权重 (用于计算 DP 的 Value)
# SELF_SELECTED_PENALTY = 8000
# NON_SELF_PENALTY = 800
# SWITCH_GAP_SEC = 60 # 换模间隙

# def clean_exam_name(name):
#     s = str(name).strip().lower()
#     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
#     s = re.sub(r'[^\w()-]', '', s)
#     return s.replace('_', '-').replace(' ', '')

# def safe_read_excel(file_path, sheet_name=0):
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

# def import_data(patient_file, duration_file):
#     # 复用原逻辑，但稍微适配以返回列表而非字典，方便 DP 索引
#     try:
#         duration_df = safe_read_excel(duration_file)
#         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
#         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

#         patient_df = safe_read_excel(patient_file)
#         patients_list = []
        
#         # 建立去重字典，避免同一 compound_id 重复添加
#         seen_cids = {}

#         for _, row in patient_df.iterrows():
#             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
#                 continue
            
#             # 构造 ID
#             raw_id = str(row['id']).strip()
#             reg_dt_str = pd.to_datetime(row['登记日期']).strftime('%Y%m%d')
#             cid = (raw_id, reg_dt_str)
            
#             exam_raw = row['检查项目']
#             exam_type = clean_exam_name(exam_raw)
#             # 转换为秒，方便 DP 计算
#             duration_min = float(exam_durations.get(exam_type, 15.0))
#             duration_sec = int(round(duration_min * 60))
            
#             is_self_selected = (str(row['是否自选时间']) == '自选时间')
#             reg_date = pd.to_datetime(row['登记日期']).date()

#             if cid not in seen_cids:
#                 p_data = {
#                     'id': raw_id,
#                     'cid': cid,
#                     'exams': [], # 支持多部位
#                     'reg_date': reg_date,
#                     'is_self_selected': is_self_selected,
#                     'scheduled': False,
#                     'main_exam_type': exam_type, # 用于快速判断
#                     'total_duration': 0
#                 }
#                 seen_cids[cid] = p_data
#                 patients_list.append(p_data)
            
#             # 添加检查部位信息
#             seen_cids[cid]['exams'].append({
#                 'exam_type': exam_type,
#                 'duration_sec': duration_sec,
#                 'exam_raw': exam_raw
#             })
#             seen_cids[cid]['total_duration'] += duration_sec

#         print(f"成功导入 {len(patients_list)} 名患者 (用于 DP)")
#         return patients_list

#     except Exception as e:
#         print(f"数据导入错误: {e}")
#         traceback.print_exc()
#         raise

# def import_device_constraints(file_path):
#     try:
#         df = safe_read_excel(file_path)
#         machine_exam_map = defaultdict(set) # 使用 set 加速查找
#         for _, row in df.iterrows():
#             mid = int(row['设备']) - 1
#             exam = clean_exam_name(row['检查项目'])
#             machine_exam_map[mid].add(exam)
#         return machine_exam_map
#     except Exception as e:
#         print(f"导入设备限制数据错误: {e}")
#         traceback.print_exc()
#         raise

# def daily_work_seconds(date_obj):
#     """计算当天某机器的工作时长（秒）"""
#     weekday = date_obj.isoweekday() # 1-7
#     hours_avail = 15.0 - WEEKDAY_END_HOURS.get(weekday, 0)
#     return int(round(hours_avail * 3600))

# # ==============================================================================
# # PART 2: 核心业务规则判断 (逻辑复用自原代码 Tensor 逻辑的 Python 版)
# # ==============================================================================

# def is_rule_feasible(p, machine_id, date_obj):
#     """
#     判断患者 p 是否满足在 machine_id 和 date_obj 进行检查的特殊硬性规则。
#     """
#     # 提取所有相关检查名称字符串
#     exam_names = [e['exam_type'] for e in p['exams']]
#     full_str = " ".join(exam_names)
    
#     weekday = date_obj.isoweekday() # 1=Mon, ..., 7=Sun
#     # 映射到 Tensor 逻辑中的 0-6 (0=Mon)
#     wd_idx = weekday - 1 
    
#     # 1. 心脏规则: 包含 '心脏' -> 必须 3号机 (index 2) 且 周二/周四 (wd 1, 3)
#     if '心脏' in full_str:
#         if machine_id != 3: # ID 3 对应 index 3 (原代码中机器编号似乎从1开始，index从0开始？)
#             # 原代码: machine_id (0-5)
#             # Tensor逻辑: assigned_machine_batch == 3
#             # 这里假定 machine_id 是 0-based
#             if machine_id != 2: # 机器3对应索引2 (通常是 id-1)，需确认
#                 return False 
#         if wd_idx not in [1, 3]: # Tue(1), Thu(3)
#             return False

#     # 2. 造影规则: 包含 '造影' -> 必须 1号机 (index 1?) (Tensor: == 1) 且 Mon/Wed/Fri
#     if '造影' in full_str:
#         if machine_id != 1: # Tensor: assigned_machine_batch == 1 (机器2?)
#             return False
#         if wd_idx not in [0, 2, 4]: # Mon, Wed, Fri
#             return False

#     # 3. 增强规则: 包含 '增强' -> 不可在周末 (Sat, Sun)
#     if '增强' in full_str:
#         if weekday >= 6:
#             return False

#     return True

# def is_device_feasible(p, machine_id, machine_exam_map):
#     """判断机器是否支持该患者的所有检查项目"""
#     allowed_exams = machine_exam_map.get(machine_id, set())
#     for e in p['exams']:
#         if e['exam_type'] not in allowed_exams:
#             return False
#     return True

# # ==============================================================================
# # PART 3: 动态规划 (Dynamic Programming) 核心逻辑
# # ==============================================================================

# def solve_knapsack_for_machine(candidates: List[dict], capacity_sec: int, date_obj) -> List[int]:
#     """
#     对单台机器单天运行 0/1 背包算法。
    
#     Args:
#         candidates: 候选患者列表
#         capacity_sec: 当天机器可用秒数
#         date_obj: 当前日期
        
#     Returns:
#         selected_indices: 被选中的 candidates 在列表中的索引
#     """
#     n = len(candidates)
#     if n == 0 or capacity_sec <= 0:
#         return []

#     # 1. 构建物品 (Weight, Value)
#     # Weight: 耗时 + 换模缓冲 (保守策略，假设每人都换模，排程时再紧凑化)
#     # Value: 等待成本 (Cost)
#     weights = []
#     values = []
    
#     # 缩放因子：为了让 DP 数组不那么大，也为了精度，这里保持秒级。
#     # 如果 capacity_sec 很大 (e.g. 50000)，Python list OK。
    
#     for p in candidates:
#         # 重量：检查时长 + 60秒换模缓冲
#         w = p['total_duration'] + SWITCH_GAP_SEC
        
#         # 价值：计算等待成本
#         wait_days = (date_obj - p['reg_date']).days
#         # 如果是未来预约的，这里 wait_days 可能为负，逻辑上应该已经被过滤掉了
#         wait_days = max(0, wait_days)
        
#         base_cost = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
#         # 价值公式： (等待天数 + 1) * 基础权重
#         # 我们希望最大化被消除的“痛苦值”
#         v = (wait_days + 1) * base_cost
        
#         # 额外加分项：如果不安排可能导致严重后果的 (如心脏只能周二做)
#         # 这里的 candidates 已经是根据规则筛选过的，所以只需关注紧迫性
        
#         weights.append(w)
#         values.append(v)

#     # 2. DP 初始化
#     # dp[w] = max_value
#     # 由于我们需要找回路径，使用二维数组或记录表会 OOM。
#     # 优化：使用一维数组 dp[w]，并额外记录 keep[i][w] (bool) 用于回溯? 
#     # 2500人太多了，O(N*C) = 2500 * 50000 = 1.25亿次计算，Python会慢。
#     # 
#     # === 性能优化策略 ===
#     # 考虑到我们只需要“填满”且“价值最高”，且物品数量 N 不大 (单日候选人通常 < 500)。
#     # 我们可以使用 standard DP。
    
#     limit = capacity_sec
#     # dp[j] 存储容量为 j 时的最大价值
#     dp = [0] * (limit + 1)

#     if n > 200:
#         # 计算性价比
#         ratios = [(values[i]/weights[i], i) for i in range(n)]
#         ratios.sort(key=lambda x: x[0], reverse=True)
#         top_indices = [x[1] for x in ratios[:200]]
        
#         # 重新映射
#         sub_candidates = [candidates[i] for i in top_indices]
#         sub_weights = [weights[i] for i in top_indices]
#         sub_values = [values[i] for i in top_indices]
        
#         # 递归调用 (使用子集)
#         sub_selected = solve_knapsack_for_machine_core(sub_weights, sub_values, limit)
#         return [top_indices[i] for i in sub_selected]
#     else:
#         return solve_knapsack_for_machine_core(weights, values, limit)

# def solve_knapsack_for_machine_core(weights, values, capacity):
#     """纯粹的 0/1 背包求解，返回被选中的索引列表"""
#     n = len(weights)
#     # dp[w]
#     dp = [0] * (capacity + 1)
#     # keep[i][w] = True if item i was picked for capacity w
#     # 为了内存优化，keep 设为 [n][capacity+1] 的 bool 矩阵 (numpy int8)
#     keep = np.zeros((n, capacity + 1), dtype=bool)

#     for i in range(n):
#         w = weights[i]
#         v = values[i]
#         for j in range(capacity, w - 1, -1):
#             if dp[j - w] + v > dp[j]:
#                 dp[j] = dp[j - w] + v
#                 keep[i][j] = True
    
#     # 回溯找出被选中的物品
#     selected = []
#     curr_c = capacity
    
#     for i in range(n - 1, -1, -1):
#         if keep[i][curr_c]:
#             selected.append(i)
#             curr_c -= weights[i]
            
#     return selected

# class RollingDPAllocator:
#     def __init__(self, patients: List[dict], machine_exam_map: Dict[int, Set[str]]):
#         self.patients = patients
#         self.machine_exam_map = machine_exam_map
#         self.total_patients = len(patients)
        
#         # 建立索引，方便快速移除
#         # 使用 list index 作为 ID
#         self.unscheduled_indices = set(range(self.total_patients))
        
#         # 结果存储
#         self.schedule_results = [] # list of dict

#     def run(self, start_date: datetime, search_days: int):
#         print(f"开始 DP 滚动排程，时间跨度: {search_days} 天...")
        
#         for day_offset in range(search_days):
#             current_date = start_date.date() + timedelta(days=day_offset)
#             daily_cap_sec = daily_work_seconds(current_date)
            
#             if daily_cap_sec <= 0:
#                 continue

#             print(f"Planning {current_date} (Cap: {daily_cap_sec}s)... Remaining: {len(self.unscheduled_indices)}")

#             # 遍历每一台机器
#             # 优化：可以随机打乱机器顺序，或者优先排特殊机器 (如机器1, 3)
#             # 这里硬编码优先排机器 1 和 3 (特殊机器)
#             machine_order = [0, 2, 1, 3, 4, 5] # 假设 Machine 1=Index 0, Machine 3=Index 2
            
#             for m_idx in machine_order:
#                 # 1. 筛选本机器今天的候选人
#                 candidates_indices = []
#                 candidates_data = []
                
#                 for p_idx in self.unscheduled_indices:
#                     p = self.patients[p_idx]
                    
#                     # A. 日期约束 (不可早于登记日)
#                     if p['reg_date'] > current_date:
#                         continue
                        
#                     # B. 设备能力约束
#                     if not is_device_feasible(p, m_idx, self.machine_exam_map):
#                         continue
                        
#                     # C. 业务规则约束 (心脏/造影等)
#                     if not is_rule_feasible(p, m_idx, current_date):
#                         continue
                    
#                     candidates_indices.append(p_idx)
#                     candidates_data.append(p)
                
#                 if not candidates_data:
#                     continue
                
#                 # 2. 运行 背包 DP
#                 # 返回的是 candidates_data 列表中的 index (0..len-1)
#                 selected_local_indices = solve_knapsack_for_machine(
#                     candidates_data, daily_cap_sec, current_date
#                 )
                
#                 if not selected_local_indices:
#                     continue
                
#                 # 3. 贪心聚类与排程生成
#                 # 我们选出了一堆人，现在要安排具体时间
#                 # 为了减少换模，按 exam_type 排序
#                 selected_p_indices = [candidates_indices[i] for i in selected_local_indices]
                
#                 # 排序：先按检查项目聚类，再按登记时间
#                 selected_p_indices.sort(key=lambda idx: (
#                     self.patients[idx]['main_exam_type'], 
#                     self.patients[idx]['reg_date']
#                 ))
                
#                 current_time_point = datetime.combine(current_date, WORK_START)
#                 prev_exam_type = None
                
#                 for p_idx in selected_p_indices:
#                     p = self.patients[p_idx]
                    
#                     # 确定开始时间 (计算换模)
#                     gap = 0
#                     if prev_exam_type is not None and p['main_exam_type'] != prev_exam_type:
#                         gap = SWITCH_GAP_SEC
                    
#                     start_dt = current_time_point + timedelta(seconds=gap)
#                     end_dt = start_dt + timedelta(seconds=p['total_duration'])
                    
#                     # 双重检查是否越界 (虽然 DP 考虑了 gap，但 DP 是估算的)
#                     # 实际排程可能比 DP 估算的紧凑，也可能因为排序问题更松散
#                     work_end_dt = datetime.combine(current_date, WORK_START) + timedelta(seconds=daily_cap_sec)
                    
#                     if end_dt > work_end_dt:
#                         # 这是一个边缘情况：DP 估算能放下，但实际排序后放不下
#                         # 策略：放不下就踢回池子，或者这一天结束
#                         # 由于 DP 预留了 60s/人，通常实际排程会比 DP 更省时间 (因为同类合并了 gap=0)
#                         # 所以这里基本不会溢出。如果溢出，break
#                         print(f"  ⚠️ Warning: 机器 {m_idx+1} 空间不足，跳过剩余 {len(selected_p_indices) - selected_p_indices.index(p_idx)} 人")
#                         break
                    
#                     # 记录结果
#                     self.schedule_results.append({
#                         '机器编号': m_idx + 1,
#                         '日期': current_date.strftime('%Y-%m-%d'),
#                         '开始时间': start_dt.strftime('%H:%M:%S'),
#                         '结束时间': end_dt.strftime('%H:%M:%S'),
#                         '检查项目': p['exams'][0]['exam_raw'], # 简化，只取第一个
#                         '患者ID': p['id'],
#                         '登记日期': p['reg_date'].strftime('%Y-%m-%d'),
#                         '是否自选': '是' if p['is_self_selected'] else '否',
#                         '等待天数': (current_date - p['reg_date']).days
#                     })
                    
#                     # 更新状态
#                     self.unscheduled_indices.remove(p_idx)
#                     current_time_point = end_dt
#                     prev_exam_type = p['main_exam_type']

#         return self.schedule_results

# # ==============================================================================
# # PART 4: 主程序入口
# # ==============================================================================

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
    
#     # 假设文件在同一目录
#     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
    
#     # 简单的文件检查
#     if not os.path.exists(patient_file):
#         print(f"文件未找到: {patient_file}")
#         return

#     print("=== 1. 读取数据 ===")
#     patients = import_data(patient_file, duration_file)
#     machine_map = import_device_constraints(device_constraint_file)
    
#     print("\n=== 2. 初始化 DP 规划器 ===")
#     allocator = RollingDPAllocator(patients, machine_map)
    
#     # 设置排程天数 (例如 45 天，确保能排完)
#     SEARCH_DAYS = 45
    
#     print("\n=== 3. 执行动态规划 (Rolling Horizon) ===")
#     t0 = datetime.now()
#     schedule = allocator.run(START_DATE, SEARCH_DAYS)
#     t1 = datetime.now()
    
#     print(f"\n排程完成! 耗时: {(t1-t0).total_seconds():.2f} 秒")
#     print(f"总计安排检查: {len(schedule)}")
#     print(f"剩余未排程人数: {len(allocator.unscheduled_indices)}")

#     print("\n=== 4. 导出结果 ===")
#     out_file = os.path.join(current_dir, 'DP_Schedule_Result.xlsx')
    
#     df_res = pd.DataFrame(schedule)
    
#     # 简单的统计
#     if not df_res.empty:
#         score = (df_res['等待天数'] * NON_SELF_PENALTY).sum() # 简化评分
#         print(f"预估总等待罚分: {score:,}")
        
#         with pd.ExcelWriter(out_file) as writer:
#             df_res.to_excel(writer, sheet_name='详细排程', index=False)
            
#             # 机日统计
#             stats = df_res.groupby(['日期', '机器编号']).size().reset_index(name='检查量')
#             stats.to_excel(writer, sheet_name='机日统计', index=False)
            
#         print(f"结果已保存至: {out_file}")
#     else:
#         print("未生成任何排程结果。")

# if __name__ == '__main__':
#     main()

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import os
from collections import defaultdict
import traceback
import re
import copy

# ==============================================================================
# PART 1: 复用代码 (来自附件 '测量时间full-GPU实验-Multi.py')
# ==============================================================================

# 全局常量
WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
WORK_START = datetime.strptime('07:00', '%H:%M').time()
TRANSITION_PENALTY = 20000
LOGICAL = 10000
SELF_SELECTED_PENALTY = 8000
NON_SELF_PENALTY = 800
START_DATE = datetime(2025, 1, 1, 7, 0)
MACHINE_COUNT = 6
DEVICE_PENALTY = 500000


def clean_exam_name(name):
    """
    将检查项目名称清洗为统一格式:
    - 转为小写
    - 将全角括号替换为半角
    - 去掉所有非字母数字和括号/横杠
    """
    s = str(name).strip().lower()
    # 替换中文括号为英文括号
    s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
    # 保留字母、数字、括号和横线
    s = re.sub(r'[^\w()-]', '', s)
    # 将下划线替换为横线，保证统一风格
    s = s.replace('_', '-').replace(' ', '')
    return s


def safe_read_excel(file_path, sheet_name=0):
    """
    针对不同后端引擎做一个兼容封装，避免用户环境中某些引擎不可用导致异常
    """
    if file_path.endswith('.xlsx'):
        engines = ['openpyxl', 'odf']
    elif file_path.endswith('.xls'):
        engines = ['xlrd']
    else:
        engines = ['openpyxl', 'xlrd', 'odf']

    last_err = None
    for engine in engines:
        try:
            return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
        except Exception as e:
            last_err = e
            continue
    # 如果所有 engine 都失败了，最后再不用 engine 参数试一次
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception:
        print(f"[safe_read_excel] 所有引擎都读取失败: {file_path}")
        if last_err is not None:
            print("最后一次错误信息:", last_err)
        raise


def import_data(patient_file, duration_file):
    # 复用原逻辑，但稍微适配以返回列表而非字典，方便 DP 索引
    try:
        duration_df = safe_read_excel(duration_file)
        duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
        exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

        patient_df = safe_read_excel(patient_file)
        patients_list = []
        
        # 建立去重字典，避免同一 compound_id 重复添加
        seen_cids = {}

        for _, row in patient_df.iterrows():
            if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
                continue
            
            # 构造 ID
            raw_id = str(row['id']).strip()
            reg_dt_str = pd.to_datetime(row['登记日期']).strftime('%Y%m%d')
            cid = (raw_id, reg_dt_str)
            
            exam_raw = row['检查项目']
            exam_type = clean_exam_name(exam_raw)
            # 转换为秒，方便 DP 计算
            duration_min = float(exam_durations.get(exam_type, 15.0))
            duration_sec = int(round(duration_min * 60))
            
            is_self_selected = (str(row['是否自选时间']) == '自选时间')
            reg_date = pd.to_datetime(row['登记日期']).date()

            if cid not in seen_cids:
                p_data = {
                    'id': raw_id,
                    'cid': cid,
                    'exams': [], # 支持多部位
                    'reg_date': reg_date,
                    'is_self_selected': is_self_selected,
                    'scheduled': False,
                    'main_exam_type': exam_type,
                    'total_duration': 0,
                }
                seen_cids[cid] = p_data
                patients_list.append(p_data)
            
            # 添加检查部位信息（保留 exam_raw，后面用来匹配“心脏 / 造影 / 增强”）
            seen_cids[cid]['exams'].append({
                'exam_type': exam_type,
                'duration_sec': duration_sec,
                'exam_raw': exam_raw
            })
            seen_cids[cid]['total_duration'] += duration_sec

        print(f"成功导入 {len(patients_list)} 名患者 (用于 DP)")
        return patients_list

    except Exception as e:
        print(f"数据导入错误: {e}")
        traceback.print_exc()
        raise


def import_device_constraints(file_path):
    try:
        df = safe_read_excel(file_path)
        machine_exam_map = defaultdict(set) # 使用 set 加速查找
        for _, row in df.iterrows():
            mid = int(row['设备']) - 1
            exam = clean_exam_name(row['检查项目'])
            machine_exam_map[mid].add(exam)
        return machine_exam_map
    except Exception as e:
        print(f"导入设备限制数据错误: {e}")
        traceback.print_exc()
        raise


def daily_work_seconds(date_obj):
    """
    根据星期几返回该天可用的总秒数。
    """
    weekday = date_obj.isoweekday()  # 1=Mon, ..., 7=Sun
    hours_avail = 15.0 - WEEKDAY_END_HOURS.get(weekday, 0)
    # 转为秒数
    return int(round(hours_avail * 3600))


# ==============================================================================
# PART 2: 规则与约束检查
# ==============================================================================

def is_rule_feasible(p, machine_id, date_obj):
    """
    判断患者 p 是否满足在 machine_id 和 date_obj 进行检查的特殊硬性规则。
    这里仅修复“心脏 / 造影 / 增强”关键字匹配的 bug，不改变原有规则逻辑。
    """
    # 使用原始检查项目字符串进行关键字匹配（修复清洗后中文丢失的 bug）
    raw_names = [str(e.get('exam_raw', '')) for e in p.get('exams', [])]
    full_raw = " ".join(raw_names)

    # 保留清洗后的检查类型字符串（如后续有需要可使用）
    exam_types = [str(e.get('exam_type', '')) for e in p.get('exams', [])]
    full_type = " ".join(exam_types)
    
    weekday = date_obj.isoweekday() # 1=Mon, ..., 7=Sun
    # 映射到 Tensor 逻辑中的 0-6 (0=Mon)
    wd_idx = weekday - 1 
    
    # 1. 心脏规则: 包含 '心脏' -> 必须 3号机 (index 2) 且 周二/周四 (wd 1, 3)
    if '心脏' in full_raw:
        if machine_id != 3: # ID 3 对应 index 3 (原代码中机器编号似乎从1开始，index从0开始？)
            # 原代码: machine_id (0-5)
            # Tensor逻辑: assigned_machine_batch == 3
            # 这里假定 machine_id 是 0-based
            if machine_id != 2: # 机器3对应索引2 (通常是 id-1)，需确认
                # 根据附件代码 `mid = int(row['设备']) - 1`，设备3 -> mid 2
                # 但附件 Tensor 逻辑写的是 `assigned_machine_batch == 3`。
                # 这是一个常见的坑。通常设备限制表里写的是机器1-6。
                # 假设 Tensor 代码是针对 `mid` (0-5) 的，那么 `==3` 意味着机器4？
                # 让我们回看附件: `mid = int(row['设备']) - 1` 再与 Tensor 代码对比。
                # 为尽量保留原逻辑，这里保守处理：允许 machine_id == 2 或 3。
                return False
        # 允许的日期: 周二(weekday=2 -> wd_idx=1), 周四(weekday=4 -> wd_idx=3)
        if wd_idx not in [1, 3]:
            return False

    # 2. 造影规则: 包含 '造影' -> 必须 2号机 (index 1) 且 周一/周三/周五 (wd 0, 2, 4)
    if '造影' in full_raw:
        if machine_id != 1:
            return False
        if wd_idx not in [0, 2, 4]: # Mon, Wed, Fri
            return False

    # 3. 增强规则: 包含 '增强' -> 不可在周末 (Sat, Sun)
    if '增强' in full_raw:
        if weekday >= 6:
            return False

    return True


def is_device_feasible(p, machine_id, machine_exam_map):
    """判断机器是否支持该患者的所有检查项目"""
    allowed_exams = machine_exam_map.get(machine_id, set())
    for e in p['exams']:
        if e['exam_type'] not in allowed_exams:
            return False
    return True

# ==============================================================================
# PART 3: 动态规划 (Dynamic Programming) 核心逻辑
# ==============================================================================

def solve_knapsack_for_machine(patients, candidate_indices, capacity_sec, date_obj, machine_id):
    """
    使用 0/1 背包为给定机器在某一天选择一组患者，以最大化加权等待时间或减少惩罚。
    
    patients: 全部患者列表
    candidate_indices: 这一天在该机器上可行的 candidate 患者索引集合(list)
    capacity_sec: 该机器这一天的总可用秒数
    date_obj: 当前日期
    machine_id: 当前机器编号 (0-based)
    
    返回:
        selected_indices: 最终选中的患者 index 列表
    """
    # 若无候选患者或容量不正，直接返回
    if not candidate_indices or capacity_sec <= 0:
        return []

    # 提取 item 列表
    items = []
    for idx in candidate_indices:
        p = patients[idx]
        duration = p['total_duration']
        # 带上换模的平均附加(简单加一个固定值)
        SWITCH_GAP_SEC = 60  # 简化处理：平均每个患者加 60 秒间隔
        w = duration + SWITCH_GAP_SEC
        # 价值函数: 按等待天数和自选权重加权
        wait_days = (date_obj - p['reg_date']).days
        base_cost = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
        val = (wait_days + 1) * base_cost
        items.append((idx, w, val))

    # 若单个患者的时长就超过容量，则直接过滤
    items = [it for it in items if it[1] <= capacity_sec]
    if not items:
        return []

    # 0/1 背包: 物品数量与容量可能较大，小心内存和时间
    # 为避免 O(N*C) 过大，这里可以做一个简单剪枝：如果物品太多，只保留性价比高的一部分
    # 例如 : 至多保留 200 个项目
    MAX_ITEMS = 200
    if len(items) > MAX_ITEMS:
        # 按 value/weight 排序，保留前 200 个
        ratios = [(v / w, idx) for (idx, w, v) in items]
        ratios.sort(reverse=True)
        top_indices = {x[1] for x in ratios[:MAX_ITEMS]}
        items = [it for it in items if it[0] in top_indices]

    n = len(items)
    if n == 0:
        return []

    # 重新编号
    idx_list = [it[0] for it in items]
    weights = [it[1] for it in items]
    values = [it[2] for it in items]

    # 背包容量
    limit = capacity_sec
    # dp[j] 存储容量为 j 时的最大价值
    dp = [0] * (limit + 1)
    
    # 记录选择：selection[i][j] = True 表示物品 i 在容量 j 时被选中
    # 注意: 这会占较大内存, 但 n <= 200, limit ~ 50000 时大致 200*50001 布尔值
    # 在当前场景还算勉强可接受
    keep = [[False] * (limit + 1) for _ in range(n)]

    for i in range(n):
        w = weights[i]
        v = values[i]
        # 倒序遍历容量，避免重复使用同一物品
        for cap in range(limit, w - 1, -1):
            if dp[cap - w] + v > dp[cap]:
                dp[cap] = dp[cap - w] + v
                keep[i][cap] = True

    # 回溯选中的物品
    selected_indices = []
    cap = limit
    for i in range(n - 1, -1, -1):
        if keep[i][cap]:
            selected_indices.append(idx_list[i])
            cap -= weights[i]
    selected_indices.reverse()
    
    return selected_indices


# ==============================================================================
# PART 4: 滚动 DP 排程框架
# ==============================================================================

class RollingDPAllocator:
    def __init__(self, patients: List[Dict], device_constraints: Dict[int, Set[str]]):
        self.patients = patients
        self.device_constraints = device_constraints
        # 未排程患者使用一个集合存储索引
        self.unscheduled_indices = set(range(len(patients)))
        self.total_patients = len(patients)
        
        # 结果存储
        self.schedule_results = [] # list of dict

    def run(self, start_date: datetime, search_days: int):
        print(f"开始 DP 滚动排程，时间跨度: {search_days} 天...")
        
        for day_offset in range(search_days):
            current_date = start_date.date() + timedelta(days=day_offset)
            daily_cap_sec = daily_work_seconds(current_date)
            
            if daily_cap_sec <= 0:
                continue

            print(f"Planning {current_date} (Cap: {daily_cap_sec}s)... Remaining: {len(self.unscheduled_indices)}")

            # 遍历每一台机器
            # 优化：可以随机打乱机器顺序，或者优先排特殊机器 (如机器1, 3)
            # 这里硬编码优先排机器 1 和 3 (特殊机器)
            machine_order = [0, 2, 1, 3, 4, 5]
            for m_id in machine_order:
                if not self.unscheduled_indices:
                    break
                self._plan_machine_for_day(m_id, current_date, daily_cap_sec)

        print(f"滚动 DP 排程结束。剩余未排程人数: {len(self.unscheduled_indices)}")

    def _plan_machine_for_day(self, machine_id: int, date_obj, daily_capacity_sec: int):
        """
        为某一天的某台机器进行 DP 背包选人。
        """
        # 选出所有当前仍未排程、且在这台机器+这一天上可行的候选患者
        candidate_indices = []
        for idx in list(self.unscheduled_indices):
            p = self.patients[idx]
            # 规则约束
            if not is_rule_feasible(p, machine_id, date_obj):
                continue
            # 设备约束
            if not is_device_feasible(p, machine_id, self.device_constraints):
                continue
            candidate_indices.append(idx)

        if not candidate_indices:
            return
        
        # 进行背包选择
        selected = solve_knapsack_for_machine(self.patients, candidate_indices, daily_capacity_sec, date_obj, machine_id)
        if not selected:
            return

        # 标记为已排程，并记录结果
        current_dt = datetime.combine(date_obj, WORK_START)
        for idx in selected:
            p = self.patients[idx]
            # 这里简单地按 total_duration 顺序排满当日 (未考虑换模精细时间)
            start_time = current_dt
            end_time = current_dt + timedelta(seconds=p['total_duration'])
            current_dt = end_time  # 简单串行
            
            self.unscheduled_indices.discard(idx)
            p['scheduled'] = True
            
            self.schedule_results.append({
                '病人ID': p['id'],
                '登记日期': p['reg_date'],
                '检查日期': date_obj,
                '机器': machine_id + 1,  # 输出为 1-based
                '检查开始时间': start_time,
                '检查结束时间': end_time,
                '检查项目': p['exams'][0]['exam_raw'] if p['exams'] else '',
                '等待天数': (date_obj - p['reg_date']).days,
                '是否自选时间': '自选时间' if p['is_self_selected'] else '非自选时间'
            })

    def to_dataframe(self) -> pd.DataFrame:
        if not self.schedule_results:
            return pd.DataFrame()
        df = pd.DataFrame(self.schedule_results)
        return df.sort_values(by=['检查日期', '机器', '检查开始时间'])


# ==============================================================================
# PART 5: 主函数入口
# ==============================================================================

def main():
    # 这里可以根据你的实际文件路径进行调整
    patient_file = '实验数据6.1small - 副本.xlsx'
    duration_file = '程序使用实际平均耗时3 - 副本.xlsx'
    device_file = '设备限制4.xlsx'
    output_file = 'DP_排程结果.xlsx'
    
    if not os.path.exists(patient_file):
        print(f"缺少患者文件: {patient_file}")
        return
    if not os.path.exists(duration_file):
        print(f"缺少检查时长文件: {duration_file}")
        return
    if not os.path.exists(device_file):
        print(f"缺少设备限制文件: {device_file}")
        return

    try:
        patients = import_data(patient_file, duration_file)
        device_constraints = import_device_constraints(device_file)
        
        allocator = RollingDPAllocator(patients, device_constraints)
        # 例如搜索 45 天
        allocator.run(START_DATE, search_days=45)
        
        df = allocator.to_dataframe()
        if not df.empty:
            df.to_excel(output_file, index=False)
            print(f"排程结果已保存到 {output_file}")
        else:
            print("未生成任何排程结果。")
    except Exception as e:
        print("运行过程中出现错误:")
        traceback.print_exc()


if __name__ == '__main__':
    main()

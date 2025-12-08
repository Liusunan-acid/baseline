# # # import pandas as pd
# # # import numpy as np
# # # from datetime import datetime, timedelta, time as datetime_time
# # # import os
# # # from collections import defaultdict
# # # import traceback
# # # import re
# # # import math
# # # import multiprocessing # 引入多进程库以检测核数
# # # from ortools.sat.python import cp_model

# # # # ===================== 全局常量 (完全对齐 GPU 实验代码) =====================
# # # WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# # # WORK_START_STR = '07:00'
# # # WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# # # START_DATE = datetime(2024, 12, 1, 7, 0)
# # # MACHINE_COUNT = 6

# # # # 求解器配置
# # # # ⚠️ 修改说明：
# # # # 1. 窗口保持 1000 以获得全局最优性
# # # # 2. 时间限制 120秒，配合多线程通常能在几十秒内找到极优解
# # # BATCH_SIZE = 1600       
# # # SEARCH_DAYS = 15        
# # # SOLVER_TIME_LIMIT = 360000000000

# # # # ===================== 评分常量 (来自 GPU 实验代码) =====================
# # # TRANSITION_PENALTY = 20000      # 换模惩罚
# # # SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
# # # NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
# # # DEVICE_PENALTY = 500000         # 设备/规则违规惩罚

# # # # ===================== 数据导入工具 (复用并对齐逻辑) =====================

# # # def clean_exam_name(name):
# # #     s = str(name).strip().lower()
# # #     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
# # #     s = re.sub(r'[^\w()-]', '', s)
# # #     return s.replace('_', '-').replace(' ', '')

# # # def safe_read_excel(file_path, sheet_name=0):
# # #     if file_path.endswith('.xlsx'):
# # #         engines = ['openpyxl', 'odf']
# # #     elif file_path.endswith('.xls'):
# # #         engines = ['xlrd']
# # #     else:
# # #         engines = ['openpyxl', 'xlrd', 'odf']
# # #     for engine in engines:
# # #         try:
# # #             return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
# # #         except Exception:
# # #             continue
# # #     return pd.read_excel(file_path, sheet_name=sheet_name)

# # # def import_data(patient_file, duration_file):
# # #     print("正在导入患者数据...")
# # #     try:
# # #         duration_df = safe_read_excel(duration_file)
# # #         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
# # #         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

# # #         patient_df = safe_read_excel(patient_file)
# # #         patients = [] 
        
# # #         for _, row in patient_df.iterrows():
# # #             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
# # #                 continue
            
# # #             raw_id = str(row['id']).strip()
# # #             reg_dt = pd.to_datetime(row['登记日期'])
# # #             cid = (raw_id, reg_dt.strftime('%Y%m%d'))
            
# # #             exam_type = clean_exam_name(row['检查项目'])
            
# # #             duration_raw = float(exam_durations.get(exam_type, 15.0))
# # #             duration_int = int(round(duration_raw)) 
            
# # #             is_self_selected = (row['是否自选时间'] == '自选时间')
            
# # #             p = {
# # #                 'id': raw_id,
# # #                 'cid': cid,
# # #                 'exam_type': exam_type,
# # #                 'duration': max(1, duration_int), 
# # #                 'reg_date': reg_dt.date(),
# # #                 'reg_datetime': reg_dt,
# # #                 'is_self_selected': is_self_selected,
# # #                 'original_row': row
# # #             }
# # #             patients.append(p)
            
# # #         patients.sort(key=lambda x: x['reg_datetime'])
# # #         print(f"成功导入 {len(patients)} 名患者。")
# # #         return patients
# # #     except Exception as e:
# # #         print(f"数据导入错误: {e}")
# # #         traceback.print_exc()
# # #         raise

# # # def import_device_constraints(file_path):
# # #     print("正在导入设备限制...")
# # #     try:
# # #         df = safe_read_excel(file_path)
# # #         machine_exam_map = defaultdict(set)
# # #         for _, row in df.iterrows():
# # #             mid = int(row['设备']) - 1
# # #             exam = clean_exam_name(row['检查项目'])
# # #             machine_exam_map[mid].add(exam)
# # #         return machine_exam_map
# # #     except Exception as e:
# # #         print(f"导入设备限制数据错误: {e}")
# # #         traceback.print_exc()
# # #         raise

# # # # ===================== 核心算法：CP-SAT 滚动调度器 =====================

# # # class RollingHorizonScheduler:
# # #     def __init__(self, patients, machine_exam_map, start_date):
# # #         self.all_patients = patients
# # #         self.machine_exam_map = machine_exam_map
# # #         self.global_start_date = start_date
# # #         self.machine_occupied_until = defaultdict(int)
# # #         self.final_schedule = []
        
# # #         self.daily_work_minutes = {}
# # #         for d in range(1, 8):
# # #             hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
# # #             self.daily_work_minutes[d] = int(round(hours_avail * 60))

# # #     def get_work_window(self, date_obj):
# # #         weekday = date_obj.isoweekday()
# # #         limit = self.daily_work_minutes.get(weekday, 0)
# # #         return 0, limit

# # #     def solve(self):
# # #         total_patients = len(self.all_patients)
# # #         # 获取CPU核心数
# # #         num_workers = multiprocessing.cpu_count()
# # #         print(f"\n🚀 开始滚动优化，已启用 {num_workers} 线程并行加速")
# # #         print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}, 搜索范围: {SEARCH_DAYS} 天")

# # #         for i in range(0, total_patients, BATCH_SIZE):
# # #             batch_patients = self.all_patients[i : min(i + BATCH_SIZE, total_patients)]
# # #             print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者 {i} - {i + len(batch_patients)}")
# # #             self.solve_batch(batch_patients, num_workers)
            
# # #         print("\n所有批次处理完毕。")

# # #     def solve_batch(self, batch_patients, num_workers):
# # #         model = cp_model.CpModel()
        
# # #         intervals = {} 
# # #         presences = {}
# # #         starts = {}
# # #         p_data = {} 
        
# # #         # 1. 建模 (同前)
# # #         for p_idx, p in enumerate(batch_patients):
# # #             p_data[p_idx] = p
# # #             possible_intervals = []
            
# # #             earliest_date = max(p['reg_date'], self.global_start_date.date())
# # #             start_day_offset = (earliest_date - self.global_start_date.date()).days
            
# # #             for d in range(SEARCH_DAYS):
# # #                 current_day_offset = start_day_offset + d
# # #                 current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
# # #                 day_start_min, day_end_min = self.get_work_window(current_date)
                
# # #                 if day_end_min <= 0: continue 
                
# # #                 for m_id in range(MACHINE_COUNT):
# # #                     if p['exam_type'] not in self.machine_exam_map[m_id]:
# # #                         continue
# # #                     occupied_until = self.machine_occupied_until[(m_id, current_date)]
# # #                     if occupied_until + p['duration'] > day_end_min:
# # #                         continue 
                        
# # #                     suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
# # #                     is_present = model.NewBoolVar(f"pres{suffix}")
# # #                     presences[(p_idx, m_id, current_day_offset)] = is_present
                    
# # #                     start_var = model.NewIntVar(occupied_until, day_end_min - p['duration'], f"start{suffix}")
# # #                     end_var = model.NewIntVar(occupied_until + p['duration'], day_end_min, f"end{suffix}")
# # #                     interval_var = model.NewOptionalIntervalVar(
# # #                         start_var, p['duration'], end_var, is_present, f"interval{suffix}"
# # #                     )
                    
# # #                     intervals[(p_idx, m_id, current_day_offset)] = interval_var
# # #                     starts[(p_idx, m_id, current_day_offset)] = start_var
# # #                     possible_intervals.append(is_present)
            
# # #             if possible_intervals:
# # #                 model.Add(sum(possible_intervals) == 1)
# # #             else:
# # #                 pass 
# # #                 # print(f"警告：患者 {p['cid']} 无可用资源")
        
# # #         # 2. 约束
# # #         machine_day_intervals = defaultdict(list)
# # #         for key, interval in intervals.items():
# # #             _, m_id, day_offset = key
# # #             machine_day_intervals[(m_id, day_offset)].append(interval)
# # #         for key, interval_list in machine_day_intervals.items():
# # #             model.AddNoOverlap(interval_list)
            
# # #         # 3. 目标优化
# # #         day_costs = []
# # #         for key, is_present in presences.items():
# # #             _, _, day_offset = key
# # #             day_costs.append(is_present * day_offset)
# # #         model.Minimize(sum(day_costs))

# # #         # 4. 求解与加速配置
# # #         solver = cp_model.CpSolver()
        
# # #         # 🔥🔥🔥 核心加速配置 🔥🔥🔥
# # #         # 启用所有 CPU 核心并行搜索
# # #         solver.parameters.num_search_workers = num_workers 
# # #         # 设置时间限制
# # #         solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
# # #         # 打印进度 (让你看到它在飞快地工作)
# # #         solver.parameters.log_search_progress = True 
        
# # #         status = solver.Solve(model)
        
# # #         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
# # #             print(f"  -> 求解成功 ({solver.StatusName(status)}), 耗时 {solver.UserTime():.2f}s")
            
# # #             current_batch_updates = defaultdict(list) 
# # #             for key, is_present in presences.items():
# # #                 if solver.Value(is_present):
# # #                     p_idx, m_id, day_offset = key
# # #                     start_val = solver.Value(starts[key])
# # #                     p = p_data[p_idx]
# # #                     duration = p['duration']
# # #                     end_val = start_val + duration
# # #                     real_date = self.global_start_date.date() + timedelta(days=day_offset)
                    
# # #                     record = {
# # #                         'patient_id': p['id'],
# # #                         'exam_type': p['exam_type'],
# # #                         'reg_date': p['reg_date'],
# # #                         'is_self_selected': p['is_self_selected'],
# # #                         'machine_id': m_id + 1, 
# # #                         'date': real_date,
# # #                         'start_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=start_val)).time(),
# # #                         'end_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=end_val)).time(),
# # #                         'wait_days': (real_date - p['reg_date']).days
# # #                     }
# # #                     self.final_schedule.append(record)
# # #                     current_batch_updates[(m_id, real_date)].append(end_val)
            
# # #             for (m_id, d_date), ends in current_batch_updates.items():
# # #                 self.machine_occupied_until[(m_id, d_date)] = max(
# # #                     self.machine_occupied_until[(m_id, d_date)], 
# # #                     max(ends)
# # #                 )
# # #         else:
# # #             print("  -> 求解失败，无可行解")

# # #     def evaluate_schedule_score(self):
# # #         if not self.final_schedule:
# # #             return 0, {}

# # #         print("\n" + "="*50)
# # #         print("🔎 正在进行 GPU 标准评分 (Python 实现版)...")
# # #         print("="*50)

# # #         total_score = 0
# # #         details = defaultdict(int)

# # #         sorted_sched = sorted(
# # #             self.final_schedule, 
# # #             key=lambda x: (x['machine_id'], x['date'], x['start_time'])
# # #         )

# # #         prev_machine = -1
# # #         prev_exam_type = None
# # #         prev_date = None

# # #         for item in sorted_sched:
# # #             wait_days = (item['date'] - item['reg_date']).days
# # #             weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
# # #             wait_cost = max(0, wait_days) * weight
# # #             total_score -= wait_cost
# # #             details['wait_cost'] += wait_cost

# # #             if (item['machine_id'] == prev_machine and 
# # #                 item['date'] == prev_date):
# # #                 if item['exam_type'] != prev_exam_type:
# # #                     total_score -= TRANSITION_PENALTY
# # #                     details['transition_cost'] += TRANSITION_PENALTY
# # #                     details['transition_count'] += 1
            
# # #             prev_machine = item['machine_id']
# # #             prev_exam_type = item['exam_type']
# # #             prev_date = item['date']

# # #             weekday = item['date'].isoweekday() 
# # #             m_idx = item['machine_id'] - 1      
# # #             exam_name = str(item['exam_type'])

# # #             is_heart = '心脏' in exam_name
# # #             is_angio = '造影' in exam_name
# # #             is_contrast = '增强' in exam_name

# # #             if is_heart:
# # #                 ok_wd = (weekday == 1 or weekday == 3)
# # #                 ok_mc = (m_idx == 3)
# # #                 if not (ok_wd and ok_mc):
# # #                     total_score -= DEVICE_PENALTY
# # #                     details['heart_violation'] += 1

# # #             if is_angio:
# # #                 ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)
# # #                 ok_mc = (m_idx == 1)
# # #                 if not (ok_wd and ok_mc):
# # #                     total_score -= DEVICE_PENALTY
# # #                     details['angio_violation'] += 1

# # #             is_weekend = (weekday == 6 or weekday == 7)
# # #             if is_contrast and is_weekend:
# # #                 total_score -= DEVICE_PENALTY
# # #                 details['weekend_contrast_violation'] += 1

# # #         print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
# # #         print("-" * 30)
# # #         print(f"  ❌ 总扣分: {-total_score:,.0f}")
# # #         print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
# # #         print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
# # #         print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
# # #         print(f"  💉 造影规则违规: {details['angio_violation']} 次")
# # #         print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")
# # #         print("="*50 + "\n")
        
# # #         return total_score, details

# # #     def export_excel(self, filename, score_data=None):
# # #         if not self.final_schedule:
# # #             print("没有排程数据可导出。")
# # #             return
            
# # #         df = pd.DataFrame(self.final_schedule)
# # #         cols = ['patient_id', 'exam_type', 'reg_date', 'is_self_selected', 
# # #                 'machine_id', 'date', 'start_time', 'end_time', 'wait_days']
# # #         df = df[cols]
# # #         df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)
        
# # #         try:
# # #             with pd.ExcelWriter(filename) as writer:
# # #                 df.to_excel(writer, sheet_name='详细排程', index=False)
# # #                 stats = df.groupby('date').size().reset_index(name='每日检查量')
# # #                 stats.to_excel(writer, sheet_name='统计', index=False)
                
# # #                 if score_data:
# # #                     score, details = score_data
# # #                     score_items = [
# # #                         ['Total Score (Fitness)', score],
# # #                         ['Total Penalty', -score],
# # #                         ['Wait Cost', details['wait_cost']],
# # #                         ['Transition Cost', details['transition_cost']],
# # #                         ['Transition Count', details['transition_count']],
# # #                         ['Heart Rule Violations', details['heart_violation']],
# # #                         ['Angio Rule Violations', details['angio_violation']],
# # #                         ['Weekend Contrast Violations', details['weekend_contrast_violation']]
# # #                     ]
# # #                     score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
# # #                     score_df.to_excel(writer, sheet_name='评分报告', index=False)
                    
# # #             print(f"排程已成功导出至: {filename}")
# # #         except Exception as e:
# # #             print(f"导出 Excel 失败: {e}")

# # # # ===================== 主程序 =====================

# # # def main():
# # #     current_dir = os.path.dirname(os.path.abspath(__file__))
# # #     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
# # #     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
# # #     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
    
# # #     for f in [patient_file, duration_file, device_constraint_file]:
# # #         if not os.path.exists(f):
# # #             print(f"❌ 错误：找不到文件 {f}")
# # #             return

# # #     patients = import_data(patient_file, duration_file)
# # #     machine_map = import_device_constraints(device_constraint_file)
    
# # #     scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
# # #     scheduler.solve()
# # #     score, details = scheduler.evaluate_schedule_score()
    
# # #     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# # #     out_file = os.path.join(current_dir, f'精确排程结果_{ts}.xlsx')
# # #     scheduler.export_excel(out_file, score_data=(score, details))

# # # if __name__ == "__main__":
# # #     main()

# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta, time as datetime_time
# # import os
# # from collections import defaultdict
# # import traceback
# # import re
# # import math
# # import multiprocessing
# # from ortools.sat.python import cp_model

# # # ===================== 全局常量 (严格对齐 GPU 实验代码) =====================
# # WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# # WORK_START_STR = '07:00'
# # WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# # START_DATE = datetime(2024, 12, 1, 7, 0)
# # MACHINE_COUNT = 6

# # # 求解器配置
# # BATCH_SIZE = 100       # 批次大小，可根据内存调整
# # SEARCH_DAYS = 30        # 搜索未来多少天的空闲（建议覆盖最大等待期）
# # SOLVER_TIME_LIMIT = 60000000  # 每个批次的求解时间限制(秒)

# # # ===================== 评分常量 (对齐 GPU 实验代码) =====================
# # TRANSITION_PENALTY = 20000      # 换模惩罚
# # SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
# # NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
# # DEVICE_PENALTY = 500000         # 设备/规则违规惩罚
# # LOGICAL_PENALTY = 10000         # 逻辑违规（如反向等待，CP-SAT中通过硬约束避免）

# # # ===================== 数据导入工具 =====================

# # def clean_exam_name(name):
# #     s = str(name).strip().lower()
# #     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
# #     s = re.sub(r'[^\w()-]', '', s)
# #     return s.replace('_', '-').replace(' ', '')

# # def safe_read_excel(file_path, sheet_name=0):
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
# #     return pd.read_excel(file_path, sheet_name=sheet_name)

# # def import_data(patient_file, duration_file):
# #     print("正在导入患者数据...")
# #     try:
# #         duration_df = safe_read_excel(duration_file)
# #         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
# #         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

# #         patient_df = safe_read_excel(patient_file)
# #         patients = [] 
        
# #         for _, row in patient_df.iterrows():
# #             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
# #                 continue
            
# #             raw_id = str(row['id']).strip()
# #             reg_dt = pd.to_datetime(row['登记日期'])
# #             cid = (raw_id, reg_dt.strftime('%Y%m%d'))
            
# #             exam_type = clean_exam_name(row['检查项目'])
            
# #             duration_raw = float(exam_durations.get(exam_type, 15.0))
# #             duration_int = int(round(duration_raw)) 
            
# #             is_self_selected = (row['是否自选时间'] == '自选时间')
            
# #             p = {
# #                 'id': raw_id,
# #                 'cid': cid,
# #                 'exam_type': exam_type,
# #                 'duration': max(1, duration_int), 
# #                 'reg_date': reg_dt.date(),
# #                 'reg_datetime': reg_dt,
# #                 'is_self_selected': is_self_selected,
# #                 'original_row': row
# #             }
# #             patients.append(p)
        
# #         # 🔥 关键优化：按检查类型排序，然后再按登记时间排序
# #         # 这会让相同检查类型的病人聚在一起，Solver 按顺序处理时自然减少换模
# #         patients.sort(key=lambda x: (x['exam_type'], x['reg_datetime']))
        
# #         print(f"成功导入 {len(patients)} 名患者。")
# #         return patients
# #     except Exception as e:
# #         print(f"数据导入错误: {e}")
# #         traceback.print_exc()
# #         raise

# # def import_device_constraints(file_path):
# #     print("正在导入设备限制...")
# #     try:
# #         df = safe_read_excel(file_path)
# #         machine_exam_map = defaultdict(set)
# #         for _, row in df.iterrows():
# #             mid = int(row['设备']) - 1
# #             exam = clean_exam_name(row['检查项目'])
# #             machine_exam_map[mid].add(exam)
# #         return machine_exam_map
# #     except Exception as e:
# #         print(f"导入设备限制数据错误: {e}")
# #         traceback.print_exc()
# #         raise

# # # ===================== 核心算法：CP-SAT 滚动调度器 (对齐版) =====================

# # class RollingHorizonScheduler:
# #     def __init__(self, patients, machine_exam_map, start_date):
# #         self.all_patients = patients
# #         self.machine_exam_map = machine_exam_map
# #         self.global_start_date = start_date
# #         # 记录每台机器每一天已经被占用到了第几分钟
# #         self.machine_occupied_until = defaultdict(int)
# #         self.final_schedule = []
        
# #         self.daily_work_minutes = {}
# #         for d in range(1, 8):
# #             hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
# #             self.daily_work_minutes[d] = int(round(hours_avail * 60))

# #     def get_work_window(self, date_obj):
# #         weekday = date_obj.isoweekday()
# #         limit = self.daily_work_minutes.get(weekday, 0)
# #         return 0, limit

# #     def solve(self):
# #         total_patients = len(self.all_patients)
# #         num_workers = multiprocessing.cpu_count()
# #         print(f"\n🚀 开始滚动优化 (已对齐规则)，启用 {num_workers} 线程")
# #         print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}")

# #         for i in range(0, total_patients, BATCH_SIZE):
# #             batch_patients = self.all_patients[i : min(i + BATCH_SIZE, total_patients)]
# #             print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者 {i} - {i + len(batch_patients)}")
# #             self.solve_batch(batch_patients, num_workers)
            
# #         print("\n所有批次处理完毕。")

# #     def solve_batch(self, batch_patients, num_workers):
# #         model = cp_model.CpModel()
        
# #         intervals = {} 
# #         presences = {}
# #         starts = {}
# #         p_data = {} 
        
# #         # 1. 建模
# #         for p_idx, p in enumerate(batch_patients):
# #             p_data[p_idx] = p
# #             possible_intervals = []
            
# #             # 基础约束：最早只能从今天开始，或者从预约/登记日期开始
# #             earliest_date = max(p['reg_date'], self.global_start_date.date())
# #             start_day_offset = (earliest_date - self.global_start_date.date()).days
            
# #             # 检查属性，用于后续规则过滤
# #             exam_name = str(p['exam_type'])
# #             is_heart = '心脏' in exam_name
# #             is_angio = '造影' in exam_name
# #             is_contrast = '增强' in exam_name

# #             # 搜索未来 N 天
# #             for d in range(SEARCH_DAYS):
# #                 current_day_offset = start_day_offset + d
# #                 current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
# #                 day_start_min, day_end_min = self.get_work_window(current_date)
                
# #                 # 如果当天没时间，跳过
# #                 if day_end_min <= 0: continue 
                
# #                 # 获取星期几 (1=Mon, 7=Sun)
# #                 weekday_iso = current_date.isoweekday()

# #                 for m_id in range(MACHINE_COUNT):
# #                     # --- 基础设备能力约束 ---
# #                     if p['exam_type'] not in self.machine_exam_map[m_id]:
# #                         continue
                    
# #                     # --- 🔥 强制对齐 GPU 规则 (Constraint Alignment) ---
# #                     # 规则1: 心脏 -> 只能是 设备4 (index 3) 且 周二(2)或周四(4)
# #                     if is_heart:
# #                         if m_id != 3 or weekday_iso not in [2, 4]:
# #                             continue

# #                     # 规则2: 造影 -> 只能是 设备2 (index 1) 且 周一(1)、周三(3)、周五(5)
# #                     if is_angio:
# #                         if m_id != 1 or weekday_iso not in [1, 3, 5]:
# #                             continue

# #                     # 规则3: 周末不能做增强
# #                     if is_contrast and weekday_iso in [6, 7]:
# #                         continue
# #                     # ------------------------------------------------

# #                     # 检查是否还有剩余时间
# #                     occupied_until = self.machine_occupied_until[(m_id, current_date)]
# #                     if occupied_until + p['duration'] > day_end_min:
# #                         continue 
                    
# #                     # 创建变量
# #                     suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
# #                     is_present = model.NewBoolVar(f"pres{suffix}")
# #                     presences[(p_idx, m_id, current_day_offset)] = is_present
                    
# #                     # Start 变量范围：[已有占用时间, 关门时间 - 耗时]
# #                     # 这里隐含了 LOGICAL 约束：start >= earliest_date (通过循环逻辑保证)
# #                     # 且 start >= occupied_until (顺序排队)
# #                     start_var = model.NewIntVar(occupied_until, day_end_min - p['duration'], f"start{suffix}")
# #                     end_var = model.NewIntVar(occupied_until + p['duration'], day_end_min, f"end{suffix}")
                    
# #                     interval_var = model.NewOptionalIntervalVar(
# #                         start_var, p['duration'], end_var, is_present, f"interval{suffix}"
# #                     )
                    
# #                     intervals[(p_idx, m_id, current_day_offset)] = interval_var
# #                     starts[(p_idx, m_id, current_day_offset)] = start_var
# #                     possible_intervals.append(is_present)
            
# #             # 每个病人必须被安排一次
# #             if possible_intervals:
# #                 model.Add(sum(possible_intervals) == 1)
# #             else:
# #                 # 如果搜了 SEARCH_DAYS 还没空位，或者规则卡死了，可能导致无解
# #                 # 实际生产中这里应该报警或扩大 SEARCH_DAYS
# #                 print(f"⚠️ 警告: 患者 {p['id']} ({p['exam_type']}) 在 {SEARCH_DAYS} 天内无符合规则的空位")
        
# #         # 2. 约束：区间不重叠
# #         # 由于我们采用了简单的 "Start >= occupied_until" 的滚动填充策略，
# #         # 实际上同一天同一台机器的 interval 都在竞争同一个 occupied_until 起跑线。
# #         # CP-SAT 的 NoOverlap 会确保它们排好队，谁先谁后由 Cost 决定。
# #         machine_day_intervals = defaultdict(list)
# #         for key, interval in intervals.items():
# #             _, m_id, day_offset = key
# #             machine_day_intervals[(m_id, day_offset)].append(interval)
        
# #         for key, interval_list in machine_day_intervals.items():
# #             model.AddNoOverlap(interval_list)
            
# #         # 3. 目标优化 (Objective Alignment)
# #         day_costs = []
# #         for key, is_present in presences.items():
# #             p_idx, _, day_offset = key
# #             p = p_data[p_idx]
            
# #             # 🔥 权重对齐：自选时间惩罚远大于非自选
# #             # 自选 = 8000/天, 非自选 = 800/天
# #             weight = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
            
# #             # Cost = 是否选择该方案 * 等待天数 * 权重
# #             day_costs.append(is_present * day_offset * weight)
            
# #         model.Minimize(sum(day_costs))

# #         # 4. 求解
# #         solver = cp_model.CpSolver()
# #         solver.parameters.num_search_workers = num_workers 
# #         solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
# #         solver.parameters.log_search_progress = False
        
# #         status = solver.Solve(model)
        
# #         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
# #             print(f"  -> 求解成功 ({solver.StatusName(status)})")
            
# #             # 提取结果并更新全局状态
# #             current_batch_updates = defaultdict(list) 
            
# #             for key, is_present in presences.items():
# #                 if solver.Value(is_present):
# #                     p_idx, m_id, day_offset = key
# #                     start_val = solver.Value(starts[key])
# #                     p = p_data[p_idx]
# #                     duration = p['duration']
# #                     end_val = start_val + duration
                    
# #                     real_date = self.global_start_date.date() + timedelta(days=day_offset)
                    
# #                     record = {
# #                         'patient_id': p['id'],
# #                         'exam_type': p['exam_type'],
# #                         'reg_date': p['reg_date'],
# #                         'is_self_selected': p['is_self_selected'],
# #                         'machine_id': m_id + 1, 
# #                         'date': real_date,
# #                         'start_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=start_val)).time(),
# #                         'end_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=end_val)).time(),
# #                         'wait_days': (real_date - p['reg_date']).days
# #                     }
# #                     self.final_schedule.append(record)
# #                     current_batch_updates[(m_id, real_date)].append(end_val)
            
# #             # 更新机器占用表：推进"起跑线"
# #             for (m_id, d_date), ends in current_batch_updates.items():
# #                 self.machine_occupied_until[(m_id, d_date)] = max(
# #                     self.machine_occupied_until[(m_id, d_date)], 
# #                     max(ends)
# #                 )
# #         else:
# #             print("  -> 求解失败，无可行解 (可能是规则太严或时间窗太短)")

# #     def evaluate_schedule_score(self):
# #         if not self.final_schedule:
# #             return 0, {}

# #         print("\n" + "="*50)
# #         print("🔎 正在进行 GPU 标准评分 (最终验证)...")
# #         print("="*50)

# #         total_score = 0
# #         details = defaultdict(int)

# #         # 必须排序才能正确计算换模
# #         sorted_sched = sorted(
# #             self.final_schedule, 
# #             key=lambda x: (x['machine_id'], x['date'], x['start_time'])
# #         )

# #         prev_machine = -1
# #         prev_exam_type = None
# #         prev_date = None

# #         for item in sorted_sched:
# #             # 1. 等待时间惩罚
# #             wait_days = (item['date'] - item['reg_date']).days
# #             # 防止逻辑错误导致 wait_days < 0 (Logical Penalty)
# #             if wait_days < 0:
# #                 total_score -= LOGICAL_PENALTY
# #                 details['logical_violation'] += 1
# #                 wait_cost = 0 # 避免重复计算
# #             else:
# #                 weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
# #                 wait_cost = wait_days * weight
            
# #             total_score -= wait_cost
# #             details['wait_cost'] += wait_cost

# #             # 2. 换模惩罚
# #             if (item['machine_id'] == prev_machine and 
# #                 item['date'] == prev_date):
# #                 if item['exam_type'] != prev_exam_type:
# #                     total_score -= TRANSITION_PENALTY
# #                     details['transition_cost'] += TRANSITION_PENALTY
# #                     details['transition_count'] += 1
            
# #             prev_machine = item['machine_id']
# #             prev_exam_type = item['exam_type']
# #             prev_date = item['date']

# #             # 3. 规则/设备惩罚 (验证是否彻底过滤)
# #             weekday = item['date'].isoweekday() 
# #             m_idx = item['machine_id'] - 1      
# #             exam_name = str(item['exam_type'])

# #             is_heart = '心脏' in exam_name
# #             is_angio = '造影' in exam_name
# #             is_contrast = '增强' in exam_name

# #             rule_violated = False

# #             if is_heart:
# #                 ok_wd = (weekday == 2 or weekday == 4) # 周二/四
# #                 ok_mc = (m_idx == 3) # 设备4
# #                 if not (ok_wd and ok_mc):
# #                     rule_violated = True
# #                     details['heart_violation'] += 1

# #             if is_angio:
# #                 ok_wd = (weekday == 1 or weekday == 3 or weekday == 5) # 周一/三/五
# #                 ok_mc = (m_idx == 1) # 设备2
# #                 if not (ok_wd and ok_mc):
# #                     rule_violated = True
# #                     details['angio_violation'] += 1

# #             is_weekend = (weekday == 6 or weekday == 7)
# #             if is_contrast and is_weekend:
# #                 rule_violated = True
# #                 details['weekend_contrast_violation'] += 1

# #             if rule_violated:
# #                 total_score -= DEVICE_PENALTY

# #         print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
# #         print("-" * 30)
# #         print(f"  ❌ 总扣分: {-total_score:,.0f}")
# #         print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
# #         print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
# #         print(f"  ⚠️ 逻辑(反向等待)违规: {details['logical_violation']} 次")
# #         print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
# #         print(f"  💉 造影规则违规: {details['angio_violation']} 次")
# #         print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")
        
# #         if details['heart_violation'] + details['angio_violation'] + details['weekend_contrast_violation'] == 0:
# #             print("\n✅ 恭喜！所有特殊规则约束已完美对齐 (违规数为0)。")
# #         else:
# #             print("\n❌ 警告！仍有规则违规，请检查约束代码。")
            
# #         print("="*50 + "\n")
        
# #         return total_score, details

# #     def export_excel(self, filename, score_data=None):
# #         if not self.final_schedule:
# #             print("没有排程数据可导出。")
# #             return
            
# #         df = pd.DataFrame(self.final_schedule)
# #         cols = ['patient_id', 'exam_type', 'reg_date', 'is_self_selected', 
# #                 'machine_id', 'date', 'start_time', 'end_time', 'wait_days']
# #         df = df[cols]
# #         df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)
        
# #         try:
# #             with pd.ExcelWriter(filename) as writer:
# #                 df.to_excel(writer, sheet_name='详细排程', index=False)
# #                 stats = df.groupby('date').size().reset_index(name='每日检查量')
# #                 stats.to_excel(writer, sheet_name='统计', index=False)
                
# #                 if score_data:
# #                     score, details = score_data
# #                     score_items = [
# #                         ['Total Score (Fitness)', score],
# #                         ['Total Penalty', -score],
# #                         ['Wait Cost', details['wait_cost']],
# #                         ['Transition Cost', details['transition_cost']],
# #                         ['Transition Count', details['transition_count']],
# #                         ['Heart Violations', details['heart_violation']],
# #                         ['Angio Violations', details['angio_violation']],
# #                         ['Weekend Contrast Violations', details['weekend_contrast_violation']]
# #                     ]
# #                     score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
# #                     score_df.to_excel(writer, sheet_name='评分报告', index=False)
                    
# #             print(f"排程已成功导出至: {filename}")
# #         except Exception as e:
# #             print(f"导出 Excel 失败: {e}")

# # # ===================== 主程序 =====================

# # def main():
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
# #     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
# #     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
# #     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
    
# #     for f in [patient_file, duration_file, device_constraint_file]:
# #         if not os.path.exists(f):
# #             print(f"❌ 错误：找不到文件 {f}")
# #             return

# #     patients = import_data(patient_file, duration_file)
# #     machine_map = import_device_constraints(device_constraint_file)
    
# #     scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
# #     scheduler.solve()
# #     score, details = scheduler.evaluate_schedule_score()
    
# #     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# #     out_file = os.path.join(current_dir, f'aligned_schedule_{ts}.xlsx')
# #     scheduler.export_excel(out_file, score_data=(score, details))

# # if __name__ == "__main__":
# #     multiprocessing.freeze_support()
# #     main()


# #秒
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, time as datetime_time
# import os
# from collections import defaultdict
# import traceback
# import re
# import math
# import multiprocessing
# from ortools.sat.python import cp_model

# # ===================== 全局常量 (严格对齐 GPU 实验代码) =====================
# WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# WORK_START_STR = '07:00'
# WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# START_DATE = datetime(2025, 1, 1, 7, 0)
# MACHINE_COUNT = 6

# # 求解器配置
# BATCH_SIZE = 200       # 批次大小，可根据内存调整
# SEARCH_DAYS = 30        # 搜索未来多少天的空闲（建议覆盖最大等待期）
# SOLVER_TIME_LIMIT = 6000000  # 每个批次的求解时间限制(秒)

# # ===================== 评分常量 (对齐 GPU 实验代码) =====================
# TRANSITION_PENALTY = 20000      # 换模惩罚
# SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重（按天）
# NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重（按天）
# DEVICE_PENALTY = 500000         # 设备/规则违规惩罚
# LOGICAL_PENALTY = 10000         # 逻辑违规（如反向等待，CP-SAT中通过硬约束避免）

# # ===================== 数据导入工具 =====================

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
#     print("正在导入患者数据...")
#     try:
#         # 读取“检查类型 → 平均耗时(分钟, 可为小数)”
#         duration_df = safe_read_excel(duration_file)
#         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
#         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

#         patient_df = safe_read_excel(patient_file)
#         patients = []

#         for _, row in patient_df.iterrows():
#             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
#                 continue

#             raw_id = str(row['id']).strip()
#             reg_dt = pd.to_datetime(row['登记日期'])
#             cid = (raw_id, reg_dt.strftime('%Y%m%d'))

#             exam_type = clean_exam_name(row['检查项目'])

#             # ---- 关键：耗时按“秒”精度处理 ----
#             # 假设 exam_durations 中的值单位是“分钟（浮点）”
#             val = exam_durations.get(exam_type, 15.0)  # 默认 15 分钟
#             try:
#                 duration_raw_min = float(val)            # 例如 10.5 分钟
#             except Exception:
#                 # 如果有奇怪格式，退回默认 15 分钟
#                 duration_raw_min = 15.0
#             duration_sec = int(round(duration_raw_min * 60))  # 分钟 → 秒
#             duration_sec = max(1, duration_sec)               # 至少 1 秒

#             is_self_selected = (row['是否自选时间'] == '自选时间')

#             p = {
#                 'id': raw_id,
#                 'cid': cid,
#                 'exam_type': exam_type,
#                 'duration': duration_sec,           # 内部统一用“秒”
#                 'reg_date': reg_dt.date(),
#                 'reg_datetime': reg_dt,
#                 'is_self_selected': is_self_selected,
#                 'original_row': row
#             }
#             patients.append(p)

#         # 🔥 关键优化：按检查类型排序，然后再按登记时间排序
#         patients.sort(key=lambda x: (x['exam_type'], x['reg_datetime']))

#         print(f"成功导入 {len(patients)} 名患者。")
#         return patients
#     except Exception as e:
#         print(f"数据导入错误: {e}")
#         traceback.print_exc()
#         raise

# def import_device_constraints(file_path):
#     print("正在导入设备限制...")
#     try:
#         df = safe_read_excel(file_path)
#         machine_exam_map = defaultdict(set)
#         for _, row in df.iterrows():
#             mid = int(row['设备']) - 1
#             exam = clean_exam_name(row['检查项目'])
#             machine_exam_map[mid].add(exam)
#         return machine_exam_map
#     except Exception as e:
#         print(f"导入设备限制数据错误: {e}")
#         traceback.print_exc()
#         raise

# # ===================== 核心算法：CP-SAT 滚动调度器 (秒精度，对齐版) =====================

# class RollingHorizonScheduler:
#     def __init__(self, patients, machine_exam_map, start_date):
#         self.all_patients = patients
#         self.machine_exam_map = machine_exam_map
#         self.global_start_date = start_date

#         # 记录每台机器每一天已经被占用到的“秒数”（从 WORK_START 起算）
#         self.machine_occupied_until = defaultdict(int)
#         self.final_schedule = []

#         # 每天最大可用工作时间（单位：秒）
#         self.daily_work_seconds = {}
#         for d in range(1, 8):
#             hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)   # 可用小时数
#             self.daily_work_seconds[d] = int(round(hours_avail * 3600))  # 小时 → 秒

#     def get_work_window(self, date_obj):
#         """返回某天工作窗口 [0, limit_sec]，单位：秒"""
#         weekday = date_obj.isoweekday()
#         limit = self.daily_work_seconds.get(weekday, 0)
#         return 0, limit

#     def solve(self):
#         total_patients = len(self.all_patients)
#         num_workers = multiprocessing.cpu_count()
#         print(f"\n🚀 开始滚动优化 (已对齐规则，时间精度：秒)，启用 {num_workers} 线程")
#         print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}")

#         for i in range(0, total_patients, BATCH_SIZE):
#             batch_patients = self.all_patients[i: min(i + BATCH_SIZE, total_patients)]
#             print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者索引 {i} - {i + len(batch_patients) - 1}")
#             self.solve_batch(batch_patients, num_workers)

#         print("\n所有批次处理完毕。")

#     def solve_batch(self, batch_patients, num_workers):
#         model = cp_model.CpModel()

#         intervals = {}   # (p_idx, m_id, day_offset) -> IntervalVar
#         presences = {}   # (p_idx, m_id, day_offset) -> BoolVar
#         starts = {}      # (p_idx, m_id, day_offset) -> IntVar (秒)
#         p_data = {}      # p_idx -> 病人信息

#         # 1. 建模
#         for p_idx, p in enumerate(batch_patients):
#             p_data[p_idx] = p
#             possible_intervals = []

#             # 最早可以安排的日期：登记日 或 全局起始日 之后
#             earliest_date = max(p['reg_date'], self.global_start_date.date())
#             start_day_offset = (earliest_date - self.global_start_date.date()).days

#             exam_name = str(p['exam_type'])
#             is_heart = '心脏' in exam_name
#             is_angio = '造影' in exam_name
#             is_contrast = '增强' in exam_name

#             for d in range(SEARCH_DAYS):
#                 current_day_offset = start_day_offset + d
#                 current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
#                 day_start_sec, day_end_sec = self.get_work_window(current_date)

#                 if day_end_sec <= 0:
#                     continue

#                 weekday_iso = current_date.isoweekday()  # 1=Mon ... 7=Sun

#                 for m_id in range(MACHINE_COUNT):
#                     # --- 基础设备能力约束 ---
#                     if p['exam_type'] not in self.machine_exam_map[m_id]:
#                         continue

#                     # --- 特殊规则，对齐 GPU ---
#                     # 规则1: 心脏 -> 设备4(index 3) 且 周二(2) or 周四(4)
#                     if is_heart:
#                         if m_id != 3 or weekday_iso not in [2, 4]:
#                             continue

#                     # 规则2: 造影 -> 设备2(index 1) 且 周一(1) / 三(3) / 五(5)
#                     if is_angio:
#                         if m_id != 1 or weekday_iso not in [1, 3, 5]:
#                             continue

#                     # 规则3: 周末不能做增强
#                     if is_contrast and weekday_iso in [6, 7]:
#                         continue
#                     # -----------------------

#                     # 剩余时间是否可容纳该检查
#                     occupied_until = self.machine_occupied_until[(m_id, current_date)]  # 已占用秒数
#                     if occupied_until + p['duration'] > day_end_sec:
#                         continue

#                     suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
#                     is_present = model.NewBoolVar(f"pres{suffix}")
#                     presences[(p_idx, m_id, current_day_offset)] = is_present

#                     # 开始时间变量：单位秒
#                     start_var = model.NewIntVar(
#                         occupied_until,
#                         day_end_sec - p['duration'],
#                         f"start{suffix}"
#                     )
#                     end_var = model.NewIntVar(
#                         occupied_until + p['duration'],
#                         day_end_sec,
#                         f"end{suffix}"
#                     )

#                     interval_var = model.NewOptionalIntervalVar(
#                         start_var, p['duration'], end_var, is_present, f"interval{suffix}"
#                     )

#                     intervals[(p_idx, m_id, current_day_offset)] = interval_var
#                     starts[(p_idx, m_id, current_day_offset)] = start_var
#                     possible_intervals.append(is_present)

#             # 每个病人必须被安排一次（如果根本没有合法位置，完全不加约束）
#             if possible_intervals:
#                 model.Add(sum(possible_intervals) == 1)
#             else:
#                 print(f"⚠️ 警告: 患者 {p['id']} ({p['exam_type']}) 在 {SEARCH_DAYS} 天内无符合规则的空位")

#         # 2. 每台机每天 NoOverlap
#         machine_day_intervals = defaultdict(list)
#         for key, interval in intervals.items():
#             _, m_id, day_offset = key
#             machine_day_intervals[(m_id, day_offset)].append(interval)

#         for key, interval_list in machine_day_intervals.items():
#             model.AddNoOverlap(interval_list)

#         # 3. 目标：只优化等候天数（按天 * 权重），与 GPU 一致
#         day_costs = []
#         for key, is_present in presences.items():
#             p_idx, _, day_offset = key
#             p = p_data[p_idx]

#             weight = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
#             # day_offset = (assigned_day - START_DATE) 的天数
#             day_costs.append(is_present * day_offset * weight)

#         model.Minimize(sum(day_costs))

#         # 4. 求解
#         solver = cp_model.CpSolver()
#         solver.parameters.num_search_workers = num_workers
#         solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
#         solver.parameters.log_search_progress = False

#         status = solver.Solve(model)

#         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#             print(f"  -> 求解成功 ({solver.StatusName(status)})")

#             current_batch_updates = defaultdict(list)

#             for key, is_present in presences.items():
#                 if solver.Value(is_present):
#                     p_idx, m_id, day_offset = key
#                     start_val = solver.Value(starts[key])  # 秒
#                     p = p_data[p_idx]
#                     duration = p['duration']
#                     end_val = start_val + duration

#                     real_date = self.global_start_date.date() + timedelta(days=day_offset)

#                     record = {
#                         'patient_id': p['id'],
#                         'exam_type': p['exam_type'],
#                         'reg_date': p['reg_date'],
#                         'is_self_selected': p['is_self_selected'],
#                         'machine_id': m_id + 1,
#                         'date': real_date,
#                         # 关键：用秒构造真实时间
#                         'start_time': (
#                             datetime.combine(real_date, WORK_START) +
#                             timedelta(seconds=start_val)
#                         ).time(),
#                         'end_time': (
#                             datetime.combine(real_date, WORK_START) +
#                             timedelta(seconds=end_val)
#                         ).time(),
#                         'wait_days': (real_date - p['reg_date']).days
#                     }
#                     self.final_schedule.append(record)
#                     current_batch_updates[(m_id, real_date)].append(end_val)

#             # 更新机器占用表
#             for (m_id, d_date), ends in current_batch_updates.items():
#                 self.machine_occupied_until[(m_id, d_date)] = max(
#                     self.machine_occupied_until[(m_id, d_date)],
#                     max(ends)
#                 )
#         else:
#             print("  -> 求解失败，无可行解 (可能是规则太严或时间窗太短)")

#     def evaluate_schedule_score(self):
#         if not self.final_schedule:
#             return 0, {}

#         print("\n" + "="*50)
#         print("🔎 正在进行 GPU 标准评分 (最终验证)...")
#         print("="*50)

#         total_score = 0
#         details = defaultdict(int)

#         # 按 (machine, date, start_time) 排序，才能正确算换模
#         sorted_sched = sorted(
#             self.final_schedule,
#             key=lambda x: (x['machine_id'], x['date'], x['start_time'])
#         )

#         prev_machine = -1
#         prev_exam_type = None
#         prev_date = None

#         for item in sorted_sched:
#             # 1. 等待时间惩罚（按天）
#             wait_days = (item['date'] - item['reg_date']).days
#             if wait_days < 0:
#                 total_score -= LOGICAL_PENALTY
#                 details['logical_violation'] += 1
#                 wait_cost = 0
#             else:
#                 weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
#                 wait_cost = wait_days * weight

#             total_score -= wait_cost
#             details['wait_cost'] += wait_cost

#             # 2. 换模惩罚（同机同日，前后检查类型不同）
#             if (item['machine_id'] == prev_machine and
#                 item['date'] == prev_date):
#                 if item['exam_type'] != prev_exam_type:
#                     total_score -= TRANSITION_PENALTY
#                     details['transition_cost'] += TRANSITION_PENALTY
#                     details['transition_count'] += 1

#             prev_machine = item['machine_id']
#             prev_exam_type = item['exam_type']
#             prev_date = item['date']

#             # 3. 规则/设备惩罚（验证是否有漏网之鱼）
#             weekday = item['date'].isoweekday()  # 1=Mon ... 7=Sun
#             m_idx = item['machine_id'] - 1
#             exam_name = str(item['exam_type'])

#             is_heart = '心脏' in exam_name
#             is_angio = '造影' in exam_name
#             is_contrast = '增强' in exam_name

#             rule_violated = False

#             if is_heart:
#                 ok_wd = (weekday == 2 or weekday == 4)  # 周二/四
#                 ok_mc = (m_idx == 3)                    # 设备4
#                 if not (ok_wd and ok_mc):
#                     rule_violated = True
#                     details['heart_violation'] += 1

#             if is_angio:
#                 ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)  # 周一/三/五
#                 ok_mc = (m_idx == 1)                                    # 设备2
#                 if not (ok_wd and ok_mc):
#                     rule_violated = True
#                     details['angio_violation'] += 1

#             is_weekend = (weekday == 6 or weekday == 7)
#             if is_contrast and is_weekend:
#                 rule_violated = True
#                 details['weekend_contrast_violation'] += 1

#             if rule_violated:
#                 total_score -= DEVICE_PENALTY

#         print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
#         print("-" * 30)
#         print(f"  ❌ 总扣分: {-total_score:,.0f}")
#         print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
#         print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
#         print(f"  ⚠️ 逻辑(反向等待)违规: {details['logical_violation']} 次")
#         print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
#         print(f"  💉 造影规则违规: {details['angio_violation']} 次")
#         print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")

#         if (details['heart_violation'] +
#             details['angio_violation'] +
#             details['weekend_contrast_violation']) == 0:
#             print("\n✅ 所有特殊规则约束已满足（违规数为0）。")
#         else:
#             print("\n❌ 警告：存在规则违规，请检查约束和数据。")

#         print("="*50 + "\n")

#         return total_score, details

#     def export_excel(self, filename, score_data=None):
#         if not self.final_schedule:
#             print("没有排程数据可导出。")
#             return

#         df = pd.DataFrame(self.final_schedule)
#         cols = [
#             'patient_id', 'exam_type', 'reg_date', 'is_self_selected',
#             'machine_id', 'date', 'start_time', 'end_time', 'wait_days'
#         ]
#         df = df[cols]
#         df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)

#         try:
#             with pd.ExcelWriter(filename) as writer:
#                 df.to_excel(writer, sheet_name='详细排程', index=False)

#                 stats = df.groupby('date').size().reset_index(name='每日检查量')
#                 stats.to_excel(writer, sheet_name='统计', index=False)

#                 if score_data:
#                     score, details = score_data
#                     score_items = [
#                         ['Total Score (Fitness)', score],
#                         ['Total Penalty', -score],
#                         ['Wait Cost', details['wait_cost']],
#                         ['Transition Cost', details['transition_cost']],
#                         ['Transition Count', details['transition_count']],
#                         ['Heart Violations', details['heart_violation']],
#                         ['Angio Violations', details['angio_violation']],
#                         ['Weekend Contrast Violations', details['weekend_contrast_violation']]
#                     ]
#                     score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
#                     score_df.to_excel(writer, sheet_name='评分报告', index=False)

#             print(f"排程已成功导出至: {filename}")
#         except Exception as e:
#             print(f"导出 Excel 失败: {e}")

# # ===================== 主程序 =====================

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

#     for f in [patient_file, duration_file, device_constraint_file]:
#         if not os.path.exists(f):
#             print(f"❌ 错误：找不到文件 {f}")
#             return

#     patients = import_data(patient_file, duration_file)
#     machine_map = import_device_constraints(device_constraint_file)

#     scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
#     scheduler.solve()
#     score, details = scheduler.evaluate_schedule_score()

#     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out_file = os.path.join(current_dir, f'aligned_schedule_seconds_{ts}.xlsx')
#     scheduler.export_excel(out_file, score_data=(score, details))

# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()



# # import pandas as pd
# # import numpy as np
# # from datetime import datetime, timedelta, time as datetime_time
# # import os
# # from collections import defaultdict
# # import traceback
# # import re
# # import math
# # import multiprocessing # 引入多进程库以检测核数
# # from ortools.sat.python import cp_model

# # # ===================== 全局常量 (完全对齐 GPU 实验代码) =====================
# # WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# # WORK_START_STR = '07:00'
# # WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# # START_DATE = datetime(2024, 12, 1, 7, 0)
# # MACHINE_COUNT = 6

# # # 求解器配置
# # # ⚠️ 修改说明：
# # # 1. 窗口保持 1000 以获得全局最优性
# # # 2. 时间限制 120秒，配合多线程通常能在几十秒内找到极优解
# # BATCH_SIZE = 1600       
# # SEARCH_DAYS = 15        
# # SOLVER_TIME_LIMIT = 360000000000

# # # ===================== 评分常量 (来自 GPU 实验代码) =====================
# # TRANSITION_PENALTY = 20000      # 换模惩罚
# # SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
# # NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
# # DEVICE_PENALTY = 500000         # 设备/规则违规惩罚

# # # ===================== 数据导入工具 (复用并对齐逻辑) =====================

# # def clean_exam_name(name):
# #     s = str(name).strip().lower()
# #     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
# #     s = re.sub(r'[^\w()-]', '', s)
# #     return s.replace('_', '-').replace(' ', '')

# # def safe_read_excel(file_path, sheet_name=0):
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
# #     return pd.read_excel(file_path, sheet_name=sheet_name)

# # def import_data(patient_file, duration_file):
# #     print("正在导入患者数据...")
# #     try:
# #         duration_df = safe_read_excel(duration_file)
# #         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
# #         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

# #         patient_df = safe_read_excel(patient_file)
# #         patients = [] 
        
# #         for _, row in patient_df.iterrows():
# #             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
# #                 continue
            
# #             raw_id = str(row['id']).strip()
# #             reg_dt = pd.to_datetime(row['登记日期'])
# #             cid = (raw_id, reg_dt.strftime('%Y%m%d'))
            
# #             exam_type = clean_exam_name(row['检查项目'])
            
# #             duration_raw = float(exam_durations.get(exam_type, 15.0))
# #             duration_int = int(round(duration_raw)) 
            
# #             is_self_selected = (row['是否自选时间'] == '自选时间')
            
# #             p = {
# #                 'id': raw_id,
# #                 'cid': cid,
# #                 'exam_type': exam_type,
# #                 'duration': max(1, duration_int), 
# #                 'reg_date': reg_dt.date(),
# #                 'reg_datetime': reg_dt,
# #                 'is_self_selected': is_self_selected,
# #                 'original_row': row
# #             }
# #             patients.append(p)
            
# #         patients.sort(key=lambda x: x['reg_datetime'])
# #         print(f"成功导入 {len(patients)} 名患者。")
# #         return patients
# #     except Exception as e:
# #         print(f"数据导入错误: {e}")
# #         traceback.print_exc()
# #         raise

# # def import_device_constraints(file_path):
# #     print("正在导入设备限制...")
# #     try:
# #         df = safe_read_excel(file_path)
# #         machine_exam_map = defaultdict(set)
# #         for _, row in df.iterrows():
# #             mid = int(row['设备']) - 1
# #             exam = clean_exam_name(row['检查项目'])
# #             machine_exam_map[mid].add(exam)
# #         return machine_exam_map
# #     except Exception as e:
# #         print(f"导入设备限制数据错误: {e}")
# #         traceback.print_exc()
# #         raise

# # # ===================== 核心算法：CP-SAT 滚动调度器 =====================

# # class RollingHorizonScheduler:
# #     def __init__(self, patients, machine_exam_map, start_date):
# #         self.all_patients = patients
# #         self.machine_exam_map = machine_exam_map
# #         self.global_start_date = start_date
# #         self.machine_occupied_until = defaultdict(int)
# #         self.final_schedule = []
        
# #         self.daily_work_minutes = {}
# #         for d in range(1, 8):
# #             hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
# #             self.daily_work_minutes[d] = int(round(hours_avail * 60))

# #     def get_work_window(self, date_obj):
# #         weekday = date_obj.isoweekday()
# #         limit = self.daily_work_minutes.get(weekday, 0)
# #         return 0, limit

# #     def solve(self):
# #         total_patients = len(self.all_patients)
# #         # 获取CPU核心数
# #         num_workers = multiprocessing.cpu_count()
# #         print(f"\n🚀 开始滚动优化，已启用 {num_workers} 线程并行加速")
# #         print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}, 搜索范围: {SEARCH_DAYS} 天")

# #         for i in range(0, total_patients, BATCH_SIZE):
# #             batch_patients = self.all_patients[i : min(i + BATCH_SIZE, total_patients)]
# #             print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者 {i} - {i + len(batch_patients)}")
# #             self.solve_batch(batch_patients, num_workers)
            
# #         print("\n所有批次处理完毕。")

# #     def solve_batch(self, batch_patients, num_workers):
# #         model = cp_model.CpModel()
        
# #         intervals = {} 
# #         presences = {}
# #         starts = {}
# #         p_data = {} 
        
# #         # 1. 建模 (同前)
# #         for p_idx, p in enumerate(batch_patients):
# #             p_data[p_idx] = p
# #             possible_intervals = []
            
# #             earliest_date = max(p['reg_date'], self.global_start_date.date())
# #             start_day_offset = (earliest_date - self.global_start_date.date()).days
            
# #             for d in range(SEARCH_DAYS):
# #                 current_day_offset = start_day_offset + d
# #                 current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
# #                 day_start_min, day_end_min = self.get_work_window(current_date)
                
# #                 if day_end_min <= 0: continue 
                
# #                 for m_id in range(MACHINE_COUNT):
# #                     if p['exam_type'] not in self.machine_exam_map[m_id]:
# #                         continue
# #                     occupied_until = self.machine_occupied_until[(m_id, current_date)]
# #                     if occupied_until + p['duration'] > day_end_min:
# #                         continue 
                        
# #                     suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
# #                     is_present = model.NewBoolVar(f"pres{suffix}")
# #                     presences[(p_idx, m_id, current_day_offset)] = is_present
                    
# #                     start_var = model.NewIntVar(occupied_until, day_end_min - p['duration'], f"start{suffix}")
# #                     end_var = model.NewIntVar(occupied_until + p['duration'], day_end_min, f"end{suffix}")
# #                     interval_var = model.NewOptionalIntervalVar(
# #                         start_var, p['duration'], end_var, is_present, f"interval{suffix}"
# #                     )
                    
# #                     intervals[(p_idx, m_id, current_day_offset)] = interval_var
# #                     starts[(p_idx, m_id, current_day_offset)] = start_var
# #                     possible_intervals.append(is_present)
            
# #             if possible_intervals:
# #                 model.Add(sum(possible_intervals) == 1)
# #             else:
# #                 pass 
# #                 # print(f"警告：患者 {p['cid']} 无可用资源")
        
# #         # 2. 约束
# #         machine_day_intervals = defaultdict(list)
# #         for key, interval in intervals.items():
# #             _, m_id, day_offset = key
# #             machine_day_intervals[(m_id, day_offset)].append(interval)
# #         for key, interval_list in machine_day_intervals.items():
# #             model.AddNoOverlap(interval_list)
            
# #         # 3. 目标优化
# #         day_costs = []
# #         for key, is_present in presences.items():
# #             _, _, day_offset = key
# #             day_costs.append(is_present * day_offset)
# #         model.Minimize(sum(day_costs))

# #         # 4. 求解与加速配置
# #         solver = cp_model.CpSolver()
        
# #         # 🔥🔥🔥 核心加速配置 🔥🔥🔥
# #         # 启用所有 CPU 核心并行搜索
# #         solver.parameters.num_search_workers = num_workers 
# #         # 设置时间限制
# #         solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
# #         # 打印进度 (让你看到它在飞快地工作)
# #         solver.parameters.log_search_progress = True 
        
# #         status = solver.Solve(model)
        
# #         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
# #             print(f"  -> 求解成功 ({solver.StatusName(status)}), 耗时 {solver.UserTime():.2f}s")
            
# #             current_batch_updates = defaultdict(list) 
# #             for key, is_present in presences.items():
# #                 if solver.Value(is_present):
# #                     p_idx, m_id, day_offset = key
# #                     start_val = solver.Value(starts[key])
# #                     p = p_data[p_idx]
# #                     duration = p['duration']
# #                     end_val = start_val + duration
# #                     real_date = self.global_start_date.date() + timedelta(days=day_offset)
                    
# #                     record = {
# #                         'patient_id': p['id'],
# #                         'exam_type': p['exam_type'],
# #                         'reg_date': p['reg_date'],
# #                         'is_self_selected': p['is_self_selected'],
# #                         'machine_id': m_id + 1, 
# #                         'date': real_date,
# #                         'start_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=start_val)).time(),
# #                         'end_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=end_val)).time(),
# #                         'wait_days': (real_date - p['reg_date']).days
# #                     }
# #                     self.final_schedule.append(record)
# #                     current_batch_updates[(m_id, real_date)].append(end_val)
            
# #             for (m_id, d_date), ends in current_batch_updates.items():
# #                 self.machine_occupied_until[(m_id, d_date)] = max(
# #                     self.machine_occupied_until[(m_id, d_date)], 
# #                     max(ends)
# #                 )
# #         else:
# #             print("  -> 求解失败，无可行解")

# #     def evaluate_schedule_score(self):
# #         if not self.final_schedule:
# #             return 0, {}

# #         print("\n" + "="*50)
# #         print("🔎 正在进行 GPU 标准评分 (Python 实现版)...")
# #         print("="*50)

# #         total_score = 0
# #         details = defaultdict(int)

# #         sorted_sched = sorted(
# #             self.final_schedule, 
# #             key=lambda x: (x['machine_id'], x['date'], x['start_time'])
# #         )

# #         prev_machine = -1
# #         prev_exam_type = None
# #         prev_date = None

# #         for item in sorted_sched:
# #             wait_days = (item['date'] - item['reg_date']).days
# #             weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
# #             wait_cost = max(0, wait_days) * weight
# #             total_score -= wait_cost
# #             details['wait_cost'] += wait_cost

# #             if (item['machine_id'] == prev_machine and 
# #                 item['date'] == prev_date):
# #                 if item['exam_type'] != prev_exam_type:
# #                     total_score -= TRANSITION_PENALTY
# #                     details['transition_cost'] += TRANSITION_PENALTY
# #                     details['transition_count'] += 1
            
# #             prev_machine = item['machine_id']
# #             prev_exam_type = item['exam_type']
# #             prev_date = item['date']

# #             weekday = item['date'].isoweekday() 
# #             m_idx = item['machine_id'] - 1      
# #             exam_name = str(item['exam_type'])

# #             is_heart = '心脏' in exam_name
# #             is_angio = '造影' in exam_name
# #             is_contrast = '增强' in exam_name

# #             if is_heart:
# #                 ok_wd = (weekday == 1 or weekday == 3)
# #                 ok_mc = (m_idx == 3)
# #                 if not (ok_wd and ok_mc):
# #                     total_score -= DEVICE_PENALTY
# #                     details['heart_violation'] += 1

# #             if is_angio:
# #                 ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)
# #                 ok_mc = (m_idx == 1)
# #                 if not (ok_wd and ok_mc):
# #                     total_score -= DEVICE_PENALTY
# #                     details['angio_violation'] += 1

# #             is_weekend = (weekday == 6 or weekday == 7)
# #             if is_contrast and is_weekend:
# #                 total_score -= DEVICE_PENALTY
# #                 details['weekend_contrast_violation'] += 1

# #         print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
# #         print("-" * 30)
# #         print(f"  ❌ 总扣分: {-total_score:,.0f}")
# #         print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
# #         print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
# #         print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
# #         print(f"  💉 造影规则违规: {details['angio_violation']} 次")
# #         print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")
# #         print("="*50 + "\n")
        
# #         return total_score, details

# #     def export_excel(self, filename, score_data=None):
# #         if not self.final_schedule:
# #             print("没有排程数据可导出。")
# #             return
            
# #         df = pd.DataFrame(self.final_schedule)
# #         cols = ['patient_id', 'exam_type', 'reg_date', 'is_self_selected', 
# #                 'machine_id', 'date', 'start_time', 'end_time', 'wait_days']
# #         df = df[cols]
# #         df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)
        
# #         try:
# #             with pd.ExcelWriter(filename) as writer:
# #                 df.to_excel(writer, sheet_name='详细排程', index=False)
# #                 stats = df.groupby('date').size().reset_index(name='每日检查量')
# #                 stats.to_excel(writer, sheet_name='统计', index=False)
                
# #                 if score_data:
# #                     score, details = score_data
# #                     score_items = [
# #                         ['Total Score (Fitness)', score],
# #                         ['Total Penalty', -score],
# #                         ['Wait Cost', details['wait_cost']],
# #                         ['Transition Cost', details['transition_cost']],
# #                         ['Transition Count', details['transition_count']],
# #                         ['Heart Rule Violations', details['heart_violation']],
# #                         ['Angio Rule Violations', details['angio_violation']],
# #                         ['Weekend Contrast Violations', details['weekend_contrast_violation']]
# #                     ]
# #                     score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
# #                     score_df.to_excel(writer, sheet_name='评分报告', index=False)
                    
# #             print(f"排程已成功导出至: {filename}")
# #         except Exception as e:
# #             print(f"导出 Excel 失败: {e}")

# # # ===================== 主程序 =====================

# # def main():
# #     current_dir = os.path.dirname(os.path.abspath(__file__))
# #     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
# #     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
# #     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
    
# #     for f in [patient_file, duration_file, device_constraint_file]:
# #         if not os.path.exists(f):
# #             print(f"❌ 错误：找不到文件 {f}")
# #             return

# #     patients = import_data(patient_file, duration_file)
# #     machine_map = import_device_constraints(device_constraint_file)
    
# #     scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
# #     scheduler.solve()
# #     score, details = scheduler.evaluate_schedule_score()
    
# #     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# #     out_file = os.path.join(current_dir, f'精确排程结果_{ts}.xlsx')
# #     scheduler.export_excel(out_file, score_data=(score, details))

# # if __name__ == "__main__":
# #     main()

# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta, time as datetime_time
# import os
# from collections import defaultdict
# import traceback
# import re
# import math
# import multiprocessing
# from ortools.sat.python import cp_model

# # ===================== 全局常量 (严格对齐 GPU 实验代码) =====================
# WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# WORK_START_STR = '07:00'
# WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
# START_DATE = datetime(2024, 12, 1, 7, 0)
# MACHINE_COUNT = 6

# # 求解器配置
# BATCH_SIZE = 100       # 批次大小，可根据内存调整
# SEARCH_DAYS = 30        # 搜索未来多少天的空闲（建议覆盖最大等待期）
# SOLVER_TIME_LIMIT = 60000000  # 每个批次的求解时间限制(秒)

# # ===================== 评分常量 (对齐 GPU 实验代码) =====================
# TRANSITION_PENALTY = 20000      # 换模惩罚
# SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
# NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
# DEVICE_PENALTY = 500000         # 设备/规则违规惩罚
# LOGICAL_PENALTY = 10000         # 逻辑违规（如反向等待，CP-SAT中通过硬约束避免）

# # ===================== 数据导入工具 =====================

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
#     print("正在导入患者数据...")
#     try:
#         duration_df = safe_read_excel(duration_file)
#         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
#         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

#         patient_df = safe_read_excel(patient_file)
#         patients = [] 
        
#         for _, row in patient_df.iterrows():
#             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
#                 continue
            
#             raw_id = str(row['id']).strip()
#             reg_dt = pd.to_datetime(row['登记日期'])
#             cid = (raw_id, reg_dt.strftime('%Y%m%d'))
            
#             exam_type = clean_exam_name(row['检查项目'])
            
#             duration_raw = float(exam_durations.get(exam_type, 15.0))
#             duration_int = int(round(duration_raw)) 
            
#             is_self_selected = (row['是否自选时间'] == '自选时间')
            
#             p = {
#                 'id': raw_id,
#                 'cid': cid,
#                 'exam_type': exam_type,
#                 'duration': max(1, duration_int), 
#                 'reg_date': reg_dt.date(),
#                 'reg_datetime': reg_dt,
#                 'is_self_selected': is_self_selected,
#                 'original_row': row
#             }
#             patients.append(p)
        
#         # 🔥 关键优化：按检查类型排序，然后再按登记时间排序
#         # 这会让相同检查类型的病人聚在一起，Solver 按顺序处理时自然减少换模
#         patients.sort(key=lambda x: (x['exam_type'], x['reg_datetime']))
        
#         print(f"成功导入 {len(patients)} 名患者。")
#         return patients
#     except Exception as e:
#         print(f"数据导入错误: {e}")
#         traceback.print_exc()
#         raise

# def import_device_constraints(file_path):
#     print("正在导入设备限制...")
#     try:
#         df = safe_read_excel(file_path)
#         machine_exam_map = defaultdict(set)
#         for _, row in df.iterrows():
#             mid = int(row['设备']) - 1
#             exam = clean_exam_name(row['检查项目'])
#             machine_exam_map[mid].add(exam)
#         return machine_exam_map
#     except Exception as e:
#         print(f"导入设备限制数据错误: {e}")
#         traceback.print_exc()
#         raise

# # ===================== 核心算法：CP-SAT 滚动调度器 (对齐版) =====================

# class RollingHorizonScheduler:
#     def __init__(self, patients, machine_exam_map, start_date):
#         self.all_patients = patients
#         self.machine_exam_map = machine_exam_map
#         self.global_start_date = start_date
#         # 记录每台机器每一天已经被占用到了第几分钟
#         self.machine_occupied_until = defaultdict(int)
#         self.final_schedule = []
        
#         self.daily_work_minutes = {}
#         for d in range(1, 8):
#             hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
#             self.daily_work_minutes[d] = int(round(hours_avail * 60))

#     def get_work_window(self, date_obj):
#         weekday = date_obj.isoweekday()
#         limit = self.daily_work_minutes.get(weekday, 0)
#         return 0, limit

#     def solve(self):
#         total_patients = len(self.all_patients)
#         num_workers = multiprocessing.cpu_count()
#         print(f"\n🚀 开始滚动优化 (已对齐规则)，启用 {num_workers} 线程")
#         print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}")

#         for i in range(0, total_patients, BATCH_SIZE):
#             batch_patients = self.all_patients[i : min(i + BATCH_SIZE, total_patients)]
#             print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者 {i} - {i + len(batch_patients)}")
#             self.solve_batch(batch_patients, num_workers)
            
#         print("\n所有批次处理完毕。")

#     def solve_batch(self, batch_patients, num_workers):
#         model = cp_model.CpModel()
        
#         intervals = {} 
#         presences = {}
#         starts = {}
#         p_data = {} 
        
#         # 1. 建模
#         for p_idx, p in enumerate(batch_patients):
#             p_data[p_idx] = p
#             possible_intervals = []
            
#             # 基础约束：最早只能从今天开始，或者从预约/登记日期开始
#             earliest_date = max(p['reg_date'], self.global_start_date.date())
#             start_day_offset = (earliest_date - self.global_start_date.date()).days
            
#             # 检查属性，用于后续规则过滤
#             exam_name = str(p['exam_type'])
#             is_heart = '心脏' in exam_name
#             is_angio = '造影' in exam_name
#             is_contrast = '增强' in exam_name

#             # 搜索未来 N 天
#             for d in range(SEARCH_DAYS):
#                 current_day_offset = start_day_offset + d
#                 current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
#                 day_start_min, day_end_min = self.get_work_window(current_date)
                
#                 # 如果当天没时间，跳过
#                 if day_end_min <= 0: continue 
                
#                 # 获取星期几 (1=Mon, 7=Sun)
#                 weekday_iso = current_date.isoweekday()

#                 for m_id in range(MACHINE_COUNT):
#                     # --- 基础设备能力约束 ---
#                     if p['exam_type'] not in self.machine_exam_map[m_id]:
#                         continue
                    
#                     # --- 🔥 强制对齐 GPU 规则 (Constraint Alignment) ---
#                     # 规则1: 心脏 -> 只能是 设备4 (index 3) 且 周二(2)或周四(4)
#                     if is_heart:
#                         if m_id != 3 or weekday_iso not in [2, 4]:
#                             continue

#                     # 规则2: 造影 -> 只能是 设备2 (index 1) 且 周一(1)、周三(3)、周五(5)
#                     if is_angio:
#                         if m_id != 1 or weekday_iso not in [1, 3, 5]:
#                             continue

#                     # 规则3: 周末不能做增强
#                     if is_contrast and weekday_iso in [6, 7]:
#                         continue
#                     # ------------------------------------------------

#                     # 检查是否还有剩余时间
#                     occupied_until = self.machine_occupied_until[(m_id, current_date)]
#                     if occupied_until + p['duration'] > day_end_min:
#                         continue 
                    
#                     # 创建变量
#                     suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
#                     is_present = model.NewBoolVar(f"pres{suffix}")
#                     presences[(p_idx, m_id, current_day_offset)] = is_present
                    
#                     # Start 变量范围：[已有占用时间, 关门时间 - 耗时]
#                     # 这里隐含了 LOGICAL 约束：start >= earliest_date (通过循环逻辑保证)
#                     # 且 start >= occupied_until (顺序排队)
#                     start_var = model.NewIntVar(occupied_until, day_end_min - p['duration'], f"start{suffix}")
#                     end_var = model.NewIntVar(occupied_until + p['duration'], day_end_min, f"end{suffix}")
                    
#                     interval_var = model.NewOptionalIntervalVar(
#                         start_var, p['duration'], end_var, is_present, f"interval{suffix}"
#                     )
                    
#                     intervals[(p_idx, m_id, current_day_offset)] = interval_var
#                     starts[(p_idx, m_id, current_day_offset)] = start_var
#                     possible_intervals.append(is_present)
            
#             # 每个病人必须被安排一次
#             if possible_intervals:
#                 model.Add(sum(possible_intervals) == 1)
#             else:
#                 # 如果搜了 SEARCH_DAYS 还没空位，或者规则卡死了，可能导致无解
#                 # 实际生产中这里应该报警或扩大 SEARCH_DAYS
#                 print(f"⚠️ 警告: 患者 {p['id']} ({p['exam_type']}) 在 {SEARCH_DAYS} 天内无符合规则的空位")
        
#         # 2. 约束：区间不重叠
#         # 由于我们采用了简单的 "Start >= occupied_until" 的滚动填充策略，
#         # 实际上同一天同一台机器的 interval 都在竞争同一个 occupied_until 起跑线。
#         # CP-SAT 的 NoOverlap 会确保它们排好队，谁先谁后由 Cost 决定。
#         machine_day_intervals = defaultdict(list)
#         for key, interval in intervals.items():
#             _, m_id, day_offset = key
#             machine_day_intervals[(m_id, day_offset)].append(interval)
        
#         for key, interval_list in machine_day_intervals.items():
#             model.AddNoOverlap(interval_list)
            
#         # 3. 目标优化 (Objective Alignment)
#         day_costs = []
#         for key, is_present in presences.items():
#             p_idx, _, day_offset = key
#             p = p_data[p_idx]
            
#             # 🔥 权重对齐：自选时间惩罚远大于非自选
#             # 自选 = 8000/天, 非自选 = 800/天
#             weight = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
            
#             # Cost = 是否选择该方案 * 等待天数 * 权重
#             day_costs.append(is_present * day_offset * weight)
            
#         model.Minimize(sum(day_costs))

#         # 4. 求解
#         solver = cp_model.CpSolver()
#         solver.parameters.num_search_workers = num_workers 
#         solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
#         solver.parameters.log_search_progress = False
        
#         status = solver.Solve(model)
        
#         if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#             print(f"  -> 求解成功 ({solver.StatusName(status)})")
            
#             # 提取结果并更新全局状态
#             current_batch_updates = defaultdict(list) 
            
#             for key, is_present in presences.items():
#                 if solver.Value(is_present):
#                     p_idx, m_id, day_offset = key
#                     start_val = solver.Value(starts[key])
#                     p = p_data[p_idx]
#                     duration = p['duration']
#                     end_val = start_val + duration
                    
#                     real_date = self.global_start_date.date() + timedelta(days=day_offset)
                    
#                     record = {
#                         'patient_id': p['id'],
#                         'exam_type': p['exam_type'],
#                         'reg_date': p['reg_date'],
#                         'is_self_selected': p['is_self_selected'],
#                         'machine_id': m_id + 1, 
#                         'date': real_date,
#                         'start_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=start_val)).time(),
#                         'end_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=end_val)).time(),
#                         'wait_days': (real_date - p['reg_date']).days
#                     }
#                     self.final_schedule.append(record)
#                     current_batch_updates[(m_id, real_date)].append(end_val)
            
#             # 更新机器占用表：推进"起跑线"
#             for (m_id, d_date), ends in current_batch_updates.items():
#                 self.machine_occupied_until[(m_id, d_date)] = max(
#                     self.machine_occupied_until[(m_id, d_date)], 
#                     max(ends)
#                 )
#         else:
#             print("  -> 求解失败，无可行解 (可能是规则太严或时间窗太短)")

#     def evaluate_schedule_score(self):
#         if not self.final_schedule:
#             return 0, {}

#         print("\n" + "="*50)
#         print("🔎 正在进行 GPU 标准评分 (最终验证)...")
#         print("="*50)

#         total_score = 0
#         details = defaultdict(int)

#         # 必须排序才能正确计算换模
#         sorted_sched = sorted(
#             self.final_schedule, 
#             key=lambda x: (x['machine_id'], x['date'], x['start_time'])
#         )

#         prev_machine = -1
#         prev_exam_type = None
#         prev_date = None

#         for item in sorted_sched:
#             # 1. 等待时间惩罚
#             wait_days = (item['date'] - item['reg_date']).days
#             # 防止逻辑错误导致 wait_days < 0 (Logical Penalty)
#             if wait_days < 0:
#                 total_score -= LOGICAL_PENALTY
#                 details['logical_violation'] += 1
#                 wait_cost = 0 # 避免重复计算
#             else:
#                 weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
#                 wait_cost = wait_days * weight
            
#             total_score -= wait_cost
#             details['wait_cost'] += wait_cost

#             # 2. 换模惩罚
#             if (item['machine_id'] == prev_machine and 
#                 item['date'] == prev_date):
#                 if item['exam_type'] != prev_exam_type:
#                     total_score -= TRANSITION_PENALTY
#                     details['transition_cost'] += TRANSITION_PENALTY
#                     details['transition_count'] += 1
            
#             prev_machine = item['machine_id']
#             prev_exam_type = item['exam_type']
#             prev_date = item['date']

#             # 3. 规则/设备惩罚 (验证是否彻底过滤)
#             weekday = item['date'].isoweekday() 
#             m_idx = item['machine_id'] - 1      
#             exam_name = str(item['exam_type'])

#             is_heart = '心脏' in exam_name
#             is_angio = '造影' in exam_name
#             is_contrast = '增强' in exam_name

#             rule_violated = False

#             if is_heart:
#                 ok_wd = (weekday == 2 or weekday == 4) # 周二/四
#                 ok_mc = (m_idx == 3) # 设备4
#                 if not (ok_wd and ok_mc):
#                     rule_violated = True
#                     details['heart_violation'] += 1

#             if is_angio:
#                 ok_wd = (weekday == 1 or weekday == 3 or weekday == 5) # 周一/三/五
#                 ok_mc = (m_idx == 1) # 设备2
#                 if not (ok_wd and ok_mc):
#                     rule_violated = True
#                     details['angio_violation'] += 1

#             is_weekend = (weekday == 6 or weekday == 7)
#             if is_contrast and is_weekend:
#                 rule_violated = True
#                 details['weekend_contrast_violation'] += 1

#             if rule_violated:
#                 total_score -= DEVICE_PENALTY

#         print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
#         print("-" * 30)
#         print(f"  ❌ 总扣分: {-total_score:,.0f}")
#         print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
#         print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
#         print(f"  ⚠️ 逻辑(反向等待)违规: {details['logical_violation']} 次")
#         print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
#         print(f"  💉 造影规则违规: {details['angio_violation']} 次")
#         print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")
        
#         if details['heart_violation'] + details['angio_violation'] + details['weekend_contrast_violation'] == 0:
#             print("\n✅ 恭喜！所有特殊规则约束已完美对齐 (违规数为0)。")
#         else:
#             print("\n❌ 警告！仍有规则违规，请检查约束代码。")
            
#         print("="*50 + "\n")
        
#         return total_score, details

#     def export_excel(self, filename, score_data=None):
#         if not self.final_schedule:
#             print("没有排程数据可导出。")
#             return
            
#         df = pd.DataFrame(self.final_schedule)
#         cols = ['patient_id', 'exam_type', 'reg_date', 'is_self_selected', 
#                 'machine_id', 'date', 'start_time', 'end_time', 'wait_days']
#         df = df[cols]
#         df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)
        
#         try:
#             with pd.ExcelWriter(filename) as writer:
#                 df.to_excel(writer, sheet_name='详细排程', index=False)
#                 stats = df.groupby('date').size().reset_index(name='每日检查量')
#                 stats.to_excel(writer, sheet_name='统计', index=False)
                
#                 if score_data:
#                     score, details = score_data
#                     score_items = [
#                         ['Total Score (Fitness)', score],
#                         ['Total Penalty', -score],
#                         ['Wait Cost', details['wait_cost']],
#                         ['Transition Cost', details['transition_cost']],
#                         ['Transition Count', details['transition_count']],
#                         ['Heart Violations', details['heart_violation']],
#                         ['Angio Violations', details['angio_violation']],
#                         ['Weekend Contrast Violations', details['weekend_contrast_violation']]
#                     ]
#                     score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
#                     score_df.to_excel(writer, sheet_name='评分报告', index=False)
                    
#             print(f"排程已成功导出至: {filename}")
#         except Exception as e:
#             print(f"导出 Excel 失败: {e}")

# # ===================== 主程序 =====================

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#     duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#     device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
    
#     for f in [patient_file, duration_file, device_constraint_file]:
#         if not os.path.exists(f):
#             print(f"❌ 错误：找不到文件 {f}")
#             return

#     patients = import_data(patient_file, duration_file)
#     machine_map = import_device_constraints(device_constraint_file)
    
#     scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
#     scheduler.solve()
#     score, details = scheduler.evaluate_schedule_score()
    
#     ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#     out_file = os.path.join(current_dir, f'aligned_schedule_{ts}.xlsx')
#     scheduler.export_excel(out_file, score_data=(score, details))

# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()

#转换成秒级别的目标函数
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as datetime_time
import os
from collections import defaultdict
import traceback
import re
import math
import multiprocessing
from ortools.sat.python import cp_model

# ===================== 全局常量 (严格对齐 GPU 实验代码) =====================
# 每日截止时间（小时），用于计算每日可用时长
WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}

# 工作开始时间
WORK_START_STR = '07:00'
WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()

# 全局排程起始日期
START_DATE = datetime(2025, 1, 1, 7, 0)
MACHINE_COUNT = 6

# ===================== 求解器配置 =====================
BATCH_SIZE = 200        # 批次大小 (固定分块)
SEARCH_DAYS = 30        # 搜索未来多少天的空闲
SOLVER_TIME_LIMIT = 3600000 # 每个批次的求解时间限制(秒)

# ===================== 评分常量 =====================
TRANSITION_PENALTY = 20000      # 换模惩罚
SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
DEVICE_PENALTY = 500000         # 设备/规则违规惩罚
LOGICAL_PENALTY = 10000         # 逻辑违规

# ===================== 数据导入工具 =====================

def clean_exam_name(name):
    s = str(name).strip().lower()
    s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
    s = re.sub(r'[^\w()-]', '', s)
    return s.replace('_', '-').replace(' ', '')

def safe_read_excel(file_path, sheet_name=0):
    if file_path.endswith('.xlsx'):
        engines = ['openpyxl', 'odf']
    elif file_path.endswith('.xls'):
        engines = ['xlrd']
    else:
        engines = ['openpyxl', 'xlrd', 'odf']
    for engine in engines:
        try:
            return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
        except Exception:
            continue
    return pd.read_excel(file_path, sheet_name=sheet_name)

def import_data(patient_file, duration_file):
    print("正在导入患者数据...")
    try:
        # 读取“检查类型 → 平均耗时(分钟)”
        duration_df = safe_read_excel(duration_file)
        duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
        exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

        patient_df = safe_read_excel(patient_file)
        patients = []

        for _, row in patient_df.iterrows():
            if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
                continue

            raw_id = str(row['id']).strip()
            reg_dt = pd.to_datetime(row['登记日期'])
            cid = (raw_id, reg_dt.strftime('%Y%m%d'))

            exam_type = clean_exam_name(row['检查项目'])

            # ---- 修改：耗时转换为秒 ----
            val = exam_durations.get(exam_type, 15.0)  # 默认 15 分钟
            try:
                duration_raw_min = float(val)
            except Exception:
                duration_raw_min = 15.0
            
            # 分钟 -> 秒
            duration_sec = int(round(duration_raw_min * 60))
            duration_sec = max(1, duration_sec)

            is_self_selected = (row['是否自选时间'] == '自选时间')

            p = {
                'id': raw_id,
                'cid': cid,
                'exam_type': exam_type,
                'duration': duration_sec,  # 单位：秒
                'reg_date': reg_dt.date(),
                'reg_datetime': reg_dt,
                'is_self_selected': is_self_selected,
                'original_row': row
            }
            patients.append(p)

        # ---- 修改：仅按登记日期排序，移除 exam_type 聚类 ----
        patients.sort(key=lambda x: x['reg_datetime'])

        print(f"成功导入 {len(patients)} 名患者。")
        return patients
    except Exception as e:
        print(f"数据导入错误: {e}")
        traceback.print_exc()
        raise

def import_device_constraints(file_path):
    print("正在导入设备限制...")
    try:
        df = safe_read_excel(file_path)
        machine_exam_map = defaultdict(set)
        for _, row in df.iterrows():
            mid = int(row['设备']) - 1
            exam = clean_exam_name(row['检查项目'])
            machine_exam_map[mid].add(exam)
        return machine_exam_map
    except Exception as e:
        print(f"导入设备限制数据错误: {e}")
        traceback.print_exc()
        raise

# ===================== 核心算法：CP-SAT 滚动调度器 (秒级精度) =====================

class RollingHorizonScheduler:
    def __init__(self, patients, machine_exam_map, start_date):
        self.all_patients = patients
        self.machine_exam_map = machine_exam_map
        self.global_start_date = start_date

        # 记录每台机器每一天已经被占用到的“秒数”（从 WORK_START 起算）
        self.machine_occupied_until = defaultdict(int)
        self.final_schedule = []

        # ---- 修改：每日最大可用工作时间（单位：秒）----
        self.daily_work_seconds = {}
        for d in range(1, 8):
            hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)   # 可用小时数
            self.daily_work_seconds[d] = int(round(hours_avail * 3600))  # 小时 → 秒

    def get_work_window(self, date_obj):
        """返回某天工作窗口 [0, limit_sec]，单位：秒"""
        weekday = date_obj.isoweekday()
        limit = self.daily_work_seconds.get(weekday, 0)
        return 0, limit

    def solve(self):
        total_patients = len(self.all_patients)
        num_workers = multiprocessing.cpu_count()
        print(f"\n🚀 开始滚动优化 (秒级精度，仅按登记时间排序)，启用 {num_workers} 线程")
        print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}")

        for i in range(0, total_patients, BATCH_SIZE):
            batch_patients = self.all_patients[i: min(i + BATCH_SIZE, total_patients)]
            print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者索引 {i} - {i + len(batch_patients) - 1}")
            self.solve_batch(batch_patients, num_workers)

        print("\n所有批次处理完毕。")

    def solve_batch(self, batch_patients, num_workers):
        model = cp_model.CpModel()

        intervals = {}   # (p_idx, m_id, day_offset) -> IntervalVar
        presences = {}   # (p_idx, m_id, day_offset) -> BoolVar
        starts = {}      # (p_idx, m_id, day_offset) -> IntVar (秒)
        p_data = {}      # p_idx -> 病人信息

        # 1. 建模
        for p_idx, p in enumerate(batch_patients):
            p_data[p_idx] = p
            possible_intervals = []

            earliest_date = max(p['reg_date'], self.global_start_date.date())
            start_day_offset = (earliest_date - self.global_start_date.date()).days

            exam_name = str(p['exam_type'])
            is_heart = '心脏' in exam_name
            is_angio = '造影' in exam_name
            is_contrast = '增强' in exam_name

            for d in range(SEARCH_DAYS):
                current_day_offset = start_day_offset + d
                current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
                day_start_sec, day_end_sec = self.get_work_window(current_date)

                if day_end_sec <= 0:
                    continue

                weekday_iso = current_date.isoweekday()

                for m_id in range(MACHINE_COUNT):
                    # --- 基础设备能力约束 ---
                    if p['exam_type'] not in self.machine_exam_map[m_id]:
                        continue

                    # --- 特殊规则 ---
                    # 规则1: 心脏 -> 设备4(index 3) 且 周二(2) or 周四(4)
                    if is_heart:
                        if m_id != 3 or weekday_iso not in [2, 4]:
                            continue

                    # 规则2: 造影 -> 设备2(index 1) 且 周一(1) / 三(3) / 五(5)
                    if is_angio:
                        if m_id != 1 or weekday_iso not in [1, 3, 5]:
                            continue

                    # 规则3: 周末不能做增强
                    if is_contrast and weekday_iso in [6, 7]:
                        continue

                    # 剩余时间是否可容纳该检查 (秒级比较)
                    occupied_until = self.machine_occupied_until[(m_id, current_date)]
                    if occupied_until + p['duration'] > day_end_sec:
                        continue

                    suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
                    is_present = model.NewBoolVar(f"pres{suffix}")
                    presences[(p_idx, m_id, current_day_offset)] = is_present

                    # 开始时间变量：单位秒
                    start_var = model.NewIntVar(
                        occupied_until,
                        day_end_sec - p['duration'],
                        f"start{suffix}"
                    )
                    end_var = model.NewIntVar(
                        occupied_until + p['duration'],
                        day_end_sec,
                        f"end{suffix}"
                    )

                    interval_var = model.NewOptionalIntervalVar(
                        start_var, p['duration'], end_var, is_present, f"interval{suffix}"
                    )

                    intervals[(p_idx, m_id, current_day_offset)] = interval_var
                    starts[(p_idx, m_id, current_day_offset)] = start_var
                    possible_intervals.append(is_present)

            # 每个病人必须被安排一次
            if possible_intervals:
                model.Add(sum(possible_intervals) == 1)
            else:
                print(f"⚠️ 警告: 患者 {p['id']} ({p['exam_type']}) 在 {SEARCH_DAYS} 天内无符合规则的空位")

        # 2. 每台机每天 NoOverlap
        machine_day_intervals = defaultdict(list)
        for key, interval in intervals.items():
            _, m_id, day_offset = key
            machine_day_intervals[(m_id, day_offset)].append(interval)

        for key, interval_list in machine_day_intervals.items():
            model.AddNoOverlap(interval_list)

        # 3. 目标：最小化加权等待天数
        day_costs = []
        for key, is_present in presences.items():
            p_idx, _, day_offset = key
            p = p_data[p_idx]

            weight = SELF_SELECTED_PENALTY if p['is_self_selected'] else NON_SELF_PENALTY
            day_costs.append(is_present * day_offset * weight)

        model.Minimize(sum(day_costs))

        # 4. 求解
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = num_workers
        solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  -> 求解成功 ({solver.StatusName(status)})")

            current_batch_updates = defaultdict(list)

            for key, is_present in presences.items():
                if solver.Value(is_present):
                    p_idx, m_id, day_offset = key
                    start_val = solver.Value(starts[key])  # 秒
                    p = p_data[p_idx]
                    duration = p['duration']
                    end_val = start_val + duration

                    real_date = self.global_start_date.date() + timedelta(days=day_offset)

                    # ---- 修改：结果还原使用秒 ----
                    record = {
                        'patient_id': p['id'],
                        'exam_type': p['exam_type'],
                        'reg_date': p['reg_date'],
                        'is_self_selected': p['is_self_selected'],
                        'machine_id': m_id + 1,
                        'date': real_date,
                        'start_time': (
                            datetime.combine(real_date, WORK_START) +
                            timedelta(seconds=start_val)
                        ).time(),
                        'end_time': (
                            datetime.combine(real_date, WORK_START) +
                            timedelta(seconds=end_val)
                        ).time(),
                        'wait_days': (real_date - p['reg_date']).days
                    }
                    self.final_schedule.append(record)
                    current_batch_updates[(m_id, real_date)].append(end_val)

            # 更新机器占用表 (秒)
            for (m_id, d_date), ends in current_batch_updates.items():
                self.machine_occupied_until[(m_id, d_date)] = max(
                    self.machine_occupied_until[(m_id, d_date)],
                    max(ends)
                )
        else:
            print("  -> 求解失败，无可行解")

    def evaluate_schedule_score(self):
        if not self.final_schedule:
            return 0, {}

        print("\n" + "="*50)
        print("🔎 正在进行评分 (最终验证)...")
        print("="*50)

        total_score = 0
        details = defaultdict(int)

        sorted_sched = sorted(
            self.final_schedule,
            key=lambda x: (x['machine_id'], x['date'], x['start_time'])
        )

        prev_machine = -1
        prev_exam_type = None
        prev_date = None

        for item in sorted_sched:
            # 1. 等待时间惩罚
            wait_days = (item['date'] - item['reg_date']).days
            if wait_days < 0:
                total_score -= LOGICAL_PENALTY
                details['logical_violation'] += 1
                wait_cost = 0
            else:
                weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
                wait_cost = wait_days * weight

            total_score -= wait_cost
            details['wait_cost'] += wait_cost

            # 2. 换模惩罚
            if (item['machine_id'] == prev_machine and
                item['date'] == prev_date):
                if item['exam_type'] != prev_exam_type:
                    total_score -= TRANSITION_PENALTY
                    details['transition_cost'] += TRANSITION_PENALTY
                    details['transition_count'] += 1

            prev_machine = item['machine_id']
            prev_exam_type = item['exam_type']
            prev_date = item['date']

            # 3. 规则/设备惩罚
            weekday = item['date'].isoweekday()
            m_idx = item['machine_id'] - 1
            exam_name = str(item['exam_type'])

            is_heart = '心脏' in exam_name
            is_angio = '造影' in exam_name
            is_contrast = '增强' in exam_name

            rule_violated = False

            if is_heart:
                ok_wd = (weekday == 2 or weekday == 4)
                ok_mc = (m_idx == 3)
                if not (ok_wd and ok_mc):
                    rule_violated = True
                    details['heart_violation'] += 1

            if is_angio:
                ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)
                ok_mc = (m_idx == 1)
                if not (ok_wd and ok_mc):
                    rule_violated = True
                    details['angio_violation'] += 1

            is_weekend = (weekday == 6 or weekday == 7)
            if is_contrast and is_weekend:
                rule_violated = True
                details['weekend_contrast_violation'] += 1

            if rule_violated:
                total_score -= DEVICE_PENALTY

        print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
        print("-" * 30)
        print(f"  ❌ 总扣分: {-total_score:,.0f}")
        print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
        print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
        print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
        print(f"  💉 造影规则违规: {details['angio_violation']} 次")
        print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")

        return total_score, details

    def export_excel(self, filename, score_data=None):
        if not self.final_schedule:
            print("没有排程数据可导出。")
            return

        df = pd.DataFrame(self.final_schedule)
        cols = [
            'patient_id', 'exam_type', 'reg_date', 'is_self_selected',
            'machine_id', 'date', 'start_time', 'end_time', 'wait_days'
        ]
        df = df[cols]
        df.sort_values(by=['date', 'machine_id', 'start_time'], inplace=True)

        try:
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name='详细排程', index=False)

                stats = df.groupby('date').size().reset_index(name='每日检查量')
                stats.to_excel(writer, sheet_name='统计', index=False)

                if score_data:
                    score, details = score_data
                    score_items = [
                        ['Total Score (Fitness)', score],
                        ['Total Penalty', -score],
                        ['Wait Cost', details['wait_cost']],
                        ['Transition Cost', details['transition_cost']],
                        ['Transition Count', details['transition_count']],
                        ['Heart Violations', details['heart_violation']],
                        ['Angio Violations', details['angio_violation']],
                        ['Weekend Contrast Violations', details['weekend_contrast_violation']]
                    ]
                    score_df = pd.DataFrame(score_items, columns=['Metric', 'Value'])
                    score_df.to_excel(writer, sheet_name='评分报告', index=False)

            print(f"排程已成功导出至: {filename}")
        except Exception as e:
            print(f"导出 Excel 失败: {e}")

# ===================== 主程序 =====================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
    duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
    device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

    for f in [patient_file, duration_file, device_constraint_file]:
        if not os.path.exists(f):
            print(f"❌ 错误：找不到文件 {f}")
            return

    patients = import_data(patient_file, duration_file)
    machine_map = import_device_constraints(device_constraint_file)

    scheduler = RollingHorizonScheduler(patients, machine_map, START_DATE)
    scheduler.solve()
    score, details = scheduler.evaluate_schedule_score()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(current_dir, f'schedule_seconds_fifo_{ts}.xlsx')
    scheduler.export_excel(out_file, score_data=(score, details))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
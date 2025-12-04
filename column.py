import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as datetime_time
import os
from collections import defaultdict
import traceback
import re
import math
import multiprocessing # 引入多进程库以检测核数
from ortools.sat.python import cp_model

# ===================== 全局常量 (完全对齐 GPU 实验代码) =====================
WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
WORK_START_STR = '07:00'
WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()
START_DATE = datetime(2024, 12, 1, 7, 0)
MACHINE_COUNT = 6

# 求解器配置
# ⚠️ 修改说明：
# 1. 窗口保持 1000 以获得全局最优性
# 2. 时间限制 120秒，配合多线程通常能在几十秒内找到极优解
BATCH_SIZE = 3000       
SEARCH_DAYS = 15        
SOLVER_TIME_LIMIT = 360000000000

# ===================== 评分常量 (来自 GPU 实验代码) =====================
TRANSITION_PENALTY = 20000      # 换模惩罚
SELF_SELECTED_PENALTY = 8000    # 自选时间等待惩罚权重
NON_SELF_PENALTY = 800          # 非自选时间等待惩罚权重
DEVICE_PENALTY = 500000         # 设备/规则违规惩罚

# ===================== 数据导入工具 (复用并对齐逻辑) =====================

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
            
            duration_raw = float(exam_durations.get(exam_type, 15.0))
            duration_int = int(round(duration_raw)) 
            
            is_self_selected = (row['是否自选时间'] == '自选时间')
            
            p = {
                'id': raw_id,
                'cid': cid,
                'exam_type': exam_type,
                'duration': max(1, duration_int), 
                'reg_date': reg_dt.date(),
                'reg_datetime': reg_dt,
                'is_self_selected': is_self_selected,
                'original_row': row
            }
            patients.append(p)
            
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

# ===================== 核心算法：CP-SAT 滚动调度器 =====================

class RollingHorizonScheduler:
    def __init__(self, patients, machine_exam_map, start_date):
        self.all_patients = patients
        self.machine_exam_map = machine_exam_map
        self.global_start_date = start_date
        self.machine_occupied_until = defaultdict(int)
        self.final_schedule = []
        
        self.daily_work_minutes = {}
        for d in range(1, 8):
            hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
            self.daily_work_minutes[d] = int(round(hours_avail * 60))

    def get_work_window(self, date_obj):
        weekday = date_obj.isoweekday()
        limit = self.daily_work_minutes.get(weekday, 0)
        return 0, limit

    def solve(self):
        total_patients = len(self.all_patients)
        # 获取CPU核心数
        num_workers = multiprocessing.cpu_count()
        print(f"\n🚀 开始滚动优化，已启用 {num_workers} 线程并行加速")
        print(f"总计 {total_patients} 名患者，批次大小: {BATCH_SIZE}, 搜索范围: {SEARCH_DAYS} 天")

        for i in range(0, total_patients, BATCH_SIZE):
            batch_patients = self.all_patients[i : min(i + BATCH_SIZE, total_patients)]
            print(f"\n>>> 处理批次 {i // BATCH_SIZE + 1}: 患者 {i} - {i + len(batch_patients)}")
            self.solve_batch(batch_patients, num_workers)
            
        print("\n所有批次处理完毕。")

    def solve_batch(self, batch_patients, num_workers):
        model = cp_model.CpModel()
        
        intervals = {} 
        presences = {}
        starts = {}
        p_data = {} 
        
        # 1. 建模 (同前)
        for p_idx, p in enumerate(batch_patients):
            p_data[p_idx] = p
            possible_intervals = []
            
            earliest_date = max(p['reg_date'], self.global_start_date.date())
            start_day_offset = (earliest_date - self.global_start_date.date()).days
            
            for d in range(SEARCH_DAYS):
                current_day_offset = start_day_offset + d
                current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
                day_start_min, day_end_min = self.get_work_window(current_date)
                
                if day_end_min <= 0: continue 
                
                for m_id in range(MACHINE_COUNT):
                    if p['exam_type'] not in self.machine_exam_map[m_id]:
                        continue
                    occupied_until = self.machine_occupied_until[(m_id, current_date)]
                    if occupied_until + p['duration'] > day_end_min:
                        continue 
                        
                    suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
                    is_present = model.NewBoolVar(f"pres{suffix}")
                    presences[(p_idx, m_id, current_day_offset)] = is_present
                    
                    start_var = model.NewIntVar(occupied_until, day_end_min - p['duration'], f"start{suffix}")
                    end_var = model.NewIntVar(occupied_until + p['duration'], day_end_min, f"end{suffix}")
                    interval_var = model.NewOptionalIntervalVar(
                        start_var, p['duration'], end_var, is_present, f"interval{suffix}"
                    )
                    
                    intervals[(p_idx, m_id, current_day_offset)] = interval_var
                    starts[(p_idx, m_id, current_day_offset)] = start_var
                    possible_intervals.append(is_present)
            
            if possible_intervals:
                model.Add(sum(possible_intervals) == 1)
            else:
                pass 
                # print(f"警告：患者 {p['cid']} 无可用资源")
        
        # 2. 约束
        machine_day_intervals = defaultdict(list)
        for key, interval in intervals.items():
            _, m_id, day_offset = key
            machine_day_intervals[(m_id, day_offset)].append(interval)
        for key, interval_list in machine_day_intervals.items():
            model.AddNoOverlap(interval_list)
            
        # 3. 目标优化
        day_costs = []
        for key, is_present in presences.items():
            _, _, day_offset = key
            day_costs.append(is_present * day_offset)
        model.Minimize(sum(day_costs))

        # 4. 求解与加速配置
        solver = cp_model.CpSolver()
        
        # 🔥🔥🔥 核心加速配置 🔥🔥🔥
        # 启用所有 CPU 核心并行搜索
        solver.parameters.num_search_workers = num_workers 
        # 设置时间限制
        solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
        # 打印进度 (让你看到它在飞快地工作)
        solver.parameters.log_search_progress = True 
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  -> 求解成功 ({solver.StatusName(status)}), 耗时 {solver.UserTime():.2f}s")
            
            current_batch_updates = defaultdict(list) 
            for key, is_present in presences.items():
                if solver.Value(is_present):
                    p_idx, m_id, day_offset = key
                    start_val = solver.Value(starts[key])
                    p = p_data[p_idx]
                    duration = p['duration']
                    end_val = start_val + duration
                    real_date = self.global_start_date.date() + timedelta(days=day_offset)
                    
                    record = {
                        'patient_id': p['id'],
                        'exam_type': p['exam_type'],
                        'reg_date': p['reg_date'],
                        'is_self_selected': p['is_self_selected'],
                        'machine_id': m_id + 1, 
                        'date': real_date,
                        'start_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=start_val)).time(),
                        'end_time': (datetime.combine(real_date, WORK_START) + timedelta(minutes=end_val)).time(),
                        'wait_days': (real_date - p['reg_date']).days
                    }
                    self.final_schedule.append(record)
                    current_batch_updates[(m_id, real_date)].append(end_val)
            
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
        print("🔎 正在进行 GPU 标准评分 (Python 实现版)...")
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
            wait_days = (item['date'] - item['reg_date']).days
            weight = SELF_SELECTED_PENALTY if item['is_self_selected'] else NON_SELF_PENALTY
            wait_cost = max(0, wait_days) * weight
            total_score -= wait_cost
            details['wait_cost'] += wait_cost

            if (item['machine_id'] == prev_machine and 
                item['date'] == prev_date):
                if item['exam_type'] != prev_exam_type:
                    total_score -= TRANSITION_PENALTY
                    details['transition_cost'] += TRANSITION_PENALTY
                    details['transition_count'] += 1
            
            prev_machine = item['machine_id']
            prev_exam_type = item['exam_type']
            prev_date = item['date']

            weekday = item['date'].isoweekday() 
            m_idx = item['machine_id'] - 1      
            exam_name = str(item['exam_type'])

            is_heart = '心脏' in exam_name
            is_angio = '造影' in exam_name
            is_contrast = '增强' in exam_name

            if is_heart:
                ok_wd = (weekday == 1 or weekday == 3)
                ok_mc = (m_idx == 3)
                if not (ok_wd and ok_mc):
                    total_score -= DEVICE_PENALTY
                    details['heart_violation'] += 1

            if is_angio:
                ok_wd = (weekday == 1 or weekday == 3 or weekday == 5)
                ok_mc = (m_idx == 1)
                if not (ok_wd and ok_mc):
                    total_score -= DEVICE_PENALTY
                    details['angio_violation'] += 1

            is_weekend = (weekday == 6 or weekday == 7)
            if is_contrast and is_weekend:
                total_score -= DEVICE_PENALTY
                details['weekend_contrast_violation'] += 1

        print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
        print("-" * 30)
        print(f"  ❌ 总扣分: {-total_score:,.0f}")
        print(f"  ⏳ 等待时间惩罚: {details['wait_cost']:,.0f}")
        print(f"  🔄 换模惩罚:     {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
        print(f"  💔 心脏规则违规: {details['heart_violation']} 次")
        print(f"  💉 造影规则违规: {details['angio_violation']} 次")
        print(f"  🚫 周末增强违规: {details['weekend_contrast_violation']} 次")
        print("="*50 + "\n")
        
        return total_score, details

    def export_excel(self, filename, score_data=None):
        if not self.final_schedule:
            print("没有排程数据可导出。")
            return
            
        df = pd.DataFrame(self.final_schedule)
        cols = ['patient_id', 'exam_type', 'reg_date', 'is_self_selected', 
                'machine_id', 'date', 'start_time', 'end_time', 'wait_days']
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
                        ['Heart Rule Violations', details['heart_violation']],
                        ['Angio Rule Violations', details['angio_violation']],
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
    out_file = os.path.join(current_dir, f'精确排程结果_{ts}.xlsx')
    scheduler.export_excel(out_file, score_data=(score, details))

if __name__ == "__main__":
    main()
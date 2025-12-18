import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict
import traceback
import re
import multiprocessing
from ortools.sat.python import cp_model

# ===================== 全局常量 =====================
WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
WORK_START_STR = '07:00'
WORK_START = datetime.strptime(WORK_START_STR, '%H:%M').time()

START_DATE = datetime(2025, 1, 1, 7, 0)  # 你可按需调整
MACHINE_COUNT = 6

# 求解器配置
BATCH_SIZE = 100          # ✅ 仅按患者个数分块
SEARCH_DAYS = 1
SOLVER_TIME_LIMIT = 60000000   # 每批求解时间上限(秒)

# ===================== 评分常量（用于 evaluate，可保留你的口径） =====================
TRANSITION_PENALTY = 20000
SELF_SELECTED_PENALTY = 8000
NON_SELF_PENALTY = 800
DEVICE_PENALTY = 500000
LOGICAL_PENALTY = 10000

# ===================== ✅ 秒级等待目标权重 =====================
# 不引入“同类型聚类”软目标
WAIT_WEIGHT_SELF = 5
WAIT_WEIGHT_NON = 1


# ===================== 数据导入工具 =====================

def clean_exam_name(name):
    """标准化检查项目名称"""
    s = str(name).strip().lower()
    s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
    s = re.sub(r'[^\w()-]', '', s)
    return s.replace('_', '-').replace(' ', '')

def safe_read_excel(file_path, sheet_name=0):
    """兼容读取不同 Excel 引擎"""
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
    """
    导入患者 + 耗时，并将耗时统一为“秒”。
    ✅ 全局只按 reg_datetime 排序（登记时间决定骨架）
    """
    print("正在导入患者数据...")
    try:
        duration_df = safe_read_excel(duration_file)
        duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
        exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

        patient_df = safe_read_excel(patient_file)
        patients = []

        for _, row in patient_df.iterrows():
            if pd.isnull(row.get('id')) or pd.isnull(row.get('登记日期')):
                continue

            raw_id = str(row['id']).strip()
            reg_dt = pd.to_datetime(row['登记日期'])

            cid = (raw_id, reg_dt.strftime('%Y%m%d'))
            exam_type = clean_exam_name(row['检查项目'])

            # ---- 耗时处理：分钟 -> 秒（允许小数） ----
            val = exam_durations.get(exam_type, 15.0)
            try:
                duration_raw_min = float(val)
            except Exception:
                duration_raw_min = 15.0

            duration_sec = int(round(duration_raw_min * 60))
            duration_sec = max(1, duration_sec)

            is_self_selected = (row.get('是否自选时间') == '自选时间')

            p = {
                'id': raw_id,
                'cid': cid,
                'exam_type': exam_type,
                'duration': duration_sec,
                'reg_date': reg_dt.date(),
                'reg_datetime': reg_dt,
                'is_self_selected': is_self_selected,
                'original_row': row
            }
            patients.append(p)

        # ✅ 只按登记时间排序
        patients.sort(key=lambda x: x['reg_datetime'])

        print(f"成功导入 {len(patients)} 名患者。")
        return patients

    except Exception as e:
        print(f"数据导入错误: {e}")
        traceback.print_exc()
        raise

def import_device_constraints(file_path):
    """导入“设备-检查项目可做映射”"""
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

        # 记录每台机器每一天已经被占用到的“秒数”
        self.machine_occupied_until = defaultdict(int)

        # 记录每台机器每一天“最后一个检查的类型” (用于批次间换模判断)
        self.machine_last_exam_type = defaultdict(lambda: None)

        self.final_schedule = []

        # 预计算每天的工作时长（秒）
        self.daily_work_seconds = {}
        for d in range(1, 8):
            hours_avail = 15.0 - WEEKDAY_END_HOURS.get(d, 0)
            self.daily_work_seconds[d] = int(round(hours_avail * 3600))

    def get_work_window(self, date_obj):
        weekday = date_obj.isoweekday()
        limit = self.daily_work_seconds.get(weekday, 0)
        return 0, limit

    # ✅ 仅按人数分批（保持登记时间排序后的顺序）
    def build_count_batches(self):
        patients = self.all_patients
        if not patients:
            return []
        return [patients[i:i + BATCH_SIZE] for i in range(0, len(patients), BATCH_SIZE)]

    def solve(self):
        num_workers = multiprocessing.cpu_count()
        total_patients = len(self.all_patients)

        print(f"\n🚀 开始滚动优化（按人数分块 + 登记时间骨架 + 等待秒级目标）")
        print(f"🔥 已启用全CPU核心加速: {num_workers} 线程并行搜索")
        print(f"总计 {total_patients} 名患者")
        print(f"单批人数上限: {BATCH_SIZE}")

        batches = self.build_count_batches()
        print(f"共构建 {len(batches)} 个批次。")

        for bi, batch_patients in enumerate(batches, 1):
            print(f"\n>>> 处理批次 {bi}/{len(batches)}: 本批 {len(batch_patients)} 人 "
                  f"(登记时间从 {batch_patients[0]['reg_datetime']} 到 {batch_patients[-1]['reg_datetime']})")
            self.solve_batch(batch_patients, num_workers)

        print("\n所有批次处理完毕。")

    def solve_batch(self, batch_patients, num_workers):
        model = cp_model.CpModel()

        # 变量存储
        intervals = {}  # (p_idx, m_id, day_offset) -> interval_var
        presences = {}  # (p_idx, m_id, day_offset) -> bool_var
        starts = {}     # (p_idx, m_id, day_offset) -> int_var
        ends = {}       # (p_idx, m_id, day_offset) -> int_var
        waits = {}      # (p_idx, m_id, day_offset) -> int_var  ✅ 等待秒变量
        wait_mins = {}  # (p_idx, m_id, day_offset) -> int_var  ✅ 等待分钟变量（仅用于目标）

        p_data = {}

        # 辅助结构：按“机器-天”归类所有可能的任务
        machine_tasks = defaultdict(list)

        # 等待秒上界（安全冗余）
        max_wait_ub = (SEARCH_DAYS + 2) * 86400
        max_wait_min_ub = (SEARCH_DAYS + 2) * 1440  # 分钟上界

        # 1) 建模
        for p_idx, p in enumerate(batch_patients):
            p_data[p_idx] = p
            possible_pres = []

            earliest_date = max(p['reg_date'], self.global_start_date.date())
            start_day_offset = (earliest_date - self.global_start_date.date()).days

            exam_name = str(p['exam_type'])
            is_heart = '心脏' in exam_name
            is_angio = '造影' in exam_name
            is_contrast = '增强' in exam_name

            # ✅ 登记时间相对全局起点的“绝对秒”
            reg_abs_sec = int(round((p['reg_datetime'] - self.global_start_date).total_seconds()))
            reg_abs_sec = max(0, reg_abs_sec)

            for d in range(SEARCH_DAYS):
                current_day_offset = start_day_offset + d
                current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
                day_start_sec, day_end_sec = self.get_work_window(current_date)
                if day_end_sec <= 0:
                    continue

                weekday_iso = current_date.isoweekday()

                # ✅ 同一天不得早于登记时刻（相对 WORK_START 的秒下界）
                reg_time_lb = 0
                if current_date == p['reg_datetime'].date():
                    reg_t = p['reg_datetime'].time()
                    reg_dt_day = datetime.combine(current_date, reg_t)
                    work_dt_day = datetime.combine(current_date, WORK_START)
                    reg_time_lb = int(round((reg_dt_day - work_dt_day).total_seconds()))
                    reg_time_lb = max(0, reg_time_lb)

                for m_id in range(MACHINE_COUNT):

                    # --- 设备与规则过滤 ---
                    if p['exam_type'] not in self.machine_exam_map[m_id]:
                        continue
                    if is_heart and (m_id != 3 or weekday_iso not in [2, 4]):
                        continue
                    if is_angio and (m_id != 1 or weekday_iso not in [1, 3, 5]):
                        continue
                    if is_contrast and weekday_iso in [6, 7]:
                        continue

                    # ✅ 不在建模中加入任何 60s 换模间隙
                    occupied_until = self.machine_occupied_until[(m_id, current_date)]

                    # 空间检查（仅基于占用与时长）
                    if occupied_until + p['duration'] > day_end_sec:
                        continue

                    suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
                    is_present = model.NewBoolVar(f"pres{suffix}")
                    presences[(p_idx, m_id, current_day_offset)] = is_present

                    # ✅ Start 下界 = max(已占用, 登记当天时刻下界)
                    earliest_start_lb = max(occupied_until, reg_time_lb)

                    start_var = model.NewIntVar(
                        earliest_start_lb,
                        day_end_sec - p['duration'],
                        f"start{suffix}"
                    )
                    end_var = model.NewIntVar(
                        earliest_start_lb + p['duration'],
                        day_end_sec,
                        f"end{suffix}"
                    )

                    interval_var = model.NewOptionalIntervalVar(
                        start_var, p['duration'], end_var, is_present, f"interval{suffix}"
                    )

                    key = (p_idx, m_id, current_day_offset)
                    intervals[key] = interval_var
                    starts[key] = start_var
                    ends[key] = end_var
                    possible_pres.append(is_present)

                    # ✅ 等待秒变量（线性化）
                    wait_var = model.NewIntVar(0, max_wait_ub, f"wait{suffix}")
                    waits[key] = wait_var

                    scheduled_start_abs = current_day_offset * 86400 + start_var
                    model.Add(wait_var == scheduled_start_abs - reg_abs_sec).OnlyEnforceIf(is_present)
                    model.Add(wait_var == 0).OnlyEnforceIf(is_present.Not())

                    # ✅ 等待分钟变量（仅用于目标，分钟级粒度）
                    wait_min_var = model.NewIntVar(0, max_wait_min_ub, f"waitmin{suffix}")
                    wait_mins[key] = wait_min_var
                    # wait_min = wait_sec // 60
                    model.AddDivisionEquality(wait_min_var, wait_var, 60)

                    # 收集任务（用于 NoOverlap & 目标中的换模计数代理）
                    machine_tasks[(m_id, current_day_offset)].append({
                        'p_idx': p_idx,
                        'type': p['exam_type'],
                        'start': start_var,
                        'end': end_var,
                        'pres': is_present
                    })

            if possible_pres:
                model.Add(sum(possible_pres) == 1)

        # 2) 约束
        for (m_id, d_offset), task_list in machine_tasks.items():
            # A. 不重叠
            current_intervals = [
                intervals[(t['p_idx'], m_id, d_offset)] for t in task_list
            ]
            model.AddNoOverlap(current_intervals)

            # ✅ 彻底取消批内 60s 换模硬约束
            # self._add_intra_batch_gap_constraints(model, task_list)

        # 3) 目标
        # ✅ 等待时间（分钟级目标，秒级变量仍保留）
        obj_terms = []
        for key, wait_min_var in wait_mins.items():
            p_idx, _, _ = key
            p = p_data[p_idx]
            w = WAIT_WEIGHT_SELF if p['is_self_selected'] else WAIT_WEIGHT_NON
            obj_terms.append(wait_min_var * w)

        # ✅ 换模个数代理：每台机器每天使用的“检查类型数 - 1”
        switch_proxy_vars = []
        for (m_id, d_offset), task_list in machine_tasks.items():
            if not task_list:
                continue

            # 当天是否有任何任务被选中
            all_pres = [t['pres'] for t in task_list]
            any_present = model.NewBoolVar(f"any_m{m_id}_d{d_offset}")
            model.AddMaxEquality(any_present, all_pres)

            # 统计该机该日被使用的“类型数”
            type_to_pres = defaultdict(list)
            for t in task_list:
                type_to_pres[str(t['type'])].append(t['pres'])

            used_bools = []
            for idx, (_tname, pres_list) in enumerate(type_to_pres.items()):
                used = model.NewBoolVar(f"used_m{m_id}_d{d_offset}_{idx}")
                if len(pres_list) == 1:
                    model.Add(used == pres_list[0])
                else:
                    model.AddMaxEquality(used, pres_list)
                used_bools.append(used)

            tcount = len(used_bools)
            type_used_count = model.NewIntVar(0, tcount, f"type_cnt_m{m_id}_d{d_offset}")
            model.Add(type_used_count == sum(used_bools))

            # 换模个数代理 = max(0, type_used_count - 1)
            switch_proxy = model.NewIntVar(0, max(0, tcount - 1), f"sw_m{m_id}_d{d_offset}")
            model.Add(switch_proxy == type_used_count - 1).OnlyEnforceIf(any_present)
            model.Add(switch_proxy == 0).OnlyEnforceIf(any_present.Not())

            switch_proxy_vars.append(switch_proxy)

        # 目标：等待(分钟级) + 换模个数 * 系数
        if switch_proxy_vars:
            obj_terms.append(sum(switch_proxy_vars) * TRANSITION_PENALTY)

        if obj_terms:
            model.Minimize(sum(obj_terms))

        # 4) 求解
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = num_workers
        solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"  -> 求解成功 ({solver.StatusName(status)})")

            batch_results_per_machine_day = defaultdict(list)

            for key, is_present in presences.items():
                if solver.Value(is_present):
                    p_idx, m_id, day_offset = key
                    start_val = solver.Value(starts[key])
                    end_val = solver.Value(ends[key])
                    p = p_data[p_idx]

                    real_date = self.global_start_date.date() + timedelta(days=day_offset)

                    record = {
                        'patient_id': p['id'],
                        'exam_type': p['exam_type'],
                        'reg_date': p['reg_date'],
                        'reg_datetime': p['reg_datetime'],
                        'is_self_selected': p['is_self_selected'],
                        'machine_id': m_id + 1,
                        'date': real_date,
                        'start_time': (datetime.combine(real_date, WORK_START) + timedelta(seconds=start_val)).time(),
                        'end_time': (datetime.combine(real_date, WORK_START) + timedelta(seconds=end_val)).time(),
                        'wait_days': (real_date - p['reg_date']).days
                    }
                    self.final_schedule.append(record)
                    batch_results_per_machine_day[(m_id, real_date)].append((end_val, p['exam_type']))

            # 批次间状态更新（保持不变）
            for (m_id, d_date), results in batch_results_per_machine_day.items():
                max_end_time, last_exam_type = max(results, key=lambda x: x[0])

                self.machine_occupied_until[(m_id, d_date)] = max(
                    self.machine_occupied_until[(m_id, d_date)],
                    max_end_time
                )
                self.machine_last_exam_type[(m_id, d_date)] = last_exam_type

        else:
            print("  -> 求解失败，无可行解")

    # def solve_batch(self, batch_patients, num_workers):
    #     model = cp_model.CpModel()

    #     # 变量存储
    #     intervals = {}  # (p_idx, m_id, day_offset) -> interval_var
    #     presences = {}  # (p_idx, m_id, day_offset) -> bool_var
    #     starts = {}     # (p_idx, m_id, day_offset) -> int_var
    #     ends = {}       # (p_idx, m_id, day_offset) -> int_var
    #     waits = {}      # (p_idx, m_id, day_offset) -> int_var  ✅ 等待秒变量

    #     p_data = {}

    #     # 辅助结构：按“机器-天”归类所有可能的任务
    #     machine_tasks = defaultdict(list)

    #     # 等待秒上界（安全冗余）
    #     max_wait_ub = (SEARCH_DAYS + 2) * 86400

    #     # 1) 建模
    #     for p_idx, p in enumerate(batch_patients):
    #         p_data[p_idx] = p
    #         possible_pres = []

    #         earliest_date = max(p['reg_date'], self.global_start_date.date())
    #         start_day_offset = (earliest_date - self.global_start_date.date()).days

    #         exam_name = str(p['exam_type'])
    #         is_heart = '心脏' in exam_name
    #         is_angio = '造影' in exam_name
    #         is_contrast = '增强' in exam_name

    #         # ✅ 登记时间相对全局起点的“绝对秒”
    #         reg_abs_sec = int(round((p['reg_datetime'] - self.global_start_date).total_seconds()))
    #         reg_abs_sec = max(0, reg_abs_sec)

    #         for d in range(SEARCH_DAYS):
    #             current_day_offset = start_day_offset + d
    #             current_date = self.global_start_date.date() + timedelta(days=current_day_offset)
    #             day_start_sec, day_end_sec = self.get_work_window(current_date)
    #             if day_end_sec <= 0:
    #                 continue

    #             weekday_iso = current_date.isoweekday()

    #             # ✅ 同一天不得早于登记时刻（相对 WORK_START 的秒下界）
    #             reg_time_lb = 0
    #             if current_date == p['reg_datetime'].date():
    #                 reg_t = p['reg_datetime'].time()
    #                 reg_dt_day = datetime.combine(current_date, reg_t)
    #                 work_dt_day = datetime.combine(current_date, WORK_START)
    #                 reg_time_lb = int(round((reg_dt_day - work_dt_day).total_seconds()))
    #                 reg_time_lb = max(0, reg_time_lb)

    #             for m_id in range(MACHINE_COUNT):

    #                 # --- 设备与规则过滤 ---
    #                 if p['exam_type'] not in self.machine_exam_map[m_id]:
    #                     continue
    #                 if is_heart and (m_id != 3 or weekday_iso not in [2, 4]):
    #                     continue
    #                 if is_angio and (m_id != 1 or weekday_iso not in [1, 3, 5]):
    #                     continue
    #                 if is_contrast and weekday_iso in [6, 7]:
    #                     continue

    #                 # --- ✅ 不再在建模中加入 60s 换模间隙 ---
    #                 occupied_until = self.machine_occupied_until[(m_id, current_date)]

    #                 # 空间检查（仅基于占用与时长）
    #                 if occupied_until + p['duration'] > day_end_sec:
    #                     continue

    #                 suffix = f"_p{p_idx}_m{m_id}_d{current_day_offset}"
    #                 is_present = model.NewBoolVar(f"pres{suffix}")
    #                 presences[(p_idx, m_id, current_day_offset)] = is_present

    #                 # ✅ Start 下界 = max(已占用, 登记当天时刻下界)
    #                 earliest_start_lb = max(occupied_until, reg_time_lb)

    #                 start_var = model.NewIntVar(
    #                     earliest_start_lb,
    #                     day_end_sec - p['duration'],
    #                     f"start{suffix}"
    #                 )
    #                 end_var = model.NewIntVar(
    #                     earliest_start_lb + p['duration'],
    #                     day_end_sec,
    #                     f"end{suffix}"
    #                 )

    #                 interval_var = model.NewOptionalIntervalVar(
    #                     start_var, p['duration'], end_var, is_present, f"interval{suffix}"
    #                 )

    #                 key = (p_idx, m_id, current_day_offset)
    #                 intervals[key] = interval_var
    #                 starts[key] = start_var
    #                 ends[key] = end_var
    #                 possible_pres.append(is_present)

    #                 # ✅ 等待秒变量（线性化）
    #                 wait_var = model.NewIntVar(0, max_wait_ub, f"wait{suffix}")
    #                 waits[key] = wait_var

    #                 scheduled_start_abs = current_day_offset * 86400 + start_var

    #                 model.Add(wait_var == scheduled_start_abs - reg_abs_sec).OnlyEnforceIf(is_present)
    #                 model.Add(wait_var == 0).OnlyEnforceIf(is_present.Not())

    #                 # 收集任务（用于 NoOverlap & 目标中的换模计数代理）
    #                 machine_tasks[(m_id, current_day_offset)].append({
    #                     'p_idx': p_idx,
    #                     'type': p['exam_type'],
    #                     'start': start_var,
    #                     'end': end_var,
    #                     'pres': is_present
    #                 })

    #         if possible_pres:
    #             model.Add(sum(possible_pres) == 1)

    #     # 2) 约束
    #     for (m_id, d_offset), task_list in machine_tasks.items():
    #         # A. 不重叠
    #         current_intervals = [
    #             intervals[(t['p_idx'], m_id, d_offset)] for t in task_list
    #         ]
    #         model.AddNoOverlap(current_intervals)

    #         # ✅ 取消批次内 60s 换模硬约束
    #         # self._add_intra_batch_gap_constraints(model, task_list)

    #     # 3) 目标
    #     # ✅ 等待时间（秒级）
    #     obj_terms = []
    #     for key, wait_var in waits.items():
    #         p_idx, _, _ = key
    #         p = p_data[p_idx]
    #         w = WAIT_WEIGHT_SELF if p['is_self_selected'] else WAIT_WEIGHT_NON
    #         obj_terms.append(wait_var * w)

    #     # ✅ 换模个数代理：每台机器每天使用的“检查类型数 - 1”
    #     switch_proxy_vars = []
    #     for (m_id, d_offset), task_list in machine_tasks.items():
    #         if not task_list:
    #             continue

    #         # 当天是否有任何任务被选中
    #         all_pres = [t['pres'] for t in task_list]
    #         any_present = model.NewBoolVar(f"any_m{m_id}_d{d_offset}")
    #         model.AddMaxEquality(any_present, all_pres)

    #         # 统计该机该日被使用的“类型数”
    #         type_to_pres = defaultdict(list)
    #         for t in task_list:
    #             type_to_pres[str(t['type'])].append(t['pres'])

    #         used_bools = []
    #         for idx, (_tname, pres_list) in enumerate(type_to_pres.items()):
    #             used = model.NewBoolVar(f"used_m{m_id}_d{d_offset}_{idx}")
    #             if len(pres_list) == 1:
    #                 model.Add(used == pres_list[0])
    #             else:
    #                 model.AddMaxEquality(used, pres_list)
    #             used_bools.append(used)

    #         tcount = len(used_bools)
    #         type_used_count = model.NewIntVar(0, tcount, f"type_cnt_m{m_id}_d{d_offset}")
    #         model.Add(type_used_count == sum(used_bools))

    #         # 换模个数代理 = max(0, type_used_count - 1)
    #         switch_proxy = model.NewIntVar(0, max(0, tcount - 1), f"sw_m{m_id}_d{d_offset}")
    #         # 若当天有排程：sw = types - 1
    #         model.Add(switch_proxy == type_used_count - 1).OnlyEnforceIf(any_present)
    #         # 若当天无排程：sw = 0
    #         model.Add(switch_proxy == 0).OnlyEnforceIf(any_present.Not())

    #         switch_proxy_vars.append(switch_proxy)

    #     if switch_proxy_vars:
    #         obj_terms.append(sum(switch_proxy_vars) * TRANSITION_PENALTY)

    #     if obj_terms:
    #         model.Minimize(sum(obj_terms))

    #     # 4) 求解
    #     solver = cp_model.CpSolver()
    #     solver.parameters.num_search_workers = num_workers
    #     solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT
    #     solver.parameters.log_search_progress = False

    #     status = solver.Solve(model)

    #     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    #         print(f"  -> 求解成功 ({solver.StatusName(status)})")

    #         batch_results_per_machine_day = defaultdict(list)

    #         for key, is_present in presences.items():
    #             if solver.Value(is_present):
    #                 p_idx, m_id, day_offset = key
    #                 start_val = solver.Value(starts[key])
    #                 end_val = solver.Value(ends[key])
    #                 p = p_data[p_idx]

    #                 real_date = self.global_start_date.date() + timedelta(days=day_offset)

    #                 record = {
    #                     'patient_id': p['id'],
    #                     'exam_type': p['exam_type'],
    #                     'reg_date': p['reg_date'],
    #                     'reg_datetime': p['reg_datetime'],
    #                     'is_self_selected': p['is_self_selected'],
    #                     'machine_id': m_id + 1,
    #                     'date': real_date,
    #                     'start_time': (datetime.combine(real_date, WORK_START) + timedelta(seconds=start_val)).time(),
    #                     'end_time': (datetime.combine(real_date, WORK_START) + timedelta(seconds=end_val)).time(),
    #                     'wait_days': (real_date - p['reg_date']).days
    #                 }
    #                 self.final_schedule.append(record)
    #                 batch_results_per_machine_day[(m_id, real_date)].append((end_val, p['exam_type']))

    #         # 批次间状态更新（保持不变）
    #         for (m_id, d_date), results in batch_results_per_machine_day.items():
    #             max_end_time, last_exam_type = max(results, key=lambda x: x[0])

    #             self.machine_occupied_until[(m_id, d_date)] = max(
    #                 self.machine_occupied_until[(m_id, d_date)],
    #                 max_end_time
    #             )
    #             self.machine_last_exam_type[(m_id, d_date)] = last_exam_type

    #     else:
    #         print("  -> 求解失败，无可行解")

    def _add_intra_batch_gap_constraints(self, model, task_list):
        """
        批次内部换模硬约束：
        若类型不同，则两任务之间间隔必须 >= 60 秒。
        """
        n = len(task_list)
        for i in range(n):
            for j in range(i + 1, n):
                task_a = task_list[i]
                task_b = task_list[j]

                if task_a['type'] != task_b['type']:
                    is_a_before_b = model.NewBoolVar(
                        f"order_{task_a['p_idx']}_{task_b['p_idx']}"
                    )

                    model.Add(task_b['start'] >= task_a['end'] + 60).OnlyEnforceIf(
                        [task_a['pres'], task_b['pres'], is_a_before_b]
                    )

                    model.Add(task_a['start'] >= task_b['end'] + 60).OnlyEnforceIf(
                        [task_a['pres'], task_b['pres'], is_a_before_b.Not()]
                    )

    # ===================== 评分函数（用于你的对齐/分析） =====================
    def evaluate_schedule_score(self):
        if not self.final_schedule:
            return 0, {}

        print("\n" + "=" * 50)
        print("🔎 正在进行规则评分验证...")
        print("=" * 50)

        total_score = 0
        details = defaultdict(int)

        sorted_sched = sorted(
            self.final_schedule,
            key=lambda x: (x['machine_id'], x['date'], x['start_time'])
        )

        prev_machine = -1
        prev_exam_type = None
        prev_date = None
        prev_end_time = None

        for item in sorted_sched:
            # 1) 等待惩罚（仍保留“天级”评分口径）
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

            # 2) 换模惩罚 & 间隙验证
            current_start_dt = datetime.combine(item['date'], item['start_time'])

            if (item['machine_id'] == prev_machine and item['date'] == prev_date):
                if item['exam_type'] != prev_exam_type:
                    total_score -= TRANSITION_PENALTY
                    details['transition_cost'] += TRANSITION_PENALTY
                    details['transition_count'] += 1

                    if prev_end_time:
                        gap = (current_start_dt - prev_end_time).total_seconds()
                        if gap < 60:
                            print(
                                f"❌ 严重错误: 发现换模间隙不足! "
                                f"{prev_end_time.time()} -> {item['start_time']} (Gap={gap}s)"
                            )
                            details['gap_violation'] += 1

            prev_machine = item['machine_id']
            prev_exam_type = item['exam_type']
            prev_date = item['date']
            prev_end_time = datetime.combine(item['date'], item['end_time'])

            # 3) 设备/规则惩罚
            weekday = item['date'].isoweekday()
            m_idx = item['machine_id'] - 1
            exam_name = str(item['exam_type'])
            is_heart = '心脏' in exam_name
            is_angio = '造影' in exam_name
            is_contrast = '增强' in exam_name

            rule_violated = False
            if is_heart and not ((weekday in [2, 4]) and m_idx == 3):
                rule_violated = True
            if is_angio and not ((weekday in [1, 3, 5]) and m_idx == 1):
                rule_violated = True
            if is_contrast and weekday in [6, 7]:
                rule_violated = True

            if rule_violated:
                total_score -= DEVICE_PENALTY
                details['device_violation'] += 1

        print(f"📊 最终 Fitness 得分: {total_score:,.0f}")
        print(f"  ❌ 总扣分: {-total_score:,.0f}")
        print(f"  ⏳ 等待时间惩罚(天级报告口径): {details['wait_cost']:,.0f}")
        print(f"  🔄 换模惩罚: {details['transition_cost']:,.0f} (发生 {details['transition_count']} 次)")
        print(f"  ⚡ 间隙违规(Gap < 60s): {details['gap_violation']} 次")
        print(f"  🔧 设备/规则违规: {details['device_violation']} 次")

        return total_score, details

    def export_excel(self, filename, score_data=None):
        if not self.final_schedule:
            return

        df = pd.DataFrame(self.final_schedule)

        cols = [
            'patient_id', 'exam_type', 'reg_date', 'reg_datetime',
            'is_self_selected', 'machine_id', 'date',
            'start_time', 'end_time', 'wait_days'
        ]
        df = df[cols].sort_values(by=['date', 'machine_id', 'start_time'])

        try:
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name='详细排程', index=False)
                df.groupby('date').size().reset_index(name='每日检查量').to_excel(
                    writer, sheet_name='统计', index=False
                )
                if score_data:
                    score, details = score_data
                    pd.DataFrame(
                        [['Total Score', score]] + [[k, v] for k, v in details.items()],
                        columns=['Metric', 'Value']
                    ).to_excel(writer, sheet_name='评分报告', index=False)
            print(f"排程已成功导出至: {filename}")
        except Exception as e:
            print(f"导出 Excel 失败: {e}")


# ===================== 主程序 =====================

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 你原来的默认文件名（如需改名，直接改这里）
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
    out_file = os.path.join(current_dir, f'schedule_countblock_waitsec_{ts}.xlsx')
    scheduler.export_excel(out_file, score_data=(score, details))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

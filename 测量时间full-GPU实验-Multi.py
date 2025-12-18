# from __future__ import annotations
# from typing import List, Dict, Set, Tuple, Any
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import os
# from collections import defaultdict
# import traceback
# import re
# import json
# import multiprocessing
# import time
# import torch

# # ===================== 全局常量 =====================
# WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
# WORK_START = datetime.strptime('07:00', '%H:%M').time()
# TRANSITION_PENALTY = 20000
# LOGICAL = 10000
# SELF_SELECTED_PENALTY = 8000
# NON_SELF_PENALTY = 800
# START_DATE = datetime(2025, 1, 1, 7, 0)
# MACHINE_COUNT = 6
# DEVICE_PENALTY = 500000

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("⚠️ 未检测到 GPU，将使用 CPU（速度极慢）")
    
# DTYPE_LONG = torch.long
# DTYPE_FLOAT = torch.float32

# # ===================== 工具函数 =====================
# # (工具函数 ... clean_exam_name, safe_read_excel, import_data, import_device_constraints ... 保持不变)

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
#     try:
#         duration_df = safe_read_excel(duration_file)
#         duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
#         exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

#         patient_df = safe_read_excel(patient_file)
#         patients = {}
#         for _, row in patient_df.iterrows():
#             if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
#                 continue
#             cid = (str(row['id']).strip(), pd.to_datetime(row['登记日期']).strftime('%Y%m%d'))
#             exam_type = clean_exam_name(row['检查项目'])
#             duration = float(exam_durations.get(exam_type, 15.0))
#             is_self_selected = (row['是否自选时间'] == '自选时间')
#             appt_date = pd.to_datetime(row['预约日期']).date() if not pd.isnull(row['预约日期']) else None
#             if cid not in patients:
#                 patients[cid] = {
#                     'compound_id': cid,
#                     'exams': [],
#                     'reg_date': pd.to_datetime(cid[1]).date(),
#                     'is_self_selected': is_self_selected,
#                     'appt_date': appt_date,
#                     'scheduled': False,
#                 }
#             patients[cid]['exams'].append([
#                 str(row['检查部位']).strip(),
#                 exam_type,
#                 duration,
#                 pd.to_datetime(row['登记日期']).date(),
#             ])
#         print(f"成功导入{len(patients)}患者，共{sum(len(p['exams']) for p in patients.values())}个检查")
#         return patients
#     except Exception as e:
#         print(f"数据导入错误: {e}")
#         traceback.print_exc()
#         raise


# def import_device_constraints(file_path):
#     try:
#         df = safe_read_excel(file_path)
#         machine_exam_map = defaultdict(list)
#         for _, row in df.iterrows():
#             mid = int(row['设备']) - 1
#             exam = clean_exam_name(row['检查项目'])
#             machine_exam_map[mid].append(exam)
#         return machine_exam_map
#     except Exception as e:
#         print(f"导入设备限制数据错误: {e}")
#         traceback.print_exc()
#         raise

# # ===================== 导出所需（CPU 精排仅用于导出） =====================
# class MachineSchedule:
#     def __init__(self, machine_id, allowed_exams):
#         self.machine_id = machine_id
#         self.allowed_exams = allowed_exams
#         self.timeline = defaultdict(list)
#         self.day_end_time = defaultdict(lambda: None)
#         self._work_end_cache = {}
#         # [新增] 记录该机器上一次进行的检查类型，用于判断是否需要换模
#         self.last_exam_type = defaultdict(lambda: None) 

#     def get_work_end(self, date):
#         if date not in self._work_end_cache:
#             weekday = date.isoweekday()
#             base = datetime.combine(date, WORK_START)
#             work_duration = 15.0 - WEEKDAY_END_HOURS[weekday]
#             self._work_end_cache[date] = base + timedelta(hours=work_duration)
#         return self._work_end_cache[date]

#     def add_exam(self, date, start_time, duration_minutes, exam_type, patient_info):
#         duration = timedelta(minutes=float(duration_minutes))
#         end_time = start_time + duration
#         self.timeline[date].append((
#             start_time, end_time, exam_type,
#             patient_info['compound_id'][0],
#             patient_info['reg_date'],
#             patient_info['is_self_selected']
#         ))
#         self.day_end_time[date] = end_time
#         # [新增] 更新该日期该机器的最后检查类型
#         self.last_exam_type[date] = exam_type
#         return end_time

# class SchedulingSystem:
#     def __init__(self, machine_exam_map, start_date=None):
#         self.machines = [MachineSchedule(mid, machine_exam_map.get(mid, [])) for mid in range(MACHINE_COUNT)]
#         self.current_date = start_date if start_date else START_DATE.date()
#         self.start_date = self.current_date
#         self.current_machine = 0

#     def reset(self):
#         self.current_date = self.start_date
#         self.current_machine = 0

#     def move_to_next(self):
#         self.current_machine += 1
#         if self.current_machine >= MACHINE_COUNT:
#             self.current_machine = 0
#             self.current_date += timedelta(days=1)

#     def find_available_slot(self, duration_minutes, exam_type, patient_info):
#         duration = timedelta(minutes=float(duration_minutes))
#         # 换模时间成本：1分钟
#         SWITCH_GAP = timedelta(minutes=1) 
        
#         for _ in range(365):
#             m = self.machines[self.current_machine]
#             last_end = m.day_end_time[self.current_date]
#             last_type = m.last_exam_type[self.current_date]
            
#             # 计算开始时间
#             if last_end is None:
#                 # 当天第一个病人，无换模时间
#                 start = datetime.combine(self.current_date, WORK_START)
#             else:
#                 # 后续病人
#                 base_start = last_end
#                 # 如果类型不同，增加1分钟间隙
#                 if last_type is not None and last_type != exam_type:
#                     start = base_start + SWITCH_GAP
#                 else:
#                     start = base_start
            
#             end = start + duration
            
#             # 检查是否超出下班时间
#             if end <= m.get_work_end(self.current_date):
#                 return m, start
                
#             self.move_to_next()
#         raise TimeoutError("无法在365天内找到可用时段")

#     def generate_exam_dates(self, individual, patients):
#         # 保持原逻辑不变，因为它调用的是 find_available_slot
#         self.reset()
#         exam_dates = {}
#         for cid in individual:
#             p = patients.get(cid)
#             if p and not p['scheduled']:
#                 exam_type = clean_exam_name(p['exams'][0][1])
#                 duration = p['exams'][0][2]
#                 try:
#                     m, start = self.find_available_slot(duration, exam_type, p)
#                     exam_dates[cid] = start.date()
#                     m.add_exam(start.date(), start, duration, exam_type, p)
#                 except Exception:
#                     exam_dates[cid] = p['reg_date']
#         return exam_dates
        
# # ===================== GPU 适配度引擎 =====================

# def _weekday_minutes_matrix_from_end_hours(M: int) -> torch.Tensor:
#     hours = [int(round((15.0 - WEEKDAY_END_HOURS[d]) * 60)) for d in range(1, 8)]
#     return torch.tensor([[m] * M for m in hours], dtype=DTYPE_LONG, device=DEVICE)


# def _build_capacity_bins(weekday_machine_minutes: torch.Tensor, start_weekday: int, total_minutes_needed: int):
#     weekday_machine_minutes = weekday_machine_minutes.to(DEVICE)
#     M = weekday_machine_minutes.size(1)
#     daily_totals = weekday_machine_minutes.sum(dim=1)
#     min_daily = torch.clamp(daily_totals.min(), min=1)
#     est_days = int((total_minutes_needed // int(min_daily.item())) + 3)
#     days_idx = (torch.arange(est_days, device=DEVICE) + start_weekday) % 7
#     caps_per_day = weekday_machine_minutes.index_select(0, days_idx)  # [D,M]
#     caps_flat = caps_per_day.reshape(-1)
#     caps_cumsum = torch.cumsum(caps_flat, dim=0)
#     while caps_cumsum[-1].item() < total_minutes_needed:
#         caps_cumsum = torch.cat([caps_cumsum, caps_cumsum[-1] + torch.cumsum(caps_flat, dim=0)])
#         caps_per_day = torch.cat([caps_per_day, caps_per_day], dim=0)
#         caps_flat = caps_per_day.reshape(-1)
#     Bins = caps_cumsum.size(0)
#     idx = torch.arange(Bins, device=DEVICE)
#     bin_day = idx // M
#     bin_machine = idx % M
#     return caps_cumsum, bin_day, bin_machine


# def _assign_bins_batch_by_prefix(durations_batch: torch.Tensor, caps_cumsum: torch.Tensor) -> torch.Tensor:
#     T = torch.cumsum(durations_batch, dim=1)
#     return torch.searchsorted(caps_cumsum, T, right=False)

# def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
#     B, N = bin_idx_batch.shape
#     arng = torch.arange(N, device=DEVICE)
#     arng_expanded = arng.expand(B, N)
    
#     key = bin_idx_batch.long() * (N + 1) + arng_expanded
#     _, sort_idx = torch.sort(key, dim=1)
#     bin_sorted = bin_idx_batch.gather(1, sort_idx)
    
#     is_start = torch.zeros_like(bin_sorted, dtype=torch.bool)
#     is_start[:, 1:] = bin_sorted[:, 1:] != bin_sorted[:, :-1]
#     is_start[:, 0] = True
    
#     start_pos = torch.where(is_start, arng_expanded, -1)
#     last_start_pos = torch.cummax(start_pos, dim=1)[0]
#     rank_in_sorted = arng_expanded - last_start_pos
    
#     order_idx = torch.empty_like(rank_in_sorted, dtype=DTYPE_LONG)
#     order_idx.scatter_(1, sort_idx, rank_in_sorted)
    
#     return order_idx


# class _GPUMatrixFitnessBatch:
#     def __init__(self, *,
#                  weekday_machine_minutes: torch.Tensor,
#                  start_weekday: int,
#                  patient_durations: torch.Tensor,
#                  reg_day_offsets: torch.Tensor,
#                  is_self_selected: torch.Tensor,
#                  has_contrast: torch.Tensor,
#                  has_heart: torch.Tensor,
#                  has_angio: torch.Tensor,
#                  patient_main_type_id: torch.Tensor, # <<< 新增参数：接收患者主要检查类型ID
#                  patient_exam_mask: torch.Tensor | None,
#                  machine_exam_mask: torch.Tensor | None,
#                  patient_main_exam_id: torch.Tensor | None = None, # 仅保留用于可能的兼容性，但逻辑上被 patient_main_type_id 替代
#                  exam_count: int | None = None):
        
#         self.weekday_machine_minutes = weekday_machine_minutes.to(DEVICE).long()
#         self.start_weekday = int(start_weekday)
#         self.patient_durations = patient_durations.to(DEVICE).long()
#         self.reg_day_offsets = reg_day_offsets.to(DEVICE).long()
#         self.is_self_selected = is_self_selected.to(DEVICE).bool()
#         self.has_contrast = has_contrast.to(DEVICE).bool()
#         self.has_heart = has_heart.to(DEVICE).bool()
#         self.has_angio = has_angio.to(DEVICE).bool()
        
#         # <<< 修改：存储类型ID，用于计算换模 >>>
#         self.patient_main_type_id = patient_main_type_id.to(DEVICE).long()
        
#         self.patient_exam_mask = patient_exam_mask.to(DEVICE).bool() if patient_exam_mask is not None else None
#         self.machine_exam_mask = machine_exam_mask.to(DEVICE).bool() if machine_exam_mask is not None else None
#         self.patient_main_exam_id = patient_main_exam_id.to(DEVICE).long() if patient_main_exam_id is not None else None
#         self.exam_count = int(exam_count) if exam_count is not None else None

#         N = self.patient_durations.size(0)
#         base_minutes = int(self.patient_durations.sum().item())
#         # 预留 N 分钟 buffer，确保 effective_durations 就算每人都换模也不会越界
#         total_minutes_needed = base_minutes + N
#         self.caps_cumsum, self.bin_day, self.bin_machine = _build_capacity_bins(
#             self.weekday_machine_minutes, self.start_weekday, total_minutes_needed
#         )

#     def _penalty_waiting(self, assigned_day_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         reg = self.reg_day_offsets.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         delta = (assigned_day_batch - reg).to(torch.int64)
#         pos_wait = torch.clamp(delta, min=0).to(DTYPE_FLOAT)
#         neg_wait = torch.clamp(-delta, min=0).to(DTYPE_FLOAT)
#         is_self = self.is_self_selected.index_select(0, perms.reshape(-1)).reshape(perms.shape).to(DTYPE_FLOAT)
#         non_self = 1.0 - is_self
#         return pos_wait * (is_self * SELF_SELECTED_PENALTY + non_self * NON_SELF_PENALTY) + neg_wait * LOGICAL

#     def _device_violate(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         if (self.patient_exam_mask is None) or (self.machine_exam_mask is None):
#             return torch.zeros_like(assigned_machine_batch, dtype=torch.bool)
#         mach_mask = self.machine_exam_mask[assigned_machine_batch]  # [B,N,E]
#         pat_mask = self.patient_exam_mask[perms]                     # [B,N,E]
#         invalid = pat_mask & (~mach_mask)
#         return invalid.any(dim=2)

#     def _penalty_device_cover(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         violate = self._device_violate(assigned_machine_batch, perms)
#         return violate.to(DTYPE_FLOAT) * DEVICE_PENALTY

#     def _special_violates(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
#         heart_mask = self.has_heart.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         ok_wd_h = (weekday_batch == 1) | (weekday_batch == 3)  # Tue/Thu
#         ok_mc_h = (assigned_machine_batch == 3)                # 3号机
#         heart_violate = heart_mask & (~(ok_wd_h & ok_mc_h))

#         angio_mask = self.has_angio.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         ok_wd_a = (weekday_batch == 0) | (weekday_batch == 2) | (weekday_batch == 4)  # Mon/Wed/Fri
#         ok_mc_a = (assigned_machine_batch == 1)                                      # 1号机
#         angio_violate = angio_mask & (~(ok_wd_a & ok_mc_a))

#         weekend = (weekday_batch == 5) | (weekday_batch == 6)
#         contrast_mask = self.has_contrast.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         weekend_violate = contrast_mask & weekend
#         return heart_violate, angio_violate, weekend_violate

#     def _penalty_special_rules(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
#         heart_v, angio_v, weekend_v = self._special_violates(weekday_batch, assigned_machine_batch, perms)
#         p = (heart_v | angio_v | weekend_v).to(DTYPE_FLOAT) * DEVICE_PENALTY
#         return p, heart_v.to(torch.int32), angio_v.to(torch.int32), weekend_v.to(torch.int32)

#     # <<< 新增：计算机器换模损耗 (Inter-patient switching) >>>
#     def _penalty_machine_switching(self, bin_idx_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         """
#         计算机器上的换模惩罚。
#         逻辑：如果 bin[i] == bin[i-1] (同一机器连续排程) 且 type[i] != type[i-1]，则罚分。
#         """
#         B, N = perms.shape
#         if TRANSITION_PENALTY <= 0:
#             return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)

#         # 1. 获取当前排列下，每个位置的患者检查类型 ID [B, N]
#         current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
        
#         # 2. 右移一位，获取"前一个"位置的信息
#         prev_types = torch.roll(current_types, shifts=1, dims=1)
#         prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
        
#         # 3. 比较
#         same_bin = (bin_idx_batch == prev_bins)     # 是否在同一台机器的连续时段
#         diff_type = (current_types != prev_types)   # 检查类型是否改变
        
#         # 4. 计算有效切换 (排除每行的第0个元素，因为roll会将最后一个元素移到第0个)
#         is_transition = same_bin & diff_type
#         is_transition[:, 0] = False 
        
#         return is_transition.to(DTYPE_FLOAT) * TRANSITION_PENALTY

#     def fitness_batch(self, perms: torch.Tensor, return_assignment: bool = False):
#         perms = perms.to(DEVICE)
#         B, N = perms.shape
        
#         # 1. 获取基础耗时
#         base_durations = self.patient_durations.unsqueeze(0).expand(B, N)
#         durations_batch = torch.gather(base_durations, 1, perms) # [B, N]

#         # 2. [新增] 计算动态换模间隙 (Dynamic Switch Gap)
#         # 获取排列后的检查类型ID
#         current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
#         # 比较当前位置与前一个位置
#         prev_types = torch.roll(current_types, shifts=1, dims=1)
#         # 是否类型不同
#         diff_type = (current_types != prev_types)
#         # 第一位不算切换（因为没有前一位）
#         diff_type[:, 0] = False
        
#         gap_minutes = 1
#         effective_durations = durations_batch + (diff_type.long() * gap_minutes)

#         # 3. 使用"有效耗时"（含间隙）来进行分箱（分配机器/日期）
#         # 注意这里传入的是 effective_durations
#         bin_idx_batch = _assign_bins_batch_by_prefix(effective_durations, self.caps_cumsum)
        
#         assigned_day_batch = self.bin_day.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         assigned_machine_batch = self.bin_machine.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         weekday_batch = (self.start_weekday + assigned_day_batch) % 7

#         # 4. 计算各项惩罚
#         p_wait  = self._penalty_waiting(assigned_day_batch, perms)
#         p_dev   = self._penalty_device_cover(assigned_machine_batch, perms)
#         p_spec, heart_v_i, angio_v_i, weekend_v_i = self._penalty_special_rules(weekday_batch, assigned_machine_batch, perms)
        
#         prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
#         same_bin = (bin_idx_batch == prev_bins)
#         same_bin[:, 0] = False
        
#         real_transitions = same_bin & diff_type
#         p_tran = real_transitions.to(DTYPE_FLOAT) * TRANSITION_PENALTY

#         total_penalty = p_wait + p_dev + p_spec + p_tran
#         fitness = - total_penalty.sum(dim=1)
        
#         out = {
#             'fitness': fitness,
#             'assigned_day': assigned_day_batch if return_assignment else None,
#             'assigned_machine': assigned_machine_batch if return_assignment else None,
#             'order_in_machine': _compute_order_in_bin_batch(bin_idx_batch) if return_assignment else None,
#             'heart_cnt': heart_v_i.sum(dim=1),
#             'angio_cnt': angio_v_i.sum(dim=1),
#             'weekend_cnt': weekend_v_i.sum(dim=1),
#             'device_cnt': (p_dev > 0).sum(dim=1),
#             'any_violate_mask': (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)).any(dim=1) 
#         }
        
#         viol_mask_b_n = (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0))
#         out['any_violate_mask_b_n'] = viol_mask_b_n 
        
#         return out


# # ===================== GA 主体（Megabatch 版） =====================
# class MultiRunOptimizer:
#     def __init__(self, patients, machine_exam_map, num_parallel_runs, pop_size_per_run, block_start_date=None):
#         self.patients = patients
#         self.machine_exam_map = machine_exam_map
#         self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
#         self.current_generation = 0
        
#         self.K = num_parallel_runs
#         self.B = pop_size_per_run
#         self.total_pop_size = self.K * self.B
#         self.N = len(self.sorted_patients)
        
#         self.block_start_date = block_start_date
        
#         self.population_tensor: torch.Tensor | None = None # 维度: [K, B, N]
#         self.fitness_history: List[List[float]] = [[] for _ in range(self.K)]
        
#         self._gpu_engine = None
#         self._cid_to_idx = None
#         self._idx_to_cid = None
#         self._patient_main_exam_id = None
#         self._E = None

#     # ------- GPU 引擎准备 -------
#     def _ensure_gpu_engine(self):
#         # (此函数 ... _ensure_gpu_engine ... 与原版 100% 相同)
#         if self._gpu_engine is not None:
#             return
#         idx_to_cid = list(self.sorted_patients)
#         cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
#         self._idx_to_cid = idx_to_cid
#         self._cid_to_idx = cid_to_idx
#         N = len(idx_to_cid)

#         patient_durations = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
#         reg_day_offsets = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
#         is_self_selected = torch.zeros(N, dtype=torch.bool, device=DEVICE)
#         has_contrast = torch.zeros(N, dtype=torch.bool, device=DEVICE)
#         has_heart = torch.zeros(N, dtype=torch.bool, device=DEVICE)
#         has_angio = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        
#         # [新增] 初始化主要类型ID张量 (用于换模惩罚)
#         patient_main_type_id = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)

#         exam_set = set()
#         for cid in idx_to_cid:
#             for _, et, _, _ in self.patients[cid]['exams']:
#                 exam_set.add(clean_exam_name(et))
#         for mid, exams in self.machine_exam_map.items():
#             for e in exams:
#                 exam_set.add(clean_exam_name(e))
#         exam_list = sorted(list(exam_set))
#         exam_to_eidx = {e: i for i, e in enumerate(exam_list)}
#         E = len(exam_list)
#         self._E = E

#         patient_exam_mask = torch.zeros((N, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
#         machine_exam_mask = torch.zeros((MACHINE_COUNT, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
#         patient_main_exam_id = torch.full((N,), -1, dtype=DTYPE_LONG, device=DEVICE)

#         base_date = self.block_start_date if self.block_start_date else START_DATE.date()
#         start_weekday = base_date.isoweekday() - 1

#         for i, cid in enumerate(idx_to_cid):
#             p = self.patients[cid]
#             total_minutes = 0
#             any_contrast = False
#             any_heart = False
#             any_angio = False
#             exam_types_seq = []
            
#             # 临时列表，用于确定患者的主要检查类型
#             p_exam_types = []
            
#             counter: Dict[int, int] = defaultdict(int)
#             for _, et, dur, _ in p['exams']:
#                 etn = clean_exam_name(et)
#                 total_minutes += int(round(float(dur)))
#                 exam_types_seq.append(etn)
#                 p_exam_types.append(etn) # 收集类型
#                 if E > 0:
#                     eidx = exam_to_eidx.get(etn, None)
#                     if eidx is not None:
#                         patient_exam_mask[i, eidx] = True
#                         counter[eidx] += 1
#                 any_contrast = any_contrast or ('增强' in etn)
#                 any_heart = any_heart or ('心脏' in etn)
#                 any_angio = any_angio or ('造影' in etn)
            
#             # 兼容性逻辑：用于 mutation 的 ID (出现次数最多的)
#             if len(counter) > 0:
#                 best_cnt = max(counter.values())
#                 main_eidx = min([k for k, v in counter.items() if v == best_cnt])
#                 patient_main_exam_id[i] = main_eidx
            
#             # [新增逻辑] 确定患者的"主要类型ID"用于换模惩罚
#             # 取第一个检查项目作为判断换模的依据
#             if p_exam_types:
#                 main_type = p_exam_types[0]
#                 patient_main_type_id[i] = exam_to_eidx.get(main_type, 0)
#             else:
#                 patient_main_type_id[i] = 0

#             # 移除旧的 switch_penalty 计算 (switch_penalty[i] = ...) 

#             patient_durations[i] = max(1, total_minutes)
#             reg_day_offsets[i] = (p['reg_date'] - base_date).days
#             is_self_selected[i] = bool(p.get('is_self_selected', False))
#             has_contrast[i] = any_contrast
#             has_heart[i] = any_heart
#             has_angio[i] = any_angio

#         if machine_exam_mask is not None:
#             for mid in range(MACHINE_COUNT):
#                 for e in self.machine_exam_map.get(mid, []):
#                     et = clean_exam_name(e)
#                     eidx = exam_to_eidx.get(et, None)
#                     if eidx is not None:
#                         machine_exam_mask[mid, eidx] = True

#         weekday_machine_minutes = _weekday_minutes_matrix_from_end_hours(MACHINE_COUNT)
        
#         # 必须将局部变量 patient_main_type_id 保存到 self 属性中
#         # 否则后续的变异算子 (_mutate_batch 等) 调用 self._patient_main_exam_id 时会报错
#         # 注意：这里我们使用 patient_main_exam_id (基于计数的) 赋值给 self._patient_main_exam_id，
#         # 保持 mutation 逻辑不变，同时传递 patient_main_type_id 给引擎做罚分。
#         self._patient_main_exam_id = patient_main_exam_id
        
#         self._gpu_engine = _GPUMatrixFitnessBatch(
#             weekday_machine_minutes=weekday_machine_minutes,
#             start_weekday=start_weekday,
#             patient_durations=patient_durations,
#             reg_day_offsets=reg_day_offsets,
#             is_self_selected=is_self_selected,
#             has_contrast=has_contrast,
#             has_heart=has_heart,
#             has_angio=has_angio,
#             patient_main_type_id=patient_main_type_id, # [新增]
#             patient_exam_mask=patient_exam_mask,
#             machine_exam_mask=machine_exam_mask,
#             patient_main_exam_id=patient_main_exam_id, # 保留用于兼容
#             exam_count=E,
#         )

#     # ------- 索引 ↔ cid (仅用于最后转换) -------
#     def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
#         # 确保 _idx_to_cid 存在
#         if self._idx_to_cid is None:
#             self._idx_to_cid = list(self.sorted_patients)
#         return [self._idx_to_cid[int(x)] for x in row.tolist()]

#     def initialize_population(self):
#         if self.N == 0:
#             print("警告：患者列表为空，无法初始化种群。")
#             return

#         # 确保 cid 映射存在
#         if self._idx_to_cid is None:
#             self._idx_to_cid = list(self.sorted_patients)
#             self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}

#         indices = torch.arange(self.N, device=DEVICE)
#         block_size = max(30, self.N // 20)
        
#         # 核心变更：创建 [K, B, N] 张量
#         pop_indices = torch.empty((self.K, self.B, self.N), dtype=DTYPE_LONG, device=DEVICE)
        
#         # 预先生成所有随机数
#         rand_matrices = torch.rand(self.K, self.B, self.N, device=DEVICE)
        
#         # CPU 循环遍历 *块*
#         for i in range(0, self.N, block_size):
#             end = min(i + block_size, self.N)
#             block_len = end - i
#             if block_len == 0:
#                 continue
                
#             # 核心变更：在 dim=2 (N 维度) 上排序
#             block_rand = rand_matrices[:, :, i:end] # [K, B, block_len]
#             block_perm_idx = torch.argsort(block_rand, dim=2) # [K, B, block_len]
            
#             block_indices = indices[i:end]
#             block_indices_expanded = block_indices.expand(self.K, self.B, -1) # [K, B, block_len]
            
#             # 核心变更：在 dim=2 上 gather
#             pop_indices[:, :, i:end] = torch.gather(block_indices_expanded, 2, block_perm_idx)

#         # 种群现在是一个保存在 GPU 上的张量
#         self.population_tensor = pop_indices
#         print(f"已生成 {self.K} 个并行种群 (每个 {self.B} 个个体)，总计 {self.total_pop_size} 个个体")


#     # ------- GA 主循环 (Megabatch 版) -------
#     def evolve_gpu(self, generations=100, elite_size=5):
#         self._ensure_gpu_engine()
#         if self.population_tensor is None:
#             raise RuntimeError("种群为空，请先 initialize_population")
        
#         # pop 是我们的 [K, B, N] GPU 张量
#         pop = self.population_tensor
#         N = self.N # 基因数
        
#         for gen_idx in range(generations):
#             # 1. 评估 (Megabatch)
#             # 将 [K, B, N] 展平为 [K*B, N]
#             pop_flat = pop.view(self.total_pop_size, N)
            
#             # 使用现有引擎，它已向量化
#             out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            
#             # 将 [K*B] 结果重塑回 [K, B]
#             fitness_flat = out['fitness']
#             fitness = fitness_flat.view(self.K, self.B)
            
#             # 将 [K*B, N] 违规掩码重塑回 [K, B, N]
#             viol_mask_flat = out['any_violate_mask_b_n'] 
#             violate_mask = viol_mask_flat.view(self.K, self.B, N)
            
#             # 2. 精英选择 (按 K 独立进行)
#             # 核心变更：沿 dim=1 (B 维度) 排序
#             topk_vals, topk_idx = torch.topk(fitness, k=self.B, largest=True, dim=1)
            
#             # 记录 K 个种群各自的最佳适应度
#             best_fitness_per_run = topk_vals[:, 0].cpu().tolist()
#             for k in range(self.K):
#                 self.fitness_history[k].append(best_fitness_per_run[k])
            
#             elite_size = min(elite_size, self.B)
#             elite_idx = topk_idx[:, :elite_size] # [K, elite_size]
            
#             # 核心变更：使用 gather 从 [K, B, N] 中挑选精英
#             # 索引需要扩展到 [K, elite_size, N]
#             idx_expanded = elite_idx.unsqueeze(2).expand(self.K, elite_size, N)
#             elites = torch.gather(pop, 1, idx_expanded) # [K, elite_size, N]

#             # 3. 父代选择 (按 K 独立进行)
#             parent_count = max(1, int(0.2 * self.B))
#             parent_idx = topk_idx[:, :parent_count] # [K, parent_count]
            
#             # 从 pop 中 gather 父代
#             idx_expanded = parent_idx.unsqueeze(2).expand(self.K, parent_count, N)
#             parents = torch.gather(pop, 1, idx_expanded) # [K, parent_count, N]
            
#             # 从 violate_mask 中 gather 对应的违规掩码
#             parent_viol = torch.gather(violate_mask, 1, idx_expanded) # [K, parent_count, N]

#             # 4. 交叉 (Megabatch)
#             num_children = self.B - elite_size
#             if num_children > 0:
#                 # 核心变更：为 K 个种群各自生成配对索引
#                 p_idx1 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE) # [K, num_children]
#                 p_idx2 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE) # [K, num_children]
                
#                 # 从 parents [K, parent_count, N] 中 gather
#                 P1 = torch.gather(parents, 1, p_idx1.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
#                 P2 = torch.gather(parents, 1, p_idx2.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
                
#                 # Gather 对应的违规掩码
#                 Vmask_choice = torch.gather(parent_viol, 1, p_idx1.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
                
#                 # 展平为 [K*num_children, N] 以
#                 # 喂给无需修改的交叉/变异函数
#                 P1_flat = P1.view(self.K * num_children, N)
#                 P2_flat = P2.view(self.K * num_children, N)
                
#                 children_flat = self._ordered_crossover_batch_gpu(P1_flat, P2_flat)
                
#                 # 5. 变异 (Megabatch)
#                 Vmask_flat = Vmask_choice.view(self.K * num_children, N)
                
#                 children_flat = self._mutate_batch_gpu(children_flat, Vmask_flat, self.current_generation)
                
#                 # 重塑回 [K, num_children, N]
#                 children = children_flat.view(self.K, num_children, N)
                
#                 # 6. 形成新种群
#                 pop = torch.cat([elites, children], dim=1) # 沿 B 维度拼接
#             else:
#                 pop = elites.clone()
            
#             if (gen_idx + 1) % 50 == 0:
#                 # 报告 K 个种群的平均最佳适应度
#                 avg_best_fit = sum(best_fitness_per_run) / self.K
#                 print(f"Generation {self.current_generation+1} | Avg Best Fitness (K={self.K}): {avg_best_fit:.2f}")

#             self.current_generation += 1

#         # 7. 进化结束，返回 K 个种群的最终最佳个体
#         print("进化完成。正在提取 K 个最佳个体...")
        
#         # 最终评估一次
#         pop_flat = pop.view(self.total_pop_size, N)
#         final_out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
#         final_fitness = final_out['fitness'].view(self.K, self.B) # [K, B]
        
#         # 找到 K 个种群中各自的最佳 (k=1)
#         final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1) # [K, 1], [K, 1]
        
#         final_best_vals = final_best_vals.flatten() # [K]
        
#         # Gather 最佳个体
#         # 修复：final_best_idx_in_B 是 [K, 1] (即 [8, 1])。
#         # 我们需要 .unsqueeze(2) 将其变为 [K, 1, 1] (即 [8, 1, 1])，
#         # 然后才能 .expand() 到 [K, 1, N] (即 [8, 1, 2379])。
#         idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(self.K, 1, N) # [K, 1, N]
#         best_individuals_tensor = torch.gather(pop, 1, idx_expanded).squeeze(1) # [K, N]
        
#         # 转换为 CPU 列表
#         best_individuals_cpu = best_individuals_tensor.cpu()
#         best_fitnesses_cpu = final_best_vals.cpu().tolist()
        
#         # 转换为 CIDs
#         results = []
#         for k in range(self.K):
#             cids = self._tensor_row_to_cids(best_individuals_cpu[k])
#             results.append({
#                 "run_id": k,
#                 "individual_cids": cids,
#                 "fitness": best_fitnesses_cpu[k]
#             })
            
#         self.population_tensor = pop # 保存最终状态
#         return results

#     # ------- 交叉和变异函数 (无需修改) -------
#     # ( ... _random_cuts, _ordered_crossover_batch_gpu ... )
#     # ( ... _mutate_step1_violations, _mutate_step2_base_swap ... )
#     # ( ... _mutate_step3_greedy_cluster, _mutate_batch_gpu ... )
#     # ( ... _greedy_cluster_mutation_gpu_inplace ... )
#     # ( ... 保持不变 ... )
    
#     @staticmethod
#     def _random_cuts(num_rows: int, N: int):
#         a = torch.randint(0, N, (num_rows,), device=DEVICE)
#         b = torch.randint(0, N, (num_rows,), device=DEVICE)
#         start = torch.minimum(a, b)
#         end = torch.maximum(a, b)
#         return start, end

#     def _ordered_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
#         C, N = P1.shape
#         arangeN = torch.arange(N, device=DEVICE).expand(C, N)
        
#         start, end = self._random_cuts(C, N)
#         s_exp = start.unsqueeze(1)
#         e_exp = end.unsqueeze(1)
        
#         mask_frag = (arangeN >= s_exp) & (arangeN <= e_exp)
        
#         children = torch.full_like(P1, -1)
#         children[mask_frag] = P1[mask_frag]
        
#         P2_expanded = P2.unsqueeze(2)  # (C, N, 1)
#         P1_expanded = P1.unsqueeze(1)  # (C, 1, N)
        
#         equality_matrix = (P2_expanded == P1_expanded)
#         mask_frag_expanded = mask_frag.unsqueeze(1)
        
#         isin_matrix = (equality_matrix & mask_frag_expanded).any(dim=2)
#         mask_tail = ~isin_matrix
#         P2_tails_flat = P2[mask_tail]
#         mask_fill = (children == -1)
#         children[mask_fill] = P2_tails_flat
        
#         return children
    
#     def _mutate_step1_violations(self, X: torch.Tensor, parent_violate_mask: torch.Tensor) -> torch.Tensor:
#         C, N = X.shape
#         # 修复：parent_violate_mask 是 [C, N]
#         any_viol_per_row = torch.any(parent_violate_mask, dim=1) # [C]
#         viol_rows_idx = torch.nonzero(any_viol_per_row, as_tuple=False).flatten()
#         R = viol_rows_idx.numel()
#         if R == 0:
#             return X

#         viol_mask_subset = parent_violate_mask[viol_rows_idx] # [R, N]
        
#         viol_idx_in_row = torch.multinomial(viol_mask_subset.float(), 1, replacement=True).flatten() # [R]
        
#         low = torch.clamp(viol_idx_in_row - 400, min=0)
#         high = torch.clamp(viol_idx_in_row + 400, max=N-1)
#         range_size = high - low + 1
        
#         range_size = torch.where(range_size <= 0, 1, range_size)
        
#         rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
#         cand_idx_in_row = low + rand_offset
        
#         cand_idx_in_row = torch.where(
#             (cand_idx_in_row == viol_idx_in_row) & (range_size > 1), 
#             torch.where(viol_idx_in_row == low, low + 1, low),
#             cand_idx_in_row
#         )
#         cand_idx_in_row = torch.clamp(cand_idx_in_row, 0, N-1)

#         val1 = X[viol_rows_idx, viol_idx_in_row]
#         val2 = X[viol_rows_idx, cand_idx_in_row]
#         X[viol_rows_idx, viol_idx_in_row] = val2
#         X[viol_rows_idx, cand_idx_in_row] = val1
        
#         return X
        
#     def _mutate_step2_base_swap(self, X: torch.Tensor, current_gen: int, base_swap_prob: float = 0.95) -> torch.Tensor:
#         C, N = X.shape
#         use_range_limit = (current_gen <= 10000)
        
#         probs = torch.rand(C, device=DEVICE)
#         rows_to_swap_mask = (probs < base_swap_prob)
#         rows_to_swap_idx = torch.nonzero(rows_to_swap_mask, as_tuple=False).flatten()
#         R = rows_to_swap_idx.numel()
#         if R == 0:
#             return X

#         idx1 = torch.randint(0, N, (R,), device=DEVICE)
        
#         if use_range_limit:
#             low = torch.clamp(idx1 - 400, min=0)
#             high = torch.clamp(idx1 + 400, max=N-1)
#             range_size = high - low + 1
#             range_size = torch.where(range_size <= 0, 1, range_size)
#             rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
#             idx2 = low + rand_offset
#             idx2 = torch.where(
#                 (idx2 == idx1) & (range_size > 1), 
#                 torch.where(idx1 == low, low + 1, low),
#                 idx2
#             )
#             idx2 = torch.clamp(idx2, 0, N-1)
#         else:
#             idx2 = torch.randint(0, N, (R,), device=DEVICE)
#             idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)
        
#         val1 = X[rows_to_swap_idx, idx1]
#         val2 = X[rows_to_swap_idx, idx2]
#         X[rows_to_swap_idx, idx1] = val2
#         X[rows_to_swap_idx, idx2] = val1
        
#         return X

#     def _mutate_step3_greedy_cluster(self, X: torch.Tensor, greedy_prob: float = 0.5) -> torch.Tensor:
#         C, N = X.shape
#         if N < 50 or self._E is None or self._patient_main_exam_id is None:
#             return X
            
#         probs2 = torch.rand(C, device=DEVICE)
#         rows_to_greedy_mask = (probs2 < greedy_prob)
#         rows_to_greedy_idx = torch.nonzero(rows_to_greedy_mask, as_tuple=False).flatten()
#         R = rows_to_greedy_idx.numel()
#         if R == 0:
#             return X
        
#         window_len = torch.randint(20, 51, (R,), device=DEVICE)
#         start = torch.randint(0, N - 49, (R,), device=DEVICE)
#         end = torch.clamp(start + window_len - 1, max=N-1)
#         window_len = end - start + 1
#         max_len = window_len.max().item()

#         arng_max = torch.arange(max_len, device=DEVICE)
#         arng_R_max = arng_max.expand(R, max_len)
        
#         window_indices = start.unsqueeze(1) + arng_R_max
        
#         valid_mask = arng_R_max < window_len.unsqueeze(1)

#         rows = X[rows_to_greedy_idx]
        
#         windows = torch.gather(rows, 1, window_indices)

#         keys = self._patient_main_exam_id[windows]
#         keys[~valid_mask] = -1

#         E = self._E
#         keys_clamped = keys.clamp(min=0, max=E-1)
#         one_hot_keys = torch.nn.functional.one_hot(keys_clamped, num_classes=E)
#         one_hot_keys[~valid_mask] = 0
#         counts = one_hot_keys.sum(dim=1)

#         size_per_pos = torch.gather(counts, 1, keys_clamped)
#         size_per_pos[~valid_mask] = -1

#         sort_key = (-size_per_pos).to(torch.int64) * (max_len + 1) + arng_R_max
#         sort_key[~valid_mask] = 9223372036854775807
        
#         new_order = torch.argsort(sort_key, dim=1, stable=True)

#         sorted_windows = torch.gather(windows, 1, new_order)

#         new_rows = rows.clone()
#         new_rows.scatter_(dim=1, index=window_indices, src=sorted_windows)
        
#         scatter_mask = torch.zeros_like(rows, dtype=torch.bool, device=DEVICE)
#         scatter_mask.scatter_(dim=1, index=window_indices, src=valid_mask)

#         final_rows = torch.where(scatter_mask, new_rows, rows)

#         X[rows_to_greedy_idx] = final_rows
            
#         return X

#     def _mutate_batch_gpu(self, X: torch.Tensor, parent_violate_mask: torch.Tensor, current_gen: int,
#                           base_swap_prob: float = 0.95, greedy_prob: float = 0.5) -> torch.Tensor:
        
#         X = self._mutate_step1_violations(X, parent_violate_mask)
#         X = self._mutate_step2_base_swap(X, current_gen, base_swap_prob)
#         X = self._mutate_step3_greedy_cluster(X, greedy_prob)
        
#         return X

#     def _greedy_cluster_mutation_gpu_inplace(self, X: torch.Tensor, row: int, start: int, end: int):
#         window = X[row, start:end+1]
#         keys = self._patient_main_exam_id.index_select(0, window)
#         W = keys.numel()
#         if W <= 1 or (self._E is None):
#             return
#         counts = torch.bincount(torch.clamp(keys, min=0), minlength=max(1, self._E))
#         size_per_pos = counts[keys.clamp(min=0)]
#         arng = torch.arange(W, device=DEVICE)
#         sort_key = (-size_per_pos).to(torch.int64) * (W + 1) + arng
#         new_order = torch.argsort(sort_key, stable=True)
#         X[row, start:end+1] = window.index_select(0, new_order)


#     # ------- 导出（可选） -------
#     def generate_schedule(self, individual):
#         system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
#         for cid in individual:
#             p = self.patients.get(cid)
#             if p and not p['scheduled']:
#                 for exam in p['exams']:
#                     exam_type = clean_exam_name(exam[1])
#                     duration = exam[2]
#                     try:
#                         m, start_time = system.find_available_slot(duration, exam_type, p)
#                         m.add_exam(start_time.date(), start_time, duration, exam_type, p)
#                     except Exception:
#                         pass
#         return system


# # ===================== 导出 Excel =====================
# # (export_schedule ... 保持不变 ...)
# def export_schedule(system, patients, filename):
#     with pd.ExcelWriter(filename) as writer:
#         rows = []
#         for machine in system.machines:
#             for date in sorted(machine.timeline):
#                 slots = sorted(machine.timeline[date], key=lambda x: x[0])
#                 for (start, end, exam, pid, reg_date, is_self) in slots:
#                     rows.append({
#                         '机器编号': machine.machine_id,
#                         '日期': date.strftime('%Y-%m-%d'),
#                         '开始时间': start.strftime('%H:%M:%S'),
#                         '结束时间': end.strftime('%H:%M:%S'),
#                         '检查项目': exam,
#                         '患者ID': pid,
#                         '登记日期': reg_date.strftime('%Y-%m-%d'),
#                         '是否自选': '是' if is_self else '否',
#                     })
#         df = pd.DataFrame(rows)
#         if df.empty:
#             pd.DataFrame(columns=['机器编号','日期','开始时间','结束时间','检查项目','患者ID','登记日期','是否自选']).to_excel(writer, sheet_name='总排程', index=False)
#         else:
#             df.sort_values(by=['机器编号', '日期', '开始时间']).to_excel(writer, sheet_name='总排程', index=False)

# # ===================== main (Megabatch 版) =====================
# def main():
#     try:
#         # ================== 配置 ==================
        
#         # 你希望并行运行多少个独立的GA实验？
#         # 这会成为 K 维度
#         NUM_PARALLEL_RUNS = 8 
        
#         # 每个独立实验的种群大小
#         # 这会成为 B 维度
#         POP_SIZE_PER_RUN = 50 
        
#         # 进化代数
#         GENERATIONS_TO_RUN = 10000    
#         # ==========================================
        
#         print(f"启动 Megabatch 模式: K={NUM_PARALLEL_RUNS} (并行实验), B={POP_SIZE_PER_RUN} (个体/实验)")
#         print(f"总 GPU 批量: {NUM_PARALLEL_RUNS * POP_SIZE_PER_RUN} 个体")
        
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#         duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#         device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
#         for f in [patient_file, duration_file, device_constraint_file]:
#             if not os.path.exists(f):
#                 print(f"❌ 错误：找不到文件 {f}")
#                 return
#         print("✓ 所有数据文件均已找到。")

#         print("正在导入数据...")
#         patients = import_data(patient_file, duration_file)
#         machine_exam_map = import_device_constraints(device_constraint_file)

#         print("\n===== 启动并行遗传算法优化 (Megabatch GPU) =====")
#         optimizer = MultiRunOptimizer(
#             patients, 
#             machine_exam_map, 
#             num_parallel_runs=NUM_PARALLEL_RUNS, 
#             pop_size_per_run=POP_SIZE_PER_RUN
#         )
        
#         t0_init = time.perf_counter()
#         optimizer.initialize_population()
#         t_init = time.perf_counter() - t0_init
#         print(f"✓ 已生成 {NUM_PARALLEL_RUNS} 个初始种群，耗时: {t_init:.4f}s")


#         print(f"\n开始 {GENERATIONS_TO_RUN} 代进化 (K={NUM_PARALLEL_RUNS})...")
#         t0 = time.perf_counter()
        
#         # evolve_gpu 现在返回一个包含 K 个结果的列表
#         final_results = optimizer.evolve_gpu(generations=GENERATIONS_TO_RUN, elite_size=5)
        
#         total_evolution_time = time.perf_counter() - t0
#         print(f"\n✓ 进化完成 (K={NUM_PARALLEL_RUNS})，总耗时: {total_evolution_time:.2f}s")
#         print(f"  平均每代耗时: {total_evolution_time / GENERATIONS_TO_RUN:.4f} s/gen")
#         print(f"  (总计 {GENERATIONS_TO_RUN * NUM_PARALLEL_RUNS} 个 'run-generations')")


#         print(f"\n===== 正在导出 {NUM_PARALLEL_RUNS} 个最佳排程 =====")
#         out_dir = 'output_schedules'; os.makedirs(out_dir, exist_ok=True)
#         ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         all_fitnesses = []
        
#         for result in final_results:
#             run_id = result['run_id']
#             best_individual = result['individual_cids']
#             best_fitness = result['fitness']
#             all_fitnesses.append(best_fitness)
            
#             print(f"  Run {run_id}: Best Fitness: {best_fitness:.2f}")
            
#             # 导出 Excel
#             xlsx = os.path.join(out_dir, f'final_schedule_RUN{run_id}_{ts}_fit_{best_fitness:.0f}.xlsx')
#             final_system = optimizer.generate_schedule(best_individual)
#             export_schedule(final_system, patients, xlsx)
#             print(f"    ✓ 已导出至 {xlsx}")

#         print("\n===== 最终统计 (K={NUM_PARALLEL_RUNS}) =====")
#         mean_fitness = np.mean(all_fitnesses)
#         std_fitness = np.std(all_fitnesses)
#         min_fitness = np.min(all_fitnesses)
#         max_fitness = np.max(all_fitnesses)
        
#         print(f"  最佳适应度 (均值): {mean_fitness:.2f}")
#         print(f"  最佳适应度 (标准差): {std_fitness:.2f}")
#         print(f"  最佳适应度 (范围): {min_fitness:.2f} ... {max_fitness:.2f}")
#         print("\n所有运行均已完成。")

#     except Exception as e:
#         print(f"运行时错误: {e}")
#         traceback.print_exc()
#     finally:
#         pass


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()

#没有一分钟换模时间
from __future__ import annotations
from typing import List, Dict, Set, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict
import traceback
import re
import json
import multiprocessing
import time
import torch

# ===================== 全局常量 =====================
WEEKDAY_END_HOURS = {1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7}
WORK_START = datetime.strptime('07:00', '%H:%M').time()
TRANSITION_PENALTY = 20000
LOGICAL = 10000
SELF_SELECTED_PENALTY = 8000
NON_SELF_PENALTY = 800
START_DATE = datetime(2024, 12, 1, 7, 0)
MACHINE_COUNT = 6
DEVICE_PENALTY = 500000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_LONG = torch.long
DTYPE_FLOAT = torch.float32

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
    try:
        duration_df = safe_read_excel(duration_file)
        duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
        exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()

        patient_df = safe_read_excel(patient_file)
        patients = {}
        for _, row in patient_df.iterrows():
            if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
                continue
            cid = (str(row['id']).strip(), pd.to_datetime(row['登记日期']).strftime('%Y%m%d'))
            exam_type = clean_exam_name(row['检查项目'])
            duration = float(exam_durations.get(exam_type, 15.0))
            is_self_selected = (row['是否自选时间'] == '自选时间')
            appt_date = pd.to_datetime(row['预约日期']).date() if not pd.isnull(row['预约日期']) else None
            if cid not in patients:
                patients[cid] = {
                    'compound_id': cid,
                    'exams': [],
                    'reg_date': pd.to_datetime(cid[1]).date(),
                    'is_self_selected': is_self_selected,
                    'appt_date': appt_date,
                    'scheduled': False,
                }
            patients[cid]['exams'].append([
                str(row['检查部位']).strip(),
                exam_type,
                duration,
                pd.to_datetime(row['登记日期']).date(),
            ])
        print(f"成功导入{len(patients)}患者，共{sum(len(p['exams']) for p in patients.values())}个检查")
        return patients
    except Exception as e:
        print(f"数据导入错误: {e}")
        traceback.print_exc()
        raise


def import_device_constraints(file_path):
    try:
        df = safe_read_excel(file_path)
        machine_exam_map = defaultdict(list)
        for _, row in df.iterrows():
            mid = int(row['设备']) - 1
            exam = clean_exam_name(row['检查项目'])
            machine_exam_map[mid].append(exam)
        return machine_exam_map
    except Exception as e:
        print(f"导入设备限制数据错误: {e}")
        traceback.print_exc()
        raise

# ===================== 导出所需（CPU 精排仅用于导出） =====================
class MachineSchedule:
    def __init__(self, machine_id, allowed_exams):
        self.machine_id = machine_id
        self.allowed_exams = allowed_exams
        self.timeline = defaultdict(list)
        self.day_end_time = defaultdict(lambda: None)
        self._work_end_cache = {}

    def get_work_end(self, date):
        if date not in self._work_end_cache:
            weekday = date.isoweekday()
            base = datetime.combine(date, WORK_START)
            work_duration = 15.0 - WEEKDAY_END_HOURS[weekday]
            self._work_end_cache[date] = base + timedelta(hours=work_duration)
        return self._work_end_cache[date]

    def add_exam(self, date, start_time, duration_minutes, exam_type, patient_info):
        duration = timedelta(minutes=float(duration_minutes))
        end_time = start_time + duration
        self.timeline[date].append((
            start_time, end_time, exam_type,
            patient_info['compound_id'][0],
            patient_info['reg_date'],
            patient_info['is_self_selected']
        ))
        self.day_end_time[date] = end_time
        return end_time


class SchedulingSystem:
    def __init__(self, machine_exam_map, start_date=None):
        self.machines = [MachineSchedule(mid, machine_exam_map.get(mid, [])) for mid in range(MACHINE_COUNT)]
        self.current_date = start_date if start_date else START_DATE.date()
        self.start_date = self.current_date
        self.current_machine = 0

    def reset(self):
        self.current_date = self.start_date
        self.current_machine = 0

    def move_to_next(self):
        self.current_machine += 1
        if self.current_machine >= MACHINE_COUNT:
            self.current_machine = 0
            self.current_date += timedelta(days=1)

    def find_available_slot(self, duration_minutes, exam_type, patient_info):
        duration = timedelta(minutes=float(duration_minutes))
        for _ in range(365):
            m = self.machines[self.current_machine]
            last_end = m.day_end_time[self.current_date]
            start = datetime.combine(self.current_date, WORK_START) if last_end is None else last_end
            end = start + duration
            if end <= m.get_work_end(self.current_date):
                return m, start
            self.move_to_next()
        raise TimeoutError("无法在365天内找到可用时段")

    def generate_exam_dates(self, individual, patients):
        self.reset()
        exam_dates = {}
        for cid in individual:
            p = patients.get(cid)
            if p and not p['scheduled']:
                exam_type = clean_exam_name(p['exams'][0][1])
                duration = p['exams'][0][2]
                try:
                    m, start = self.find_available_slot(duration, exam_type, p)
                    exam_dates[cid] = start.date()
                    m.add_exam(start.date(), start, duration, exam_type, p)
                except Exception:
                    exam_dates[cid] = p['reg_date']
        return exam_dates
        
# ===================== GPU 适配度引擎 =====================

def _weekday_minutes_matrix_from_end_hours(M: int) -> torch.Tensor:
    hours = [int(round((15.0 - WEEKDAY_END_HOURS[d]) * 60)) for d in range(1, 8)]
    return torch.tensor([[m] * M for m in hours], dtype=DTYPE_LONG, device=DEVICE)


def _build_capacity_bins(weekday_machine_minutes: torch.Tensor, start_weekday: int, total_minutes_needed: int):
    weekday_machine_minutes = weekday_machine_minutes.to(DEVICE)
    M = weekday_machine_minutes.size(1)
    daily_totals = weekday_machine_minutes.sum(dim=1)
    min_daily = torch.clamp(daily_totals.min(), min=1)
    est_days = int((total_minutes_needed // int(min_daily.item())) + 3)
    days_idx = (torch.arange(est_days, device=DEVICE) + start_weekday) % 7
    caps_per_day = weekday_machine_minutes.index_select(0, days_idx)  # [D,M]
    caps_flat = caps_per_day.reshape(-1)
    caps_cumsum = torch.cumsum(caps_flat, dim=0)
    while caps_cumsum[-1].item() < total_minutes_needed:
        caps_cumsum = torch.cat([caps_cumsum, caps_cumsum[-1] + torch.cumsum(caps_flat, dim=0)])
        caps_per_day = torch.cat([caps_per_day, caps_per_day], dim=0)
        caps_flat = caps_per_day.reshape(-1)
    Bins = caps_cumsum.size(0)
    idx = torch.arange(Bins, device=DEVICE)
    bin_day = idx // M
    bin_machine = idx % M
    return caps_cumsum, bin_day, bin_machine


def _assign_bins_batch_by_prefix(durations_batch: torch.Tensor, caps_cumsum: torch.Tensor) -> torch.Tensor:
    T = torch.cumsum(durations_batch, dim=1)
    return torch.searchsorted(caps_cumsum, T, right=False)

def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
    B, N = bin_idx_batch.shape
    arng = torch.arange(N, device=DEVICE)
    arng_expanded = arng.expand(B, N)
    
    key = bin_idx_batch.long() * (N + 1) + arng_expanded
    _, sort_idx = torch.sort(key, dim=1)
    bin_sorted = bin_idx_batch.gather(1, sort_idx)
    
    is_start = torch.zeros_like(bin_sorted, dtype=torch.bool)
    is_start[:, 1:] = bin_sorted[:, 1:] != bin_sorted[:, :-1]
    is_start[:, 0] = True
    
    start_pos = torch.where(is_start, arng_expanded, -1)
    last_start_pos = torch.cummax(start_pos, dim=1)[0]
    rank_in_sorted = arng_expanded - last_start_pos
    
    order_idx = torch.empty_like(rank_in_sorted, dtype=DTYPE_LONG)
    order_idx.scatter_(1, sort_idx, rank_in_sorted)
    
    return order_idx


class _GPUMatrixFitnessBatch:
    def __init__(self, *,
                 weekday_machine_minutes: torch.Tensor,
                 start_weekday: int,
                 patient_durations: torch.Tensor,
                 reg_day_offsets: torch.Tensor,
                 is_self_selected: torch.Tensor,
                 has_contrast: torch.Tensor,
                 has_heart: torch.Tensor,
                 has_angio: torch.Tensor,
                 patient_main_type_id: torch.Tensor, # <<< 新增参数：接收患者主要检查类型ID
                 patient_exam_mask: torch.Tensor | None,
                 machine_exam_mask: torch.Tensor | None,
                 patient_main_exam_id: torch.Tensor | None = None, # 仅保留用于可能的兼容性，但逻辑上被 patient_main_type_id 替代
                 exam_count: int | None = None):
        
        self.weekday_machine_minutes = weekday_machine_minutes.to(DEVICE).long()
        self.start_weekday = int(start_weekday)
        self.patient_durations = patient_durations.to(DEVICE).long()
        self.reg_day_offsets = reg_day_offsets.to(DEVICE).long()
        self.is_self_selected = is_self_selected.to(DEVICE).bool()
        self.has_contrast = has_contrast.to(DEVICE).bool()
        self.has_heart = has_heart.to(DEVICE).bool()
        self.has_angio = has_angio.to(DEVICE).bool()
        
        # <<< 修改：存储类型ID，用于计算换模 >>>
        self.patient_main_type_id = patient_main_type_id.to(DEVICE).long()
        
        self.patient_exam_mask = patient_exam_mask.to(DEVICE).bool() if patient_exam_mask is not None else None
        self.machine_exam_mask = machine_exam_mask.to(DEVICE).bool() if machine_exam_mask is not None else None
        self.patient_main_exam_id = patient_main_exam_id.to(DEVICE).long() if patient_main_exam_id is not None else None
        self.exam_count = int(exam_count) if exam_count is not None else None

        total_minutes_needed = int(self.patient_durations.sum().item())
        self.caps_cumsum, self.bin_day, self.bin_machine = _build_capacity_bins(
            self.weekday_machine_minutes, self.start_weekday, total_minutes_needed
        )

    def _penalty_waiting(self, assigned_day_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        reg = self.reg_day_offsets.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        delta = (assigned_day_batch - reg).to(torch.int64)
        pos_wait = torch.clamp(delta, min=0).to(DTYPE_FLOAT)
        neg_wait = torch.clamp(-delta, min=0).to(DTYPE_FLOAT)
        is_self = self.is_self_selected.index_select(0, perms.reshape(-1)).reshape(perms.shape).to(DTYPE_FLOAT)
        non_self = 1.0 - is_self
        return pos_wait * (is_self * SELF_SELECTED_PENALTY + non_self * NON_SELF_PENALTY) + neg_wait * LOGICAL

    def _device_violate(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        if (self.patient_exam_mask is None) or (self.machine_exam_mask is None):
            return torch.zeros_like(assigned_machine_batch, dtype=torch.bool)
        mach_mask = self.machine_exam_mask[assigned_machine_batch]  # [B,N,E]
        pat_mask = self.patient_exam_mask[perms]                     # [B,N,E]
        invalid = pat_mask & (~mach_mask)
        return invalid.any(dim=2)

    def _penalty_device_cover(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        violate = self._device_violate(assigned_machine_batch, perms)
        return violate.to(DTYPE_FLOAT) * DEVICE_PENALTY

    def _special_violates(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
        heart_mask = self.has_heart.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        ok_wd_h = (weekday_batch == 1) | (weekday_batch == 3)  # Tue/Thu
        ok_mc_h = (assigned_machine_batch == 3)                # 3号机
        heart_violate = heart_mask & (~(ok_wd_h & ok_mc_h))

        angio_mask = self.has_angio.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        ok_wd_a = (weekday_batch == 0) | (weekday_batch == 2) | (weekday_batch == 4)  # Mon/Wed/Fri
        ok_mc_a = (assigned_machine_batch == 1)                                      # 1号机
        angio_violate = angio_mask & (~(ok_wd_a & ok_mc_a))

        weekend = (weekday_batch == 5) | (weekday_batch == 6)
        contrast_mask = self.has_contrast.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        weekend_violate = contrast_mask & weekend
        return heart_violate, angio_violate, weekend_violate

    def _penalty_special_rules(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
        heart_v, angio_v, weekend_v = self._special_violates(weekday_batch, assigned_machine_batch, perms)
        p = (heart_v | angio_v | weekend_v).to(DTYPE_FLOAT) * DEVICE_PENALTY
        return p, heart_v.to(torch.int32), angio_v.to(torch.int32), weekend_v.to(torch.int32)

    # <<< 新增：计算机器换模损耗 (Inter-patient switching) >>>
    def _penalty_machine_switching(self, bin_idx_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        """
        计算机器上的换模惩罚。
        逻辑：如果 bin[i] == bin[i-1] (同一机器连续排程) 且 type[i] != type[i-1]，则罚分。
        """
        B, N = perms.shape
        if TRANSITION_PENALTY <= 0:
            return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)

        # 1. 获取当前排列下，每个位置的患者检查类型 ID [B, N]
        current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
        
        # 2. 右移一位，获取"前一个"位置的信息
        prev_types = torch.roll(current_types, shifts=1, dims=1)
        prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
        
        # 3. 比较
        same_bin = (bin_idx_batch == prev_bins)     # 是否在同一台机器的连续时段
        diff_type = (current_types != prev_types)   # 检查类型是否改变
        
        # 4. 计算有效切换 (排除每行的第0个元素，因为roll会将最后一个元素移到第0个)
        is_transition = same_bin & diff_type
        is_transition[:, 0] = False 
        
        return is_transition.to(DTYPE_FLOAT) * TRANSITION_PENALTY

    def fitness_batch(self, perms: torch.Tensor, return_assignment: bool = False):
        perms = perms.to(DEVICE)
        B, N = perms.shape
        base = self.patient_durations.unsqueeze(0).expand(B, N)
        durations_batch = torch.gather(base, 1, perms)

        bin_idx_batch = _assign_bins_batch_by_prefix(durations_batch, self.caps_cumsum)
        assigned_day_batch = self.bin_day.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
        assigned_machine_batch = self.bin_machine.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
        weekday_batch = (self.start_weekday + assigned_day_batch) % 7

        p_wait  = self._penalty_waiting(assigned_day_batch, perms)
        p_dev   = self._penalty_device_cover(assigned_machine_batch, perms)
        p_spec, heart_v_i, angio_v_i, weekend_v_i = self._penalty_special_rules(weekday_batch, assigned_machine_batch, perms)
        
        # <<< 修改：使用新的换模惩罚逻辑 >>>
        p_tran  = self._penalty_machine_switching(bin_idx_batch, perms)

        total_penalty = p_wait + p_dev + p_spec + p_tran
        fitness = - total_penalty.sum(dim=1)
        out = {
            'fitness': fitness,
            'assigned_day': assigned_day_batch if return_assignment else None,
            'assigned_machine': assigned_machine_batch if return_assignment else None,
            'order_in_machine': _compute_order_in_bin_batch(bin_idx_batch) if return_assignment else None,
            'heart_cnt': heart_v_i.sum(dim=1),
            'angio_cnt': angio_v_i.sum(dim=1),
            'weekend_cnt': weekend_v_i.sum(dim=1),
            'device_cnt': (p_dev > 0).sum(dim=1),
            'any_violate_mask': (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)).any(dim=1) 
        }
        
        viol_mask_b_n = (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0))
        out['any_violate_mask_b_n'] = viol_mask_b_n # [B, N]
        
        return out


# ===================== GA 主体（Megabatch 版） =====================
class MultiRunOptimizer:
    def __init__(self, patients, machine_exam_map, num_parallel_runs, pop_size_per_run, block_start_date=None):
        self.patients = patients
        self.machine_exam_map = machine_exam_map
        self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
        self.current_generation = 0
        
        self.K = num_parallel_runs
        self.B = pop_size_per_run
        self.total_pop_size = self.K * self.B
        self.N = len(self.sorted_patients)
        
        self.block_start_date = block_start_date
        
        self.population_tensor: torch.Tensor | None = None # 维度: [K, B, N]
        self.fitness_history: List[List[float]] = [[] for _ in range(self.K)]
        
        self._gpu_engine = None
        self._cid_to_idx = None
        self._idx_to_cid = None
        self._patient_main_exam_id = None
        self._E = None

    # ------- GPU 引擎准备 -------
    def _ensure_gpu_engine(self):
        # (此函数 ... _ensure_gpu_engine ... 与原版 100% 相同)
        if self._gpu_engine is not None:
            return
        idx_to_cid = list(self.sorted_patients)
        cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
        self._idx_to_cid = idx_to_cid
        self._cid_to_idx = cid_to_idx
        N = len(idx_to_cid)

        patient_durations = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        reg_day_offsets = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        is_self_selected = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_contrast = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_heart = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_angio = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        
        # [新增] 初始化主要类型ID张量 (用于换模惩罚)
        patient_main_type_id = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)

        exam_set = set()
        for cid in idx_to_cid:
            for _, et, _, _ in self.patients[cid]['exams']:
                exam_set.add(clean_exam_name(et))
        for mid, exams in self.machine_exam_map.items():
            for e in exams:
                exam_set.add(clean_exam_name(e))
        exam_list = sorted(list(exam_set))
        exam_to_eidx = {e: i for i, e in enumerate(exam_list)}
        E = len(exam_list)
        self._E = E

        patient_exam_mask = torch.zeros((N, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
        machine_exam_mask = torch.zeros((MACHINE_COUNT, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
        patient_main_exam_id = torch.full((N,), -1, dtype=DTYPE_LONG, device=DEVICE)

        base_date = self.block_start_date if self.block_start_date else START_DATE.date()
        start_weekday = base_date.isoweekday() - 1

        for i, cid in enumerate(idx_to_cid):
            p = self.patients[cid]
            total_minutes = 0
            any_contrast = False
            any_heart = False
            any_angio = False
            exam_types_seq = []
            
            # 临时列表，用于确定患者的主要检查类型
            p_exam_types = []
            
            counter: Dict[int, int] = defaultdict(int)
            for _, et, dur, _ in p['exams']:
                etn = clean_exam_name(et)
                total_minutes += int(round(float(dur)))
                exam_types_seq.append(etn)
                p_exam_types.append(etn) # 收集类型
                if E > 0:
                    eidx = exam_to_eidx.get(etn, None)
                    if eidx is not None:
                        patient_exam_mask[i, eidx] = True
                        counter[eidx] += 1
                any_contrast = any_contrast or ('增强' in etn)
                any_heart = any_heart or ('心脏' in etn)
                any_angio = any_angio or ('造影' in etn)
            
            # 兼容性逻辑：用于 mutation 的 ID (出现次数最多的)
            if len(counter) > 0:
                best_cnt = max(counter.values())
                main_eidx = min([k for k, v in counter.items() if v == best_cnt])
                patient_main_exam_id[i] = main_eidx
            
            # [新增逻辑] 确定患者的"主要类型ID"用于换模惩罚
            # 取第一个检查项目作为判断换模的依据
            if p_exam_types:
                main_type = p_exam_types[0]
                patient_main_type_id[i] = exam_to_eidx.get(main_type, 0)
            else:
                patient_main_type_id[i] = 0

            # 移除旧的 switch_penalty 计算 (switch_penalty[i] = ...) 

            patient_durations[i] = max(1, total_minutes)
            reg_day_offsets[i] = (p['reg_date'] - base_date).days
            is_self_selected[i] = bool(p.get('is_self_selected', False))
            has_contrast[i] = any_contrast
            has_heart[i] = any_heart
            has_angio[i] = any_angio

        if machine_exam_mask is not None:
            for mid in range(MACHINE_COUNT):
                for e in self.machine_exam_map.get(mid, []):
                    et = clean_exam_name(e)
                    eidx = exam_to_eidx.get(et, None)
                    if eidx is not None:
                        machine_exam_mask[mid, eidx] = True

        weekday_machine_minutes = _weekday_minutes_matrix_from_end_hours(MACHINE_COUNT)
        
        # 必须将局部变量 patient_main_type_id 保存到 self 属性中
        # 否则后续的变异算子 (_mutate_batch 等) 调用 self._patient_main_exam_id 时会报错
        # 注意：这里我们使用 patient_main_exam_id (基于计数的) 赋值给 self._patient_main_exam_id，
        # 保持 mutation 逻辑不变，同时传递 patient_main_type_id 给引擎做罚分。
        self._patient_main_exam_id = patient_main_exam_id
        
        self._gpu_engine = _GPUMatrixFitnessBatch(
            weekday_machine_minutes=weekday_machine_minutes,
            start_weekday=start_weekday,
            patient_durations=patient_durations,
            reg_day_offsets=reg_day_offsets,
            is_self_selected=is_self_selected,
            has_contrast=has_contrast,
            has_heart=has_heart,
            has_angio=has_angio,
            patient_main_type_id=patient_main_type_id, # [新增]
            patient_exam_mask=patient_exam_mask,
            machine_exam_mask=machine_exam_mask,
            patient_main_exam_id=patient_main_exam_id, # 保留用于兼容
            exam_count=E,
        )

    # ------- 索引 ↔ cid (仅用于最后转换) -------
    def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
        # 确保 _idx_to_cid 存在
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
        return [self._idx_to_cid[int(x)] for x in row.tolist()]

    def initialize_population(self):
        if self.N == 0:
            print("警告：患者列表为空，无法初始化种群。")
            return

        # 确保 cid 映射存在
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
            self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}

        indices = torch.arange(self.N, device=DEVICE)
        block_size = max(30, self.N // 20)
        
        # 核心变更：创建 [K, B, N] 张量
        pop_indices = torch.empty((self.K, self.B, self.N), dtype=DTYPE_LONG, device=DEVICE)
        
        # 预先生成所有随机数
        rand_matrices = torch.rand(self.K, self.B, self.N, device=DEVICE)
        
        # CPU 循环遍历 *块*
        for i in range(0, self.N, block_size):
            end = min(i + block_size, self.N)
            block_len = end - i
            if block_len == 0:
                continue
                
            # 核心变更：在 dim=2 (N 维度) 上排序
            block_rand = rand_matrices[:, :, i:end] # [K, B, block_len]
            block_perm_idx = torch.argsort(block_rand, dim=2) # [K, B, block_len]
            
            block_indices = indices[i:end]
            block_indices_expanded = block_indices.expand(self.K, self.B, -1) # [K, B, block_len]
            
            # 核心变更：在 dim=2 上 gather
            pop_indices[:, :, i:end] = torch.gather(block_indices_expanded, 2, block_perm_idx)

        # 种群现在是一个保存在 GPU 上的张量
        self.population_tensor = pop_indices
        print(f"已生成 {self.K} 个并行种群 (每个 {self.B} 个个体)，总计 {self.total_pop_size} 个个体")


    # ------- GA 主循环 (Megabatch 版) -------
    def evolve_gpu(self, generations=100, elite_size=5):
        self._ensure_gpu_engine()
        if self.population_tensor is None:
            raise RuntimeError("种群为空，请先 initialize_population")
        
        # pop 是我们的 [K, B, N] GPU 张量
        pop = self.population_tensor
        N = self.N # 基因数
        
        for gen_idx in range(generations):
            # 1. 评估 (Megabatch)
            # 将 [K, B, N] 展平为 [K*B, N]
            pop_flat = pop.view(self.total_pop_size, N)
            out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            fitness_flat = out['fitness']
            fitness = fitness_flat.view(self.K, self.B)
            viol_mask_flat = out['any_violate_mask_b_n'] 
            violate_mask = viol_mask_flat.view(self.K, self.B, N)
            topk_vals, topk_idx = torch.topk(fitness, k=self.B, largest=True, dim=1)
            best_fitness_per_run = topk_vals[:, 0].cpu().tolist()
            for k in range(self.K):
                self.fitness_history[k].append(best_fitness_per_run[k])
            
            elite_size = min(elite_size, self.B)
            elite_idx = topk_idx[:, :elite_size] # [K, elite_size]
            
            # 核心变更：使用 gather 从 [K, B, N] 中挑选精英
            # 索引需要扩展到 [K, elite_size, N]
            idx_expanded = elite_idx.unsqueeze(2).expand(self.K, elite_size, N)
            elites = torch.gather(pop, 1, idx_expanded) # [K, elite_size, N]

            # 3. 父代选择 (按 K 独立进行)
            parent_count = max(1, int(0.2 * self.B))
            parent_idx = topk_idx[:, :parent_count] # [K, parent_count]
            
            # 从 pop 中 gather 父代
            idx_expanded = parent_idx.unsqueeze(2).expand(self.K, parent_count, N)
            parents = torch.gather(pop, 1, idx_expanded) # [K, parent_count, N]
            
            # 从 violate_mask 中 gather 对应的违规掩码
            parent_viol = torch.gather(violate_mask, 1, idx_expanded) # [K, parent_count, N]

            # 4. 交叉 (Megabatch)
            num_children = self.B - elite_size
            if num_children > 0:
                # 核心变更：为 K 个种群各自生成配对索引
                p_idx1 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE) # [K, num_children]
                p_idx2 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE) # [K, num_children]
                
                # 从 parents [K, parent_count, N] 中 gather
                P1 = torch.gather(parents, 1, p_idx1.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
                P2 = torch.gather(parents, 1, p_idx2.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
                
                # Gather 对应的违规掩码
                Vmask_choice = torch.gather(parent_viol, 1, p_idx1.unsqueeze(2).expand(-1, -1, N)) # [K, num_children, N]
                
                # 展平为 [K*num_children, N] 以
                # 喂给无需修改的交叉/变异函数
                P1_flat = P1.view(self.K * num_children, N)
                P2_flat = P2.view(self.K * num_children, N)
                
                children_flat = self._ordered_crossover_batch_gpu(P1_flat, P2_flat)
                
                # 5. 变异 (Megabatch)
                Vmask_flat = Vmask_choice.view(self.K * num_children, N)
                
                children_flat = self._mutate_batch_gpu(children_flat, Vmask_flat, self.current_generation)
                
                # 重塑回 [K, num_children, N]
                children = children_flat.view(self.K, num_children, N)
                
                # 6. 形成新种群
                pop = torch.cat([elites, children], dim=1) # 沿 B 维度拼接
            else:
                pop = elites.clone()
            
            if (gen_idx + 1) % 50 == 0:
                # 报告 K 个种群的平均最佳适应度
                avg_best_fit = sum(best_fitness_per_run) / self.K
                print(f"Generation {self.current_generation+1} | Avg Best Fitness (K={self.K}): {avg_best_fit:.2f}")

            self.current_generation += 1

        # 7. 进化结束，返回 K 个种群的最终最佳个体
        print("进化完成。正在提取 K 个最佳个体...")
        
        # 最终评估一次
        pop_flat = pop.view(self.total_pop_size, N)
        final_out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
        final_fitness = final_out['fitness'].view(self.K, self.B) # [K, B]
        
        # 找到 K 个种群中各自的最佳 (k=1)
        final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1) # [K, 1], [K, 1]
        
        final_best_vals = final_best_vals.flatten() # [K]
        
        # Gather 最佳个体
        # 修复：final_best_idx_in_B 是 [K, 1] (即 [8, 1])。
        # 我们需要 .unsqueeze(2) 将其变为 [K, 1, 1] (即 [8, 1, 1])，
        # 然后才能 .expand() 到 [K, 1, N] (即 [8, 1, 2379])。
        idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(self.K, 1, N) # [K, 1, N]
        best_individuals_tensor = torch.gather(pop, 1, idx_expanded).squeeze(1) # [K, N]
        
        # 转换为 CPU 列表
        best_individuals_cpu = best_individuals_tensor.cpu()
        best_fitnesses_cpu = final_best_vals.cpu().tolist()
        
        # 转换为 CIDs
        results = []
        for k in range(self.K):
            cids = self._tensor_row_to_cids(best_individuals_cpu[k])
            results.append({
                "run_id": k,
                "individual_cids": cids,
                "fitness": best_fitnesses_cpu[k]
            })
            
        self.population_tensor = pop # 保存最终状态
        return results

    @staticmethod
    def _random_cuts(num_rows: int, N: int):
        a = torch.randint(0, N, (num_rows,), device=DEVICE)
        b = torch.randint(0, N, (num_rows,), device=DEVICE)
        start = torch.minimum(a, b)
        end = torch.maximum(a, b)
        return start, end

    def _ordered_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        C, N = P1.shape
        arangeN = torch.arange(N, device=DEVICE).expand(C, N)
        
        start, end = self._random_cuts(C, N)
        s_exp = start.unsqueeze(1)
        e_exp = end.unsqueeze(1)
        
        mask_frag = (arangeN >= s_exp) & (arangeN <= e_exp)
        
        children = torch.full_like(P1, -1)
        children[mask_frag] = P1[mask_frag]
        
        P2_expanded = P2.unsqueeze(2)  # (C, N, 1)
        P1_expanded = P1.unsqueeze(1)  # (C, 1, N)
        
        equality_matrix = (P2_expanded == P1_expanded)
        mask_frag_expanded = mask_frag.unsqueeze(1)
        
        isin_matrix = (equality_matrix & mask_frag_expanded).any(dim=2)
        mask_tail = ~isin_matrix
        P2_tails_flat = P2[mask_tail]
        mask_fill = (children == -1)
        children[mask_fill] = P2_tails_flat
        
        return children
    
    def _mutate_step1_violations(self, X: torch.Tensor, parent_violate_mask: torch.Tensor) -> torch.Tensor:
        C, N = X.shape
        # 修复：parent_violate_mask 是 [C, N]
        any_viol_per_row = torch.any(parent_violate_mask, dim=1) # [C]
        viol_rows_idx = torch.nonzero(any_viol_per_row, as_tuple=False).flatten()
        R = viol_rows_idx.numel()
        if R == 0:
            return X

        viol_mask_subset = parent_violate_mask[viol_rows_idx] # [R, N]
        
        viol_idx_in_row = torch.multinomial(viol_mask_subset.float(), 1, replacement=True).flatten() # [R]
        
        low = torch.clamp(viol_idx_in_row - 400, min=0)
        high = torch.clamp(viol_idx_in_row + 400, max=N-1)
        range_size = high - low + 1
        
        range_size = torch.where(range_size <= 0, 1, range_size)
        
        rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
        cand_idx_in_row = low + rand_offset
        
        cand_idx_in_row = torch.where(
            (cand_idx_in_row == viol_idx_in_row) & (range_size > 1), 
            torch.where(viol_idx_in_row == low, low + 1, low),
            cand_idx_in_row
        )
        cand_idx_in_row = torch.clamp(cand_idx_in_row, 0, N-1)

        val1 = X[viol_rows_idx, viol_idx_in_row]
        val2 = X[viol_rows_idx, cand_idx_in_row]
        X[viol_rows_idx, viol_idx_in_row] = val2
        X[viol_rows_idx, cand_idx_in_row] = val1
        
        return X
        
    def _mutate_step2_base_swap(self, X: torch.Tensor, current_gen: int, base_swap_prob: float = 0.95) -> torch.Tensor:
        C, N = X.shape
        use_range_limit = (current_gen <= 10000)
        
        probs = torch.rand(C, device=DEVICE)
        rows_to_swap_mask = (probs < base_swap_prob)
        rows_to_swap_idx = torch.nonzero(rows_to_swap_mask, as_tuple=False).flatten()
        R = rows_to_swap_idx.numel()
        if R == 0:
            return X

        idx1 = torch.randint(0, N, (R,), device=DEVICE)
        
        if use_range_limit:
            low = torch.clamp(idx1 - 400, min=0)
            high = torch.clamp(idx1 + 400, max=N-1)
            range_size = high - low + 1
            range_size = torch.where(range_size <= 0, 1, range_size)
            rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
            idx2 = low + rand_offset
            idx2 = torch.where(
                (idx2 == idx1) & (range_size > 1), 
                torch.where(idx1 == low, low + 1, low),
                idx2
            )
            idx2 = torch.clamp(idx2, 0, N-1)
        else:
            idx2 = torch.randint(0, N, (R,), device=DEVICE)
            idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)
        
        val1 = X[rows_to_swap_idx, idx1]
        val2 = X[rows_to_swap_idx, idx2]
        X[rows_to_swap_idx, idx1] = val2
        X[rows_to_swap_idx, idx2] = val1
        
        return X

    def _mutate_step3_greedy_cluster(self, X: torch.Tensor, greedy_prob: float = 0.5) -> torch.Tensor:
        C, N = X.shape
        if N < 50 or self._E is None or self._patient_main_exam_id is None:
            return X
            
        probs2 = torch.rand(C, device=DEVICE)
        rows_to_greedy_mask = (probs2 < greedy_prob)
        rows_to_greedy_idx = torch.nonzero(rows_to_greedy_mask, as_tuple=False).flatten()
        R = rows_to_greedy_idx.numel()
        if R == 0:
            return X
        
        window_len = torch.randint(20, 51, (R,), device=DEVICE)
        start = torch.randint(0, N - 49, (R,), device=DEVICE)
        end = torch.clamp(start + window_len - 1, max=N-1)
        window_len = end - start + 1
        max_len = window_len.max().item()

        arng_max = torch.arange(max_len, device=DEVICE)
        arng_R_max = arng_max.expand(R, max_len)
        
        window_indices = start.unsqueeze(1) + arng_R_max
        
        valid_mask = arng_R_max < window_len.unsqueeze(1)

        rows = X[rows_to_greedy_idx]
        
        windows = torch.gather(rows, 1, window_indices)

        keys = self._patient_main_exam_id[windows]
        keys[~valid_mask] = -1

        E = self._E
        keys_clamped = keys.clamp(min=0, max=E-1)
        one_hot_keys = torch.nn.functional.one_hot(keys_clamped, num_classes=E)
        one_hot_keys[~valid_mask] = 0
        counts = one_hot_keys.sum(dim=1)

        size_per_pos = torch.gather(counts, 1, keys_clamped)
        size_per_pos[~valid_mask] = -1

        sort_key = (-size_per_pos).to(torch.int64) * (max_len + 1) + arng_R_max
        sort_key[~valid_mask] = 9223372036854775807
        
        new_order = torch.argsort(sort_key, dim=1, stable=True)

        sorted_windows = torch.gather(windows, 1, new_order)

        new_rows = rows.clone()
        new_rows.scatter_(dim=1, index=window_indices, src=sorted_windows)
        
        scatter_mask = torch.zeros_like(rows, dtype=torch.bool, device=DEVICE)
        scatter_mask.scatter_(dim=1, index=window_indices, src=valid_mask)

        final_rows = torch.where(scatter_mask, new_rows, rows)

        X[rows_to_greedy_idx] = final_rows
            
        return X

    def _mutate_batch_gpu(self, X: torch.Tensor, parent_violate_mask: torch.Tensor, current_gen: int,
                          base_swap_prob: float = 0.95, greedy_prob: float = 0.5) -> torch.Tensor:
        
        X = self._mutate_step1_violations(X, parent_violate_mask)
        X = self._mutate_step2_base_swap(X, current_gen, base_swap_prob)
        X = self._mutate_step3_greedy_cluster(X, greedy_prob)
        
        return X

    def _greedy_cluster_mutation_gpu_inplace(self, X: torch.Tensor, row: int, start: int, end: int):
        window = X[row, start:end+1]
        keys = self._patient_main_exam_id.index_select(0, window)
        W = keys.numel()
        if W <= 1 or (self._E is None):
            return
        counts = torch.bincount(torch.clamp(keys, min=0), minlength=max(1, self._E))
        size_per_pos = counts[keys.clamp(min=0)]
        arng = torch.arange(W, device=DEVICE)
        sort_key = (-size_per_pos).to(torch.int64) * (W + 1) + arng
        new_order = torch.argsort(sort_key, stable=True)
        X[row, start:end+1] = window.index_select(0, new_order)


    # ------- 导出（可选） -------
    # def generate_schedule(self, individual):
    #     system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
    #     for cid in individual:
    #         p = self.patients.get(cid)
    #         if p and not p['scheduled']:
    #             # 修复：确保所有检查都被安排
    #             for exam in p['exams']:
    #                 exam_type = clean_exam_name(exam[1])
    #                 duration = exam[2]
    #                 try:
    #                     m, start_time = system.find_available_slot(duration, exam_type, p)
    #                     m.add_exam(system.current_date, start_time, duration, exam_type, p)
    #                 except Exception as e:
    #                     # 避免在真实排程中打印
    #                     # print(f"排程错误: {e}") 
    #                     pass # 忽略错误并继续
    #     return system
    def generate_schedule(self, individual_cids):
            self._ensure_gpu_engine()
            if self._cid_to_idx is None or self._idx_to_cid is None:
                self._idx_to_cid = list(self.sorted_patients)
                self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}
            base_date = self.block_start_date if self.block_start_date else START_DATE.date()
            perm_idx = torch.tensor(
                [self._cid_to_idx[cid] for cid in individual_cids],
                dtype=DTYPE_LONG,
                device=DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                out = self._gpu_engine.fitness_batch(perm_idx, return_assignment=True)

            assigned_day = out["assigned_day"][0].cpu().tolist()      # [N]
            assigned_machine = out["assigned_machine"][0].cpu().tolist() # [N]
            order_in_machine = out["order_in_machine"][0].cpu().tolist() # [N]
            perm_cpu = perm_idx[0].cpu().tolist()  # [N] 患者的 index
            bins = defaultdict(list)
            for pos, pid_idx in enumerate(perm_cpu):
                d = int(assigned_day[pos])
                m = int(assigned_machine[pos])
                o = int(order_in_machine[pos])
                bins[(d, m)].append((o, int(pid_idx)))

            system = SchedulingSystem(self.machine_exam_map, start_date=base_date)
            for (d, m), lst in sorted(bins.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                current_date_obj = base_date + timedelta(days=int(d))
                machine_obj = system.machines[int(m)]
                cur_dt = datetime.combine(current_date_obj, WORK_START)
                for _, pid_idx in sorted(lst, key=lambda x: x[0]):
                    cid = self._idx_to_cid[int(pid_idx)]
                    p = self.patients.get(cid)
                    if not p:
                        continue
                    for exam in p["exams"]:
                        exam_type = clean_exam_name(exam[1])
                        duration = exam[2]
                        cur_dt = machine_obj.add_exam(current_date_obj, cur_dt, duration, exam_type, p)

            return system

# ===================== 导出 Excel =====================
# (export_schedule ... 保持不变 ...)
def export_schedule(system, patients, filename):
    with pd.ExcelWriter(filename) as writer:
        rows = []
        for machine in system.machines:
            for date in sorted(machine.timeline):
                slots = sorted(machine.timeline[date], key=lambda x: x[0])
                for (start, end, exam, pid, reg_date, is_self) in slots:
                    rows.append({
                        '机器编号': machine.machine_id,
                        '日期': date.strftime('%Y-%m-%d'),
                        '开始时间': start.strftime('%H:%M:%S'),
                        '结束时间': end.strftime('%H:%M:%S'),
                        '检查项目': exam,
                        '患者ID': pid,
                        '登记日期': reg_date.strftime('%Y-%m-%d'),
                        '是否自选': '是' if is_self else '否',
                    })
        df = pd.DataFrame(rows)
        if df.empty:
            pd.DataFrame(columns=['机器编号','日期','开始时间','结束时间','检查项目','患者ID','登记日期','是否自选']).to_excel(writer, sheet_name='总排程', index=False)
        else:
            df.sort_values(by=['机器编号', '日期', '开始时间']).to_excel(writer, sheet_name='总排程', index=False)

# ===================== main (Megabatch 版) =====================
def main():
    try:
        print(START_DATE)
        # ================== 配置 ==================
        
        # 你希望并行运行多少个独立的GA实验？
        # 这会成为 K 维度
        NUM_PARALLEL_RUNS = 4 
        
        # 每个独立实验的种群大小
        # 这会成为 B 维度
        POP_SIZE_PER_RUN = 50 
        
        # 进化代数
        GENERATIONS_TO_RUN = 10000    
        # ==========================================
        
        print(f"启动 Megabatch 模式: K={NUM_PARALLEL_RUNS} (并行实验), B={POP_SIZE_PER_RUN} (个体/实验)")
        print(f"总 GPU 批量: {NUM_PARALLEL_RUNS * POP_SIZE_PER_RUN} 个体")
        
        current_dir = "/home/preprocess/_funsearch/baseline/data"
        patient_file = os.path.join(current_dir, '实验数据6.1 - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
        for f in [patient_file, duration_file, device_constraint_file]:
            if not os.path.exists(f):
                print(f"❌ 错误：找不到文件 {f}")
                return

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        print("\n===== 启动并行遗传算法优化 (Megabatch GPU) =====")
        optimizer = MultiRunOptimizer(
            patients, 
            machine_exam_map, 
            num_parallel_runs=NUM_PARALLEL_RUNS, 
            pop_size_per_run=POP_SIZE_PER_RUN
        )
        
        t0_init = time.perf_counter()
        optimizer.initialize_population()
        t_init = time.perf_counter() - t0_init
        print(f"✓ 已生成 {NUM_PARALLEL_RUNS} 个初始种群，耗时: {t_init:.4f}s")


        print(f"\n开始 {GENERATIONS_TO_RUN} 代进化 (K={NUM_PARALLEL_RUNS})...")
        t0 = time.perf_counter()
        
        # evolve_gpu 现在返回一个包含 K 个结果的列表
        final_results = optimizer.evolve_gpu(generations=GENERATIONS_TO_RUN, elite_size=5)
        
        total_evolution_time = time.perf_counter() - t0
        print(f"\n✓ 进化完成 (K={NUM_PARALLEL_RUNS})，总耗时: {total_evolution_time:.2f}s")
        print(f"  平均每代耗时: {total_evolution_time / GENERATIONS_TO_RUN:.4f} s/gen")
        print(f"  (总计 {GENERATIONS_TO_RUN * NUM_PARALLEL_RUNS} 个 'run-generations')")


        print(f"\n===== 正在导出 {NUM_PARALLEL_RUNS} 个最佳排程 =====")
        out_dir = 'output_schedules'; os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        all_fitnesses = []
        
        for result in final_results:
            run_id = result['run_id']
            best_individual = result['individual_cids']
            best_fitness = result['fitness']
            all_fitnesses.append(best_fitness)
            
            print(f"  Run {run_id}: Best Fitness: {best_fitness:.2f}")
            
            # 导出 Excel
            xlsx = os.path.join(out_dir, f'final_schedule_RUN{run_id}_{ts}_fit_{best_fitness:.0f}.xlsx')
            final_system = optimizer.generate_schedule(best_individual)
            export_schedule(final_system, patients, xlsx)
            print(f"    ✓ 已导出至 {xlsx}")

        print("\n===== 最终统计 (K={NUM_PARALLEL_RUNS}) =====")
        mean_fitness = np.mean(all_fitnesses)
        std_fitness = np.std(all_fitnesses)
        min_fitness = np.min(all_fitnesses)
        max_fitness = np.max(all_fitnesses)
        
        print(f"  最佳适应度 (均值): {mean_fitness:.2f}")
        print(f"  最佳适应度 (标准差): {std_fitness:.2f}")
        print(f"  最佳适应度 (范围): {min_fitness:.2f} ... {max_fitness:.2f}")
        print("\n所有运行均已完成。")

    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()
    finally:
        # 移除 input() 以便自动退出
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
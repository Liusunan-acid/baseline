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
# START_DATE = datetime(2024, 12, 1, 7, 0)
# MACHINE_COUNT = 6
# DEVICE_PENALTY = 500000
# POPULATION_FILE = 'population_state.json'

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE_LONG = torch.long
# DTYPE_FLOAT = torch.float32

# # ===================== 工具函数 =====================

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

# # ===================== 导出所需（CPU 精排仅用于导出，不改变 GA 思想） =====================
# class MachineSchedule:
#     def __init__(self, machine_id, allowed_exams):
#         self.machine_id = machine_id
#         self.allowed_exams = allowed_exams
#         self.timeline = defaultdict(list)
#         self.day_end_time = defaultdict(lambda: None)
#         self._work_end_cache = {}

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
#         for _ in range(365):
#             m = self.machines[self.current_machine]
#             last_end = m.day_end_time[self.current_date]
#             start = datetime.combine(self.current_date, WORK_START) if last_end is None else last_end
#             end = start + duration
#             if end <= m.get_work_end(self.current_date):
#                 return m, start
#             self.move_to_next()
#         raise TimeoutError("无法在365天内找到可用时段")

#     def generate_exam_dates(self, individual, patients):
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


# def _compute_order_in_bin_row(bin_idx_row: torch.Tensor) -> torch.Tensor:
#     N = bin_idx_row.numel()
#     arng = torch.arange(N, device=bin_idx_row.device)
#     sort_idx = torch.argsort(bin_idx_row, stable=True)
#     bin_sorted = bin_idx_row[sort_idx]
#     is_start = torch.ones(N, dtype=torch.bool, device=bin_idx_row.device)
#     is_start[1:] = bin_sorted[1:] != bin_sorted[:-1]
#     start_pos = torch.where(is_start, arng, torch.full((N,), -1, device=bin_idx_row.device))
#     last_start_pos = torch.cummax(start_pos, dim=0)[0]
#     rank_in_sorted = arng - last_start_pos
#     order_idx = torch.empty_like(rank_in_sorted)
#     order_idx[sort_idx] = rank_in_sorted
#     return order_idx


# def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
#     B, N = bin_idx_batch.shape
#     out = torch.empty_like(bin_idx_batch)
#     for b in range(B):
#         out[b] = _compute_order_in_bin_row(bin_idx_batch[b])
#     return out

# # @torch.no_grad()
# # def _compute_order_in_bin_batch(self, bin_idx_batch: torch.Tensor) -> torch.Tensor:
# #     """
# #     输入:
# #         bin_idx_batch: LongTensor [B, N]，每个位置所属的 bin 编号（可重复、可无序）
# #     输出:
# #         order_in_bin:  LongTensor [B, N]，该位置在其 bin 内的出现次序 (0-based)
# #                        如 [2,2,5,2,5] -> [0,1,0,2,1]
# #     说明:
# #         向量化实现：按 (bin, pos) 排序 -> 分段前缀计数 -> 反 scatter 回原位
# #     """
# #     assert bin_idx_batch.dim() == 2 and bin_idx_batch.dtype == torch.long
# #     B, N = bin_idx_batch.shape
# #     device = bin_idx_batch.device

# #     # --- 处理可能出现的负 bin：整体平移到非负 ---
# #     min_bin = int(bin_idx_batch.min())
# #     if min_bin < 0:
# #         bins = bin_idx_batch - min_bin
# #     else:
# #         bins = bin_idx_batch

# #     # (1) 生成列索引 pos，并构造组合键 key = bins * N + pos
# #     pos = torch.arange(N, device=device, dtype=torch.long).expand(B, N)   # [B, N]
# #     key = bins * N + pos                                                  # [B, N]，同 bin 内按原位置递增

# #     # 行内排序：得到每行 (bin, pos) 升序的下标
# #     idx_sorted = torch.argsort(key, dim=1, stable=True)                   # [B, N]
# #     bins_sorted = torch.gather(bins, 1, idx_sorted)                       # [B, N]
# #     pos_sorted  = torch.gather(pos,  1, idx_sorted)                       # [B, N]

# #     # (2) 标记新段起点（bin 发生变化处），并做“分段前缀计数”
# #     start = torch.ones_like(bins_sorted, dtype=torch.bool)                # [B, N]
# #     start[:, 1:] = bins_sorted[:, 1:] != bins_sorted[:, :-1]              # True 表示一个新 bin 段开始

# #     # 把段起点的 pos_sorted 作为“基准”，非起点填极小值；行向 cummax 得到最近段起点位置
# #     NEG_INF = torch.iinfo(torch.int64).min // 4
# #     base = torch.where(start, pos_sorted, torch.full_like(pos_sorted, NEG_INF))
# #     base_cummax, _ = torch.cummax(base, dim=1)                            # [B, N] 最近段起点的 pos
# #     order_sorted = (pos_sorted - base_cummax).to(torch.long)              # 段内 0,1,2,...

# #     # (3) 反 scatter 回原来的列位置
# #     order_in_bin = torch.empty_like(order_sorted)
# #     order_in_bin.scatter_(dim=1, index=idx_sorted, src=order_sorted)      # [B, N]

# #     return order_in_bin



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
#                  switch_penalty: torch.Tensor,
#                  patient_exam_mask: torch.Tensor | None,
#                  machine_exam_mask: torch.Tensor | None,
#                  patient_main_exam_id: torch.Tensor | None = None,
#                  exam_count: int | None = None):
#         self.weekday_machine_minutes = weekday_machine_minutes.to(DEVICE).long()
#         self.start_weekday = int(start_weekday)
#         self.patient_durations = patient_durations.to(DEVICE).long()
#         self.reg_day_offsets = reg_day_offsets.to(DEVICE).long()
#         self.is_self_selected = is_self_selected.to(DEVICE).bool()
#         self.has_contrast = has_contrast.to(DEVICE).bool()
#         self.has_heart = has_heart.to(DEVICE).bool()
#         self.has_angio = has_angio.to(DEVICE).bool()
#         self.switch_penalty = switch_penalty.to(DEVICE).to(DTYPE_FLOAT)
#         self.patient_exam_mask = patient_exam_mask.to(DEVICE).bool() if patient_exam_mask is not None else None
#         self.machine_exam_mask = machine_exam_mask.to(DEVICE).bool() if machine_exam_mask is not None else None
#         self.patient_main_exam_id = patient_main_exam_id.to(DEVICE).long() if patient_main_exam_id is not None else None
#         self.exam_count = int(exam_count) if exam_count is not None else None

#         total_minutes_needed = int(self.patient_durations.sum().item())
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

#     def _penalty_transition(self, perms: torch.Tensor) -> torch.Tensor:
#         B, N = perms.shape
#         if not torch.any(self.switch_penalty > 0):
#             return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)
#         base = self.switch_penalty.unsqueeze(0).expand(B, -1)
#         return torch.gather(base, 1, perms)

#     def fitness_batch(self, perms: torch.Tensor, return_assignment: bool = False):
#         perms = perms.to(DEVICE)
#         B, N = perms.shape
#         base = self.patient_durations.unsqueeze(0).expand(B, N)
#         durations_batch = torch.gather(base, 1, perms)

#         bin_idx_batch = _assign_bins_batch_by_prefix(durations_batch, self.caps_cumsum)
#         assigned_day_batch = self.bin_day.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         assigned_machine_batch = self.bin_machine.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         weekday_batch = (self.start_weekday + assigned_day_batch) % 7

#         p_wait  = self._penalty_waiting(assigned_day_batch, perms)
#         p_dev   = self._penalty_device_cover(assigned_machine_batch, perms)
#         p_spec, heart_v_i, angio_v_i, weekend_v_i = self._penalty_special_rules(weekday_batch, assigned_machine_batch, perms)
#         p_tran  = self._penalty_transition(perms)

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
#             'any_violate_mask': (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0))
#         }
#         return out


# # ===================== GA 主体（GPU 版） =====================
# class BlockGeneticOptimizer:
#     def __init__(self, patients, machine_exam_map, pop_size=50, block_start_date=None):
#         self.patients = patients
#         self.machine_exam_map = machine_exam_map
#         self.population: List[List[Any]] = []
#         self.fitness_history: List[float] = []
#         self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
#         self.current_generation = 0
#         self.pop_size = pop_size
#         self.block_start_date = block_start_date
#         self._gpu_engine = None
#         self._cid_to_idx = None
#         self._idx_to_cid = None
#         self._patient_main_exam_id = None
#         self._E = None

#     # ------- GPU 引擎准备 -------
#     def _ensure_gpu_engine(self):
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
#         switch_penalty = torch.zeros(N, dtype=DTYPE_FLOAT, device=DEVICE)

#         # 构建检查全集
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
#             counter: Dict[int, int] = defaultdict(int)
#             for _, et, dur, _ in p['exams']:
#                 etn = clean_exam_name(et)
#                 total_minutes += int(round(float(dur)))
#                 exam_types_seq.append(etn)
#                 if E > 0:
#                     eidx = exam_to_eidx.get(etn, None)
#                     if eidx is not None:
#                         patient_exam_mask[i, eidx] = True
#                         counter[eidx] += 1
#                 any_contrast = any_contrast or ('增强' in etn)
#                 any_heart = any_heart or ('心脏' in etn)
#                 any_angio = any_angio or ('造影' in etn)
#             if len(counter) > 0:
#                 best_cnt = max(counter.values())
#                 main_eidx = min([k for k, v in counter.items() if v == best_cnt])
#                 patient_main_exam_id[i] = main_eidx

#             switches = 0
#             for k in range(1, len(exam_types_seq)):
#                 if exam_types_seq[k] != exam_types_seq[k-1]:
#                     switches += 1
#             switch_penalty[i] = float(switches) * float(TRANSITION_PENALTY) if switches > 0 else 0.0

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
#         self._gpu_engine = _GPUMatrixFitnessBatch(
#             weekday_machine_minutes=weekday_machine_minutes,
#             start_weekday=start_weekday,
#             patient_durations=patient_durations,
#             reg_day_offsets=reg_day_offsets,
#             is_self_selected=is_self_selected,
#             has_contrast=has_contrast,
#             has_heart=has_heart,
#             has_angio=has_angio,
#             switch_penalty=switch_penalty,
#             patient_exam_mask=patient_exam_mask,
#             machine_exam_mask=machine_exam_mask,
#             patient_main_exam_id=patient_main_exam_id,
#             exam_count=E,
#         )
#         self._patient_main_exam_id = patient_main_exam_id

#     # ------- 索引 ↔ cid -------
#     def _individual_to_perm(self, individual: List[Any]) -> torch.Tensor:
#         return torch.tensor([self._cid_to_idx[cid] for cid in individual], dtype=DTYPE_LONG, device=DEVICE)

#     def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
#         return [self._idx_to_cid[int(x)] for x in row.tolist()]

#     # ------- 初始化种群（保持原“分块洗牌”思想） -------
#     def initialize_population(self, pop_size=None):
#         if pop_size is None:
#             pop_size = self.pop_size
#         block_size = max(30, len(self.sorted_patients) // 20)
#         blocks = [self.sorted_patients[i:i + block_size] for i in range(0, len(self.sorted_patients), block_size)]
#         self.population = []
#         rng = np.random.default_rng()
#         for _ in range(pop_size):
#             individual = []
#             for block in blocks:
#                 idx = rng.permutation(len(block))
#                 individual.extend([block[j] for j in idx])
#             self.population.append(individual)
#         print(f"已生成包含{len(self.population)}个个体的种群")

#     # ------- 评估（GPU 粗排） -------
#     def evaluate_population_gpu(self, population: List[List[Any]]):
#         self._ensure_gpu_engine()
#         perms = torch.stack([self._individual_to_perm(ind) for ind in population], dim=0)
#         out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
#         fitness = out['fitness'].tolist()
#         heart_cnt = out['heart_cnt'].tolist()
#         angio_cnt = out['angio_cnt'].tolist()
#         weekend_cnt = out['weekend_cnt'].tolist()
#         device_cnt = out['device_cnt'].tolist()
#         violate_mask = out['any_violate_mask'].cpu().numpy()
#         indiv_viol_sets = []
#         for b in range(len(population)):
#             bad_idxs = np.where(violate_mask[b])[0].tolist()
#             bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
#             indiv_viol_sets.append(bad_cids)
#         results = []
#         for i, ind in enumerate(population):
#             results.append((ind, float(fitness[i]), int(heart_cnt[i]), int(angio_cnt[i]), int(device_cnt[i]), int(weekend_cnt[i]), indiv_viol_sets[i]))
#         return results

#     # ------- GA 主循环（GPU 版） -------
#     def evolve_gpu(self, generations=100, elite_size=5):
#         self._ensure_gpu_engine()
#         if len(self.population) == 0:
#             raise RuntimeError("种群为空，请先 initialize_population")
#         B = len(self.population)
#         pop = torch.stack([self._individual_to_perm(ind) for ind in self.population], dim=0)
#         for _ in range(generations):
#             out = self._gpu_engine.fitness_batch(pop, return_assignment=False)
#             fitness = out['fitness']
#             violate_mask = out['any_violate_mask']

#             elite_size = min(elite_size, B)
#             topk_vals, topk_idx = torch.topk(fitness, k=B, largest=True)
#             best_fitness = float(topk_vals[0].item())
#             self.current_generation += 1
#             self.fitness_history.append(best_fitness)
#             elites = pop.index_select(0, topk_idx[:elite_size])

#             parent_count = max(1, int(0.2 * B))
#             parents = pop.index_select(0, topk_idx[:parent_count])
#             parent_viol = violate_mask.index_select(0, topk_idx[:parent_count])

#             num_children = B - elite_size
#             if num_children > 0:
#                 p_idx1 = torch.randint(0, parent_count, (num_children,), device=DEVICE)
#                 p_idx2 = torch.randint(0, parent_count, (num_children,), device=DEVICE)
#                 P1 = parents.index_select(0, p_idx1)
#                 P2 = parents.index_select(0, p_idx2)
#                 Vmask_choice = parent_viol.index_select(0, p_idx1)
#                 children = self._ordered_crossover_batch_gpu(P1, P2)
#                 children = self._mutate_batch_gpu(children, Vmask_choice, self.current_generation)
#                 pop = torch.cat([elites, children], dim=0)
#             else:
#                 pop = elites.clone()

#             print(f"Generation {self.current_generation} | Best Fitness: {best_fitness:.2f}")

#         final_population = [self._tensor_row_to_cids(pop[i]) for i in range(pop.size(0))]
#         self.population = final_population
#         return final_population

#     # ------- OX 交叉（GPU） -------
#     @staticmethod
#     def _random_cuts(num_rows: int, N: int):
#         a = torch.randint(0, N, (num_rows,), device=DEVICE)
#         b = torch.randint(0, N, (num_rows,), device=DEVICE)
#         start = torch.minimum(a, b)
#         end = torch.maximum(a, b)
#         return start, end

#     def _ordered_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
#         C, N = P1.shape
#         start, end = self._random_cuts(C, N)
#         children = torch.empty_like(P1)
#         arangeN = torch.arange(N, device=DEVICE)
#         for i in range(C):
#             s = int(start[i]); e = int(end[i])
#             frag = P1[i, s:e+1]
#             isin = torch.isin(P2[i], frag)
#             tail = P2[i][~isin]
#             child = torch.empty(N, dtype=P1.dtype, device=DEVICE)
#             child[s:e+1] = frag
#             rest_pos = torch.cat([arangeN[:s], arangeN[e+1:]])
#             child[rest_pos] = tail[:rest_pos.numel()]
#             children[i] = child
#         return children

#     # ------- 变异（GPU）：定向交换 + 基础交换 + 贪婪分组 -------
#     def _mutate_batch_gpu(self, X: torch.Tensor, parent_violate_mask: torch.Tensor, current_gen: int,
#                           base_swap_prob: float = 0.95, greedy_prob: float = 0.5) -> torch.Tensor:
#         C, N = X.shape
#         # 定向：修复父代违规 — 交换违规基因与±400范围的随机位
#         for i in range(C):
#             viol_mask = parent_violate_mask[i]
#             if torch.any(viol_mask):
#                 idxs = torch.nonzero(viol_mask, as_tuple=False).flatten()
#                 violator_idx = int(idxs[torch.randint(0, idxs.numel(), (1,), device=DEVICE)])
#                 low = max(0, violator_idx - 400)
#                 high = min(N - 1, violator_idx + 400)
#                 if high - low >= 1:
#                     cand = int(torch.randint(low, high + 1, (1,), device=DEVICE))
#                     cand = cand if cand != violator_idx else (low if violator_idx < high else high)
#                     tmp = X[i, violator_idx].clone(); X[i, violator_idx] = X[i, cand]; X[i, cand] = tmp
#         # 基础随机交换
#         use_range_limit = (current_gen <= 10000)
#         probs = torch.rand((C,), device=DEVICE)
#         for i in range(C):
#             if probs[i].item() < base_swap_prob:
#                 idx1 = int(torch.randint(0, N, (1,), device=DEVICE))
#                 if use_range_limit:
#                     low = max(0, idx1 - 400)
#                     high = min(N - 1, idx1 + 400)
#                     if high - low >= 1: 
#                         idx2 = int(torch.randint(low, high + 1, (1,), device=DEVICE))
#                         idx2 = idx2 if idx2 != idx1 else (low if idx1 < high else high)
#                     else:
#                         idx2 = int(torch.randint(0, N, (1,), device=DEVICE))
#                         while idx2 == idx1:
#                             idx2 = int(torch.randint(0, N, (1,), device=DEVICE))
#                 else:
#                     idx2 = int(torch.randint(0, N, (1,), device=DEVICE))
#                     while idx2 == idx1:
#                         idx2 = int(torch.randint(0, N, (1,), device=DEVICE))
#                 tmp = X[i, idx1].clone(); X[i, idx1] = X[i, idx2]; X[i, idx2] = tmp
#         # 贪婪分组变异
#         probs2 = torch.rand((C,), device=DEVICE)
#         for i in range(C):
#             if probs2[i].item() < greedy_prob and N >= 50:
#                 start = int(torch.randint(0, N - 49, (1,), device=DEVICE))
#                 end = min(N - 1, start + int(torch.randint(20, 51, (1,), device=DEVICE)))
#                 self._greedy_cluster_mutation_gpu_inplace(X, i, start, end)
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

#     # ------- 单个体适配度（GPU 粗排） -------
#     def calculate_fitness(self, schedule):
#         if not schedule:
#             return -float('inf'), 0, 0, 0, 0, set()
#         self._ensure_gpu_engine()
#         perms = self._individual_to_perm(schedule).unsqueeze(0)
#         out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
#         fitness = float(out['fitness'][0].item())
#         heart_cnt = int(out['heart_cnt'][0].item())
#         angio_cnt = int(out['angio_cnt'][0].item())
#         device_cnt = int(out['device_cnt'][0].item())
#         weekend_cnt = int(out['weekend_cnt'][0].item())
#         mask = out['any_violate_mask'][0].cpu().numpy()
#         bad_idxs = np.where(mask)[0].tolist()
#         bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
#         return fitness, heart_cnt, angio_cnt, device_cnt, weekend_cnt, bad_cids

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
#                         m.add_exam(system.current_date, start_time, duration, exam_type, p)
#                     except Exception as e:
#                         print(f"排程错误: {e}")
#         return system

#     def save_state(self, filename):
#         state = {
#             'current_generation': self.current_generation,
#             'population': [[list(cid) for cid in ind] for ind in self.population],
#             'fitness_history': self.fitness_history,
#             'block_start_date': self.block_start_date.strftime('%Y-%m-%d') if self.block_start_date else None,
#         }
#         with open(filename, 'w') as f:
#             json.dump(state, f, indent=2)
#         print(f"✅ 已保存第{self.current_generation}代状态")

#     @classmethod
#     def load_state(cls, filename, patients, machine_exam_map, pop_size=50):
#         if not os.path.exists(filename):
#             return None
#         with open(filename, 'r') as f:
#             state = json.load(f)
#         valid_population = []
#         for serialized_ind in state['population']:
#             try:
#                 individual = [tuple(cid) for cid in serialized_ind]
#                 if all(cid in patients for cid in individual):
#                     valid_population.append(individual)
#             except Exception:
#                 pass
#         block_start_date = None
#         if state.get('block_start_date'):
#             block_start_date = datetime.strptime(state['block_start_date'], '%Y-%m-%d').date()
#         opt = cls(patients, machine_exam_map, pop_size, block_start_date)
#         opt.population = valid_population
#         opt.current_generation = state['current_generation']
#         opt.fitness_history = state['fitness_history']
#         if len(valid_population) < pop_size:
#             print(f"⚠️ 补充{pop_size - len(valid_population)}个新个体")
#             opt.initialize_population(pop_size - len(valid_population))
#             opt.population = valid_population + opt.population
#         print(f"成功加载第{opt.current_generation}代状态")
#         return opt

# # ===================== 导出 Excel =====================

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

# # ===================== main =====================

# def main():
#     try:
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
#         duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
#         device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
#         for f in [patient_file, duration_file, device_constraint_file]:
#             if not os.path.exists(f):
#                 print(f"❌ 错误：找不到文件 {f}")
#                 try:
#                     input("按回车退出...")
#                 except Exception:
#                     pass
#                 return
#         print("✓ 所有数据文件均已找到。")

#         print("正在导入数据...")
#         patients = import_data(patient_file, duration_file)
#         machine_exam_map = import_device_constraints(device_constraint_file)

#         pop_size = 50
#         print("\n===== 启动遗传算法优化（GPU 粗排 + GPU 主体） =====")
#         optimizer = BlockGeneticOptimizer(patients, machine_exam_map, pop_size=pop_size)
#         optimizer.initialize_population(pop_size=pop_size)
#         print(f"✓ 已生成初始种群 (大小: {len(optimizer.population)})")

#         generations_to_run = 100
#         print(f"\n开始 {generations_to_run} 代进化...")
#         t0 = time.perf_counter()
#         final_population = optimizer.evolve_gpu(generations=generations_to_run, elite_size=5)
#         total_evolution_time = time.perf_counter() - t0
#         print(f"✓ 进化完成，耗时: {total_evolution_time:.2f}s")

#         print("正在评估最终种群以选择最佳个体（GPU 适配度）...")
#         scored_final = []
#         for ind in final_population:
#             fitness, *_ = optimizer.calculate_fitness(ind)
#             scored_final.append((ind, fitness))
#         if scored_final:
#             scored_final.sort(key=lambda x: x[1], reverse=True)
#             best_individual = scored_final[0][0]
#             best_fitness = scored_final[0][1]
#             print(f"✓ 最佳个体 Fitness: {best_fitness:.2f}")
#             out_dir = 'output_schedules'; os.makedirs(out_dir, exist_ok=True)
#             ts = datetime.now().strftime('%Y%m%d_%H%M%S')
#             xlsx = os.path.join(out_dir, f'final_schedule_{ts}_fit_{best_fitness:.0f}.xlsx')
#             final_system = optimizer.generate_schedule(best_individual)
#             export_schedule(final_system, patients, xlsx)
#             print(f"✓ 已导出最佳排程至 {xlsx}")
#         else:
#             print("⚠️ 最终种群为空，无法选择最佳个体。")

#     except Exception as e:
#         print(f"运行时错误: {e}")
#         traceback.print_exc()
#     finally:
#         try:
#             input("按回车退出...")
#         except Exception:
#             pass


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()


###########################################深度使用GPU，全部GPU化

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
POPULATION_FILE = 'population_state.json'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_LONG = torch.long
DTYPE_FLOAT = torch.float32

# ===================== 工具函数 =====================

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

# ===================== 导出所需（CPU 精排仅用于导出，不改变 GA 思想） =====================
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


# ----------------------------------------------------------------
# 优化点 5: 向量化 _compute_order_in_bin_batch
# 原始的 _compute_order_in_bin_batch (和 _compute_order_in_bin_row) 
# 在CPU端循环 (for b in range(B))，为种群中每个个体启动一次GPU核函数。
#
# 新实现：
# 1. 使用唯一的排序键 (bin_idx * (N+1) + arange) 来实现按bin_idx的稳定排序。
# 2. 在批处理维度 (B) 上并行计算每个bin内的排序 (rank)。
# 3. 寻找每个bin的起始位置，计算累积最大值 (cummax) 以找到每个元素所属bin的起始点。
# 4. 计算排序后的rank (arng - last_start_pos)。
# 5. 使用 scatter_ 将rank"反排序"回其原始位置。
# 6. 移除了 _compute_order_in_bin_row 函数。
# ----------------------------------------------------------------
def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
    B, N = bin_idx_batch.shape
    arng = torch.arange(N, device=DEVICE)
    arng_expanded = arng.expand(B, N)
    
    # 创建一个唯一的键来进行稳定排序：先按 bin_idx 排序，再按原始位置排序
    # (N+1) 确保 bin_idx 占主导地位
    key = bin_idx_batch.long() * (N + 1) + arng_expanded
    
    # 对键进行排序，获取排序索引
    _, sort_idx = torch.sort(key, dim=1)
    
    # 根据排序索引获取排序后的 bin_idx
    bin_sorted = bin_idx_batch.gather(1, sort_idx)
    
    # 找到每个bin的起始位置 (在排序后的张量中)
    is_start = torch.zeros_like(bin_sorted, dtype=torch.bool)
    is_start[:, 1:] = bin_sorted[:, 1:] != bin_sorted[:, :-1]
    is_start[:, 0] = True  # 每行的第一个元素总是一个新bin的开始
    
    # 获取起始位置的索引
    start_pos = torch.where(is_start, arng_expanded, -1)
    
    # 向前传播最后一个起始位置的索引
    last_start_pos = torch.cummax(start_pos, dim=1)[0]
    
    # 计算在排序后张量中的bin内rank
    rank_in_sorted = arng_expanded - last_start_pos
    
    # 使用 scatter_ (gather的反操作) 将rank放回其在原始perm中的位置
    order_idx = torch.empty_like(rank_in_sorted, dtype=DTYPE_LONG)
    order_idx.scatter_(1, sort_idx, rank_in_sorted)
    
    return order_idx


class _GPUMatrixFitnessBatch:
    """(来自 GPU_exp1，已修改支持机器换模损耗)"""
    def __init__(self, *,
                 weekday_machine_minutes: torch.Tensor,  # [7,M]
                 start_weekday: int,                    # 0..6
                 patient_durations: torch.Tensor,       # [N]
                 reg_day_offsets: torch.Tensor,         # [N]
                 is_self_selected: torch.Tensor,        # [N]
                 has_contrast: torch.Tensor,            # [N]
                 has_heart: torch.Tensor,               # [N]
                 has_angio: torch.Tensor,               # [N]
                 patient_main_type_id: torch.Tensor,    # [N] <<< 修改：接收患者主要检查类型ID
                 patient_exam_mask: torch.Tensor | None,   # [N,E] or None
                 machine_exam_mask: torch.Tensor | None    # [M,E] or None
                 ):
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
        penalty = pos_wait * (is_self * SELF_SELECTED_PENALTY + non_self * NON_SELF_PENALTY) + neg_wait * LOGICAL
        return penalty

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
        ok_wd_h = (weekday_batch == 1) | (weekday_batch == 3)
        ok_mc_h = (assigned_machine_batch == 3)
        heart_violate = heart_mask & (~(ok_wd_h & ok_mc_h))
        
        angio_mask = self.has_angio.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        ok_wd_a = (weekday_batch == 0) | (weekday_batch == 2) | (weekday_batch == 4)
        ok_mc_a = (assigned_machine_batch == 1)
        angio_violate = angio_mask & (~(ok_wd_a & ok_mc_a))
        
        weekend = (weekday_batch == 5) | (weekday_batch == 6)
        contrast_mask = self.has_contrast.index_select(0, perms.reshape(-1)).reshape(perms.shape)
        weekend_violate = contrast_mask & weekend
        return heart_violate, angio_violate, weekend_violate

    def _penalty_special_rules(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        total_penalty_per_patient = p_wait + p_dev + p_spec + p_tran  # [B,N]
        fitness = - total_penalty_per_patient.sum(dim=1)              # [B]

        out = {
            "fitness": fitness,
            "assigned_day": assigned_day_batch if return_assignment else None,
            "assigned_machine": assigned_machine_batch if return_assignment else None,
            "order_in_machine": _compute_order_in_bin_batch(bin_idx_batch) if return_assignment else None,
            "heart_cnt": heart_v_i.sum(dim=1),
            "angio_cnt": angio_v_i.sum(dim=1),
            "weekend_cnt": weekend_v_i.sum(dim=1),
            "device_cnt": (p_dev > 0).sum(dim=1),
            "any_violate_mask": (heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0))
        }
        return out


# ===================== GA 主体（GPU 版） =====================
class BlockGeneticOptimizer:
    def __init__(self, patients, machine_exam_map, pop_size=50, block_start_date=None):
        self.patients = patients
        self.machine_exam_map = machine_exam_map
        self.population: List[List[Any]] = []
        self.fitness_history: List[float] = []
        self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
        self.current_generation = 0
        self.pop_size = pop_size
        self.block_start_date = block_start_date
        self._gpu_engine = None
        self._cid_to_idx = None
        self._idx_to_cid = None
        self._patient_main_exam_id = None
        self._E = None

    # ------- GPU 引擎准备 -------
    def _ensure_gpu_engine(self):
        if self._gpu_engine is not None:
            return

        # 1. 建立索引映射 (CID <-> Index)
        idx_to_cid = list(self.sorted_patients)
        cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
        self._idx_to_cid = idx_to_cid
        self._cid_to_idx = cid_to_idx
        N = len(idx_to_cid)

        # 2. 初始化基础张量
        patient_durations = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        reg_day_offsets = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        is_self_selected = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_contrast = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_heart = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_angio = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        
        # [新增] 初始化主要类型ID张量 (用于换模惩罚)
        patient_main_type_id = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)

        # 3. 建立全局检查类型映射 (Exam Name -> ID)
        exam_set = set()
        for cid in idx_to_cid:
            for _, exam_type, _, _ in self.patients[cid]['exams']:
                exam_set.add(clean_exam_name(exam_type))
        # 同时也加上机器限制里的类型，确保字典完整
        for m, exams in self.machine_exam_map.items():
            for e in exams:
                exam_set.add(clean_exam_name(e))
        
        exam_list = sorted(list(exam_set))
        exam_to_eidx = {e: i for i, e in enumerate(exam_list)}
        E = len(exam_list)

        patient_exam_mask = torch.zeros((N, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
        machine_exam_mask = torch.zeros((MACHINE_COUNT, E), dtype=torch.bool, device=DEVICE) if E > 0 else None

        base_date = self.block_start_date if self.block_start_date else START_DATE.date()
        start_weekday = base_date.isoweekday() - 1  # 0..6

        # 4. 遍历填充患者数据
        for i, cid in enumerate(idx_to_cid):
            p = self.patients[cid]
            total_minutes = 0
            any_contrast = False
            any_heart = False
            any_angio = False
            
            # 临时列表，用于确定患者的主要检查类型
            p_exam_types = []

            if patient_exam_mask is not None:
                for _, exam_type, duration, _ in p['exams']:
                    et = clean_exam_name(exam_type)
                    total_minutes += int(round(float(duration)))
                    eidx = exam_to_eidx.get(et, None)
                    if eidx is not None:
                        patient_exam_mask[i, eidx] = True
                    p_exam_types.append(et)
                    if '增强' in et: any_contrast = True
                    if '心脏' in et: any_heart = True
                    if '造影' in et: any_angio = True
            else:
                for _, exam_type, duration, _ in p['exams']:
                    et = clean_exam_name(exam_type)
                    total_minutes += int(round(float(duration)))
                    p_exam_types.append(et)
                    if '增强' in et: any_contrast = True
                    if '心脏' in et: any_heart = True
                    if '造影' in et: any_angio = True

            # [新增逻辑] 确定患者的"主要类型ID"
            # 取第一个检查项目作为判断换模的依据
            if p_exam_types:
                main_type = p_exam_types[0]
                patient_main_type_id[i] = exam_to_eidx.get(main_type, 0)
            else:
                patient_main_type_id[i] = 0

            # [移除] 旧的 switch_penalty 计算 (switch_penalty[i] = ...) 已被彻底删除

            patient_durations[i] = max(1, total_minutes)
            reg_day_offsets[i] = (p['reg_date'] - base_date).days
            is_self_selected[i] = bool(p.get('is_self_selected', False))
            has_contrast[i] = any_contrast
            has_heart[i] = any_heart
            has_angio[i] = any_angio

        # 5. 填充机器掩码
        if machine_exam_mask is not None:
            for mid in range(MACHINE_COUNT):
                for e in self.machine_exam_map.get(mid, []):
                    et = clean_exam_name(e)
                    eidx = exam_to_eidx.get(et, None)
                    if eidx is not None:
                        machine_exam_mask[mid, eidx] = True

        weekday_machine_minutes = _weekday_minutes_matrix_from_end_hours(MACHINE_COUNT)

        # ==================== 关键修复 ====================
        # 必须将局部变量 patient_main_type_id 保存到 self 属性中
        # 否则后续的变异算子 (_mutate_batch 等) 调用 self._patient_main_exam_id 时会报错
        self._patient_main_exam_id = patient_main_type_id 
        # =================================================

        # 6. 初始化 GPU 引擎
        self._gpu_engine = _GPUMatrixFitnessBatch(
            weekday_machine_minutes=weekday_machine_minutes,
            start_weekday=start_weekday,
            patient_durations=patient_durations,
            reg_day_offsets=reg_day_offsets,
            is_self_selected=is_self_selected,
            has_contrast=has_contrast,
            has_heart=has_heart,
            has_angio=has_angio,
            patient_main_type_id=patient_main_type_id, # [新增参数] 传入类型ID
            patient_exam_mask=patient_exam_mask,
            machine_exam_mask=machine_exam_mask,
            # switch_penalty=switch_penalty [已移除旧参数]
        )

    # ------- 索引 ↔ cid -------
    def _individual_to_perm(self, individual: List[Any]) -> torch.Tensor:
        return torch.tensor([self._cid_to_idx[cid] for cid in individual], dtype=DTYPE_LONG, device=DEVICE)

    def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
        return [self._idx_to_cid[int(x)] for x in row.tolist()]

    def initialize_population(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        
        N = len(self.sorted_patients)
        if N == 0:
            print("警告：患者列表为空，无法初始化种群。")
            return

        # 确保 _idx_to_cid 已准备好，以便最后转换
        # 注意：这里不能调用 _ensure_gpu_engine()，因为它依赖于一个已排序的患者列表
        # 但我们可以在这里安全地构建 _idx_to_cid
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
            self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}

        indices = torch.arange(N, device=DEVICE)
        block_size = max(30, N // 20)
        
        pop_indices = torch.empty((pop_size, N), dtype=DTYPE_LONG, device=DEVICE)
        
        # 预先生成所有随机数
        rand_matrices = torch.rand(pop_size, N, device=DEVICE)
        
        # CPU 循环遍历 *块* (非常少)，而不是个体
        for i in range(0, N, block_size):
            end = min(i + block_size, N)
            block_len = end - i
            if block_len == 0:
                continue
            block_rand = rand_matrices[:, i:end]
            block_perm_idx = torch.argsort(block_rand, dim=1)
            block_indices = indices[i:end]
            block_indices_expanded = block_indices.expand(pop_size, -1)
            
            # 使用 gather 并行应用所有排列
            pop_indices[:, i:end] = torch.gather(block_indices_expanded, 1, block_perm_idx)

        # 批量将GPU索引张量转换回CPU上的CIDs列表
        pop_indices_cpu = pop_indices.cpu().numpy()
        
        # 修复：
        # 原始的 np.array(self._idx_to_cid) 会创建 (N, 2) 的字符串数组
        # 导致 self.population 包含 np.ndarray (不可哈希)
        # 
        # 新修复：直接使用 Python 列表索引，确保个体是 tuple (可哈希)
        idx_to_cid_list = self._idx_to_cid # 这是 List[tuple]
        self.population = [
            [idx_to_cid_list[idx] for idx in row] for row in pop_indices_cpu
        ]
        print(f"已生成包含{len(self.population)}个个体的种群")


    # ------- 评估（GPU 粗排） -------
    def evaluate_population_gpu(self, population: List[List[Any]]):
        self._ensure_gpu_engine()
        perms = torch.stack([self._individual_to_perm(ind) for ind in population], dim=0)
        out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
        fitness = out['fitness'].tolist()
        heart_cnt = out['heart_cnt'].tolist()
        angio_cnt = out['angio_cnt'].tolist()
        weekend_cnt = out['weekend_cnt'].tolist()
        device_cnt = out['device_cnt'].tolist()
        violate_mask = out['any_violate_mask'].cpu().numpy()
        indiv_viol_sets = []
        for b in range(len(population)):
            bad_idxs = np.where(violate_mask[b])[0].tolist()
            bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
            indiv_viol_sets.append(bad_cids)
        results = []
        for i, ind in enumerate(population):
            results.append((ind, float(fitness[i]), int(heart_cnt[i]), int(angio_cnt[i]), int(device_cnt[i]), int(weekend_cnt[i]), indiv_viol_sets[i]))
        return results

    # ------- GA 主循环（GPU 版） -------
    def evolve_gpu(self, generations=100, elite_size=5):
        self._ensure_gpu_engine()
        if len(self.population) == 0:
            raise RuntimeError("种群为空，请先 initialize_population")
        B = len(self.population)
        pop = torch.stack([self._individual_to_perm(ind) for ind in self.population], dim=0)
        for gen_idx in range(generations):
            out = self._gpu_engine.fitness_batch(pop, return_assignment=False)
            fitness = out['fitness']
            violate_mask = out['any_violate_mask']

            elite_size = min(elite_size, B)
            topk_vals, topk_idx = torch.topk(fitness, k=B, largest=True)
            best_fitness = float(topk_vals[0].item())
            self.current_generation += 1
            self.fitness_history.append(best_fitness)
            elites = pop.index_select(0, topk_idx[:elite_size])

            parent_count = max(1, int(0.2 * B))
            parents = pop.index_select(0, topk_idx[:parent_count])
            parent_viol = violate_mask.index_select(0, topk_idx[:parent_count])

            num_children = B - elite_size
            if num_children > 0:
                p_idx1 = torch.randint(0, parent_count, (num_children,), device=DEVICE)
                p_idx2 = torch.randint(0, parent_count, (num_children,), device=DEVICE)
                P1 = parents.index_select(0, p_idx1)
                P2 = parents.index_select(0, p_idx2)
                Vmask_choice = parent_viol.index_select(0, p_idx1)
                children = self._ordered_crossover_batch_gpu(P1, P2)
                children = self._mutate_batch_gpu(children, Vmask_choice, self.current_generation)
                pop = torch.cat([elites, children], dim=0)
            else:
                pop = elites.clone()
            
            if (gen_idx + 1) % 50 == 0:
                print(f"Generation {self.current_generation} | Best Fitness: {best_fitness:.2f}")

        final_population = [self._tensor_row_to_cids(pop[i]) for i in range(pop.size(0))]
        self.population = final_population
        return final_population

    # ------- OX 交叉（GPU） -------
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
        
        # 2. (C,N) 掩码，标记 P1 的片段
        mask_frag = (arangeN >= s_exp) & (arangeN <= e_exp)
        
        # 3. (C,N) 子代张量，用 P1 片段填充
        children = torch.full_like(P1, -1)
        children[mask_frag] = P1[mask_frag]
        
        # 4. 向量化 row-wise isin
        # 检查 P2 (C,N) 中的每个元素是否存在于 P1 (C,N) 的片段 (mask_frag) 中
        P2_expanded = P2.unsqueeze(2)  # (C, N, 1)
        P1_expanded = P1.unsqueeze(1)  # (C, 1, N)
        
        # (C, N, N), [i, j, k] = (P2[i, j] == P1[i, k])
        equality_matrix = (P2_expanded == P1_expanded)
        
        # (C, 1, N), 广播 P1 片段掩码
        mask_frag_expanded = mask_frag.unsqueeze(1)
        
        # (C, N)
        # [i, j] = any(P2[i, j] == P1[i, k] for k where mask_frag[i, k])
        isin_matrix = (equality_matrix & mask_frag_expanded).any(dim=2)
        
        # 5. (C,N) 掩码，标记 P2 中 *不在* P1 片段中的元素
        mask_tail = ~isin_matrix
        
        # 6. (K,) 一维张量，包含所有需要填充的元素 (按 P2 的 C,N 顺序)
        P2_tails_flat = P2[mask_tail]
        
        # 7. (C,N) 掩码，标记子代中需要填充的空位
        mask_fill = (children == -1)
        
        # 8. 一次性填充
        children[mask_fill] = P2_tails_flat
        
        return children
    
    def _mutate_step1_violations(self, X: torch.Tensor, parent_violate_mask: torch.Tensor) -> torch.Tensor:
        C, N = X.shape
        any_viol = torch.any(parent_violate_mask, dim=1)
        viol_rows_idx = torch.nonzero(any_viol, as_tuple=False).flatten()
        R = viol_rows_idx.numel()
        if R == 0:
            return X

        # (R, N) - 仅获取有违规的行的掩码
        viol_mask_subset = parent_violate_mask[viol_rows_idx]
        
        # (R,) - 批量随机选择一个违规索引
        viol_idx_in_row = torch.multinomial(viol_mask_subset.float(), 1, replacement=True).flatten()
        
        # 批量计算邻域
        low = torch.clamp(viol_idx_in_row - 400, min=0)
        high = torch.clamp(viol_idx_in_row + 400, max=N-1)
        range_size = high - low + 1
        
        # 修复 range_size == 0 (虽然不太可能)
        range_size = torch.where(range_size <= 0, 1, range_size)
        
        # 批量生成随机偏移
        rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
        cand_idx_in_row = low + rand_offset
        
        # 确保 cand != viol
        cand_idx_in_row = torch.where(
            (cand_idx_in_row == viol_idx_in_row) & (range_size > 1), 
            torch.where(viol_idx_in_row == low, low + 1, low), # 避免 cand == viol
            cand_idx_in_row
        )
        cand_idx_in_row = torch.clamp(cand_idx_in_row, 0, N-1) # 确保有效

        # 批量交换
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

        # 批量生成 idx1
        idx1 = torch.randint(0, N, (R,), device=DEVICE)
        
        if use_range_limit:
            low = torch.clamp(idx1 - 0, min=0)
            high = torch.clamp(idx1 + 400, max=N-1)
            range_size = high - low + 1
            range_size = torch.where(range_size <= 0, 1, range_size) # 修复
            rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
            idx2 = low + rand_offset
            idx2 = torch.where(
                (idx2 == idx1) & (range_size > 1), 
                torch.where(idx1 == low, low + 1, low),
                idx2
            )
            idx2 = torch.clamp(idx2, 0, N-1) # 确保有效
        else:
            idx2 = torch.randint(0, N, (R,), device=DEVICE)
            # 简单处理碰撞
            idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)
        
        # 批量交换
        val1 = X[rows_to_swap_idx, idx1]
        val2 = X[rows_to_swap_idx, idx2]
        X[rows_to_swap_idx, idx1] = val2
        X[rows_to_swap_idx, idx2] = val1
        
        return X

    def _mutate_step3_greedy_cluster(self, X: torch.Tensor, greedy_prob: float = 0.5) -> torch.Tensor:
        C, N = X.shape
        if N < 50:
            return X
            
        probs2 = torch.rand(C, device=DEVICE)
        rows_to_greedy_mask = (probs2 < greedy_prob)
        rows_to_greedy_idx = torch.nonzero(rows_to_greedy_mask, as_tuple=False).flatten()
        R = rows_to_greedy_idx.numel()
        if R == 0:
            return X
            
        # 批量生成窗口
        start = torch.randint(0, N - 49, (R,), device=DEVICE)
        end_offset = torch.randint(20, 51, (R,), device=DEVICE)
        end = torch.clamp(start + end_offset, max=N-1)
        
        # *妥协*：CPU 循环只遍历需要操作的行 (R)，而不是全部 (C)
        start_cpu = start.cpu().tolist()
        end_cpu = end.cpu().tolist()
        
        for i, row_idx in enumerate(rows_to_greedy_idx):
            s = start_cpu[i]
            e = end_cpu[i]
            # 调用已向量化的行内操作
            self._greedy_cluster_mutation_gpu_inplace(X, int(row_idx), s, e)
            
        return X

    def _mutate_batch_gpu(self, X: torch.Tensor, parent_violate_mask: torch.Tensor, current_gen: int,
                          base_swap_prob: float = 0.95, greedy_prob: float = 0.5) -> torch.Tensor:
        
        X = self._mutate_step1_violations(X, parent_violate_mask)
        X = self._mutate_step2_base_swap(X, current_gen, base_swap_prob)
        #X = self._mutate_step3_greedy_cluster(X, greedy_prob)
        
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

    # ------- 单个体适配度（GPU 粗排） -------
    def calculate_fitness(self, schedule):
        if not schedule:
            return -float('inf'), 0, 0, 0, 0, set()
        self._ensure_gpu_engine()
        perms = self._individual_to_perm(schedule).unsqueeze(0)
        out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
        fitness = float(out['fitness'][0].item())
        heart_cnt = int(out['heart_cnt'][0].item())
        angio_cnt = int(out['angio_cnt'][0].item())
        device_cnt = int(out['device_cnt'][0].item())
        weekend_cnt = int(out['weekend_cnt'][0].item())
        mask = out['any_violate_mask'][0].cpu().numpy()
        bad_idxs = np.where(mask)[0].tolist()
        bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
        return fitness, heart_cnt, angio_cnt, device_cnt, weekend_cnt, bad_cids

    # ------- 导出（可选） -------
    def generate_schedule(self, individual):
        system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
        for cid in individual:
            p = self.patients.get(cid)
            if p and not p['scheduled']:
                for exam in p['exams']:
                    exam_type = clean_exam_name(exam[1])
                    duration = exam[2]
                    try:
                        m, start_time = system.find_available_slot(duration, exam_type, p)
                        m.add_exam(system.current_date, start_time, duration, exam_type, p)
                    except Exception as e:
                        print(f"排程错误: {e}")
        return system

    def save_state(self, filename):
        state = {
            'current_generation': self.current_generation,
            'population': [[list(cid) for cid in ind] for ind in self.population],
            'fitness_history': self.fitness_history,
            'block_start_date': self.block_start_date.strftime('%Y-%m-%d') if self.block_start_date else None,
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✅ 已保存第{self.current_generation}代状态")

    @classmethod
    def load_state(cls, filename, patients, machine_exam_map, pop_size=50):
        if not os.path.exists(filename):
            return None
        with open(filename, 'r') as f:
            state = json.load(f)
        valid_population = []
        for serialized_ind in state['population']:
            try:
                individual = [tuple(cid) for cid in serialized_ind]
                if all(cid in patients for cid in individual):
                    valid_population.append(individual)
            except Exception:
                pass
        block_start_date = None
        if state.get('block_start_date'):
            block_start_date = datetime.strptime(state['block_start_date'], '%Y-%m-%d').date()
        opt = cls(patients, machine_exam_map, pop_size, block_start_date)
        opt.population = valid_population
        opt.current_generation = state['current_generation']
        opt.fitness_history = state['fitness_history']
        if len(valid_population) < pop_size:
            print(f"⚠️ 补充{pop_size - len(valid_population)}个新个体")
            new_pop_obj = cls(patients, machine_exam_map, pop_size, block_start_date)
            new_pop_obj.initialize_population(pop_size - len(valid_population))
            opt.population = valid_population + new_pop_obj.population
        print(f"成功加载第{opt.current_generation}代状态")
        return opt

# ===================== 导出 Excel =====================

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

# ===================== main =====================

def main():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
        for f in [patient_file, duration_file, device_constraint_file]:
            if not os.path.exists(f):
                print(f"❌ 错误：找不到文件 {f}")
                try:
                    input("按回车退出...")
                except Exception:
                    pass
                return
        print("✓ 所有数据文件均已找到。")

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        pop_size = 50
        print("\n===== 启动遗传算法优化（GPU 粗排 + GPU 主体） =====")
        optimizer = BlockGeneticOptimizer(patients, machine_exam_map, pop_size=pop_size)
        
        t0_init = time.perf_counter()
        optimizer.initialize_population(pop_size=pop_size)
        t_init = time.perf_counter() - t0_init
        print(f"✓ 已生成初始种群 (大小: {len(optimizer.population)})，耗时: {t_init:.4f}s")


        generations_to_run = 20000
        print(f"\n开始 {generations_to_run} 代进化...")
        t0 = time.perf_counter()
        final_population = optimizer.evolve_gpu(generations=generations_to_run, elite_size=5)
        total_evolution_time = time.perf_counter() - t0
        print(f"\n✓ 进化完成，总耗时: {total_evolution_time:.2f}s ({total_evolution_time / generations_to_run:.4f} s/gen)")


        print("正在评估最终种群以选择最佳个体（GPU 适配度）...")
        scored_final = []
        for ind in final_population:
            fitness, *_ = optimizer.calculate_fitness(ind)
            scored_final.append((ind, fitness))
        if scored_final:
            scored_final.sort(key=lambda x: x[1], reverse=True)
            best_individual = scored_final[0][0]
            best_fitness = scored_final[0][1]
            print(f"✓ 最佳个体 Fitness: {best_fitness:.2f}")
            out_dir = 'output_schedules'; os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            xlsx = os.path.join(out_dir, f'final_schedule_{ts}_fit_{best_fitness:.0f}.xlsx')
            final_system = optimizer.generate_schedule(best_individual)
            export_schedule(final_system, patients, xlsx)
            print(f"✓ 已导出最佳排程至 {xlsx}")
        else:
            print("⚠️ 最终种群为空，无法选择最佳个体。")

    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()
    finally:
        try:
            input("按回车退出...")
        except Exception:
            pass


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
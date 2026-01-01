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
# import torch.nn as nn
# import torch.nn.functional as F

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

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("⚠️ 未检测到 GPU，将使用 CPU（速度极慢）")
    
# DTYPE_LONG = torch.long
# DTYPE_FLOAT = torch.float32


# # ===================== 论文式 Step2 位置选择器（注意力） =====================
# class Step2PositionAttentionPolicy(nn.Module):
#     """Reference-conditioned attention policy for selecting idx2 in base swap.

#     输入（batch = R 行，每行 W 个候选）：
#       ref_type: Long[R]            # 参考位点（idx1）的主检查类型 id（可为 -1）
#       ref_pos_norm: Float[R,1]     # idx1 / (N-1)
#       cand_type: Long[R,W]         # 候选位置上元素的主检查类型 id（可为 -1）
#       cand_pos_norm: Float[R,W,1]  # cand_pos / (N-1)
#       rel_offset_norm: Float[W]    # offsets / window
#       cand_mask: Bool[R,W]         # True 表示候选有效且可选（!= idx1 且在范围内等）

#     输出：
#       logits: Float[R,W]           # 未做 mask 的 logits（调用方可 masked_fill）
#     """
#     def __init__(
#         self,
#         num_exam_types: int,
#         d_model: int = 128,
#         d_emb: int = 32,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         # exam_type 可能为 -1（dummy/unknown），这里统一 shift +1 映射到 [0..num_exam_types]
#         self.num_exam_types = int(num_exam_types)
#         self.type_emb = nn.Embedding(self.num_exam_types + 1, d_emb)  # 0=unknown/pad

#         # Embf / Embc（两套参数）
#         self.emb_f = nn.Sequential(
#             nn.Linear(d_emb + 1, d_model),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model),
#         )
#         self.emb_c = nn.Sequential(
#             nn.Linear(d_emb + 2, d_model),  # cand_pos_norm + rel_offset_norm
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model),
#         )

#         # attention: u_i^H = z_A^T tanh(W_A [h_i; h_f])
#         self.attn_w = nn.Linear(2 * d_model, d_model)
#         self.attn_z = nn.Linear(d_model, 1, bias=False)

#         # q = FFN([c; h_f])
#         self.q_ffn = nn.Sequential(
#             nn.Linear(2 * d_model, d_model),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, d_model),
#         )

#         # u_i = z_B^T tanh(h_i + q)
#         self.out_z = nn.Linear(d_model, 1, bias=False)

#     @staticmethod
#     def _shift_type(x: torch.Tensor) -> torch.Tensor:
#         # [-1, ..] -> [0, ..]
#         return torch.clamp(x + 1, min=0)

#     def forward(
#         self,
#         ref_type: torch.Tensor,        # [R]
#         ref_pos_norm: torch.Tensor,    # [R,1]
#         cand_type: torch.Tensor,       # [R,W]
#         cand_pos_norm: torch.Tensor,   # [R,W,1]
#         rel_offset_norm: torch.Tensor, # [W]
#         cand_mask: torch.Tensor,       # [R,W]
#     ) -> torch.Tensor:
#         R = ref_type.shape[0]
#         W = cand_type.shape[1]

#         ref_type_s = self._shift_type(ref_type).long()
#         cand_type_s = self._shift_type(cand_type).long()

#         ref_e = self.type_emb(ref_type_s)    # [R, d_emb]
#         cand_e = self.type_emb(cand_type_s)  # [R, W, d_emb]

#         hf_in = torch.cat([ref_e, ref_pos_norm], dim=-1)  # [R, d_emb+1]
#         h_f = self.emb_f(hf_in)                           # [R, d_model]

#         rel = rel_offset_norm.view(1, W, 1).expand(R, W, 1)           # [R, W, 1]
#         hc_in = torch.cat([cand_e, cand_pos_norm, rel], dim=-1)       # [R, W, d_emb+2]
#         h_i = self.emb_c(hc_in)                                       # [R, W, d_model]

#         h_f_ex = h_f.unsqueeze(1).expand(R, W, h_f.shape[-1])         # [R, W, d_model]
#         att_in = torch.cat([h_i, h_f_ex], dim=-1)                      # [R, W, 2d]
#         att_h = torch.tanh(self.attn_w(att_in))                        # [R, W, d_model]
#         att_logits = self.attn_z(att_h).squeeze(-1)                    # [R, W]
#         att_logits = att_logits.masked_fill(~cand_mask, -1e9)
#         att = torch.softmax(att_logits, dim=1)                         # [R, W]

#         c = torch.sum(att.unsqueeze(-1) * h_i, dim=1)                  # [R, d_model]
#         q = self.q_ffn(torch.cat([c, h_f], dim=-1))                    # [R, d_model]

#         u = torch.tanh(h_i + q.unsqueeze(1))                           # [R, W, d_model]
#         logits = self.out_z(u).squeeze(-1)                             # [R, W]
#         return logits


# class Step2PositionAttentionValue(nn.Module):
#     """可选 baseline（critic），用于降低策略梯度方差。"""
#     def __init__(self, num_exam_types: int, d_emb: int = 32, d_hidden: int = 128, dropout: float = 0.0):
#         super().__init__()
#         self.num_exam_types = int(num_exam_types)
#         self.type_emb = nn.Embedding(self.num_exam_types + 1, d_emb)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_emb + 3, d_hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(d_hidden, d_hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(d_hidden, 1),
#         )

#     @staticmethod
#     def _shift_type(x: torch.Tensor) -> torch.Tensor:
#         return torch.clamp(x + 1, min=0)

#     def forward(
#         self,
#         ref_type: torch.Tensor,        # [R]
#         ref_pos_norm: torch.Tensor,    # [R,1]
#         cand_mask: torch.Tensor,       # [R,W]
#         cand_pos_norm: torch.Tensor,   # [R,W,1]
#     ) -> torch.Tensor:
#         ref_t = self._shift_type(ref_type).long()
#         e = self.type_emb(ref_t)  # [R, d_emb]

#         R, W = cand_mask.shape
#         mask_f = cand_mask.float()
#         cnt = mask_f.sum(dim=1, keepdim=True) / float(max(W, 1))  # [R,1]
#         denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)     # [R,1]
#         pos = cand_pos_norm.squeeze(-1)                            # [R,W]
#         mean = (pos * mask_f).sum(dim=1, keepdim=True) / denom     # [R,1]

#         x = torch.cat([e, ref_pos_norm, cnt, mean], dim=-1)        # [R, d_emb+3]
#         v = self.mlp(x).squeeze(-1)                                # [R]
#         return v


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

# # ===================== 导出所需 =====================
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
#                  patient_main_type_id: torch.Tensor, 
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
#         self.patient_main_type_id = patient_main_type_id.to(DEVICE).long()
        
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
#         return pos_wait * (is_self * SELF_SELECTED_PENALTY + non_self * NON_SELF_PENALTY) + neg_wait * LOGICAL,pos_wait

#     def _device_violate(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         if (self.patient_exam_mask is None) or (self.machine_exam_mask is None):
#             return torch.zeros_like(assigned_machine_batch, dtype=torch.bool)
#         mach_mask = self.machine_exam_mask[assigned_machine_batch]
#         pat_mask = self.patient_exam_mask[perms]
#         invalid = pat_mask & (~mach_mask)
#         return invalid.any(dim=2)

#     def _penalty_device_cover(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         violate = self._device_violate(assigned_machine_batch, perms)
#         return violate.to(DTYPE_FLOAT) * DEVICE_PENALTY

#     def _special_violates(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
#         heart_mask = self.has_heart.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         ok_wd_h = (weekday_batch == 1) | (weekday_batch == 3)
#         ok_mc_h = (assigned_machine_batch == 3)
#         heart_violate = heart_mask & (~(ok_wd_h & ok_mc_h))

#         angio_mask = self.has_angio.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         ok_wd_a = (weekday_batch == 0) | (weekday_batch == 2) | (weekday_batch == 4)
#         ok_mc_a = (assigned_machine_batch == 1)
#         angio_violate = angio_mask & (~(ok_wd_a & ok_mc_a))

#         weekend = (weekday_batch == 5) | (weekday_batch == 6)
#         contrast_mask = self.has_contrast.index_select(0, perms.reshape(-1)).reshape(perms.shape)
#         weekend_violate = contrast_mask & weekend
#         return heart_violate, angio_violate, weekend_violate

#     def _penalty_special_rules(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
#         heart_v, angio_v, weekend_v = self._special_violates(weekday_batch, assigned_machine_batch, perms)
#         p = (heart_v | angio_v | weekend_v).to(DTYPE_FLOAT) * DEVICE_PENALTY
#         return p, heart_v.to(torch.int32), angio_v.to(torch.int32), weekend_v.to(torch.int32)

#     def _penalty_machine_switching(self, bin_idx_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
#         B, N = perms.shape
#         if TRANSITION_PENALTY <= 0:
#             return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)
#         current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
#         prev_types = torch.roll(current_types, shifts=1, dims=1)
#         prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
#         same_bin = (bin_idx_batch == prev_bins)
#         diff_type = (current_types != prev_types)
#         is_transition = same_bin & diff_type
#         is_transition[:, 0] = False 
#         return is_transition.to(DTYPE_FLOAT) * TRANSITION_PENALTY,is_transition


#     def fitness_batch(self, perms: torch.Tensor, return_assignment: bool = False):
#         perms = perms.to(DEVICE)
#         B, N = perms.shape
#         base = self.patient_durations.unsqueeze(0).expand(B, -1) 
#         durations_batch = torch.gather(base, 1, perms)

#         bin_idx_batch = _assign_bins_batch_by_prefix(durations_batch, self.caps_cumsum)
#         assigned_day_batch = self.bin_day.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         assigned_machine_batch = self.bin_machine.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
#         weekday_batch = (self.start_weekday + assigned_day_batch) % 7

#         p_wait, pos_wait_days = self._penalty_waiting(assigned_day_batch, perms)
#         p_dev   = self._penalty_device_cover(assigned_machine_batch, perms)
#         p_spec, heart_v_i, angio_v_i, weekend_v_i = self._penalty_special_rules(weekday_batch, assigned_machine_batch, perms)
#         p_tran, is_transition = self._penalty_machine_switching(bin_idx_batch, perms)

#         total_penalty = p_wait + p_dev + p_spec + p_tran
        
#         mask_valid = (durations_batch > 0)
#         total_penalty = total_penalty * mask_valid.to(DTYPE_FLOAT)

#         wait_days_sum = (pos_wait_days * mask_valid.to(DTYPE_FLOAT)).sum(dim=1)          # [B]
#         switch_cnt = (is_transition & mask_valid).sum(dim=1).to(torch.int32)            # [B]

#         fitness = - total_penalty.sum(dim=1)
#         out = {
#             'fitness': fitness,
#             'assigned_day': assigned_day_batch if return_assignment else None,
#             'assigned_machine': assigned_machine_batch if return_assignment else None,
#             'order_in_machine': _compute_order_in_bin_batch(bin_idx_batch) if return_assignment else None,
#             'heart_cnt': (heart_v_i * mask_valid).sum(dim=1),
#             'angio_cnt': (angio_v_i * mask_valid).sum(dim=1),
#             'weekend_cnt': (weekend_v_i * mask_valid).sum(dim=1),
#             'device_cnt': ((p_dev > 0) * mask_valid).sum(dim=1),
#             'any_violate_mask': ((heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)) & mask_valid).any(dim=1) 
#         }
#         out['any_violate_mask_b_n'] = ((heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)) & mask_valid)
#         out["wait_days_sum"] = wait_days_sum
#         out["switch_cnt"] = switch_cnt
#         return out


# # ===================== GA 主体（Megabatch 版 + 协同进化） =====================
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
        
#         self.population_tensor: torch.Tensor | None = None
#         self.fitness_history: List[List[float]] = [[] for _ in range(self.K)]
        
#         self._gpu_engine = None
#         self._cid_to_idx = None
#         self._idx_to_cid = None
        
#         self.patient_durations_all = None
#         self.reg_day_offsets_all = None
#         self.is_self_selected_all = None
#         self.has_contrast_all = None
#         self.has_heart_all = None
#         self.has_angio_all = None
#         self.patient_main_type_id_all = None
#         self.patient_exam_mask_all = None
#         self.patient_main_exam_id_all = None
        
#         self._E = None
#         self._machine_exam_mask = None
#         self._weekday_machine_minutes = None
        
#         self.dummy_idx = self.N 

#         # Step2: 论文式位置选择器（可选）
#         self.step2_pos_policy: Step2PositionAttentionPolicy | None = None
#         self.step2_pos_value: Step2PositionAttentionValue | None = None
#         self.step2_pos_window: int = 400
#         self.step2_pos_deterministic: bool = False
#         # 记录最近一次 step2 选择（便于外部 RL/日志）
#         self.last_step2_rows: torch.Tensor | None = None
#         self.last_step2_idx1: torch.Tensor | None = None
#         self.last_step2_idx2: torch.Tensor | None = None
#         self.last_step2_logprob: torch.Tensor | None = None

#     def _ensure_gpu_engine(self):
#         if self._gpu_engine is not None:
#             return
#         idx_to_cid = list(self.sorted_patients)
#         cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
#         self._idx_to_cid = idx_to_cid
#         self._cid_to_idx = cid_to_idx
#         N = len(idx_to_cid)

#         self.patient_durations_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)
#         self.reg_day_offsets_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)
#         self.is_self_selected_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
#         self.has_contrast_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
#         self.has_heart_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
#         self.has_angio_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
#         self.patient_main_type_id_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)

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

#         self.patient_exam_mask_all = torch.zeros((N + 1, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
#         self._machine_exam_mask = torch.zeros((MACHINE_COUNT, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
#         self.patient_main_exam_id_all = torch.full((N + 1,), -1, dtype=DTYPE_LONG, device=DEVICE)

#         base_date = self.block_start_date if self.block_start_date else START_DATE.date()
#         start_weekday = base_date.isoweekday() - 1

#         for i, cid in enumerate(idx_to_cid):
#             p = self.patients[cid]
#             total_minutes = 0
#             any_contrast = False
#             any_heart = False
#             any_angio = False
#             p_exam_types = []
            
#             counter: Dict[int, int] = defaultdict(int)
#             for _, et, dur, _ in p['exams']:
#                 etn = clean_exam_name(et)
#                 total_minutes += int(round(float(dur)))
#                 p_exam_types.append(etn)
#                 if E > 0:
#                     eidx = exam_to_eidx.get(etn, None)
#                     if eidx is not None:
#                         self.patient_exam_mask_all[i, eidx] = True
#                         counter[eidx] += 1
#                 any_contrast = any_contrast or ('增强' in etn)
#                 any_heart = any_heart or ('心脏' in etn)
#                 any_angio = any_angio or ('造影' in etn)
            
#             if len(counter) > 0:
#                 best_cnt = max(counter.values())
#                 main_eidx = min([k for k, v in counter.items() if v == best_cnt])
#                 self.patient_main_exam_id_all[i] = main_eidx
            
#             if p_exam_types:
#                 main_type = p_exam_types[0]
#                 self.patient_main_type_id_all[i] = exam_to_eidx.get(main_type, 0)
#             else:
#                 self.patient_main_type_id_all[i] = 0

#             self.patient_durations_all[i] = max(1, total_minutes)
#             self.reg_day_offsets_all[i] = (p['reg_date'] - base_date).days
#             self.is_self_selected_all[i] = bool(p.get('is_self_selected', False))
#             self.has_contrast_all[i] = any_contrast
#             self.has_heart_all[i] = any_heart
#             self.has_angio_all[i] = any_angio

#         self.patient_durations_all[N] = 0
#         self.reg_day_offsets_all[N] = 0

#         if self._machine_exam_mask is not None:
#             for mid in range(MACHINE_COUNT):
#                 for e in self.machine_exam_map.get(mid, []):
#                     et = clean_exam_name(e)
#                     eidx = exam_to_eidx.get(et, None)
#                     if eidx is not None:
#                         self._machine_exam_mask[mid, eidx] = True

#         self._weekday_machine_minutes = _weekday_minutes_matrix_from_end_hours(MACHINE_COUNT)
#         self._patient_main_exam_id = self.patient_main_exam_id_all
        
#         self._gpu_engine = _GPUMatrixFitnessBatch(
#             weekday_machine_minutes=self._weekday_machine_minutes,
#             start_weekday=start_weekday,
#             patient_durations=self.patient_durations_all,
#             reg_day_offsets=self.reg_day_offsets_all,
#             is_self_selected=self.is_self_selected_all,
#             has_contrast=self.has_contrast_all,
#             has_heart=self.has_heart_all,
#             has_angio=self.has_angio_all,
#             patient_main_type_id=self.patient_main_type_id_all, 
#             patient_exam_mask=self.patient_exam_mask_all,
#             machine_exam_mask=self._machine_exam_mask,
#             patient_main_exam_id=self.patient_main_exam_id_all,
#             exam_count=E,
#         )

#     def enable_step2_position_policy(
#         self,
#         ckpt_path: str | None = None,
#         deterministic: bool = False,
#         d_model: int = 128,
#         d_emb: int = 32,
#         dropout: float = 0.0,
#     ) -> None:
#         """启用 step2 的论文式注意力位置选择。

#         - 若 ckpt_path 提供且存在：加载权重（支持 {'policy':..., 'value':...} 或直接 state_dict）
#         - 若不提供：初始化随机权重（用于后续外部训练）
#         """
#         self._ensure_gpu_engine()
#         if self._E is None:
#             raise RuntimeError("GPU engine 未初始化 exam_count（self._E=None），无法构建 step2 位置选择器。")
#         num_exam_types = int(self._E)

#         self.step2_pos_policy = Step2PositionAttentionPolicy(
#             num_exam_types=num_exam_types,
#             d_model=d_model,
#             d_emb=d_emb,
#             dropout=dropout,
#         ).to(DEVICE)
#         self.step2_pos_value = Step2PositionAttentionValue(
#             num_exam_types=num_exam_types,
#             d_emb=d_emb,
#             d_hidden=d_model,
#             dropout=dropout,
#         ).to(DEVICE)

#         self.step2_pos_deterministic = bool(deterministic)

#         if ckpt_path is not None and os.path.isfile(ckpt_path):
#             ckpt = torch.load(ckpt_path, map_location=DEVICE)
#             if isinstance(ckpt, dict) and "policy" in ckpt:
#                 self.step2_pos_policy.load_state_dict(ckpt["policy"], strict=False)
#                 if "value" in ckpt and self.step2_pos_value is not None:
#                     self.step2_pos_value.load_state_dict(ckpt["value"], strict=False)
#             elif isinstance(ckpt, dict):
#                 self.step2_pos_policy.load_state_dict(ckpt, strict=False)
#             else:
#                 raise ValueError("ckpt 格式不支持，期望 dict 或 state_dict。")

#         self.step2_pos_policy.eval()
#         if self.step2_pos_value is not None:
#             self.step2_pos_value.eval()


#     def disable_step2_position_policy(self) -> None:
#         self.step2_pos_policy = None
#         self.step2_pos_value = None
#         self.last_step2_rows = None
#         self.last_step2_idx1 = None
#         self.last_step2_idx2 = None
#         self.last_step2_logprob = None


#     def _step2_select_idx2_with_attention(
#         self,
#         X: torch.Tensor,
#         rows: torch.Tensor,   # [R]
#         idx1: torch.Tensor,   # [R]
#         N: int,
#         window: int,
#         deterministic: bool,
#     ) -> torch.Tensor:
#         """给定 rows/idx1，为每行选一个 idx2（论文式 attention）。"""
#         assert self.step2_pos_policy is not None
#         if self._patient_main_exam_id is None:
#             idx2 = torch.randint(0, N, (rows.numel(),), device=DEVICE)
#             idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)
#             return idx2

#         R = rows.numel()
#         W = 2 * window + 1
#         offsets = torch.arange(-window, window + 1, device=DEVICE, dtype=DTYPE_LONG)  # [W]
#         cand_pos = idx1.view(R, 1) + offsets.view(1, W)  # [R,W]
#         valid = (cand_pos >= 0) & (cand_pos < N)
#         not_self = cand_pos != idx1.view(R, 1)
#         cand_mask = valid & not_self

#         any_valid = cand_mask.any(dim=1)  # [R]

#         cand_pos_safe = cand_pos.clamp(0, N - 1)
#         cand_gene = X[rows.view(R, 1), cand_pos_safe]  # [R,W]
#         ref_gene = X[rows, idx1]                       # [R]

#         cand_type = self._patient_main_exam_id[cand_gene]  # [R,W]
#         ref_type = self._patient_main_exam_id[ref_gene]    # [R]

#         denom = float(max(N - 1, 1))
#         ref_pos_norm = (idx1.float() / denom).view(R, 1)              # [R,1]
#         cand_pos_norm = (cand_pos_safe.float() / denom).view(R, W, 1) # [R,W,1]
#         rel_offset_norm = offsets.float() / float(max(window, 1))     # [W]

#         policy = self.step2_pos_policy
#         if not policy.training:
#             with torch.no_grad():
#                 logits = policy(ref_type, ref_pos_norm, cand_type, cand_pos_norm, rel_offset_norm, cand_mask)
#         else:
#             logits = policy(ref_type, ref_pos_norm, cand_type, cand_pos_norm, rel_offset_norm, cand_mask)

#         logits = logits.masked_fill(~cand_mask, -1e9)

#         if deterministic:
#             pick = torch.argmax(logits, dim=1)  # [R]
#             logprob = None
#         else:
#             dist = torch.distributions.Categorical(logits=logits)
#             pick = dist.sample()
#             logprob = dist.log_prob(pick)

#         idx2 = cand_pos_safe.gather(1, pick.view(R, 1)).squeeze(1)  # [R]
#         idx2 = torch.where(any_valid, idx2, (idx1 + 1) % N)

#         self.last_step2_rows = rows.detach()
#         self.last_step2_idx1 = idx1.detach()
#         self.last_step2_idx2 = idx2.detach()
#         # 训练时必须保留计算图；如需日志/保存用 detach 版本，请单独存一个
#         self.last_step2_logprob = logprob
#         return idx2

#     def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
#         if self._idx_to_cid is None:
#             self._idx_to_cid = list(self.sorted_patients)
#         valid_indices = row[row < self.N]
#         return [self._idx_to_cid[int(x)] for x in valid_indices.tolist()]

#     def initialize_population(self):
#         if self.N == 0:
#             print("警告：患者列表为空，无法初始化种群。")
#             return
#         if self._idx_to_cid is None:
#             self._idx_to_cid = list(self.sorted_patients)
#             self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}

#         indices = torch.arange(self.N, device=DEVICE)
#         block_size = max(30, self.N // 20)
#         pop_indices = torch.empty((self.K, self.B, self.N), dtype=DTYPE_LONG, device=DEVICE)
#         rand_matrices = torch.rand(self.K, self.B, self.N, device=DEVICE)
        
#         for i in range(0, self.N, block_size):
#             end = min(i + block_size, self.N)
#             block_len = end - i
#             if block_len == 0: continue
#             block_rand = rand_matrices[:, :, i:end]
#             block_perm_idx = torch.argsort(block_rand, dim=2)
#             block_indices = indices[i:end]
#             block_indices_expanded = block_indices.expand(self.K, self.B, -1)
#             pop_indices[:, :, i:end] = torch.gather(block_indices_expanded, 2, block_perm_idx)

#         self.population_tensor = pop_indices
#         print(f"已生成 {self.K} 个并行种群 (每个 {self.B} 个个体)，总计 {self.total_pop_size} 个个体")

#     # def _heterogeneous_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     纯 GPU 实现的异构交叉。
#     #     逻辑：Child 继承 P1 的有效基因集合。
#     #     其中属于 (P1 ∩ P2) 的基因，在 Child 中的相对顺序参考 P2。
#     #     属于 (P1 - P2) 的基因，保持在 P1 中的相对位置。
#     #     """
#     #     B, L = P1.shape
#     #     dummy = self.N
        
#     #     # 1. 识别有效区域
#     #     P1_valid = P1 < dummy
#     #     P2_valid = P2 < dummy
        
#     #     # 2. 识别 Common 元素 (在 P1 中且在 P2 中)
#     #     # 利用 Row Offset 技巧进行行内比较
#     #     row_ids = torch.arange(B, device=DEVICE).unsqueeze(1) * (self.N + 1)
        
#     #     P1_off = P1 + row_ids
#     #     P2_off = P2 + row_ids
        
#     #     # isin(A, B): 检查 A 的元素是否在 B 中
#     #     # Mask for P1: P1 的哪些位置包含 Common 元素
#     #     common_mask_p1 = torch.isin(P1_off, P2_off) & P1_valid
        
#     #     # Mask for P2: P2 的哪些位置包含 Common 元素 (用于提取值，蕴含了 P2 的顺序)
#     #     common_mask_p2 = torch.isin(P2_off, P1_off) & P2_valid
        
#     #     # 3. 提取 P2 中的 Common 值
#     #     # masked_select 返回 1D Tensor，按行优先顺序平铺
#     #     # 只要每行 Common 元素的数量一致（集合交集必然一致），顺序就会自动对齐
#     #     common_vals = torch.masked_select(P2, common_mask_p2)
        
#     #     # 4. 填入 P1
#     #     Child = P1.clone()
#     #     # masked_scatter_ 会按顺序消费 common_vals 并填入 mask 为 True 的位置
#     #     Child.masked_scatter_(common_mask_p1, common_vals)
        
#     #     return Child
#     def _heterogeneous_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
#         """
#         [对齐版] 异构交叉：Child 继承 P1 的有效基因集合；
#         对于 (P1 ∩ P2) 的基因，在 Child 中的相对顺序参考 P2；
#         (P1 - P2) 的基因保持在 P1 的相对位置。

#         重要：支持“每行交集长度不同”，不会再因为 masked_select 扁平化导致行间错位。
#         """
#         B, L = P1.shape
#         dummy = self.N

#         P1_valid = P1 < dummy
#         P2_valid = P2 < dummy

#         # 用 row offset 把“同值但不同行”区分开，加速 isin 且避免跨行误匹配
#         row_off = (torch.arange(B, device=DEVICE, dtype=P1.dtype) * (dummy + 1)).unsqueeze(1)
#         P1_off = P1 + row_off
#         P2_off = P2 + row_off

#         common_p1 = P1_valid & torch.isin(P1_off, P2_off)
#         common_p2 = P2_valid & torch.isin(P2_off, P1_off)

#         # 在各自行内做 rank（True 的位置得到 0..m-1）
#         rank_p2 = common_p2.cumsum(dim=1) - 1
#         rank_p1 = common_p1.cumsum(dim=1) - 1

#         row_idx = torch.arange(B, device=DEVICE).unsqueeze(1).expand(B, L)

#         # 1) 构造每行按 P2 顺序排列的 common 值表 common_vals[b, r] = P2 中第 r 个 common
#         common_vals = torch.full((B, L), dummy, dtype=P1.dtype, device=DEVICE)
#         flat_r = row_idx[common_p2]
#         flat_c = rank_p2[common_p2]
#         flat_v = P2[common_p2]
#         common_vals[flat_r, flat_c] = flat_v

#         # 2) 构造 rank -> P1 列位置 的映射 pos_p1[b, r] = P1 中第 r 个 common 的列号
#         pos_p1 = torch.full((B, L), -1, dtype=torch.long, device=DEVICE)
#         col_idx = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
#         flat_r1 = row_idx[common_p1]
#         flat_c1 = rank_p1[common_p1]
#         flat_v1 = col_idx[common_p1]
#         pos_p1[flat_r1, flat_c1] = flat_v1

#         # 3) 逐 rank 回填：Child[P1_common_positions] = common_vals（严格行内对齐）
#         m = common_p1.sum(dim=1)  # [B]
#         rank_mat = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
#         use = rank_mat < m.unsqueeze(1)

#         flat_r2 = row_idx[use]
#         flat_pos = pos_p1[use]
#         flat_val2 = common_vals[use]

#         ok = flat_pos >= 0
#         children = P1.clone()
#         children[flat_r2[ok], flat_pos[ok]] = flat_val2[ok]
#         return children
    
#     def _heterogeneous_mutate_violations_gpu(self, sub_pop: torch.Tensor, violate_mask: torch.Tensor) -> torch.Tensor:
#         """
#         步骤1: 纯 GPU 实现的异构定点变异（基于违规掩码）。
#         [修改说明]：变异范围限制在违规位置的 +/- 400 内。
#         """
#         B, L = sub_pop.shape
#         dummy = self.N

#         # 1. 找出存在违规的行
#         has_violation = violate_mask.any(dim=1)
#         viol_rows_idx = torch.nonzero(has_violation).flatten()
        
#         if viol_rows_idx.numel() == 0:
#             return sub_pop

#         # 2. 在违规行中，选择一个违规位置 (Source)
#         subset_mask = violate_mask[viol_rows_idx]
#         viol_idx_in_row = torch.multinomial(subset_mask.float(), 1, replacement=True).flatten()
        
#         # 3. 选择一个目标交换位置 (Target) - 限制在 +/- 400 范围内
#         # 获取选中行的有效长度
#         current_lens = (sub_pop[viol_rows_idx] < dummy).sum(dim=1)
        
#         # 计算局部窗口范围
#         low = torch.clamp(viol_idx_in_row - 400, min=0)
#         high = torch.clamp(viol_idx_in_row + 400, max=current_lens - 1)
#         span = high - low + 1
        
#         # 在窗口内随机选择位置
#         rand_offset = (torch.rand(viol_rows_idx.numel(), device=DEVICE) * span.float()).long()
#         target_idx_in_row = low + rand_offset

#         # 避免原地交换 (如果窗口大于1)
#         target_idx_in_row = torch.where(
#             (target_idx_in_row == viol_idx_in_row) & (span > 1),
#             torch.where(target_idx_in_row < high, target_idx_in_row + 1, target_idx_in_row - 1),
#             target_idx_in_row
#         )

#         # 4. 执行交换
#         row_idx = viol_rows_idx
#         idx1 = viol_idx_in_row
#         idx2 = target_idx_in_row
        
#         val1 = sub_pop[row_idx, idx1]
#         val2 = sub_pop[row_idx, idx2]
        
#         sub_pop[row_idx, idx1] = val2
#         sub_pop[row_idx, idx2] = val1
        
#         return sub_pop

#     def _heterogeneous_mutate_general_gpu(self, sub_pop: torch.Tensor, prob: float = 1) -> torch.Tensor:
#         """
#         步骤2: 纯 GPU 实现的异构一般变异（随机交换）。
#         [修改说明]：变异范围限制在选中位置的 +/- 400 内。
#         """
#         B, L = sub_pop.shape
#         dummy = self.N
        
#         # 1. 确定哪些行需要变异
#         do_mutate = torch.rand(B, device=DEVICE) < prob
        
#         # 2. 计算每行有效长度
#         valid_lens = (sub_pop < dummy).sum(dim=1)
        
#         # 过滤掉长度 < 2 的行
#         can_mutate = do_mutate & (valid_lens >= 2)
#         indices = torch.nonzero(can_mutate).flatten()
        
#         if indices.numel() == 0:
#             return sub_pop
            
#         # 3. 对需要变异的行操作
#         target_lens = valid_lens[indices]
        
#         # 生成第一个随机位置 idx1
#         rand1 = torch.rand(indices.numel(), device=DEVICE)
#         idx1 = (rand1 * target_lens.float()).long()
        
#         # 计算 idx1 周围 +/- 400 的窗口
#         low = torch.clamp(idx1 - 400, min=0)
#         high = torch.clamp(idx1 + 400, max=target_lens - 1)
#         span = high - low + 1
        
#         # 生成第二个位置 idx2
#         rand_offset = (torch.rand(indices.numel(), device=DEVICE) * span.float()).long()
#         idx2 = low + rand_offset
        
#         # 避免原地交换
#         idx2 = torch.where(
#             (idx2 == idx1) & (span > 1),
#             torch.where(idx2 < high, idx2 + 1, idx2 - 1),
#             idx2
#         )
        
#         # 4. 执行交换
#         row_idx = indices
        
#         val1 = sub_pop[row_idx, idx1]
#         val2 = sub_pop[row_idx, idx2]
        
#         sub_pop[row_idx, idx1] = val2
#         sub_pop[row_idx, idx2] = val1
        
#         return sub_pop

#     def _heterogeneous_mutate_greedy_cluster_gpu(self, sub_pop: torch.Tensor, greedy_prob: float = 0.5) -> torch.Tensor:
#         """
#         步骤3: 纯 GPU 实现的异构贪婪聚类变异。
#         """
#         B, L = sub_pop.shape
#         dummy = self.N
        
#         # 1. 筛选需要变异的行
#         probs = torch.rand(B, device=DEVICE)
#         valid_lens = (sub_pop < dummy).sum(dim=1)
        
#         # 条件：概率满足 且 有效长度足够支持聚类 (例如至少5个)
#         can_mutate = (probs < greedy_prob) & (valid_lens >= 5)
#         indices = torch.nonzero(can_mutate).flatten()
#         R = indices.numel()
#         if R == 0:
#             return sub_pop

#         # 2. 确定窗口
#         target_lens = valid_lens[indices] # [R]
        
#         # 窗口大小：在 [2, min(50, target_len)] 之间
#         max_possible_window = torch.clamp(target_lens, max=50)
#         rand_w = torch.rand(R, device=DEVICE)
#         window_lens = (rand_w * (max_possible_window - 2).float()).long() + 2
        
#         # 确定起始位置
#         max_starts = target_lens - window_lens
#         rand_s = torch.rand(R, device=DEVICE)
#         starts = (rand_s * (max_starts + 1).float()).long()
        
#         # 3. 提取并排序
#         max_w_len_batch = window_lens.max().item()
        
#         arng = torch.arange(max_w_len_batch, device=DEVICE).unsqueeze(0).expand(R, -1)
#         mask_window = arng < window_lens.unsqueeze(1)
        
#         # 构造 gather indices
#         gather_cols = starts.unsqueeze(1) + arng
#         gather_cols = torch.clamp(gather_cols, max=L-1)
        
#         rows = sub_pop[indices]
#         windows = torch.gather(rows, 1, gather_cols)
        
#         # 获取 keys (Main Exam IDs)
#         keys = self._patient_main_exam_id[windows]
#         keys[~mask_window] = -1
        
#         # Calculate counts and sort
#         E = self._E
#         keys_clamped = keys.clamp(min=0, max=E-1)
#         one_hot = torch.nn.functional.one_hot(keys_clamped, num_classes=E)
#         one_hot[~mask_window] = 0
        
#         counts = one_hot.sum(dim=1) # [R, E]
        
#         size_per_pos = torch.gather(counts, 1, keys_clamped)
#         size_per_pos[~mask_window] = -1
        
#         sort_score = (-size_per_pos).to(torch.int64) * (max_w_len_batch + 1) + arng
#         sort_score[~mask_window] = 9223372036854775807 # Push invalid to end
        
#         sort_indices = torch.argsort(sort_score, dim=1)
        
#         # Gather sorted windows
#         sorted_windows = torch.gather(windows, 1, sort_indices)
        
#         # 4. Scatter back (Manual 2D Scatter)
#         new_rows = rows.clone()
        
#         # Flatten indices and values for assignment
#         row_idx_expanded = torch.arange(R, device=DEVICE).unsqueeze(1).expand(-1, max_w_len_batch)
        
#         flat_r = row_idx_expanded[mask_window]
#         flat_c = gather_cols[mask_window]
#         flat_v = sorted_windows[mask_window]
        
#         new_rows[flat_r, flat_c] = flat_v
        
#         # Assign back to sub_pop
#         sub_pop[indices] = new_rows
        
#         return sub_pop

#     # def _evolve_heterogeneous_sub_pop(self, sub_pop: torch.Tensor, sub_engine: _GPUMatrixFitnessBatch, generations: int):
#     #     """
#     #     子种群进化循环 (In-place, 全 GPU)。
#     #     包含：交叉、定点变异、一般变异、贪婪聚类。
#     #     """
#     #     K, B, L = sub_pop.shape
        
#     #     old_main_exam_id = self._patient_main_exam_id
#     #     self._patient_main_exam_id = self.patient_main_exam_id_all
        
#     #     # 0. 初始评估
#     #     pop_flat = sub_pop.view(K*B, L)
#     #     out = sub_engine.fitness_batch(pop_flat)
#     #     current_fitness = out['fitness'].view(K, B)
#     #     # 获取初始的违规 Mask
#     #     current_viol_mask = out['any_violate_mask_b_n'].view(K, B, L)
        
#     #     for _ in range(generations):
#     #         # 准备 P1 (即当前种群)
#     #         P1_flat = sub_pop.view(K*B, L)
#     #         # P1 对应的违规 Mask
#     #         P1_viol_mask_flat = current_viol_mask.view(K*B, L)
            
#     #         # 准备 P2 (同组内随机打乱)
#     #         idx_in_b = torch.randint(0, B, (K, B), device=DEVICE)
#     #         batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B
#     #         flat_p2_idx = (batch_offsets + idx_in_b).view(-1)
#     #         P2_flat = P1_flat[flat_p2_idx]
            
#     #         # 1. 交叉 (Cross)
#     #         children_flat = self._heterogeneous_crossover_batch_gpu(P1_flat, P2_flat)
            
#     #         # 2. 定点变异 (Mutation - Violations)
#     #         children_flat = self._heterogeneous_mutate_violations_gpu(children_flat, P1_viol_mask_flat)
            
#     #         # 3. 一般变异 (Mutation - General)
#     #         children_flat = self._heterogeneous_mutate_general_gpu(children_flat, prob=0.8)

#     #         # 4. 贪婪聚类变异 (Mutation - Greedy Cluster)
#     #         children_flat = self._heterogeneous_mutate_greedy_cluster_gpu(children_flat, greedy_prob=0.3)
            
#     #         # 5. 评估子代
#     #         out_child = sub_engine.fitness_batch(children_flat)
#     #         child_fitness = out_child['fitness'].view(K, B)
#     #         child_viol_mask = out_child['any_violate_mask_b_n'].view(K, B, L)
            
#     #         # 6. 选择 (Selection) - (1+1) 策略
#     #         children = children_flat.view(K, B, L)
#     #         mask_better = child_fitness > current_fitness
#     #         mask_better_exp = mask_better.unsqueeze(2).expand(K, B, L)
            
#     #         sub_pop = torch.where(mask_better_exp, children, sub_pop)
#     #         current_fitness = torch.where(mask_better, child_fitness, current_fitness)
#     #         current_viol_mask = torch.where(mask_better_exp, child_viol_mask, current_viol_mask)
            
#     #     self._patient_main_exam_id = old_main_exam_id
#     #     return sub_pop
#     def _evolve_heterogeneous_sub_pop(self, sub_pop: torch.Tensor, sub_engine: _GPUMatrixFitnessBatch, generations: int):
#         """
#         子种群进化循环 (In-place, 全 GPU)。
#         包含：交叉、定点变异、一般变异、贪婪聚类。

#         [关键改动]：P2 选择为“同组内随机其它个体”，强制非 self，
#         让子阶段恢复跨个体的有效重组。
#         """
#         K, B, L = sub_pop.shape

#         old_main_exam_id = self._patient_main_exam_id
#         self._patient_main_exam_id = self.patient_main_exam_id_all

#         # 0. 初始评估
#         pop_flat = sub_pop.view(K * B, L)
#         out = sub_engine.fitness_batch(pop_flat)
#         current_fitness = out["fitness"].view(K, B)
#         current_viol_mask = out["any_violate_mask_b_n"].view(K, B, L)

#         for _ in range(generations):
#             # 准备 P1
#             P1_flat = sub_pop.view(K * B, L)
#             P1_viol_mask_flat = current_viol_mask.view(K * B, L)

#             # ====== P2：同组内随机“其它个体”（强制不等于自身） ======
#             if B > 1:
#                 # 对每个 (k, b)，随机一个 delta ∈ [1, B-1]，让 idx2 = (b + delta) % B
#                 delta = torch.randint(1, B, (K, B), device=DEVICE)
#                 base = torch.arange(B, device=DEVICE).unsqueeze(0).expand(K, -1)
#                 idx_in_b = (base + delta) % B
#             else:
#                 # B==1 时没法跨个体，退化为自身
#                 idx_in_b = torch.zeros((K, B), dtype=torch.long, device=DEVICE)

#             batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B

#             # ====== P2：从本 run 的精英池采样（强制 != self） ======
#             elite_pool = min(8, B)  # 你可以调 5~16；B 小就自动收缩
#             elite_idx = torch.topk(current_fitness, k=elite_pool, dim=1, largest=True).indices  # [K, elite_pool]

#             # 给每个 (k,b) 随机挑一个精英
#             pick = torch.randint(0, elite_pool, (K, B), device=DEVICE)                          # [K,B]
#             idx_in_b = elite_idx.gather(1, pick)                                                # [K,B] in [0,B)

#             # 强制 P2 != P1（如果撞上了，就换成精英池下一个）
#             base = torch.arange(B, device=DEVICE).unsqueeze(0).expand(K, B)
#             idx_in_b = torch.where(idx_in_b == base, elite_idx[:, 0:1].expand(K, B), idx_in_b)

#             batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B
#             flat_p2_idx = (batch_offsets + idx_in_b).reshape(-1)
#             P2_flat = P1_flat[flat_p2_idx]
#             # =========================================================

#             # 1) 交叉
#             children_flat = self._heterogeneous_crossover_batch_gpu(P1_flat, P2_flat)

#             # 2) 定点变异（基于违规 mask）
#             children_flat = self._heterogeneous_mutate_violations_gpu(children_flat, P1_viol_mask_flat)

#             # 3) 一般变异（局部交换）
#             children_flat = self._heterogeneous_mutate_general_gpu(children_flat, prob=0.8)

#             # 4) 贪婪聚类变异
#             children_flat = self._heterogeneous_mutate_greedy_cluster_gpu(children_flat, greedy_prob=0.3)

#             # 5) 评估子代
#             out_child = sub_engine.fitness_batch(children_flat)
#             child_fitness = out_child["fitness"].view(K, B)
#             child_viol_mask = out_child["any_violate_mask_b_n"].view(K, B, L)

#             # 6) 选择 (1+1)
#             children = children_flat.view(K, B, L)
#             mask_better = child_fitness > current_fitness
#             mask_better_exp = mask_better.unsqueeze(2).expand(K, B, L)

#             sub_pop = torch.where(mask_better_exp, children, sub_pop)
#             current_fitness = torch.where(mask_better, child_fitness, current_fitness)
#             current_viol_mask = torch.where(mask_better_exp, child_viol_mask, current_viol_mask)

#         self._patient_main_exam_id = old_main_exam_id
#         return sub_pop

#     def run_coevolution_phase(self, co_gens=50):      
#         pop_flat = self.population_tensor.view(self.total_pop_size, self.N)
#         out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=True)
#         assigned_days = out['assigned_day'] 
        
#         max_day = assigned_days.max().item()
        
#         for day_start in range(0, int(max_day) + 1, 7):
#             day_end = day_start + 7
            
#             # Mask: [Total_P, N]
#             mask = (assigned_days >= day_start) & (assigned_days < day_end)
            
#             # --- 全 GPU 切分与 Padding ---
#             # 1. 计算每行选中的元素数量
#             row_lens = mask.sum(dim=1) # [Total_P]
#             max_len = row_lens.max().item()
            
#             if max_len < 2: continue
            
#             # 2. 准备输出容器 (Padding with Dummy)
#             sub_pop_padded = torch.full((self.total_pop_size, max_len), self.N, dtype=DTYPE_LONG, device=DEVICE)
#             mask_int = mask.long()
#             # cumsum 得到每一行内的累积计数，减 1 得到 0-based index
#             col_indices = mask_int.cumsum(dim=1) - 1
#             # 仅保留 mask 为 True 位置的 col_indices
#             valid_col_indices = torch.masked_select(col_indices, mask)
#             row_indices = torch.arange(self.total_pop_size, device=DEVICE).unsqueeze(1).expand(-1, self.N)
#             valid_row_indices = torch.masked_select(row_indices, mask)
#             valid_values = torch.masked_select(self.population_tensor.view(self.total_pop_size, self.N), mask)
#             sub_pop_padded[valid_row_indices, valid_col_indices] = valid_values
            
#             sub_pop_gpu = sub_pop_padded.view(self.K, self.B, max_len)
            
#             # --- 准备子引擎 ---
#             shifted_reg_offsets = self.reg_day_offsets_all - day_start
            
#             base_date = self.block_start_date if self.block_start_date else START_DATE.date()
#             base_weekday = base_date.isoweekday() - 1
#             new_start_weekday = (base_weekday + day_start) % 7
            
#             sub_engine = _GPUMatrixFitnessBatch(
#                 weekday_machine_minutes=self._weekday_machine_minutes,
#                 start_weekday=new_start_weekday,
#                 patient_durations=self.patient_durations_all,
#                 reg_day_offsets=shifted_reg_offsets, 
#                 is_self_selected=self.is_self_selected_all,
#                 has_contrast=self.has_contrast_all,
#                 has_heart=self.has_heart_all,
#                 has_angio=self.has_angio_all,
#                 patient_main_type_id=self.patient_main_type_id_all, 
#                 patient_exam_mask=self.patient_exam_mask_all,
#                 machine_exam_mask=self._machine_exam_mask,
#                 patient_main_exam_id=self.patient_main_exam_id_all,
#                 exam_count=self._E
#             )
            
#             # --- 进化 ---
#             evolved_sub_pop = self._evolve_heterogeneous_sub_pop(sub_pop_gpu, sub_engine, co_gens)
            
#             # --- 全 GPU 合并 ---
#             # evolved_sub_pop: [K, B, Max_Len]
#             evolved_flat = evolved_sub_pop.view(self.total_pop_size, max_len)            
#             # 构造一个 range 矩阵 [Total_P, Max_Len]
#             range_mat = torch.arange(max_len, device=DEVICE).unsqueeze(0).expand(self.total_pop_size, -1)
#             row_lens_exp = row_lens.unsqueeze(1).expand(-1, max_len)
#             evolved_valid_mask = range_mat < row_lens_exp
            
#             # 提取进化后的有效值 (1D)
#             evolved_valid_values = torch.masked_select(evolved_flat, evolved_valid_mask)
#             self.population_tensor.view(self.total_pop_size, self.N).masked_scatter_(mask, evolved_valid_values)

#     def evolve_gpu(self, generations=100, elite_size=5):
#         self._ensure_gpu_engine()
#         if self.population_tensor is None:
#             raise RuntimeError("种群为空，请先 initialize_population")
        
#         pop = self.population_tensor
#         N = self.N
        
#         for gen_idx in range(generations):
#             if gen_idx > 0 and gen_idx % 50 == 0:
#                 self.run_coevolution_phase(co_gens=50)
#                 pop = self.population_tensor

#             pop_flat = pop.view(self.total_pop_size, N)
#             out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
#             fitness = out['fitness'].view(self.K, self.B)
#             viol_mask_flat = out['any_violate_mask_b_n'] 
#             violate_mask = viol_mask_flat.view(self.K, self.B, N)
            
#             topk_vals, topk_idx = torch.topk(fitness, k=self.B, largest=True, dim=1)
#             best_fitness_per_run = topk_vals[:, 0].cpu().tolist()
#             for k in range(self.K):
#                 self.fitness_history[k].append(best_fitness_per_run[k])
            
#             elite_size = min(elite_size, self.B)
#             elite_idx = topk_idx[:, :elite_size]
#             idx_expanded = elite_idx.unsqueeze(2).expand(self.K, elite_size, N)
#             elites = torch.gather(pop, 1, idx_expanded)

#             parent_count = max(1, int(0.2 * self.B))
#             parent_idx = topk_idx[:, :parent_count]
#             idx_expanded = parent_idx.unsqueeze(2).expand(self.K, parent_count, N)
#             parents = torch.gather(pop, 1, idx_expanded)
#             parent_viol = torch.gather(violate_mask, 1, idx_expanded)

#             num_children = self.B - elite_size
#             if num_children > 0:
#                 p_idx1 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE)
#                 p_idx2 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE)
#                 P1 = torch.gather(parents, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))
#                 P2 = torch.gather(parents, 1, p_idx2.unsqueeze(2).expand(-1, -1, N))
#                 Vmask_choice = torch.gather(parent_viol, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))
                
#                 P1_flat = P1.view(self.K * num_children, N)
#                 P2_flat = P2.view(self.K * num_children, N)
#                 children_flat = self._ordered_crossover_batch_gpu(P1_flat, P2_flat)
                
#                 Vmask_flat = Vmask_choice.view(self.K * num_children, N)
#                 children_flat = self._mutate_batch_gpu(children_flat, Vmask_flat, self.current_generation)
#                 children = children_flat.view(self.K, num_children, N)
                
#                 pop = torch.cat([elites, children], dim=1)
#             else:
#                 pop = elites.clone()
            
#             self.population_tensor = pop

#             if (gen_idx + 1) % 50 == 0:
#                 avg_best_fit = sum(best_fitness_per_run) / self.K
#                 flat_viols = (out['heart_cnt'] + out['angio_cnt'] + 
#                               out['weekend_cnt'] + out['device_cnt'])
#                 best_viols = torch.gather(
#                     flat_viols.view(self.K, self.B), 
#                     1, 
#                     topk_idx[:, :1]
#                 )
#                 avg_viols = best_viols.float().mean().item()
#                 print(f"Generation {(gen_idx+1)*2} | Avg Best Fitness (K={self.K}): {avg_best_fit:.2f} | Avg Violations: {avg_viols:.2f}")

#             self.current_generation += 1

#         print("进化完成。正在提取 K 个最佳个体...")
#         pop_flat = pop.view(self.total_pop_size, N)
#         final_out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
#         final_fitness = final_out['fitness'].view(self.K, self.B)
        
#         final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1)
#         final_best_vals = final_best_vals.flatten()
#         idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(self.K, 1, N)
#         best_individuals_tensor = torch.gather(pop, 1, idx_expanded).squeeze(1)
        
#         best_individuals_cpu = best_individuals_tensor.cpu()
#         best_fitnesses_cpu = final_best_vals.cpu().tolist()
        
#         results = []
#         for k in range(self.K):
#             cids = self._tensor_row_to_cids(best_individuals_cpu[k])
#             results.append({
#                 "run_id": k,
#                 "individual_cids": cids,
#                 "fitness": best_fitnesses_cpu[k]
#             })
            
#         self.population_tensor = pop
#         return results

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
#         P2_expanded = P2.unsqueeze(2)
#         P1_expanded = P1.unsqueeze(1)
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
#         any_viol_per_row = torch.any(parent_violate_mask, dim=1)
#         viol_rows_idx = torch.nonzero(any_viol_per_row, as_tuple=False).flatten()
#         R = viol_rows_idx.numel()
#         if R == 0:
#             return X
#         viol_mask_subset = parent_violate_mask[viol_rows_idx]
#         viol_idx_in_row = torch.multinomial(viol_mask_subset.float(), 1, replacement=True).flatten()
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
#         """Step2: 基础交换变异（Base Swap）。

#         原版：对每行随机选 idx1，再随机选 idx2（早期限制在 idx1±400）。
#         修改：保持 idx1 的随机性，把 idx2 的选择替换为论文式 *reference-conditioned attention*：
#              reference = (row, idx1)，候选集合 = idx1±window 内的位置集合，
#              由注意力策略输出分布并采样/取 argmax 得 idx2，然后执行 swap。

#         若未启用 step2_pos_policy（或缺少必要映射），则退化为原随机逻辑。
#         """
#         C, N = X.shape
#         if N <= 1:
#             return X

#         probs = torch.rand(C, device=DEVICE)
#         rows_to_swap_mask = (probs < base_swap_prob)
#         rows_to_swap_idx = torch.nonzero(rows_to_swap_mask, as_tuple=False).flatten()
#         R = rows_to_swap_idx.numel()
#         if R == 0:
#             return X

#         idx1 = torch.randint(0, N, (R,), device=DEVICE)

#         if self.step2_pos_policy is not None:
#             window = int(self.step2_pos_window)
#             deterministic = bool(self.step2_pos_deterministic)
#             idx2 = self._step2_select_idx2_with_attention(
#                 X=X,
#                 rows=rows_to_swap_idx,
#                 idx1=idx1,
#                 N=N,
#                 window=window,
#                 deterministic=deterministic,
#             )
#         else:
#             use_range_limit = (current_gen <= 10000)
#             if use_range_limit:
#                 low = torch.clamp(idx1 - 400, min=0)
#                 high = torch.clamp(idx1 + 400, max=N - 1)
#                 range_size = high - low + 1
#                 range_size = torch.where(range_size <= 0, 1, range_size)
#                 rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
#                 idx2 = low + rand_offset
#                 idx2 = torch.where(
#                     (idx2 == idx1) & (range_size > 1),
#                     torch.where(idx1 == low, low + 1, low),
#                     idx2
#                 )
#                 idx2 = torch.clamp(idx2, 0, N - 1)
#             else:
#                 idx2 = torch.randint(0, N, (R,), device=DEVICE)
#                 idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)

#         val1 = X[rows_to_swap_idx, idx1]
#         val2 = X[rows_to_swap_idx, idx2]
#         X[rows_to_swap_idx, idx1] = val2
#         X[rows_to_swap_idx, idx2] = val1
#         return X
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

#     # ------- 导出（可选） -------
#     # def generate_schedule(self, individual):
#     #     system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
#     #     for cid in individual:
#     #         p = self.patients.get(cid)
#     #         if p and not p['scheduled']:
#     #             # 修复：确保所有检查都被安排
#     #             for exam in p['exams']:
#     #                 exam_type = clean_exam_name(exam[1])
#     #                 duration = exam[2]
#     #                 try:
#     #                     m, start_time = system.find_available_slot(duration, exam_type, p)
#     #                     m.add_exam(system.current_date, start_time, duration, exam_type, p)
#     #                 except Exception as e:
#     #                     # 避免在真实排程中打印
#     #                     # print(f"排程错误: {e}") 
#     #                     pass # 忽略错误并继续
#     #     return system
#     def generate_schedule(self, individual_cids):
#             self._ensure_gpu_engine()
#             if self._cid_to_idx is None or self._idx_to_cid is None:
#                 self._idx_to_cid = list(self.sorted_patients)
#                 self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}
#             base_date = self.block_start_date if self.block_start_date else START_DATE.date()
#             perm_idx = torch.tensor(
#                 [self._cid_to_idx[cid] for cid in individual_cids],
#                 dtype=DTYPE_LONG,
#                 device=DEVICE
#             ).unsqueeze(0)
#             with torch.no_grad():
#                 out = self._gpu_engine.fitness_batch(perm_idx, return_assignment=True)

#             assigned_day = out["assigned_day"][0].cpu().tolist()      # [N]
#             assigned_machine = out["assigned_machine"][0].cpu().tolist() # [N]
#             order_in_machine = out["order_in_machine"][0].cpu().tolist() # [N]
#             perm_cpu = perm_idx[0].cpu().tolist()  # [N] 患者的 index
#             bins = defaultdict(list)
#             for pos, pid_idx in enumerate(perm_cpu):
#                 d = int(assigned_day[pos])
#                 m = int(assigned_machine[pos])
#                 o = int(order_in_machine[pos])
#                 bins[(d, m)].append((o, int(pid_idx)))

#             system = SchedulingSystem(self.machine_exam_map, start_date=base_date)
#             for (d, m), lst in sorted(bins.items(), key=lambda kv: (kv[0][0], kv[0][1])):
#                 current_date_obj = base_date + timedelta(days=int(d))
#                 machine_obj = system.machines[int(m)]
#                 cur_dt = datetime.combine(current_date_obj, WORK_START)
#                 for _, pid_idx in sorted(lst, key=lambda x: x[0]):
#                     cid = self._idx_to_cid[int(pid_idx)]
#                     p = self.patients.get(cid)
#                     if not p:
#                         continue
#                     for exam in p["exams"]:
#                         exam_type = clean_exam_name(exam[1])
#                         duration = exam[2]
#                         cur_dt = machine_obj.add_exam(current_date_obj, cur_dt, duration, exam_type, p)

#             return system

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

# def main():
#     try:
#         NUM_PARALLEL_RUNS = 4 
#         POP_SIZE_PER_RUN = 50 
#         GENERATIONS_TO_RUN = 5000
        
#         print(f"启动 Megabatch 模式: K={NUM_PARALLEL_RUNS} (并行实验), B={POP_SIZE_PER_RUN} (个体/实验)")
#         print(f"总 GPU 批量: {NUM_PARALLEL_RUNS * POP_SIZE_PER_RUN} 个体")
        
#         current_dir = "/home/preprocess/_funsearch/baseline/data"
#         patient_file = os.path.join(current_dir, '实验数据6.1 - 副本.xlsx')
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

#         print("\n===== 启动并行遗传算法优化 (Megabatch GPU + 协同进化) =====")
#         optimizer = MultiRunOptimizer(
#             patients, 
#             machine_exam_map, 
#             num_parallel_runs=NUM_PARALLEL_RUNS, 
#             pop_size_per_run=POP_SIZE_PER_RUN
#         )
#         # 可选：启用 step2 的论文式位置选择器
#         # 用法：export COE_STEP2_POS_CKPT=/path/to/ckpt.pt （若不提供则只初始化随机权重，便于外部训练）
#         _step2_ckpt = os.environ.get("COE_STEP2_POS_CKPT", "").strip()
#         _step2_det = os.environ.get("COE_STEP2_POS_DETERMINISTIC", "0").strip() in ("1", "true", "True", "YES", "yes")
#         if _step2_ckpt:
#             print(f"✓ 启用 Step2 位置选择器（ckpt={_step2_ckpt}, deterministic={_step2_det}）")
#             optimizer.enable_step2_position_policy(ckpt_path=_step2_ckpt, deterministic=_step2_det)
#         else:
#             # 如果你希望“无 ckpt 也强制启用并在线训练”，可以在外部显式调用：
#             # optimizer.enable_step2_position_policy(ckpt_path=None)
#             pass

        
#         t0_init = time.perf_counter()
#         optimizer.initialize_population()
#         t_init = time.perf_counter() - t0_init
#         print(f"✓ 已生成 {NUM_PARALLEL_RUNS} 个初始种群，耗时: {t_init:.4f}s")


#         print(f"\n开始 {GENERATIONS_TO_RUN} 代进化 (K={NUM_PARALLEL_RUNS})...")
#         t0 = time.perf_counter()
#         final_results = optimizer.evolve_gpu(generations=GENERATIONS_TO_RUN, elite_size=5)
        
#         total_evolution_time = time.perf_counter() - t0
#         print(f"\n✓ 进化完成 (K={NUM_PARALLEL_RUNS})，总耗时: {total_evolution_time:.2f}s")

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
            
#             xlsx = os.path.join(out_dir, f'final_schedule_RUN{run_id}_{ts}_fit_{best_fitness:.0f}.xlsx')
#             final_system = optimizer.generate_schedule(best_individual)
#             export_schedule(final_system, patients, xlsx)
#             print(f"    ✓ 已导出至 {xlsx}")

#         print("\n===== 最终统计 =====")
#         mean_fitness = np.mean(all_fitnesses)
#         print(f"  最佳适应度 (均值): {mean_fitness:.2f}")

#     except Exception as e:
#         print(f"运行时错误: {e}")
#         traceback.print_exc()

# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()

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
import torch.nn as nn
import torch.nn.functional as F

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
if torch.cuda.is_available():
    print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU（速度极慢）")
    
DTYPE_LONG = torch.long
DTYPE_FLOAT = torch.float32


# ===================== 论文式 Step2 位置选择器（注意力） =====================
class Step2PositionAttentionPolicy(nn.Module):
    """Reference-conditioned attention policy for selecting idx2 in base swap.

    新增最小化编码信息（仅三类）：
      - 检查日期（assigned_day, 连续，归一化为 day_norm）
      - 星期（weekday, 0..6, embedding）
      - 机器号（machine_id, 0..M-1, embedding）

    输入（batch = R 行，每行 W 个候选）：
      ref_type: Long[R]
      ref_pos_norm: Float[R,1]
      ref_day_norm: Float[R,1]
      ref_weekday: Long[R]              # 0..6
      ref_machine: Long[R]              # 0..M-1

      cand_type: Long[R,W]
      cand_pos_norm: Float[R,W,1]
      cand_day_norm: Float[R,W,1]
      cand_weekday: Long[R,W]
      cand_machine: Long[R,W]
      rel_offset_norm: Float[W]
      cand_mask: Bool[R,W]              # True 表示候选有效且可选（且满足设备约束 mask）

    输出：
      logits: Float[R,W]
    """
    def __init__(
        self,
        num_exam_types: int,
        machine_count: int,
        d_model: int = 128,
        d_type_emb: int = 32,
        d_week_emb: int = 8,
        d_machine_emb: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # exam_type 可能为 -1（dummy/unknown），这里统一 shift +1 映射到 [0..num_exam_types]
        self.num_exam_types = int(num_exam_types)
        self.machine_count = int(machine_count)

        self.type_emb = nn.Embedding(self.num_exam_types + 1, d_type_emb)  # 0=unknown/pad
        self.week_emb = nn.Embedding(7, d_week_emb)
        self.machine_emb = nn.Embedding(self.machine_count, d_machine_emb)

        # Embf / Embc（两套参数）
        # ref: [type_emb, pos_norm, day_norm, week_emb, machine_emb]
        in_f = d_type_emb + 1 + 1 + d_week_emb + d_machine_emb
        self.emb_f = nn.Sequential(
            nn.Linear(in_f, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        # cand: [type_emb, pos_norm, day_norm, week_emb, machine_emb, rel_offset_norm]
        in_c = d_type_emb + 1 + 1 + d_week_emb + d_machine_emb + 1
        self.emb_c = nn.Sequential(
            nn.Linear(in_c, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # attention: u_i^H = z_A^T tanh(W_A [h_i; h_f])
        self.attn_w = nn.Linear(2 * d_model, d_model)
        self.attn_z = nn.Linear(d_model, 1, bias=False)

        # q = FFN([c; h_f])
        self.q_ffn = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # u_i = z_B^T tanh(h_i + q)
        self.out_z = nn.Linear(d_model, 1, bias=False)

    @staticmethod
    def _shift_type(x: torch.Tensor) -> torch.Tensor:
        # [-1, ..] -> [0, ..]
        return torch.clamp(x + 1, min=0)

    def forward(
        self,
        ref_type: torch.Tensor,        # [R]
        ref_pos_norm: torch.Tensor,    # [R,1]
        ref_day_norm: torch.Tensor,    # [R,1]
        ref_weekday: torch.Tensor,     # [R]
        ref_machine: torch.Tensor,     # [R]

        cand_type: torch.Tensor,       # [R,W]
        cand_pos_norm: torch.Tensor,   # [R,W,1]
        cand_day_norm: torch.Tensor,   # [R,W,1]
        cand_weekday: torch.Tensor,    # [R,W]
        cand_machine: torch.Tensor,    # [R,W]

        rel_offset_norm: torch.Tensor, # [W]
        cand_mask: torch.Tensor,       # [R,W]
    ) -> torch.Tensor:
        R = ref_type.shape[0]
        W = cand_type.shape[1]

        ref_type_s = self._shift_type(ref_type).long()
        cand_type_s = self._shift_type(cand_type).long()

        ref_e = self.type_emb(ref_type_s)    # [R, d_type_emb]
        cand_e = self.type_emb(cand_type_s)  # [R, W, d_type_emb]

        ref_w = self.week_emb(ref_weekday.clamp(0, 6).long())  # [R, d_week]
        ref_m = self.machine_emb(ref_machine.clamp(0, self.machine_count - 1).long())  # [R, d_mach]

        cand_w = self.week_emb(cand_weekday.clamp(0, 6).long())  # [R, W, d_week]
        cand_m = self.machine_emb(cand_machine.clamp(0, self.machine_count - 1).long())  # [R, W, d_mach]

        hf_in = torch.cat([ref_e, ref_pos_norm, ref_day_norm, ref_w, ref_m], dim=-1)  # [R, in_f]
        h_f = self.emb_f(hf_in)                                                       # [R, d_model]

        rel = rel_offset_norm.view(1, W, 1).expand(R, W, 1)                            # [R, W, 1]
        hc_in = torch.cat([cand_e, cand_pos_norm, cand_day_norm, cand_w, cand_m, rel], dim=-1)  # [R,W,in_c]
        h_i = self.emb_c(hc_in)                                                       # [R, W, d_model]

        h_f_ex = h_f.unsqueeze(1).expand(R, W, h_f.shape[-1])                          # [R, W, d_model]
        att_in = torch.cat([h_i, h_f_ex], dim=-1)                                      # [R, W, 2d]
        att_h = torch.tanh(self.attn_w(att_in))                                        # [R, W, d_model]
        att_logits = self.attn_z(att_h).squeeze(-1)                                    # [R, W]
        att_logits = att_logits.masked_fill(~cand_mask, -1e9)
        att = torch.softmax(att_logits, dim=1)                                         # [R, W]

        c = torch.sum(att.unsqueeze(-1) * h_i, dim=1)                                  # [R, d_model]
        q = self.q_ffn(torch.cat([c, h_f], dim=-1))                                    # [R, d_model]

        u = torch.tanh(h_i + q.unsqueeze(1))                                           # [R, W, d_model]
        logits = self.out_z(u).squeeze(-1)                                             # [R, W]
        return logits


class Step2PositionAttentionValue(nn.Module):
    """可选 baseline（critic），用于降低策略梯度方差。"""
    def __init__(self, num_exam_types: int, d_emb: int = 32, d_hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.num_exam_types = int(num_exam_types)
        self.type_emb = nn.Embedding(self.num_exam_types + 1, d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb + 3, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    @staticmethod
    def _shift_type(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 1, min=0)

    def forward(
        self,
        ref_type: torch.Tensor,        # [R]
        ref_pos_norm: torch.Tensor,    # [R,1]
        cand_mask: torch.Tensor,       # [R,W]
        cand_pos_norm: torch.Tensor,   # [R,W,1]
    ) -> torch.Tensor:
        ref_t = self._shift_type(ref_type).long()
        e = self.type_emb(ref_t)  # [R, d_emb]

        R, W = cand_mask.shape
        mask_f = cand_mask.float()
        cnt = mask_f.sum(dim=1, keepdim=True) / float(max(W, 1))  # [R,1]
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)     # [R,1]
        pos = cand_pos_norm.squeeze(-1)                            # [R,W]
        mean = (pos * mask_f).sum(dim=1, keepdim=True) / denom     # [R,1]

        x = torch.cat([e, ref_pos_norm, cnt, mean], dim=-1)        # [R, d_emb+3]
        v = self.mlp(x).squeeze(-1)                                # [R]
        return v


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

# ===================== 导出所需 =====================
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
                 patient_main_type_id: torch.Tensor, 
                 patient_exam_mask: torch.Tensor | None,
                 machine_exam_mask: torch.Tensor | None,
                 patient_main_exam_id: torch.Tensor | None = None,
                 exam_count: int | None = None):
        
        self.weekday_machine_minutes = weekday_machine_minutes.to(DEVICE).long()
        self.start_weekday = int(start_weekday)
        self.patient_durations = patient_durations.to(DEVICE).long()
        self.reg_day_offsets = reg_day_offsets.to(DEVICE).long()
        self.is_self_selected = is_self_selected.to(DEVICE).bool()
        self.has_contrast = has_contrast.to(DEVICE).bool()
        self.has_heart = has_heart.to(DEVICE).bool()
        self.has_angio = has_angio.to(DEVICE).bool()
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
        return pos_wait * (is_self * SELF_SELECTED_PENALTY + non_self * NON_SELF_PENALTY) + neg_wait * LOGICAL,pos_wait

    def _device_violate(self, assigned_machine_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        if (self.patient_exam_mask is None) or (self.machine_exam_mask is None):
            return torch.zeros_like(assigned_machine_batch, dtype=torch.bool)
        mach_mask = self.machine_exam_mask[assigned_machine_batch]
        pat_mask = self.patient_exam_mask[perms]
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

    def _penalty_special_rules(self, weekday_batch: torch.Tensor, assigned_machine_batch: torch.Tensor, perms: torch.Tensor):
        heart_v, angio_v, weekend_v = self._special_violates(weekday_batch, assigned_machine_batch, perms)
        p = (heart_v | angio_v | weekend_v).to(DTYPE_FLOAT) * DEVICE_PENALTY
        return p, heart_v.to(torch.int32), angio_v.to(torch.int32), weekend_v.to(torch.int32)

    def _penalty_machine_switching(self, bin_idx_batch: torch.Tensor, perms: torch.Tensor) -> torch.Tensor:
        B, N = perms.shape
        if TRANSITION_PENALTY <= 0:
            return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)
        current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
        prev_types = torch.roll(current_types, shifts=1, dims=1)
        prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
        same_bin = (bin_idx_batch == prev_bins)
        diff_type = (current_types != prev_types)
        is_transition = same_bin & diff_type
        is_transition[:, 0] = False 
        return is_transition.to(DTYPE_FLOAT) * TRANSITION_PENALTY,is_transition


    def fitness_batch(self, perms: torch.Tensor, return_assignment: bool = False):
        perms = perms.to(DEVICE)
        B, N = perms.shape
        base = self.patient_durations.unsqueeze(0).expand(B, -1) 
        durations_batch = torch.gather(base, 1, perms)

        bin_idx_batch = _assign_bins_batch_by_prefix(durations_batch, self.caps_cumsum)
        assigned_day_batch = self.bin_day.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
        assigned_machine_batch = self.bin_machine.index_select(0, bin_idx_batch.reshape(-1)).reshape(B, N)
        weekday_batch = (self.start_weekday + assigned_day_batch) % 7

        p_wait, pos_wait_days = self._penalty_waiting(assigned_day_batch, perms)
        p_dev   = self._penalty_device_cover(assigned_machine_batch, perms)
        p_spec, heart_v_i, angio_v_i, weekend_v_i = self._penalty_special_rules(weekday_batch, assigned_machine_batch, perms)
        p_tran, is_transition = self._penalty_machine_switching(bin_idx_batch, perms)

        total_penalty = p_wait + p_dev + p_spec + p_tran
        
        mask_valid = (durations_batch > 0)
        total_penalty = total_penalty * mask_valid.to(DTYPE_FLOAT)

        wait_days_sum = (pos_wait_days * mask_valid.to(DTYPE_FLOAT)).sum(dim=1)          # [B]
        switch_cnt = (is_transition & mask_valid).sum(dim=1).to(torch.int32)            # [B]

        fitness = - total_penalty.sum(dim=1)
        out = {
            'fitness': fitness,
            'assigned_day': assigned_day_batch if return_assignment else None,
            'assigned_machine': assigned_machine_batch if return_assignment else None,
            'order_in_machine': _compute_order_in_bin_batch(bin_idx_batch) if return_assignment else None,
            'heart_cnt': (heart_v_i * mask_valid).sum(dim=1),
            'angio_cnt': (angio_v_i * mask_valid).sum(dim=1),
            'weekend_cnt': (weekend_v_i * mask_valid).sum(dim=1),
            'device_cnt': ((p_dev > 0) * mask_valid).sum(dim=1),
            'any_violate_mask': ((heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)) & mask_valid).any(dim=1) 
        }
        out['any_violate_mask_b_n'] = ((heart_v_i.bool() | angio_v_i.bool() | weekend_v_i.bool() | (p_dev > 0)) & mask_valid)
        out["wait_days_sum"] = wait_days_sum
        out["switch_cnt"] = switch_cnt
        return out


# ===================== GA 主体（Megabatch 版 + 协同进化） =====================
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
        
        self.population_tensor: torch.Tensor | None = None
        self.fitness_history: List[List[float]] = [[] for _ in range(self.K)]
        
        self._gpu_engine = None
        self._cid_to_idx = None
        self._idx_to_cid = None
        
        self.patient_durations_all = None
        self.reg_day_offsets_all = None
        self.is_self_selected_all = None
        self.has_contrast_all = None
        self.has_heart_all = None
        self.has_angio_all = None
        self.patient_main_type_id_all = None
        self.patient_exam_mask_all = None
        self.patient_main_exam_id_all = None
        
        self._E = None
        self._machine_exam_mask = None
        self._weekday_machine_minutes = None
        
        self.dummy_idx = self.N

        # Step2 辅助：assigned_day 归一化尺度（在 _ensure_gpu_engine 里赋值）
        self._step2_day_scale: float = 1.0

        # Step2: 论文式位置选择器（可选）
        self.step2_pos_policy: Step2PositionAttentionPolicy | None = None
        self.step2_pos_value: Step2PositionAttentionValue | None = None
        self.step2_pos_window: int = 400
        self.step2_pos_deterministic: bool = False
        # 记录最近一次 step2 选择（便于外部 RL/日志）
        self.last_step2_rows: torch.Tensor | None = None
        self.last_step2_idx1: torch.Tensor | None = None
        self.last_step2_idx2: torch.Tensor | None = None
        self.last_step2_logprob: torch.Tensor | None = None

    def _ensure_gpu_engine(self):
        if self._gpu_engine is not None:
            return
        idx_to_cid = list(self.sorted_patients)
        cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
        self._idx_to_cid = idx_to_cid
        self._cid_to_idx = cid_to_idx
        N = len(idx_to_cid)

        self.patient_durations_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)
        self.reg_day_offsets_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)
        self.is_self_selected_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
        self.has_contrast_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
        self.has_heart_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
        self.has_angio_all = torch.zeros(N + 1, dtype=torch.bool, device=DEVICE)
        self.patient_main_type_id_all = torch.zeros(N + 1, dtype=DTYPE_LONG, device=DEVICE)

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

        self.patient_exam_mask_all = torch.zeros((N + 1, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
        self._machine_exam_mask = torch.zeros((MACHINE_COUNT, E), dtype=torch.bool, device=DEVICE) if E > 0 else None
        self.patient_main_exam_id_all = torch.full((N + 1,), -1, dtype=DTYPE_LONG, device=DEVICE)

        base_date = self.block_start_date if self.block_start_date else START_DATE.date()
        start_weekday = base_date.isoweekday() - 1

        for i, cid in enumerate(idx_to_cid):
            p = self.patients[cid]
            total_minutes = 0
            any_contrast = False
            any_heart = False
            any_angio = False
            p_exam_types = []
            
            counter: Dict[int, int] = defaultdict(int)
            for _, et, dur, _ in p['exams']:
                etn = clean_exam_name(et)
                total_minutes += int(round(float(dur)))
                p_exam_types.append(etn)
                if E > 0:
                    eidx = exam_to_eidx.get(etn, None)
                    if eidx is not None:
                        self.patient_exam_mask_all[i, eidx] = True
                        counter[eidx] += 1
                any_contrast = any_contrast or ('增强' in etn)
                any_heart = any_heart or ('心脏' in etn)
                any_angio = any_angio or ('造影' in etn)
            
            if len(counter) > 0:
                best_cnt = max(counter.values())
                main_eidx = min([k for k, v in counter.items() if v == best_cnt])
                self.patient_main_exam_id_all[i] = main_eidx
            
            if p_exam_types:
                main_type = p_exam_types[0]
                self.patient_main_type_id_all[i] = exam_to_eidx.get(main_type, 0)
            else:
                self.patient_main_type_id_all[i] = 0

            self.patient_durations_all[i] = max(1, total_minutes)
            self.reg_day_offsets_all[i] = (p['reg_date'] - base_date).days
            self.is_self_selected_all[i] = bool(p.get('is_self_selected', False))
            self.has_contrast_all[i] = any_contrast
            self.has_heart_all[i] = any_heart
            self.has_angio_all[i] = any_angio

        self.patient_durations_all[N] = 0
        self.reg_day_offsets_all[N] = 0

        if self._machine_exam_mask is not None:
            for mid in range(MACHINE_COUNT):
                for e in self.machine_exam_map.get(mid, []):
                    et = clean_exam_name(e)
                    eidx = exam_to_eidx.get(et, None)
                    if eidx is not None:
                        self._machine_exam_mask[mid, eidx] = True

        self._weekday_machine_minutes = _weekday_minutes_matrix_from_end_hours(MACHINE_COUNT)
        self._patient_main_exam_id = self.patient_main_exam_id_all
        
        self._gpu_engine = _GPUMatrixFitnessBatch(
            weekday_machine_minutes=self._weekday_machine_minutes,
            start_weekday=start_weekday,
            patient_durations=self.patient_durations_all,
            reg_day_offsets=self.reg_day_offsets_all,
            is_self_selected=self.is_self_selected_all,
            has_contrast=self.has_contrast_all,
            has_heart=self.has_heart_all,
            has_angio=self.has_angio_all,
            patient_main_type_id=self.patient_main_type_id_all, 
            patient_exam_mask=self.patient_exam_mask_all,
            machine_exam_mask=self._machine_exam_mask,
            patient_main_exam_id=self.patient_main_exam_id_all,
            exam_count=E,
        )

        # Step2 日期归一化尺度：用容量 bins 的最大 day 作为尺度（避免手填 365）
        try:
            self._step2_day_scale = float(int(self._gpu_engine.bin_day.max().item()) + 1)
        except Exception:
            self._step2_day_scale = 1.0

    def enable_step2_position_policy(
        self,
        ckpt_path: str | None = None,
        deterministic: bool = False,
        d_model: int = 128,
        d_emb: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """启用 step2 的论文式注意力位置选择。

        - 若 ckpt_path 提供且存在：加载权重（支持 {'policy':..., 'value':...} 或直接 state_dict）
        - 若不提供：初始化随机权重（用于后续外部训练）
        """
        self._ensure_gpu_engine()
        if self._E is None:
            raise RuntimeError("GPU engine 未初始化 exam_count（self._E=None），无法构建 step2 位置选择器。")
        num_exam_types = int(self._E)

        self.step2_pos_policy = Step2PositionAttentionPolicy(
            num_exam_types=num_exam_types,
            machine_count=MACHINE_COUNT,
            d_model=d_model,
            d_type_emb=d_emb,
            dropout=dropout,
        ).to(DEVICE)
        self.step2_pos_value = Step2PositionAttentionValue(
            num_exam_types=num_exam_types,
            d_emb=d_emb,
            d_hidden=d_model,
            dropout=dropout,
        ).to(DEVICE)

        self.step2_pos_deterministic = bool(deterministic)

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                self.step2_pos_policy.load_state_dict(ckpt["policy"], strict=False)
                if "value" in ckpt and self.step2_pos_value is not None:
                    self.step2_pos_value.load_state_dict(ckpt["value"], strict=False)
            elif isinstance(ckpt, dict):
                self.step2_pos_policy.load_state_dict(ckpt, strict=False)
            else:
                raise ValueError("ckpt 格式不支持，期望 dict 或 state_dict。")

        self.step2_pos_policy.eval()
        if self.step2_pos_value is not None:
            self.step2_pos_value.eval()


    def disable_step2_position_policy(self) -> None:
        self.step2_pos_policy = None
        self.step2_pos_value = None
        self.last_step2_rows = None
        self.last_step2_idx1 = None
        self.last_step2_idx2 = None
        self.last_step2_logprob = None
    @torch.no_grad()
    def _step2_compute_slot_features(
        self,
        X_rows: torch.Tensor,  # [R,N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据当前 permutation（仅由 durations 决定）计算每个位置的 (assigned_day, assigned_machine, weekday)。

        返回：
          day:     Long[R,N]
          machine: Long[R,N]
          weekday: Long[R,N]   # 0..6
        """
        assert self._gpu_engine is not None
        # durations per position
        dur = self.patient_durations_all.index_select(0, X_rows.reshape(-1)).reshape(X_rows.shape).long()  # [R,N]
        T = torch.cumsum(dur, dim=1)  # [R,N]
        bin_idx = torch.searchsorted(self._gpu_engine.caps_cumsum, T, right=False)  # [R,N]
        day = self._gpu_engine.bin_day.index_select(0, bin_idx.reshape(-1)).reshape(bin_idx.shape).long()
        machine = self._gpu_engine.bin_machine.index_select(0, bin_idx.reshape(-1)).reshape(bin_idx.shape).long()
        weekday = (int(self._gpu_engine.start_weekday) + day) % 7
        return day, machine, weekday




    def _step2_select_idx2_with_attention(
        self,
        X: torch.Tensor,
        rows: torch.Tensor,   # [R]
        idx1: torch.Tensor,   # [R]
        N: int,
        window: int,
        deterministic: bool,
    ) -> torch.Tensor:
        """给定 rows/idx1，为每行选一个 idx2（论文式 attention）。"""
        assert self.step2_pos_policy is not None
        if self._patient_main_exam_id is None:
            idx2 = torch.randint(0, N, (rows.numel(),), device=DEVICE)
            idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)
            return idx2

        R = rows.numel()
        W = 2 * window + 1
        offsets = torch.arange(-window, window + 1, device=DEVICE, dtype=DTYPE_LONG)  # [W]
        cand_pos = idx1.view(R, 1) + offsets.view(1, W)  # [R,W]
        valid = (cand_pos >= 0) & (cand_pos < N)
        not_self = cand_pos != idx1.view(R, 1)

        # 先构造 safe cand_pos（避免越界 gather），并提前拿到 ref/cand 基因（供 mask & 编码使用）
        cand_pos_safe = cand_pos.clamp(0, N - 1)
        cand_gene = X[rows.view(R, 1), cand_pos_safe]  # [R,W]
        ref_gene = X[rows, idx1]                       # [R]
        cand_mask = valid & not_self

        # ===== 新增：slot 编码（检查日期/星期/机器号） =====
        # 仅对本次参与 step2 的 rows 计算一次全序列 slot 特征（O(R*N)）
        day_batch, machine_batch, weekday_batch = self._step2_compute_slot_features(X[rows])  # [R,N]

        # ref / cand slot 信息（来自当前 permutation 的 assignment）
        ref_day = day_batch.gather(1, idx1.view(R, 1)).squeeze(1)          # [R]
        ref_machine = machine_batch.gather(1, idx1.view(R, 1)).squeeze(1)  # [R]
        ref_weekday = weekday_batch.gather(1, idx1.view(R, 1)).squeeze(1)  # [R]

        cand_day = day_batch.gather(1, cand_pos_safe)                      # [R,W]
        cand_machine = machine_batch.gather(1, cand_pos_safe)              # [R,W]
        cand_weekday = weekday_batch.gather(1, cand_pos_safe)              # [R,W]

        # day 归一化（最小化处理：用预估尺度 self._step2_day_scale）
        day_scale = float(self._step2_day_scale) if getattr(self, "_step2_day_scale", 1.0) > 0 else 1.0
        ref_day_norm = (ref_day.float() / day_scale).clamp(0.0, 1.0).view(R, 1)             # [R,1]
        cand_day_norm = (cand_day.float() / day_scale).clamp(0.0, 1.0).view(R, W, 1)        # [R,W,1]

        # ===== 新增：设备约束 hard mask（不可行位置直接不允许选）=====
        # 仅处理“设备可覆盖检查类型”的硬约束：swap 后两边都必须与对方 machine 兼容（近似：_attach_to_current_machine_）
        if False and (self.patient_exam_mask_all is not None) and (self._machine_exam_mask is not None):
            # ref_gene -> cand_machine 可行？
            ref_pat_mask = self.patient_exam_mask_all[ref_gene]                 # [R,E]
            cand_mach_mask = self._machine_exam_mask[cand_machine]              # [R,W,E]
            invalid1 = (ref_pat_mask.unsqueeze(1) & (~cand_mach_mask)).any(dim=2)  # [R,W]

            # cand_gene -> ref_machine 可行？
            cand_pat_mask = self.patient_exam_mask_all[cand_gene]               # [R,W,E]
            ref_mach_mask = self._machine_exam_mask[ref_machine]                # [R,E]
            invalid2 = (cand_pat_mask & (~ref_mach_mask.unsqueeze(1))).any(dim=2)  # [R,W]

            feasible = ~(invalid1 | invalid2)
            cand_mask = cand_mask & feasible

        # ===== 新增：special rule hard mask（心脏/造影/周末-增强(对比剂)）=====
        # 规则与 _GPUMatrixFitnessBatch._special_violates 保持一致：
        #   - 心脏：weekday ∈ {1,3} 且 machine==3
        #   - 造影：weekday ∈ {0,2,4} 且 machine==1
        #   - 对比剂/增强：周末(weekday ∈ {5,6}) 禁止
        if False and self._gpu_engine is not None and hasattr(self._gpu_engine, "has_heart") and hasattr(self._gpu_engine, "has_angio") and hasattr(self._gpu_engine, "has_contrast"):
            # ref_gene -> cand_slot
            ref_has_heart = self._gpu_engine.has_heart.index_select(0, ref_gene).bool()         # [R]
            ref_has_angio = self._gpu_engine.has_angio.index_select(0, ref_gene).bool()         # [R]
            ref_has_con   = self._gpu_engine.has_contrast.index_select(0, ref_gene).bool()      # [R]

            ok_wd_h_c = (cand_weekday == 1) | (cand_weekday == 3)                                # [R,W]
            ok_mc_h_c = (cand_machine == 3)                                                       # [R,W]
            inv_h1 = ref_has_heart.view(R, 1) & (~(ok_wd_h_c & ok_mc_h_c))

            ok_wd_a_c = (cand_weekday == 0) | (cand_weekday == 2) | (cand_weekday == 4)
            ok_mc_a_c = (cand_machine == 1)
            inv_a1 = ref_has_angio.view(R, 1) & (~(ok_wd_a_c & ok_mc_a_c))

            weekend_c = (cand_weekday == 5) | (cand_weekday == 6)
            inv_w1 = ref_has_con.view(R, 1) & weekend_c

            # cand_gene -> ref_slot
            cand_has_heart = self._gpu_engine.has_heart.index_select(0, cand_gene.reshape(-1)).reshape(R, W).bool()
            cand_has_angio = self._gpu_engine.has_angio.index_select(0, cand_gene.reshape(-1)).reshape(R, W).bool()
            cand_has_con   = self._gpu_engine.has_contrast.index_select(0, cand_gene.reshape(-1)).reshape(R, W).bool()

            ok_wd_h_r = ((ref_weekday == 1) | (ref_weekday == 3)).view(R, 1)                     # [R,1]
            ok_mc_h_r = (ref_machine == 3).view(R, 1)
            inv_h2 = cand_has_heart & (~(ok_wd_h_r & ok_mc_h_r))

            ok_wd_a_r = ((ref_weekday == 0) | (ref_weekday == 2) | (ref_weekday == 4)).view(R, 1)
            ok_mc_a_r = (ref_machine == 1).view(R, 1)
            inv_a2 = cand_has_angio & (~(ok_wd_a_r & ok_mc_a_r))

            weekend_r = ((ref_weekday == 5) | (ref_weekday == 6)).view(R, 1)
            inv_w2 = cand_has_con & weekend_r

            feasible_spec = ~(inv_h1 | inv_a1 | inv_w1 | inv_h2 | inv_a2 | inv_w2)
            cand_mask = cand_mask & feasible_spec

        any_valid = cand_mask.any(dim=1)  # [R]

        cand_type = self._patient_main_exam_id[cand_gene]  # [R,W]
        ref_type = self._patient_main_exam_id[ref_gene]    # [R]

        denom = float(max(N - 1, 1))
        ref_pos_norm = (idx1.float() / denom).view(R, 1)              # [R,1]
        cand_pos_norm = (cand_pos_safe.float() / denom).view(R, W, 1) # [R,W,1]
        rel_offset_norm = offsets.float() / float(max(window, 1))     # [W]

        policy = self.step2_pos_policy
        if not policy.training:
            with torch.no_grad():
                logits = policy(ref_type, ref_pos_norm, ref_day_norm, ref_weekday, ref_machine,
                               cand_type, cand_pos_norm, cand_day_norm, cand_weekday, cand_machine,
                               rel_offset_norm, cand_mask)
        else:
            logits = policy(ref_type, ref_pos_norm, ref_day_norm, ref_weekday, ref_machine,
                           cand_type, cand_pos_norm, cand_day_norm, cand_weekday, cand_machine,
                           rel_offset_norm, cand_mask)

        logits = logits.masked_fill(~cand_mask, -1e9)

        if deterministic:
            pick = torch.argmax(logits, dim=1)  # [R]
            logprob = None
        else:
            dist = torch.distributions.Categorical(logits=logits)
            pick = dist.sample()
            logprob = dist.log_prob(pick)

        idx2 = cand_pos_safe.gather(1, pick.view(R, 1)).squeeze(1)  # [R]
        idx2 = torch.where(any_valid, idx2, (idx1 + 1) % N)

        self.last_step2_rows = rows.detach()
        self.last_step2_idx1 = idx1.detach()
        self.last_step2_idx2 = idx2.detach()
        # 训练时必须保留计算图；如需日志/保存用 detach 版本，请单独存一个
        self.last_step2_logprob = logprob
        return idx2

    def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
        valid_indices = row[row < self.N]
        return [self._idx_to_cid[int(x)] for x in valid_indices.tolist()]

    def initialize_population(self):
        if self.N == 0:
            print("警告：患者列表为空，无法初始化种群。")
            return
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
            self._cid_to_idx = {cid: i for i, cid in enumerate(self._idx_to_cid)}

        indices = torch.arange(self.N, device=DEVICE)
        block_size = max(30, self.N // 20)
        pop_indices = torch.empty((self.K, self.B, self.N), dtype=DTYPE_LONG, device=DEVICE)
        rand_matrices = torch.rand(self.K, self.B, self.N, device=DEVICE)
        
        for i in range(0, self.N, block_size):
            end = min(i + block_size, self.N)
            block_len = end - i
            if block_len == 0: continue
            block_rand = rand_matrices[:, :, i:end]
            block_perm_idx = torch.argsort(block_rand, dim=2)
            block_indices = indices[i:end]
            block_indices_expanded = block_indices.expand(self.K, self.B, -1)
            pop_indices[:, :, i:end] = torch.gather(block_indices_expanded, 2, block_perm_idx)

        self.population_tensor = pop_indices
        print(f"已生成 {self.K} 个并行种群 (每个 {self.B} 个个体)，总计 {self.total_pop_size} 个个体")

    # def _heterogeneous_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
    #     """
    #     纯 GPU 实现的异构交叉。
    #     逻辑：Child 继承 P1 的有效基因集合。
    #     其中属于 (P1 ∩ P2) 的基因，在 Child 中的相对顺序参考 P2。
    #     属于 (P1 - P2) 的基因，保持在 P1 中的相对位置。
    #     """
    #     B, L = P1.shape
    #     dummy = self.N
        
    #     # 1. 识别有效区域
    #     P1_valid = P1 < dummy
    #     P2_valid = P2 < dummy
        
    #     # 2. 识别 Common 元素 (在 P1 中且在 P2 中)
    #     # 利用 Row Offset 技巧进行行内比较
    #     row_ids = torch.arange(B, device=DEVICE).unsqueeze(1) * (self.N + 1)
        
    #     P1_off = P1 + row_ids
    #     P2_off = P2 + row_ids
        
    #     # isin(A, B): 检查 A 的元素是否在 B 中
    #     # Mask for P1: P1 的哪些位置包含 Common 元素
    #     common_mask_p1 = torch.isin(P1_off, P2_off) & P1_valid
        
    #     # Mask for P2: P2 的哪些位置包含 Common 元素 (用于提取值，蕴含了 P2 的顺序)
    #     common_mask_p2 = torch.isin(P2_off, P1_off) & P2_valid
        
    #     # 3. 提取 P2 中的 Common 值
    #     # masked_select 返回 1D Tensor，按行优先顺序平铺
    #     # 只要每行 Common 元素的数量一致（集合交集必然一致），顺序就会自动对齐
    #     common_vals = torch.masked_select(P2, common_mask_p2)
        
    #     # 4. 填入 P1
    #     Child = P1.clone()
    #     # masked_scatter_ 会按顺序消费 common_vals 并填入 mask 为 True 的位置
    #     Child.masked_scatter_(common_mask_p1, common_vals)
        
    #     return Child
    def _heterogeneous_crossover_batch_gpu(self, P1: torch.Tensor, P2: torch.Tensor) -> torch.Tensor:
        """
        [对齐版] 异构交叉：Child 继承 P1 的有效基因集合；
        对于 (P1 ∩ P2) 的基因，在 Child 中的相对顺序参考 P2；
        (P1 - P2) 的基因保持在 P1 的相对位置。

        重要：支持“每行交集长度不同”，不会再因为 masked_select 扁平化导致行间错位。
        """
        B, L = P1.shape
        dummy = self.N

        P1_valid = P1 < dummy
        P2_valid = P2 < dummy

        # 用 row offset 把“同值但不同行”区分开，加速 isin 且避免跨行误匹配
        row_off = (torch.arange(B, device=DEVICE, dtype=P1.dtype) * (dummy + 1)).unsqueeze(1)
        P1_off = P1 + row_off
        P2_off = P2 + row_off

        common_p1 = P1_valid & torch.isin(P1_off, P2_off)
        common_p2 = P2_valid & torch.isin(P2_off, P1_off)

        # 在各自行内做 rank（True 的位置得到 0..m-1）
        rank_p2 = common_p2.cumsum(dim=1) - 1
        rank_p1 = common_p1.cumsum(dim=1) - 1

        row_idx = torch.arange(B, device=DEVICE).unsqueeze(1).expand(B, L)

        # 1) 构造每行按 P2 顺序排列的 common 值表 common_vals[b, r] = P2 中第 r 个 common
        common_vals = torch.full((B, L), dummy, dtype=P1.dtype, device=DEVICE)
        flat_r = row_idx[common_p2]
        flat_c = rank_p2[common_p2]
        flat_v = P2[common_p2]
        common_vals[flat_r, flat_c] = flat_v

        # 2) 构造 rank -> P1 列位置 的映射 pos_p1[b, r] = P1 中第 r 个 common 的列号
        pos_p1 = torch.full((B, L), -1, dtype=torch.long, device=DEVICE)
        col_idx = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
        flat_r1 = row_idx[common_p1]
        flat_c1 = rank_p1[common_p1]
        flat_v1 = col_idx[common_p1]
        pos_p1[flat_r1, flat_c1] = flat_v1

        # 3) 逐 rank 回填：Child[P1_common_positions] = common_vals（严格行内对齐）
        m = common_p1.sum(dim=1)  # [B]
        rank_mat = torch.arange(L, device=DEVICE).unsqueeze(0).expand(B, L)
        use = rank_mat < m.unsqueeze(1)

        flat_r2 = row_idx[use]
        flat_pos = pos_p1[use]
        flat_val2 = common_vals[use]

        ok = flat_pos >= 0
        children = P1.clone()
        children[flat_r2[ok], flat_pos[ok]] = flat_val2[ok]
        return children
    
    def _heterogeneous_mutate_violations_gpu(self, sub_pop: torch.Tensor, violate_mask: torch.Tensor) -> torch.Tensor:
        """
        步骤1: 纯 GPU 实现的异构定点变异（基于违规掩码）。
        [修改说明]：变异范围限制在违规位置的 +/- 400 内。
        """
        B, L = sub_pop.shape
        dummy = self.N

        # 1. 找出存在违规的行
        has_violation = violate_mask.any(dim=1)
        viol_rows_idx = torch.nonzero(has_violation).flatten()
        
        if viol_rows_idx.numel() == 0:
            return sub_pop

        # 2. 在违规行中，选择一个违规位置 (Source)
        subset_mask = violate_mask[viol_rows_idx]
        viol_idx_in_row = torch.multinomial(subset_mask.float(), 1, replacement=True).flatten()
        
        # 3. 选择一个目标交换位置 (Target) - 限制在 +/- 400 范围内
        # 获取选中行的有效长度
        current_lens = (sub_pop[viol_rows_idx] < dummy).sum(dim=1)
        
        # 计算局部窗口范围
        low = torch.clamp(viol_idx_in_row - 400, min=0)
        high = torch.clamp(viol_idx_in_row + 400, max=current_lens - 1)
        span = high - low + 1
        
        # 在窗口内随机选择位置
        rand_offset = (torch.rand(viol_rows_idx.numel(), device=DEVICE) * span.float()).long()
        target_idx_in_row = low + rand_offset

        # 避免原地交换 (如果窗口大于1)
        target_idx_in_row = torch.where(
            (target_idx_in_row == viol_idx_in_row) & (span > 1),
            torch.where(target_idx_in_row < high, target_idx_in_row + 1, target_idx_in_row - 1),
            target_idx_in_row
        )

        # 4. 执行交换
        row_idx = viol_rows_idx
        idx1 = viol_idx_in_row
        idx2 = target_idx_in_row
        
        val1 = sub_pop[row_idx, idx1]
        val2 = sub_pop[row_idx, idx2]
        
        sub_pop[row_idx, idx1] = val2
        sub_pop[row_idx, idx2] = val1
        
        return sub_pop

    def _heterogeneous_mutate_general_gpu(self, sub_pop: torch.Tensor, prob: float = 1) -> torch.Tensor:
        """
        步骤2: 纯 GPU 实现的异构一般变异（随机交换）。
        [修改说明]：变异范围限制在选中位置的 +/- 400 内。
        """
        B, L = sub_pop.shape
        dummy = self.N
        
        # 1. 确定哪些行需要变异
        do_mutate = torch.rand(B, device=DEVICE) < prob
        
        # 2. 计算每行有效长度
        valid_lens = (sub_pop < dummy).sum(dim=1)
        
        # 过滤掉长度 < 2 的行
        can_mutate = do_mutate & (valid_lens >= 2)
        indices = torch.nonzero(can_mutate).flatten()
        
        if indices.numel() == 0:
            return sub_pop
            
        # 3. 对需要变异的行操作
        target_lens = valid_lens[indices]
        
        # 生成第一个随机位置 idx1
        rand1 = torch.rand(indices.numel(), device=DEVICE)
        idx1 = (rand1 * target_lens.float()).long()
        
        # 计算 idx1 周围 +/- 400 的窗口
        low = torch.clamp(idx1 - 400, min=0)
        high = torch.clamp(idx1 + 400, max=target_lens - 1)
        span = high - low + 1
        
        # 生成第二个位置 idx2
        rand_offset = (torch.rand(indices.numel(), device=DEVICE) * span.float()).long()
        idx2 = low + rand_offset
        
        # 避免原地交换
        idx2 = torch.where(
            (idx2 == idx1) & (span > 1),
            torch.where(idx2 < high, idx2 + 1, idx2 - 1),
            idx2
        )
        
        # 4. 执行交换
        row_idx = indices
        
        val1 = sub_pop[row_idx, idx1]
        val2 = sub_pop[row_idx, idx2]
        
        sub_pop[row_idx, idx1] = val2
        sub_pop[row_idx, idx2] = val1
        
        return sub_pop

    def _heterogeneous_mutate_greedy_cluster_gpu(self, sub_pop: torch.Tensor, greedy_prob: float = 0.5) -> torch.Tensor:
        """
        步骤3: 纯 GPU 实现的异构贪婪聚类变异。
        """
        B, L = sub_pop.shape
        dummy = self.N
        
        # 1. 筛选需要变异的行
        probs = torch.rand(B, device=DEVICE)
        valid_lens = (sub_pop < dummy).sum(dim=1)
        
        # 条件：概率满足 且 有效长度足够支持聚类 (例如至少5个)
        can_mutate = (probs < greedy_prob) & (valid_lens >= 5)
        indices = torch.nonzero(can_mutate).flatten()
        R = indices.numel()
        if R == 0:
            return sub_pop

        # 2. 确定窗口
        target_lens = valid_lens[indices] # [R]
        
        # 窗口大小：在 [2, min(50, target_len)] 之间
        max_possible_window = torch.clamp(target_lens, max=50)
        rand_w = torch.rand(R, device=DEVICE)
        window_lens = (rand_w * (max_possible_window - 2).float()).long() + 2
        
        # 确定起始位置
        max_starts = target_lens - window_lens
        rand_s = torch.rand(R, device=DEVICE)
        starts = (rand_s * (max_starts + 1).float()).long()
        
        # 3. 提取并排序
        max_w_len_batch = window_lens.max().item()
        
        arng = torch.arange(max_w_len_batch, device=DEVICE).unsqueeze(0).expand(R, -1)
        mask_window = arng < window_lens.unsqueeze(1)
        
        # 构造 gather indices
        gather_cols = starts.unsqueeze(1) + arng
        gather_cols = torch.clamp(gather_cols, max=L-1)
        
        rows = sub_pop[indices]
        windows = torch.gather(rows, 1, gather_cols)
        
        # 获取 keys (Main Exam IDs)
        keys = self._patient_main_exam_id[windows]
        keys[~mask_window] = -1
        
        # Calculate counts and sort
        E = self._E
        keys_clamped = keys.clamp(min=0, max=E-1)
        one_hot = torch.nn.functional.one_hot(keys_clamped, num_classes=E)
        one_hot[~mask_window] = 0
        
        counts = one_hot.sum(dim=1) # [R, E]
        
        size_per_pos = torch.gather(counts, 1, keys_clamped)
        size_per_pos[~mask_window] = -1
        
        sort_score = (-size_per_pos).to(torch.int64) * (max_w_len_batch + 1) + arng
        sort_score[~mask_window] = 9223372036854775807 # Push invalid to end
        
        sort_indices = torch.argsort(sort_score, dim=1)
        
        # Gather sorted windows
        sorted_windows = torch.gather(windows, 1, sort_indices)
        
        # 4. Scatter back (Manual 2D Scatter)
        new_rows = rows.clone()
        
        # Flatten indices and values for assignment
        row_idx_expanded = torch.arange(R, device=DEVICE).unsqueeze(1).expand(-1, max_w_len_batch)
        
        flat_r = row_idx_expanded[mask_window]
        flat_c = gather_cols[mask_window]
        flat_v = sorted_windows[mask_window]
        
        new_rows[flat_r, flat_c] = flat_v
        
        # Assign back to sub_pop
        sub_pop[indices] = new_rows
        
        return sub_pop

    # def _evolve_heterogeneous_sub_pop(self, sub_pop: torch.Tensor, sub_engine: _GPUMatrixFitnessBatch, generations: int):
    #     """
    #     子种群进化循环 (In-place, 全 GPU)。
    #     包含：交叉、定点变异、一般变异、贪婪聚类。
    #     """
    #     K, B, L = sub_pop.shape
        
    #     old_main_exam_id = self._patient_main_exam_id
    #     self._patient_main_exam_id = self.patient_main_exam_id_all
        
    #     # 0. 初始评估
    #     pop_flat = sub_pop.view(K*B, L)
    #     out = sub_engine.fitness_batch(pop_flat)
    #     current_fitness = out['fitness'].view(K, B)
    #     # 获取初始的违规 Mask
    #     current_viol_mask = out['any_violate_mask_b_n'].view(K, B, L)
        
    #     for _ in range(generations):
    #         # 准备 P1 (即当前种群)
    #         P1_flat = sub_pop.view(K*B, L)
    #         # P1 对应的违规 Mask
    #         P1_viol_mask_flat = current_viol_mask.view(K*B, L)
            
    #         # 准备 P2 (同组内随机打乱)
    #         idx_in_b = torch.randint(0, B, (K, B), device=DEVICE)
    #         batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B
    #         flat_p2_idx = (batch_offsets + idx_in_b).view(-1)
    #         P2_flat = P1_flat[flat_p2_idx]
            
    #         # 1. 交叉 (Cross)
    #         children_flat = self._heterogeneous_crossover_batch_gpu(P1_flat, P2_flat)
            
    #         # 2. 定点变异 (Mutation - Violations)
    #         children_flat = self._heterogeneous_mutate_violations_gpu(children_flat, P1_viol_mask_flat)
            
    #         # 3. 一般变异 (Mutation - General)
    #         children_flat = self._heterogeneous_mutate_general_gpu(children_flat, prob=0.8)

    #         # 4. 贪婪聚类变异 (Mutation - Greedy Cluster)
    #         children_flat = self._heterogeneous_mutate_greedy_cluster_gpu(children_flat, greedy_prob=0.3)
            
    #         # 5. 评估子代
    #         out_child = sub_engine.fitness_batch(children_flat)
    #         child_fitness = out_child['fitness'].view(K, B)
    #         child_viol_mask = out_child['any_violate_mask_b_n'].view(K, B, L)
            
    #         # 6. 选择 (Selection) - (1+1) 策略
    #         children = children_flat.view(K, B, L)
    #         mask_better = child_fitness > current_fitness
    #         mask_better_exp = mask_better.unsqueeze(2).expand(K, B, L)
            
    #         sub_pop = torch.where(mask_better_exp, children, sub_pop)
    #         current_fitness = torch.where(mask_better, child_fitness, current_fitness)
    #         current_viol_mask = torch.where(mask_better_exp, child_viol_mask, current_viol_mask)
            
    #     self._patient_main_exam_id = old_main_exam_id
    #     return sub_pop
    def _evolve_heterogeneous_sub_pop(self, sub_pop: torch.Tensor, sub_engine: _GPUMatrixFitnessBatch, generations: int):
        """
        子种群进化循环 (In-place, 全 GPU)。
        包含：交叉、定点变异、一般变异、贪婪聚类。

        [关键改动]：P2 选择为“同组内随机其它个体”，强制非 self，
        让子阶段恢复跨个体的有效重组。
        """
        K, B, L = sub_pop.shape

        old_main_exam_id = self._patient_main_exam_id
        self._patient_main_exam_id = self.patient_main_exam_id_all

        # 0. 初始评估
        pop_flat = sub_pop.view(K * B, L)
        out = sub_engine.fitness_batch(pop_flat)
        current_fitness = out["fitness"].view(K, B)
        current_viol_mask = out["any_violate_mask_b_n"].view(K, B, L)

        for _ in range(generations):
            # 准备 P1
            P1_flat = sub_pop.view(K * B, L)
            P1_viol_mask_flat = current_viol_mask.view(K * B, L)

            # ====== P2：同组内随机“其它个体”（强制不等于自身） ======
            if B > 1:
                # 对每个 (k, b)，随机一个 delta ∈ [1, B-1]，让 idx2 = (b + delta) % B
                delta = torch.randint(1, B, (K, B), device=DEVICE)
                base = torch.arange(B, device=DEVICE).unsqueeze(0).expand(K, -1)
                idx_in_b = (base + delta) % B
            else:
                # B==1 时没法跨个体，退化为自身
                idx_in_b = torch.zeros((K, B), dtype=torch.long, device=DEVICE)

            batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B

            # ====== P2：从本 run 的精英池采样（强制 != self） ======
            elite_pool = min(8, B)  # 你可以调 5~16；B 小就自动收缩
            elite_idx = torch.topk(current_fitness, k=elite_pool, dim=1, largest=True).indices  # [K, elite_pool]

            # 给每个 (k,b) 随机挑一个精英
            pick = torch.randint(0, elite_pool, (K, B), device=DEVICE)                          # [K,B]
            idx_in_b = elite_idx.gather(1, pick)                                                # [K,B] in [0,B)

            # 强制 P2 != P1（如果撞上了，就换成精英池下一个）
            base = torch.arange(B, device=DEVICE).unsqueeze(0).expand(K, B)
            idx_in_b = torch.where(idx_in_b == base, elite_idx[:, 0:1].expand(K, B), idx_in_b)

            batch_offsets = torch.arange(K, device=DEVICE).unsqueeze(1) * B
            flat_p2_idx = (batch_offsets + idx_in_b).reshape(-1)
            P2_flat = P1_flat[flat_p2_idx]
            # =========================================================

            # 1) 交叉
            children_flat = self._heterogeneous_crossover_batch_gpu(P1_flat, P2_flat)

            # 2) 定点变异（基于违规 mask）
            children_flat = self._heterogeneous_mutate_violations_gpu(children_flat, P1_viol_mask_flat)

            # 3) 一般变异（局部交换）
            children_flat = self._heterogeneous_mutate_general_gpu(children_flat, prob=0.8)

            # 4) 贪婪聚类变异
            children_flat = self._heterogeneous_mutate_greedy_cluster_gpu(children_flat, greedy_prob=0.3)

            # 5) 评估子代
            out_child = sub_engine.fitness_batch(children_flat)
            child_fitness = out_child["fitness"].view(K, B)
            child_viol_mask = out_child["any_violate_mask_b_n"].view(K, B, L)

            # 6) 选择 (1+1)
            children = children_flat.view(K, B, L)
            mask_better = child_fitness > current_fitness
            mask_better_exp = mask_better.unsqueeze(2).expand(K, B, L)

            sub_pop = torch.where(mask_better_exp, children, sub_pop)
            current_fitness = torch.where(mask_better, child_fitness, current_fitness)
            current_viol_mask = torch.where(mask_better_exp, child_viol_mask, current_viol_mask)

        self._patient_main_exam_id = old_main_exam_id
        return sub_pop

    def run_coevolution_phase(self, co_gens=50):      
        pop_flat = self.population_tensor.view(self.total_pop_size, self.N)
        out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=True)
        assigned_days = out['assigned_day'] 
        
        max_day = assigned_days.max().item()
        
        for day_start in range(0, int(max_day) + 1, 7):
            day_end = day_start + 7
            
            # Mask: [Total_P, N]
            mask = (assigned_days >= day_start) & (assigned_days < day_end)
            
            # --- 全 GPU 切分与 Padding ---
            # 1. 计算每行选中的元素数量
            row_lens = mask.sum(dim=1) # [Total_P]
            max_len = row_lens.max().item()
            
            if max_len < 2: continue
            
            # 2. 准备输出容器 (Padding with Dummy)
            sub_pop_padded = torch.full((self.total_pop_size, max_len), self.N, dtype=DTYPE_LONG, device=DEVICE)
            mask_int = mask.long()
            # cumsum 得到每一行内的累积计数，减 1 得到 0-based index
            col_indices = mask_int.cumsum(dim=1) - 1
            # 仅保留 mask 为 True 位置的 col_indices
            valid_col_indices = torch.masked_select(col_indices, mask)
            row_indices = torch.arange(self.total_pop_size, device=DEVICE).unsqueeze(1).expand(-1, self.N)
            valid_row_indices = torch.masked_select(row_indices, mask)
            valid_values = torch.masked_select(self.population_tensor.view(self.total_pop_size, self.N), mask)
            sub_pop_padded[valid_row_indices, valid_col_indices] = valid_values
            
            sub_pop_gpu = sub_pop_padded.view(self.K, self.B, max_len)
            
            # --- 准备子引擎 ---
            shifted_reg_offsets = self.reg_day_offsets_all - day_start
            
            base_date = self.block_start_date if self.block_start_date else START_DATE.date()
            base_weekday = base_date.isoweekday() - 1
            new_start_weekday = (base_weekday + day_start) % 7
            
            sub_engine = _GPUMatrixFitnessBatch(
                weekday_machine_minutes=self._weekday_machine_minutes,
                start_weekday=new_start_weekday,
                patient_durations=self.patient_durations_all,
                reg_day_offsets=shifted_reg_offsets, 
                is_self_selected=self.is_self_selected_all,
                has_contrast=self.has_contrast_all,
                has_heart=self.has_heart_all,
                has_angio=self.has_angio_all,
                patient_main_type_id=self.patient_main_type_id_all, 
                patient_exam_mask=self.patient_exam_mask_all,
                machine_exam_mask=self._machine_exam_mask,
                patient_main_exam_id=self.patient_main_exam_id_all,
                exam_count=self._E
            )
            
            # --- 进化 ---
            evolved_sub_pop = self._evolve_heterogeneous_sub_pop(sub_pop_gpu, sub_engine, co_gens)
            
            # --- 全 GPU 合并 ---
            # evolved_sub_pop: [K, B, Max_Len]
            evolved_flat = evolved_sub_pop.view(self.total_pop_size, max_len)            
            # 构造一个 range 矩阵 [Total_P, Max_Len]
            range_mat = torch.arange(max_len, device=DEVICE).unsqueeze(0).expand(self.total_pop_size, -1)
            row_lens_exp = row_lens.unsqueeze(1).expand(-1, max_len)
            evolved_valid_mask = range_mat < row_lens_exp
            
            # 提取进化后的有效值 (1D)
            evolved_valid_values = torch.masked_select(evolved_flat, evolved_valid_mask)
            self.population_tensor.view(self.total_pop_size, self.N).masked_scatter_(mask, evolved_valid_values)

    def evolve_gpu(self, generations=100, elite_size=5):
        self._ensure_gpu_engine()
        if self.population_tensor is None:
            raise RuntimeError("种群为空，请先 initialize_population")
        
        pop = self.population_tensor
        N = self.N
        
        for gen_idx in range(generations):
            if gen_idx > 0 and gen_idx % 50 == 0:
                self.run_coevolution_phase(co_gens=50)
                pop = self.population_tensor

            pop_flat = pop.view(self.total_pop_size, N)
            out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
            fitness = out['fitness'].view(self.K, self.B)
            viol_mask_flat = out['any_violate_mask_b_n'] 
            violate_mask = viol_mask_flat.view(self.K, self.B, N)
            
            topk_vals, topk_idx = torch.topk(fitness, k=self.B, largest=True, dim=1)
            best_fitness_per_run = topk_vals[:, 0].cpu().tolist()
            for k in range(self.K):
                self.fitness_history[k].append(best_fitness_per_run[k])
            
            elite_size = min(elite_size, self.B)
            elite_idx = topk_idx[:, :elite_size]
            idx_expanded = elite_idx.unsqueeze(2).expand(self.K, elite_size, N)
            elites = torch.gather(pop, 1, idx_expanded)

            parent_count = max(1, int(0.2 * self.B))
            parent_idx = topk_idx[:, :parent_count]
            idx_expanded = parent_idx.unsqueeze(2).expand(self.K, parent_count, N)
            parents = torch.gather(pop, 1, idx_expanded)
            parent_viol = torch.gather(violate_mask, 1, idx_expanded)

            num_children = self.B - elite_size
            if num_children > 0:
                p_idx1 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE)
                p_idx2 = torch.randint(0, parent_count, (self.K, num_children), device=DEVICE)
                P1 = torch.gather(parents, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))
                P2 = torch.gather(parents, 1, p_idx2.unsqueeze(2).expand(-1, -1, N))
                Vmask_choice = torch.gather(parent_viol, 1, p_idx1.unsqueeze(2).expand(-1, -1, N))
                
                P1_flat = P1.view(self.K * num_children, N)
                P2_flat = P2.view(self.K * num_children, N)
                children_flat = self._ordered_crossover_batch_gpu(P1_flat, P2_flat)
                
                Vmask_flat = Vmask_choice.view(self.K * num_children, N)
                children_flat = self._mutate_batch_gpu(children_flat, Vmask_flat, self.current_generation)
                children = children_flat.view(self.K, num_children, N)
                
                pop = torch.cat([elites, children], dim=1)
            else:
                pop = elites.clone()
            
            self.population_tensor = pop

            if (gen_idx + 1) % 50 == 0:
                avg_best_fit = sum(best_fitness_per_run) / self.K
                flat_viols = (out['heart_cnt'] + out['angio_cnt'] + 
                              out['weekend_cnt'] + out['device_cnt'])
                best_viols = torch.gather(
                    flat_viols.view(self.K, self.B), 
                    1, 
                    topk_idx[:, :1]
                )
                avg_viols = best_viols.float().mean().item()
                print(f"Generation {(gen_idx+1)*2} | Avg Best Fitness (K={self.K}): {avg_best_fit:.2f} | Avg Violations: {avg_viols:.2f}")

            self.current_generation += 1

        print("进化完成。正在提取 K 个最佳个体...")
        pop_flat = pop.view(self.total_pop_size, N)
        final_out = self._gpu_engine.fitness_batch(pop_flat, return_assignment=False)
        final_fitness = final_out['fitness'].view(self.K, self.B)
        
        final_best_vals, final_best_idx_in_B = torch.topk(final_fitness, k=1, dim=1)
        final_best_vals = final_best_vals.flatten()
        idx_expanded = final_best_idx_in_B.unsqueeze(2).expand(self.K, 1, N)
        best_individuals_tensor = torch.gather(pop, 1, idx_expanded).squeeze(1)
        
        best_individuals_cpu = best_individuals_tensor.cpu()
        best_fitnesses_cpu = final_best_vals.cpu().tolist()
        
        results = []
        for k in range(self.K):
            cids = self._tensor_row_to_cids(best_individuals_cpu[k])
            results.append({
                "run_id": k,
                "individual_cids": cids,
                "fitness": best_fitnesses_cpu[k]
            })
            
        self.population_tensor = pop
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
        P2_expanded = P2.unsqueeze(2)
        P1_expanded = P1.unsqueeze(1)
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
        any_viol_per_row = torch.any(parent_violate_mask, dim=1)
        viol_rows_idx = torch.nonzero(any_viol_per_row, as_tuple=False).flatten()
        R = viol_rows_idx.numel()
        if R == 0:
            return X
        viol_mask_subset = parent_violate_mask[viol_rows_idx]
        viol_idx_in_row = torch.multinomial(viol_mask_subset.float(), 1, replacement=True).flatten()
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
        """Step2: 基础交换变异（Base Swap）。

        原版：对每行随机选 idx1，再随机选 idx2（早期限制在 idx1±400）。
        修改：保持 idx1 的随机性，把 idx2 的选择替换为论文式 *reference-conditioned attention*：
             reference = (row, idx1)，候选集合 = idx1±window 内的位置集合，
             由注意力策略输出分布并采样/取 argmax 得 idx2，然后执行 swap。

        若未启用 step2_pos_policy（或缺少必要映射），则退化为原随机逻辑。
        """
        C, N = X.shape
        if N <= 1:
            return X

        probs = torch.rand(C, device=DEVICE)
        rows_to_swap_mask = (probs < base_swap_prob)
        rows_to_swap_idx = torch.nonzero(rows_to_swap_mask, as_tuple=False).flatten()
        R = rows_to_swap_idx.numel()
        if R == 0:
            return X

        idx1 = torch.randint(0, N, (R,), device=DEVICE)

        if self.step2_pos_policy is not None:
            window = int(self.step2_pos_window)
            deterministic = bool(self.step2_pos_deterministic)
            idx2 = self._step2_select_idx2_with_attention(
                X=X,
                rows=rows_to_swap_idx,
                idx1=idx1,
                N=N,
                window=window,
                deterministic=deterministic,
            )
        else:
            use_range_limit = (current_gen <= 10000)
            if use_range_limit:
                low = torch.clamp(idx1 - 400, min=0)
                high = torch.clamp(idx1 + 400, max=N - 1)
                range_size = high - low + 1
                range_size = torch.where(range_size <= 0, 1, range_size)
                rand_offset = torch.floor(torch.rand(R, device=DEVICE) * range_size).long()
                idx2 = low + rand_offset
                idx2 = torch.where(
                    (idx2 == idx1) & (range_size > 1),
                    torch.where(idx1 == low, low + 1, low),
                    idx2
                )
                idx2 = torch.clamp(idx2, 0, N - 1)
            else:
                idx2 = torch.randint(0, N, (R,), device=DEVICE)
                idx2 = torch.where(idx2 == idx1, (idx1 + 1) % N, idx2)

        val1 = X[rows_to_swap_idx, idx1]
        val2 = X[rows_to_swap_idx, idx2]
        X[rows_to_swap_idx, idx1] = val2
        X[rows_to_swap_idx, idx2] = val1
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
        #X = self._mutate_step1_violations(X, parent_violate_mask)
        X = self._mutate_step2_base_swap(X, current_gen, base_swap_prob)
        #X = self._mutate_step3_greedy_cluster(X, greedy_prob)
        return X

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

def main():
    try:
        NUM_PARALLEL_RUNS = 4 
        POP_SIZE_PER_RUN = 50 
        GENERATIONS_TO_RUN = 5000
        
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
        print("✓ 所有数据文件均已找到。")

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        print("\n===== 启动并行遗传算法优化 (Megabatch GPU + 协同进化) =====")
        optimizer = MultiRunOptimizer(
            patients, 
            machine_exam_map, 
            num_parallel_runs=NUM_PARALLEL_RUNS, 
            pop_size_per_run=POP_SIZE_PER_RUN
        )
        # 可选：启用 step2 的论文式位置选择器
        # 用法：export COE_STEP2_POS_CKPT=/path/to/ckpt.pt （若不提供则只初始化随机权重，便于外部训练）
        _step2_ckpt = os.environ.get("COE_STEP2_POS_CKPT", "").strip()
        _step2_det = os.environ.get("COE_STEP2_POS_DETERMINISTIC", "0").strip() in ("1", "true", "True", "YES", "yes")
        if _step2_ckpt:
            print(f"✓ 启用 Step2 位置选择器（ckpt={_step2_ckpt}, deterministic={_step2_det}）")
            optimizer.enable_step2_position_policy(ckpt_path=_step2_ckpt, deterministic=_step2_det)
        else:
            # 如果你希望“无 ckpt 也强制启用并在线训练”，可以在外部显式调用：
            # optimizer.enable_step2_position_policy(ckpt_path=None)
            pass

        
        t0_init = time.perf_counter()
        optimizer.initialize_population()
        t_init = time.perf_counter() - t0_init
        print(f"✓ 已生成 {NUM_PARALLEL_RUNS} 个初始种群，耗时: {t_init:.4f}s")


        print(f"\n开始 {GENERATIONS_TO_RUN} 代进化 (K={NUM_PARALLEL_RUNS})...")
        t0 = time.perf_counter()
        final_results = optimizer.evolve_gpu(generations=GENERATIONS_TO_RUN, elite_size=5)
        
        total_evolution_time = time.perf_counter() - t0
        print(f"\n✓ 进化完成 (K={NUM_PARALLEL_RUNS})，总耗时: {total_evolution_time:.2f}s")

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
            
            xlsx = os.path.join(out_dir, f'final_schedule_RUN{run_id}_{ts}_fit_{best_fitness:.0f}.xlsx')
            final_system = optimizer.generate_schedule(best_individual)
            export_schedule(final_system, patients, xlsx)
            print(f"    ✓ 已导出至 {xlsx}")

        print("\n===== 最终统计 =====")
        mean_fitness = np.mean(all_fitnesses)
        print(f"  最佳适应度 (均值): {mean_fitness:.2f}")

    except Exception as e:
        print(f"运行时错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

'''

训练
CUDA_VISIBLE_DEVICES=1 python /home/preprocess/_funsearch/baseline/0EMBC/消融/train-nomask.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/消融/trans-nomask.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 40000000000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir ./step2_ckpts \
  --save_every 1000 \
  --log_every 50

CUDA_VISIBLE_DEVICES=1 nohup python /home/preprocess/_funsearch/baseline/0EMBC/消融/train-nomask.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/消融/trans-nomask.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 40000000000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir ./step2_ckpts \
  --save_every 1000 \
  --log_every 50 > nomask.log 2>&1 &

继续训练
CUDA_VISIBLE_DEVICES=3 python /home/preprocess/_funsearch/baseline/0EMBC/train.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/trans.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 40000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir /home/preprocess/_funsearch/baseline/step2_ckpts \
  --save_every 1000 \
  --log_every 50 \
  --resume /home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_final.pt

export COE_STEP2_POS_CKPT=/home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_final.pt
export COE_STEP2_POS_CKPT=/home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_NOMASK_step30000.pt
export COE_STEP2_POS_DETERMINISTIC=1

CUDA_VISIBLE_DEVICES=3 python /home/preprocess/_funsearch/baseline/0EMBC/trans.py


'''
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
if torch.cuda.is_available():
    print(f"✓ 检测到 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 未检测到 GPU，将使用 CPU（速度极慢）")
    
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
        out['any_violate_mask_b_n'] = viol_mask_b_n 
        return out


# ===================== GA/ACO 基础类 =====================
class MultiRunOptimizer:
    def __init__(self, patients, machine_exam_map, num_parallel_runs=1, pop_size_per_run=10, block_start_date=None):
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
        self._patient_main_exam_id = None
        self._E = None

    # ------- GPU 引擎准备 -------
    def _ensure_gpu_engine(self):
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
            
            p_exam_types = []
            
            counter: Dict[int, int] = defaultdict(int)
            for _, et, dur, _ in p['exams']:
                etn = clean_exam_name(et)
                total_minutes += int(round(float(dur)))
                exam_types_seq.append(etn)
                p_exam_types.append(etn)
                if E > 0:
                    eidx = exam_to_eidx.get(etn, None)
                    if eidx is not None:
                        patient_exam_mask[i, eidx] = True
                        counter[eidx] += 1
                any_contrast = any_contrast or ('增强' in etn)
                any_heart = any_heart or ('心脏' in etn)
                any_angio = any_angio or ('造影' in etn)
            
            if len(counter) > 0:
                best_cnt = max(counter.values())
                main_eidx = min([k for k, v in counter.items() if v == best_cnt])
                patient_main_exam_id[i] = main_eidx
            
            if p_exam_types:
                main_type = p_exam_types[0]
                patient_main_type_id[i] = exam_to_eidx.get(main_type, 0)
            else:
                patient_main_type_id[i] = 0

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
            patient_main_type_id=patient_main_type_id,
            patient_exam_mask=patient_exam_mask,
            machine_exam_mask=machine_exam_mask,
            patient_main_exam_id=patient_main_exam_id,
            exam_count=E,
        )

    def _tensor_row_to_cids(self, row: torch.Tensor) -> List[Any]:
        if self._idx_to_cid is None:
            self._idx_to_cid = list(self.sorted_patients)
        return [self._idx_to_cid[int(x)] for x in row.tolist()]

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
                        pass
        return system


# ===================== 蚁群算法优化器（GPU 版） =====================
class BlockAntColonyOptimizer(MultiRunOptimizer):
    """
    并行多维蚁群算法（Parallel Multi-Colony ACO）：
    - 同时在 GPU 上运行 K 个独立的蚁群 (K = num_parallel_runs)。
    - 每个蚁群有 A 只蚂蚁 (A = num_ants_per_run)。
    - 总并行度 = K * A。
    - 信息素矩阵 tau 维度升级为: [K, N, N]。
    """

    def __init__(self,
                 patients,
                 machine_exam_map,
                 block_start_date=None,
                 num_parallel_runs: int = 16,  # [关键] 并行蚁群数量 (建议设为 8~32)
                 num_ants_per_run: int = 50,   # [关键] 每个群的蚂蚁数 (建议设为 50~100)
                 num_iterations: int = 2000,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q: float = 100000.0,
                 elite_weight: float = 2.0):
        
        # 初始化父类，建立 GPU 引擎
        # 这里 pop_size_per_run 实际上对应 total_ants_per_run，但父类主要用它来做一些形状推断
        super().__init__(patients, machine_exam_map, 
                         num_parallel_runs=num_parallel_runs, 
                         pop_size_per_run=num_ants_per_run, 
                         block_start_date=block_start_date)

        # ACO 参数
        self.num_ants = num_ants_per_run
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight

        # 运行时变量
        # tau 形状: [K, N, N]
        self._tau = None
        # eta 形状: [1, N] (所有群共享相同的启发式信息)
        self._eta = None
        
        self._tau_min = 1e-4
        self._tau_max = 10.0
        
        # 记录每个群的历史最优 [K, Iter]
        self._history_best_fitness = [] 

    # ------- 启发式信息 η_j -------
    def _build_heuristic_eta(self) -> torch.Tensor:
        """
        构建启发式信息，形状 [1, N]，可广播。
        """
        self._ensure_gpu_engine()
        eng = self._gpu_engine

        durations = eng.patient_durations.to(DEVICE).float()
        reg_offsets = eng.reg_day_offsets.to(DEVICE).float()
        is_self_selected = eng.is_self_selected.to(DEVICE)

        dur_norm = durations / (durations.mean() + 1e-6)
        
        # 登记时间归一化
        reg_min = reg_offsets.min()
        reg_norm = (reg_offsets - reg_min) / ((reg_offsets - reg_min).mean() + 1e-6)

        non_self = (~is_self_selected).float()

        # 启发式公式：偏好短作业、早登记、非自选
        eta = 1.0 / (1.0 + 0.5 * dur_norm + 0.3 * reg_norm) + 0.2 * non_self
        eta = eta.clamp(min=1e-3)
        return eta.unsqueeze(0) # [1, N]

    # ------- 初始化并行信息素矩阵 -------
    def _init_pheromone_matrix(self, N: int):
        """
        初始化 K 个独立的信息素矩阵。
        self._tau: [K, N, N]
        """
        device = DEVICE
        K = self.K # 并行运行数
        tau0 = 1.0
        
        # [K, N, N] 的张量
        self._tau = torch.full((K, N, N), tau0, dtype=torch.float32, device=device)
        self._eta = self._build_heuristic_eta() # [1, N]

    # ------- 构造解 (高度并行化) -------
    def _construct_solutions_parallel(self) -> torch.Tensor:
        """
        同时为 K 个群，每个群 A 只蚂蚁构造解。
        返回 perms: [K, A, N]
        """
        K = self.K
        A = self.num_ants
        N = self._tau.size(1)
        device = self._tau.device

        # 准备数据结构
        # perms: [K, A, N]
        perms = torch.empty((K, A, N), dtype=torch.long, device=device)
        
        # visited: [K, A, N]
        visited = torch.zeros((K, A, N), dtype=torch.bool, device=device)
        
        # 预计算 eta^beta: [1, N]
        eta_beta = self._eta.pow(self.beta) 

        # 辅助索引 [K, A]
        batch_indices = torch.arange(K, device=device).unsqueeze(1).expand(K, A)
        ant_indices = torch.arange(A, device=device).unsqueeze(0).expand(K, A)

        # 逐位置构造
        for pos in range(N):
            # 1. 获取当前位置的信息素
            # self._tau: [K, N, N] -> 取第 pos 行 -> [K, N]
            # unsqueeze -> [K, 1, N] 以便广播给 A 只蚂蚁
            tau_pos = self._tau[:, pos, :].unsqueeze(1) # [K, 1, N]
            
            # 2. 计算概率分子 (Numerator)
            # [K, 1, N] * [1, 1, N] -> [K, 1, N]
            probs_raw = (tau_pos.pow(self.alpha)) * eta_beta.unsqueeze(0)
            
            # 3. 扩展到所有蚂蚁 [K, A, N]
            probs = probs_raw.expand(K, A, N).clone()
            
            # 4. 屏蔽已访问节点
            probs[visited] = 0.0
            
            # 5. 处理全0情况 (防止随机游走死锁)
            row_sums = probs.sum(dim=2, keepdim=True) # [K, A, 1]
            zero_mask = (row_sums <= 1e-9).squeeze(2) # [K, A]
            if zero_mask.any():
                # 对全0行，允许访问任何未访问节点（均匀概率）
                probs[zero_mask] = (~visited[zero_mask]).float()
                row_sums = probs.sum(dim=2, keepdim=True)
            
            # 6. 归一化
            probs.div_(row_sums + 1e-10)
            
            # 7. 采样 (Batch Sampling)
            # torch.multinomial 只能处理 2D，所以需要 reshape
            # [K*A, N]
            probs_flat = probs.view(-1, N)
            chosen_flat = torch.multinomial(probs_flat, 1).squeeze(1) # [K*A]
            
            # 8. 记录
            chosen = chosen_flat.view(K, A)
            perms[:, :, pos] = chosen
            
            # 更新 visited
            # 使用 scatter 更新 visited 矩阵
            # visited[k, a, chosen[k,a]] = True
            # 为了效率，使用 flatten 索引
            # flat_idx = k*A*N + a*N + chosen_val
            # 但这里我们利用高级索引
            visited[batch_indices, ant_indices, chosen] = True

        return perms

    # ------- 信息素挥发 (并行) -------
    def _evaporate_pheromone(self):
        with torch.no_grad():
            self._tau.mul_(1.0 - self.rho)
            self._tau.clamp_(self._tau_min, self._tau_max)

    # ------- 信息素更新 (并行) -------
    def _deposit_pheromone_parallel(self, 
                                    perms: torch.Tensor, 
                                    fitness: torch.Tensor, 
                                    weight: float = 1.0):
        """
        为 K 个群同时更新信息素。
        perms: [K, N] (每个群只传最优的那个蚂蚁路径)
        fitness: [K] (对应的适应度)
        """
        K, N = perms.shape
        device = self._tau.device
        
        # 1. 计算增量 Delta
        # Cost = -Fitness
        costs = -fitness # [K]
        costs = torch.max(costs, torch.tensor(1.0, device=device)) # 避免除零
        
        delta_vals = (self.q * weight) / costs # [K]
        delta_vals = delta_vals / float(N) # 平均分配到每步
        
        # 2. 批量更新
        # 我们需要更新 self._tau[k, pos, perms[k, pos]] += delta_vals[k]
        
        # 构建索引
        k_indices = torch.arange(K, device=device).unsqueeze(1).expand(K, N) # [K, N]
        pos_indices = torch.arange(N, device=device).unsqueeze(0).expand(K, N) # [K, N]
        patient_indices = perms # [K, N]
        
        # 准备增量矩阵 [K, N]
        update_values = delta_vals.unsqueeze(1).expand(K, N)
        
        with torch.no_grad():
            # 高级索引更新
            self._tau[k_indices, pos_indices, patient_indices] += update_values
            # 再次截断，防止溢出
            self._tau.clamp_(self._tau_min, self._tau_max)

    # ------- 主运行循环 -------
    def run_aco(self, verbose: bool = True):
        self._ensure_gpu_engine()
        N = len(self._idx_to_cid)
        K = self.K
        A = self.num_ants
        
        if N == 0: raise RuntimeError("N=0")

        self._init_pheromone_matrix(N)
        
        # 全局最优 (每个群维护一个)
        global_best_fitness = torch.full((K,), -float('inf'), device=DEVICE)
        global_best_perms = torch.zeros((K, N), dtype=torch.long, device=DEVICE)
        
        print(f"🚀 启动并行多维 ACO")
        print(f"   - 并行群数 (Runs): {K}")
        print(f"   - 每群蚂蚁 (Ants): {A}")
        print(f"   - 总并发数 (Batch): {K * A}")
        print(f"   - 显存利用: 信息素矩阵大小 {K}x{N}x{N} ({K*N*N*4/1024/1024:.1f} MB)")

        t_start = time.perf_counter()

        for it in range(self.num_iterations):
            # 1. 并行构造解 [K, A, N]
            batch_perms = self._construct_solutions_parallel()
            
            # 2. 评估解
            # 需要 flatten 成 [K*A, N] 给引擎
            flat_perms = batch_perms.view(-1, N)
            out = self._gpu_engine.fitness_batch(flat_perms, return_assignment=False)
            flat_fitness = out['fitness'] # [K*A]
            
            # Reshape 回 [K, A]
            fitness_matrix = flat_fitness.view(K, A)
            
            # 3. 找到每群的当代最优 (Iteration Best)
            iter_best_vals, iter_best_indices = torch.max(fitness_matrix, dim=1) # [K]
            
            # 提取对应的路径 [K, N]
            # batch_perms[k, iter_best_indices[k], :]
            # 使用 gather 比较方便
            # unsqueeze index: [K, 1, N]
            gather_idx = iter_best_indices.view(K, 1, 1).expand(K, 1, N)
            iter_best_perms = batch_perms.gather(1, gather_idx).squeeze(1) # [K, N]
            
            # 4. 更新全局最优 (Global Best)
            update_mask = iter_best_vals > global_best_fitness
            global_best_fitness[update_mask] = iter_best_vals[update_mask]
            # 仅更新变好的那些群的 perm
            # 这是一个 mask copy 操作
            # 为了简便，我们逐行覆盖 (或使用 where)
            if update_mask.any():
                # 扩展 mask [K, N]
                mask_expanded = update_mask.unsqueeze(1).expand(K, N)
                global_best_perms = torch.where(mask_expanded, iter_best_perms, global_best_perms)
            
            # 5. 信息素更新
            self._evaporate_pheromone()
            
            # 策略：每个群只根据自己的 IterBest 和 GlobalBest 更新自己的 Tau
            # 当代最优强化
            self._deposit_pheromone_parallel(iter_best_perms, iter_best_vals, weight=1.0)
            # 全局最优强化 (Elitism)
            self._deposit_pheromone_parallel(global_best_perms, global_best_fitness, weight=self.elite_weight)
            
            # 6. 日志
            if verbose and ((it + 1) % 50 == 0 or it == 0):
                # 计算 K 个群中的最好值展示
                best_of_all = global_best_fitness.max().item()
                mean_of_all = global_best_fitness.mean().item()
                t_now = time.perf_counter()
                print(f"[Iter {it+1}/{self.num_iterations}] Best: {best_of_all:.0f} | Mean: {mean_of_all:.0f} | FPS: {K*A/(t_now-t_start)* (it+1) :.1f} ants/sec")

        # 结束，返回最好的一个群的结果
        final_best_idx = torch.argmax(global_best_fitness).item()
        best_perm = global_best_perms[final_best_idx]
        best_val = global_best_fitness[final_best_idx].item()
        
        return self._tensor_row_to_cids(best_perm), best_val


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


def main_aco_optimized():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')

        for f in [patient_file, duration_file, device_constraint_file]:
            if not os.path.exists(f):
                print(f"❌ 错误：找不到文件 {f}")
                return
                
        print("✓ 数据文件检查通过。")
        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        # ========= 参数调优建议 =========
        # 300M 显存虽然小，但对于 N=2000 左右的数据：
        # Tau 矩阵: K * 2000 * 2000 * 4 bytes
        # 若 K=8: 8 * 4MB * 4 = 128MB。
        # 加上其他 Tensor，K=8, Ants=100 是安全的。
        # 
        # 如果你的显存其实有更多（比如 4G+），可以把 RUNS 开到 32 或 64。
        # 这里按低配优化：
        
        NUM_RUNS = 8      # 并行运行 8 个独立的蚁群
        ANTS_PER_RUN = 64 # 每个群 64 只蚂蚁
        ITERATIONS = 500  # 迭代次数
        
        print("\n===== 启动并行多维 ACO (High-GPU Utilization) =====")
        optimizer = BlockAntColonyOptimizer(
            patients,
            machine_exam_map,
            num_parallel_runs=NUM_RUNS,
            num_ants_per_run=ANTS_PER_RUN,
            num_iterations=ITERATIONS,
            alpha=1.0,
            beta=2.5,       # 提高启发式权重，加速初期收敛
            rho=0.1,        # 挥发率
            q=100000.0,     # 匹配 Penalty 量级
            elite_weight=3.0
        )

        t0 = time.perf_counter()
        best_individual, best_fitness = optimizer.run_aco(verbose=True)
        total_time = time.perf_counter() - t0
        
        print(f"\n✓ 优化完成。总耗时: {total_time:.2f}s")
        print(f"✓ 全局最佳 Fitness: {best_fitness:.2f}")

        # 导出
        out_dir = 'output_schedules'
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        xlsx = os.path.join(out_dir, f'ACO_Parallel_Best_{ts}_fit_{abs(best_fitness):.0f}.xlsx')
        
        final_system = optimizer.generate_schedule(best_individual)
        export_schedule(final_system, patients, xlsx)
        print(f"✓ 结果已导出: {xlsx}")

    except Exception as e:
        print(f"运行时发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_aco_optimized()
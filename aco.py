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

# 自动选择最佳设备，优先使用 GPU
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

# ===================== 排程系统 (CPU/Export用) =====================
# 用于将优化后的解转化为具体的排班表
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

# ===================== GPU 核心引擎 (保持不变) =====================
# 这部分模拟了 ai4co 中 environment/problem 的角色
# 负责快速评估一组解（路径）的质量

def _weekday_minutes_matrix_from_end_hours(M: int) -> torch.Tensor:
    hours = [int(round((15.0 - WEEKDAY_END_HOURS[d]) * 60)) for d in range(1, 8)]
    return torch.tensor([[m] * M for m in hours], dtype=DTYPE_LONG, device=DEVICE)


def _build_capacity_bins(weekday_machine_minutes: torch.Tensor, start_weekday: int, total_minutes_needed: int):
    # 构建每一天、每一台机器的容量“桶”
    weekday_machine_minutes = weekday_machine_minutes.to(DEVICE)
    M = weekday_machine_minutes.size(1)
    daily_totals = weekday_machine_minutes.sum(dim=1)
    min_daily = torch.clamp(daily_totals.min(), min=1)
    est_days = int((total_minutes_needed // int(min_daily.item())) + 3)
    days_idx = (torch.arange(est_days, device=DEVICE) + start_weekday) % 7
    caps_per_day = weekday_machine_minutes.index_select(0, days_idx)  # [D,M]
    caps_flat = caps_per_day.reshape(-1)
    caps_cumsum = torch.cumsum(caps_flat, dim=0)
    
    # 扩展容量直到足够覆盖所有需求
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
    # 快速计算每个患者落在哪个时间桶里
    T = torch.cumsum(durations_batch, dim=1)
    return torch.searchsorted(caps_cumsum, T, right=False)


def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
    # 计算每个患者在其所在机器时间桶内的顺序（用于精排）
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
                 machine_exam_mask: torch.Tensor | None
                 ):
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
        # 计算序列依赖的换模惩罚（核心）
        B, N = perms.shape
        if TRANSITION_PENALTY <= 0:
            return torch.zeros((B, N), dtype=DTYPE_FLOAT, device=DEVICE)
        current_types = self.patient_main_type_id.index_select(0, perms.reshape(-1)).reshape(B, N)
        prev_types = torch.roll(current_types, shifts=1, dims=1)
        prev_bins = torch.roll(bin_idx_batch, shifts=1, dims=1)
        same_bin = (bin_idx_batch == prev_bins) # 只有在同一机器同一天连续做才算
        diff_type = (current_types != prev_types) # 类型不同才罚
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

        total_penalty_per_patient = p_wait + p_dev + p_spec + p_tran
        fitness = - total_penalty_per_patient.sum(dim=1)

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


# ===================== ACO 主体 (修复与优化版) =====================

class GPUEngineMixin:
    """负责数据预处理和GPU引擎初始化的 Mixin"""
    def __init__(self, patients, machine_exam_map, block_start_date=None):
        self.patients = patients
        self.machine_exam_map = machine_exam_map
        self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
        self.block_start_date = block_start_date
        self._gpu_engine = None
        self._cid_to_idx = None
        self._idx_to_cid = None
        self._patient_main_exam_id = None
        self._E = None

    def _ensure_gpu_engine(self):
        if self._gpu_engine is not None:
            return

        idx_to_cid = list(self.sorted_patients)
        cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
        self._idx_to_cid = idx_to_cid
        self._cid_to_idx = cid_to_idx
        N = len(idx_to_cid)

        # 初始化各种数据张量
        patient_durations = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        reg_day_offsets = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)
        is_self_selected = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_contrast = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_heart = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        has_angio = torch.zeros(N, dtype=torch.bool, device=DEVICE)
        patient_main_type_id = torch.zeros(N, dtype=DTYPE_LONG, device=DEVICE)

        # 构建检查映射
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

        base_date = self.block_start_date if self.block_start_date else START_DATE.date()
        start_weekday = base_date.isoweekday() - 1

        for i, cid in enumerate(idx_to_cid):
            p = self.patients[cid]
            total_minutes = 0
            any_contrast = False
            any_heart = False
            any_angio = False
            p_exam_types = []

            if patient_exam_mask is not None:
                for _, et, dur, _ in p['exams']:
                    etn = clean_exam_name(et)
                    total_minutes += int(round(float(dur)))
                    eidx = exam_to_eidx.get(etn, None)
                    if eidx is not None:
                        patient_exam_mask[i, eidx] = True
                    p_exam_types.append(etn)
                    if '增强' in etn: any_contrast = True
                    if '心脏' in etn: any_heart = True
                    if '造影' in etn: any_angio = True
            else:
                for _, et, dur, _ in p['exams']:
                    etn = clean_exam_name(et)
                    total_minutes += int(round(float(dur)))
                    p_exam_types.append(etn)
                    if '增强' in etn: any_contrast = True
                    if '心脏' in etn: any_heart = True
                    if '造影' in etn: any_angio = True

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
        self._patient_main_exam_id = patient_main_type_id

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
        )


class ACOScheduler(GPUEngineMixin):
    def __init__(self,
                 patients,
                 machine_exam_map,
                 num_ants: int = 100,   # 增加蚂蚁数量以提高搜索广度
                 max_iter: int = 200,
                 alpha: float = 1.0,    # 信息素重要程度
                 beta: float = 2.0,     # 启发式重要程度
                 rho: float = 0.1,      # 信息素挥发率 (Reference code usually uses 0.1)
                 q: float = 100.0,      # 信息素增强强度
                 block_start_date=None):
        super().__init__(patients, machine_exam_map, block_start_date=block_start_date)
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        self._tau: torch.Tensor | None = None          # 路径信息素矩阵 [N, N] (对应 CVRP 的 edge pheromone)
        self._eta_trans: torch.Tensor | None = None    # 换模启发式矩阵 [N, N] (对应 CVRP 的 distance)
        self._eta_urgency: torch.Tensor | None = None  # 紧急度启发式向量 [N] (对应 CVRP 的 demand/time window urgency)
        
        self.best_perm_indices: torch.Tensor | None = None
        self.best_fitness: float = float("-inf")
        self.aco_history: list[float] = []

    def _prepare_aco(self):
        """初始化 ACO 数据结构 (信息素矩阵 + 启发式矩阵)"""
        self._ensure_gpu_engine()
        assert self._idx_to_cid is not None
        N = len(self._idx_to_cid)

        # 1. 初始化路径信息素 [N, N]
        # tau[i, j] 表示从患者 i 之后紧接着做患者 j 的倾向
        self._tau = torch.ones((N, N), dtype=DTYPE_FLOAT, device=DEVICE)

        # 2. 启发式 A: 紧急度 (Urgency Heuristic) [N]
        # 对应 CVRP 中优先服务时间窗紧迫的客户
        base_date = self.block_start_date if self.block_start_date else START_DATE.date()
        reg_offsets = []
        for cid in self._idx_to_cid:
            p = self.patients[cid]
            days = (p["reg_date"] - base_date).days
            reg_offsets.append(days)
        reg_t = torch.tensor(reg_offsets, dtype=DTYPE_FLOAT, device=DEVICE)
        min_day = reg_t.min()
        if min_day < 0:
            reg_t = reg_t - min_day
        # 启发式：等待时间越久(days越小)，分数越高
        self._eta_urgency = 10.0 / (1.0 + reg_t) 

        # 3. 启发式 B: 换模聚类 (Clustering Heuristic) [N, N]
        # 对应 CVRP/TSP 中的距离矩阵 (Distance Matrix)
        # 如果 Type[i] == Type[j]，则 eta[i, j] 很高，相当于距离很近
        types = self._patient_main_exam_id  # [N]
        same_type_mask = (types.unsqueeze(1) == types.unsqueeze(0))
        
        # 基础分 1.0，同类型奖励分 10.0 (减少换模惩罚)
        self._eta_trans = torch.ones((N, N), dtype=DTYPE_FLOAT, device=DEVICE)
        self._eta_trans[same_type_mask] = 10.0
        self._eta_trans.fill_diagonal_(0.0)

        self.best_perm_indices = None
        self.best_fitness = float("-inf")
        self.aco_history = []

    def _construct_ants_batch(self) -> torch.Tensor:
        """
        [向量化构造]
        同时为 num_ants 只蚂蚁构建解。利用 GPU 矩阵运算替代循环。
        """
        N = len(self._idx_to_cid)
        num_ants = self.num_ants
        
        # 结果容器: [num_ants, N]
        paths = torch.zeros((num_ants, N), dtype=DTYPE_LONG, device=DEVICE)
        
        # 访问掩码: [num_ants, N], True 表示已访问
        visited = torch.zeros((num_ants, N), dtype=torch.bool, device=DEVICE)
        
        # 1. 选择起始点 (Step 0)
        # 依据 "紧急度" 概率随机选择起点
        start_probs = self._eta_urgency.expand(num_ants, N) # [A, N]
        current_nodes = torch.multinomial(start_probs, 1).squeeze(1) # [A]
        
        paths[:, 0] = current_nodes
        visited.scatter_(1, current_nodes.unsqueeze(1), True)
        
        # 为了速度，提前取引用
        tau = self._tau            # [N, N]
        eta_trans = self._eta_trans # [N, N]
        eta_urg = self._eta_urgency # [N]
        
        # 2. 逐步构造 (Step 1 to N-1)
        for step in range(1, N):
            # 获取当前节点到所有节点的边信息素 & 启发式
            # current_nodes: [A]
            # Matrix Indexing: [A, N] = Matrix[current_nodes]
            current_tau = tau[current_nodes] 
            current_eta_trans = eta_trans[current_nodes]
            current_eta_urg = eta_urg.unsqueeze(0)
            
            # 计算转移概率 (Transition Probability)
            # P_ij ~ tau_ij^alpha * eta_trans_ij^beta * eta_urg_j^beta
            probs = (current_tau ** self.alpha) * \
                    (current_eta_trans ** self.beta) * \
                    (current_eta_urg ** self.beta)
            
            # 掩盖已访问节点 (Masking)
            probs[visited] = 0.0
            
            # 极小概率修正 (数值稳定性)
            prob_sum = probs.sum(dim=1, keepdim=True)
            zero_mask = (prob_sum <= 1e-9).squeeze(1)
            if zero_mask.any():
                uniform_probs = (~visited[zero_mask]).float()
                probs[zero_mask] = uniform_probs
            
            # 轮盘赌采样 (Sampling)
            next_nodes = torch.multinomial(probs, 1).squeeze(1)
            
            # 更新状态
            paths[:, step] = next_nodes
            visited.scatter_(1, next_nodes.unsqueeze(1), True)
            current_nodes = next_nodes
            
        return paths

    def run_aco(self, max_iter: int | None = None):
        if max_iter is None:
            max_iter = self.max_iter

        self._prepare_aco()
        N = len(self._idx_to_cid)
        print(f"ACO 初始化完成: N={N}, Ants={self.num_ants}, Alpha={self.alpha}, Beta={self.beta}")

        for it in range(max_iter):
            t0 = time.time()
            
            # 1. 批量构造解 (Batch Construction)
            perms_idx = self._construct_ants_batch() # [Ants, N]
            
            # 2. 批量评估 (Batch Evaluation)
            out = self._gpu_engine.fitness_batch(perms_idx, return_assignment=False)
            fitness = out["fitness"] # [Ants]
            
            # 3. 更新最优解 (Elitism)
            iter_best_val, iter_best_idx = torch.max(fitness, dim=0)
            iter_best_val_f = float(iter_best_val.item())
            
            if iter_best_val_f > self.best_fitness:
                self.best_fitness = iter_best_val_f
                self.best_perm_indices = perms_idx[iter_best_idx].detach().clone()
                print(f"[ACO] New Best at iter {it}: {self.best_fitness:.2f}")

            self.aco_history.append(self.best_fitness)
            
            # 4. 信息素更新 (Pheromone Update)
            # 4.1 挥发 (Evaporation)
            self._tau *= (1.0 - self.rho)
            
            # 4.2 增强 (Depositing) - 只增强本轮最优路径
            best_path = perms_idx[iter_best_idx]
            from_nodes = best_path[:-1]
            to_nodes = best_path[1:]
            
            # 使用固定增量或基于 Fitness 的增量
            delta_tau = self.q / float(N) # 简单策略
            
            # 向量化加法
            self._tau[from_nodes, to_nodes] += delta_tau
            
            # 4.3 限制范围 (MMAS Strategy: Min-Max Limits)
            # 防止信息素过大导致收敛过早，或过小导致搜索停滞
            self._tau.clamp_(min=0.01, max=20.0)

            if (it + 1) % 10 == 0:
                print(f"Iter {it+1}/{max_iter} | Best: {self.best_fitness:.2f} | Time: {time.time()-t0:.3f}s")

        if self.best_perm_indices is None:
            return None, None

        best_indices_cpu = self.best_perm_indices.cpu().tolist()
        best_individual = [self._idx_to_cid[int(i)] for i in best_indices_cpu]
        return best_individual, self.best_fitness

    def build_best_schedule(self):
        best_individual, best_fitness = self.run_aco()
        if best_individual is None:
            return None, None

        system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
        for cid in best_individual:
            p = self.patients.get(cid)
            if p and not p.get("scheduled", False):
                for exam in p["exams"]:
                    exam_type = clean_exam_name(exam[1])
                    duration = exam[2]
                    try:
                        m, start_time = system.find_available_slot(duration, exam_type, p)
                        m.add_exam(system.current_date, start_time, duration, exam_type, p)
                    except Exception as e:
                        pass
        return system, best_fitness


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

# ===================== Main =====================

def main():
    multiprocessing.freeze_support()
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
        
        if not os.path.exists(patient_file):
            print(f"找不到文件: {patient_file}")
            return

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        print("\n===== 启动 GPU 向量化蚁群算法 (ACO) =====")
        aco = ACOScheduler(
            patients,
            machine_exam_map,
            num_ants=100,     # GPU并行评估100个解
            max_iter=300,     # 迭代300次
            alpha=1.0,        # 关注历史经验(信息素)
            beta=5.0,         # 重点关注启发式(同类型聚类)
            rho=0.1,          # 标准挥发率
            q=10.0,
        )

        t0 = time.perf_counter()
        system, best_fitness = aco.build_best_schedule()
        total_time = time.perf_counter() - t0
        
        print(f"\n✓ 优化结束，总耗时: {total_time:.2f}s")

        if system is not None:
            print(f"✓ 最佳 Fitness: {best_fitness:.2f}")
            out_dir = 'output_schedules_aco'
            os.makedirs(out_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            xlsx = os.path.join(out_dir, f'aco_gpu_schedule_{ts}.xlsx')
            export_schedule(system, patients, xlsx)
            print(f"✓ 文件已导出: {xlsx}")
        else:
            print("未能生成有效排程")

    except Exception as e:
        print(f"Err: {e}")
        traceback.print_exc()
    finally:
        pass

if __name__ == "__main__":
    main()
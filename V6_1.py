from __future__ import annotations
from typing import List, Dict, Set, Tuple, Any, Collection, Sequence
import pandas as pd
import random as rd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback
import re
import json
import signal
import sys
import types
import multiprocessing
import time
import tiktoken
import http.client
import inspect
import copy  # 导入copy模块
import functools # 新增：GPU实验1的性能分析需要
import torch
from funsearch_impl import config as config_lib
from funsearch_impl import funsearch
from funsearch_impl import code_manipulation
from funsearch_impl import evaluator
from funsearch_impl import sampler

WEEKDAY_END_HOURS = {
    1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7
}
WORK_START = datetime.strptime('07:00', '%H:%M').time()
TRANSITION_PENALTY = 20000
WAIT_PENALTY = 500
LOGICAL = 10000
SELF_SELECTED_PENALTY = 8000
NON_SELF_PENALTY = 800
START_DATE = datetime(2024, 12, 1, 7, 0)
MAX_RETRIES = 3
MACHINE_COUNT = 6
DEVICE_PENALTY = 500000
POPULATION_FILE = 'population_state.json'
BLOCK_SIZE_DAYS = 7
_GLOBAL_OPTIMIZER = None # 保留，以防FunSearch的某些部分可能依赖它（尽管evolve不再依赖）

# ===== 新增：GPU 常量 (来自 GPU_exp1) =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE_LONG = torch.long
DTYPE_FLOAT = torch.float32

# ======================== 新增：性能分析模块 (来自 GPU_exp1) ========================

# --- 全局性能分析存储 ---
TIMER_STORAGE = defaultdict(float)
TIMER_COUNTS = defaultdict(int)


def profile_me(func):
    """一个简单的性能分析装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        TIMER_STORAGE[func.__name__] += elapsed
        TIMER_COUNTS[func.__name__] += 1
        return result
    return wrapper

def print_profiling_report(total_evolution_time):
    """打印收集到的性能分析数据"""
    print(f"\n{'='*60}")
    print(f"遗传算法 (100代) 内部性能分析:")
    print(f"(总进化时间: {total_evolution_time:.2f}s)")
    print(f"{'='*60}")
    
    # 按照总耗时降序排序
    sorted_items = sorted(TIMER_STORAGE.items(), key=lambda item: item[1], reverse=True)
    
    for name, total_time in sorted_items:
        count = TIMER_COUNTS[name]
        avg_time = total_time / count if count > 0 else 0
        percentage = (total_time / total_evolution_time) * 100 if total_evolution_time > 0 else 0
        
        print(f"  模块: {name}")
        print(f"    总耗时: {total_time:.4f}s")
        print(f"    调用次数: {count}")
        print(f"    占总时间: {percentage:.2f}%")

    print(f"\n  > 关键说明:")
    print(f"    - BlockGeneticOptimizer: 'calculate_fitness', 'crossover', 'mutate', 'greedy_cluster_mutation'")
    print(f"    - 'calculate_fitness' (GA) 采用 GPU 粗排。")
    print(f"{'='*60}\n")

def reset_profiler():
    """重置全局分析器状态"""
    TIMER_STORAGE.clear()
    TIMER_COUNTS.clear()

# ======================== 性能分析模块结束 ========================


def clean_exam_name(name):
    """标准化检查项目名称"""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', cleaned)
    cleaned = re.sub(r'[^\w()-]', '', cleaned)
    return cleaned.replace('_', '-').replace(' ', '')


def safe_read_excel(file_path, sheet_name=0):
    """安全读取Excel文件，自动尝试不同引擎"""
    if file_path.endswith('.xlsx'):
        engines = ['openpyxl', 'odf']
    elif file_path.endswith('.xls'):
        engines = ['xlrd']
    else:
        engines = ['openpyxl', 'xlrd', 'odf']
    for engine in engines:
        try:
            return pd.read_excel(file_path, engine=engine, sheet_name=sheet_name)
        except:
            continue

    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")


def import_data(patient_file, duration_file):
    """数据导入函数 (来自 V6)"""
    try:
        duration_df = safe_read_excel(duration_file)
        duration_df['cleaned_exam'] = duration_df['检查项目'].apply(clean_exam_name)
        exam_durations = duration_df.set_index('cleaned_exam')['实际平均耗时'].to_dict()
        patient_df = safe_read_excel(patient_file)
        patients = {}

        for _, row in patient_df.iterrows():
            if pd.isnull(row['id']) or pd.isnull(row['登记日期']):
                continue

            compound_id = (str(row['id']).strip(),
                             pd.to_datetime(row['登记日期']).strftime('%Y%m%d'))

            exam_type = clean_exam_name(row['检查项目'])
            duration = exam_durations.get(exam_type, 15.0)

            is_self_selected = (row['是否自选时间'] == '自选时间')
            appt_date = pd.to_datetime(row['预约日期']).date() if not pd.isnull(row['预约日期']) else None

            if compound_id not in patients:
                patients[compound_id] = {
                    'compound_id': compound_id,
                    'exams': [],
                    'reg_date': pd.to_datetime(compound_id[1]).date(),
                    'is_self_selected': is_self_selected,
                    'appt_date': appt_date,
                    'scheduled': False
                }
            patients[compound_id]['exams'].append([
                str(row['检查部位']).strip(),
                exam_type,
                duration,
                pd.to_datetime(row['登记日期']).date()
            ])

        print(f"成功导入{len(patients)}患者，共{sum(len(p['exams']) for p in patients.values())}个检查")
        return patients
    except Exception as e:
        print(f"数据导入错误: {str(e)}")
        traceback.print_exc()
        raise

def import_device_constraints(file_path):
    """导入设备限制数据 (来自 V6)"""
    try:
        df = safe_read_excel(file_path)
        machine_exam_map = defaultdict(list)
        for _, row in df.iterrows():
            machine_id = int(row['设备']) - 1
            exam = clean_exam_name(row['检查项目'])
            machine_exam_map[machine_id].append(exam)
        return machine_exam_map
    except Exception as e:
        print(f"导入设备限制数据错误: {str(e)}")
        traceback.print_exc()
        raise


class MachineSchedule:
    def __init__(self, machine_id, allowed_exams):
        self.machine_id = machine_id
        self.allowed_exams = allowed_exams
        self.timeline = defaultdict(list)
        self.day_end_time = defaultdict(lambda: None)  # ✅ 缓存每天的结束时间
        self._work_end_cache = {}  # ✅ 缓存工作结束时间

    def get_work_end(self, date):
        """缓存工作结束时间计算"""
        if date not in self._work_end_cache:
            weekday = date.isoweekday()
            base_time = datetime.combine(date, WORK_START)
            work_duration = 15.0 - WEEKDAY_END_HOURS[weekday]
            self._work_end_cache[date] = base_time + timedelta(hours=work_duration)
        return self._work_end_cache[date]

    def add_exam(self, date, start_time, duration_minutes, exam_type, patient_info):
        """O(1) 追加操作"""
        duration = timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        
        self.timeline[date].append((
            start_time, end_time, exam_type,
            patient_info['compound_id'][0],
            patient_info['reg_date'],
            patient_info['is_self_selected']
        ))
        
        # ✅ 直接更新结束时间，无需遍历
        self.day_end_time[date] = end_time
        return end_time


class SchedulingSystem:
    """多机器排程系统（保留，用于导出时精排；适配度阶段不再调用）"""

    def __init__(self, machine_exam_map, start_date=None):
        self.machines = []
        for machine_id in range(MACHINE_COUNT):
            allowed_exams = machine_exam_map.get(machine_id, [])
            self.machines.append(MachineSchedule(machine_id, allowed_exams))
        if start_date is None:
            self.current_date = START_DATE.date()
        else:
            self.current_date = start_date

        self.current_machine = 0
        self.start_date = self.current_date

    def reset(self):
        """重置排程系统状态（注意：MachineSchedule 未实现 reset_timeline，未使用）"""
        self.current_date = self.start_date
        self.current_machine = 0

    def find_available_slot(self, duration_minutes, exam_type, patient_info):
        """优化后的O(1)查找（导出时使用）"""
        duration = timedelta(minutes=duration_minutes)
        max_iterations = 365
        
        for _ in range(max_iterations):
            machine = self.machines[self.current_machine]
            last_end_time = machine.day_end_time[self.current_date]
            start_time = datetime.combine(self.current_date, WORK_START) if last_end_time is None else last_end_time
            end_time = start_time + duration
            work_end = machine.get_work_end(self.current_date)
            if end_time <= work_end:
                return machine, start_time
            self.move_to_next()
        raise TimeoutError("无法在365天内找到可用时段")

    def move_to_next(self):
        self.current_machine += 1
        if self.current_machine >= MACHINE_COUNT:
            self.current_machine = 0
            self.current_date += timedelta(days=1)


# ======================== 新增：GPU 粗排批量适配度引擎 (来自 GPU_exp1) ========================

def _weekday_minutes_matrix_from_end_hours(num_machines: int) -> torch.Tensor:
    hours = [15.0 - WEEKDAY_END_HOURS[d] for d in range(1, 8)]  # 1..7
    minutes = [int(round(h * 60)) for h in hours]
    mat = torch.tensor([[m] * num_machines for m in minutes], dtype=DTYPE_LONG, device=DEVICE)  # [7, M]
    return mat


def _build_capacity_bins(weekday_machine_minutes: torch.Tensor, start_weekday: int, total_minutes_needed: int):
    weekday_machine_minutes = weekday_machine_minutes.to(DEVICE)
    M = weekday_machine_minutes.size(1)
    daily_totals = weekday_machine_minutes.sum(dim=1)
    min_daily = torch.clamp(daily_totals.min(), min=1)
    est_days = int((total_minutes_needed // int(min_daily.item())) + 3)
    days_idx = (torch.arange(est_days, device=DEVICE) + start_weekday) % 7
    caps_per_day = weekday_machine_minutes.index_select(dim=0, index=days_idx)   # [D, M]
    caps_flat = caps_per_day.reshape(-1)
    caps_cumsum = torch.cumsum(caps_flat, dim=0)
    while caps_cumsum[-1].item() < total_minutes_needed:
        caps_cumsum = torch.cat([caps_cumsum, caps_cumsum[-1] + torch.cumsum(caps_flat, dim=0)])
        caps_per_day = torch.cat([caps_per_day, caps_per_day], dim=0)
        caps_flat = caps_per_day.reshape(-1)
    D = caps_per_day.size(0)
    B = D * M
    bin_idx = torch.arange(B, device=DEVICE)
    bin_day = bin_idx // M
    bin_machine = bin_idx % M
    caps_len = caps_cumsum.size(0)
    return caps_cumsum, bin_day[:caps_len], bin_machine[:caps_len]


def _assign_bins_batch_by_prefix(durations_batch: torch.Tensor, caps_cumsum: torch.Tensor) -> torch.Tensor:
    T_batch = torch.cumsum(durations_batch, dim=1)
    return torch.searchsorted(caps_cumsum, T_batch, right=False)


def _compute_order_in_bin_row(bin_idx_row: torch.Tensor) -> torch.Tensor:
    N = bin_idx_row.numel()
    arng = torch.arange(N, device=bin_idx_row.device)
    sort_idx = torch.argsort(bin_idx_row, stable=True)
    bin_sorted = bin_idx_row[sort_idx]
    is_start = torch.ones(N, dtype=torch.bool, device=bin_idx_row.device)
    is_start[1:] = bin_sorted[1:] != bin_sorted[:-1]
    start_pos = torch.where(is_start, arng, torch.full((N,), -1, device=bin_idx_row.device))
    last_start_pos = torch.cummax(start_pos, dim=0)[0]
    rank_in_sorted = arng - last_start_pos
    order_idx = torch.empty_like(rank_in_sorted)
    order_idx[sort_idx] = rank_in_sorted
    return order_idx


def _compute_order_in_bin_batch(bin_idx_batch: torch.Tensor) -> torch.Tensor:
    B, N = bin_idx_batch.shape
    out = torch.empty_like(bin_idx_batch)
    for b in range(B):
        out[b] = _compute_order_in_bin_row(bin_idx_batch[b])
    return out



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
# ======================== 遗传算法 (修改版) ========================

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

    def _individual_to_perm(self, individual: List[Any]) -> torch.Tensor:
        """(来自 GPU_exp1)"""
        idxs = [self._cid_to_idx[cid] for cid in individual]
        return torch.tensor(idxs, dtype=DTYPE_LONG, device=DEVICE)

    def evaluate_population_gpu(self, population: List[List[Any]]):
        """(来自 GPU_exp1)"""
        self._ensure_gpu_engine()
        B = len(population)
        N = len(self._idx_to_cid)
        perms = torch.stack([self._individual_to_perm(ind) for ind in population], dim=0)  # [B,N]
        out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
        fitness = out['fitness'].tolist()
        heart_cnt = out['heart_cnt'].tolist()
        angio_cnt = out['angio_cnt'].tolist()
        weekend_cnt = out['weekend_cnt'].tolist()
        device_cnt = out['device_cnt'].tolist()
        violate_mask = out['any_violate_mask'].cpu().numpy()  # [B,N]
        indiv_viol_sets = []
        for b in range(B):
            bad_idxs = np.where(violate_mask[b])[0].tolist()
            bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
            indiv_viol_sets.append(bad_cids)
        results = []
        for i, ind in enumerate(population):
            results.append((ind, float(fitness[i]), int(heart_cnt[i]), int(angio_cnt[i]), int(device_cnt[i]), int(weekend_cnt[i]), indiv_viol_sets[i]))
        return results

    # ========== GA 常规 (来自 V6) ==========

    def initialize_population(self, pop_size=None):
        """初始化种群 (来自 V6)"""
        if pop_size is None:
            pop_size = self.pop_size

        block_size = max(30, len(self.sorted_patients) // 20)

        blocks = [
            self.sorted_patients[i:i + block_size]
            for i in range(0, len(self.sorted_patients), block_size)
        ]

        self.population = []
        for _ in range(pop_size):
            individual = []
            for block in blocks:
                shuffled_block = rd.sample(block, len(block))
                individual.extend(shuffled_block)
            self.population.append(individual)
        print(f"已生成包含{len(self.population)}个个体的种群")

    # ========== 分块进化 (来自 V6) - 保留，但注意：main() 并未调用 ==========
    def split_population_into_blocks(self):
        """根据实际检查日期将种群划分为块"""
        exam_dates = self.scheduling_system.get_exam_dates(self.population[0], self.patients)

        min_date = min(exam_dates.values())
        max_date = max(exam_dates.values())

        date_blocks = []
        current_start = min_date
        while current_start <= max_date:
            current_end = current_start + timedelta(days=BLOCK_SIZE_DAYS - 1)
            date_blocks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        patient_blocks = defaultdict(list)
        for cid, exam_date in exam_dates.items():
            for idx, (start, end) in enumerate(date_blocks):
                if start <= exam_date <= end:
                    patient_blocks[idx].append(cid)
                    break

        block_populations = defaultdict(list)
        for individual in self.population:
            individual_blocks = defaultdict(list)
            for cid in individual:
                exam_date = exam_dates.get(cid, min_date)
                for block_idx, (start, end) in enumerate(date_blocks):
                    if start <= exam_date <= end:
                        individual_blocks[block_idx].append(cid)
                        break
                else:
                    individual_blocks[len(date_blocks) - 1].append(cid)

            for block_idx, patient_ids in individual_blocks.items():
                block_populations[block_idx].append(patient_ids)

        return block_populations, date_blocks

    def create_block_optimizers(self, block_populations, date_blocks):
        """为每个时间块创建优化器"""
        block_optimizers = {}

        for block_idx, population_block in block_populations.items():
            block_patients = {}
            for cid in {pid for ind in population_block for pid in ind}:
                if cid in self.patients:
                    block_patients[cid] = self.patients[cid]

            if not block_patients:
                continue

            block_start_date = date_blocks[block_idx][0]

            block_opt = BlockGeneticOptimizer(
                block_patients,
                self.machine_exam_map,
                pop_size=len(population_block),
                block_start_date=block_start_date
            )
            block_opt.population = population_block
            block_optimizers[block_idx] = block_opt

        return block_optimizers

    def merge_blocks(self, block_optimizers):
        """将进化后的块合并回完整种群"""
        sorted_blocks = sorted(block_optimizers.keys())

        merged_population = []
        num_individuals = len(block_optimizers[sorted_blocks[0]].population) if sorted_blocks else 0

        for i in range(num_individuals):
            full_individual = []
            for block_idx in sorted_blocks:
                if i < len(block_optimizers[block_idx].population):
                    full_individual.extend(block_optimizers[block_idx].population[i])
            merged_population.append(full_individual)

        return merged_population

    def block_evolution(self, block_generations=100, full_generations=100):
        """分块进化过程"""
        print("\n=== 开始分块进化 ===")

        block_populations, date_blocks = self.split_population_into_blocks()
        print(f"已将种群划分为 {len(block_populations)} 个时间块")

        block_optimizers = self.create_block_optimizers(block_populations, date_blocks)

        for block_idx, block_opt in block_optimizers.items():
            block_start = date_blocks[block_idx][0]
            print(f"时间块 #{block_idx + 1} ({block_start} 至 {date_blocks[block_idx][1]}) 开始进化...")
            try:
                start_gen = block_opt.current_generation
                best_ind, best_fitness = block_opt.evolve(block_generations)
                end_gen = block_opt.current_generation
                print(
                    f"时间块 #{block_idx + 1} 进化完成: 第{start_gen + 1}-{end_gen}代, 最佳适应度: {best_fitness:.2f}")
            except Exception as e:
                print(f"时间块 #{block_idx + 1} 进化出错: {str(e)}")
                traceback.print_exc()

        self.population = self.merge_blocks(block_optimizers)
        print("已合并所有子种群")

        print("开始整体进化...")
        start_gen = self.current_generation
        best_ind, best_fitness = self.evolve(full_generations)
        end_gen = self.current_generation
        print(f"整体进化完成: 第{start_gen + 1}-{end_gen}代, 最佳适应度: {best_fitness:.2f}")

        return best_ind, best_fitness
    
    def evolve(self, generations=100, elite_size=5):
            try:
                start_gen = self.current_generation
                end_gen = start_gen + generations
                expected_count = len(self.patients)

                for gen in range(start_gen, end_gen):
                    self.current_generation = gen + 1

                    is_check_generation = (self.current_generation == 100) or \
                                        (self.current_generation == end_gen)

                    if is_check_generation:
                        for i, individual in enumerate(self.population):
                            if len(individual) != expected_count:
                                print(f"❌ [致命错误] 第{self.current_generation}代，个体#{i} 长度异常！")
                                print(f"   预期: {expected_count}, 实际: {len(individual)}")
                                return None

                            if len(set(individual)) != expected_count:
                                print(f"❌ [致命错误] 第{self.current_generation}代，个体#{i} 发现重复ID！")
                                return None

                    eval_results = self.evaluate_population_gpu(self.population)

                    scored = []
                    indiv_violations_by_id = {}

                    for ind, fit, h, a, d, w, violset in eval_results:
                        if fit <= -1e14:
                            return None
                        scored.append((ind, fit))
                        indiv_violations_by_id[id(ind)] = set(violset)

                    scored.sort(key=lambda x: x[1], reverse=True)
                    best_fitness = scored[0][1]
                    self.fitness_history.append(best_fitness)

                    # =========================================================
                    # [新增] 仅在最后一代输出结果
                    if self.current_generation == end_gen:
                        print(f"Generation {gen + 1} | Best Fitness: {best_fitness:.2f}")
                    # =========================================================

                    new_population = [ind.copy() for ind, _ in scored[:elite_size]]
                    parents = [ind for ind, _ in scored[:int(0.2 * len(scored))]]

                    if not parents:
                        parents = new_population if new_population else ([scored[0][0]] if scored else [])
                        if not parents:
                            return None

                    while len(new_population) < len(self.population):
                        parent1, parent2 = rd.choices(parents, k=2)
                        child = self.crossover(parent1, parent2)

                        p_choice = parent1 if rd.random() < 0.5 else parent2
                        parent_viol = indiv_violations_by_id.get(id(p_choice), set())

                        mutated_child = self.mutate(child.copy(), parent_viol)
                        new_population.append(mutated_child)

                    self.population = new_population

                return self.population

            except KeyboardInterrupt:
                print("用户中断，已保存当前状态")
                raise
            except Exception as e:
                print(f"❌ 进化过程发生未捕获异常: {e}")
                traceback.print_exc()
                return None
        
    @profile_me # 新增 (来自 GPU_exp1)
    def mutate(self, individual, parent_violations=None, base_rate=0.3):
        """(来自 V6，但添加了 @profile_me)"""
        current_gen = self.current_generation
        use_range_limit = current_gen <= 10000

        # 优先处理违规患者（100%概率触发）
        if parent_violations and rd.random() < 1:
            violating = [cid for cid in individual if cid in parent_violations]

            if len(violating) >= 1:
                try:
                    violator = rd.choice(violating)
                    violator_idx = individual.index(violator)
                    if use_range_limit:
                        low_bound = max(0, violator_idx - 400)
                        high_bound = min(len(individual) - 1, violator_idx + 400)
                        possible_positions = [
                            i for i in range(low_bound, high_bound + 1)
                            if i != violator_idx
                        ]
                    else:
                        possible_positions = [
                            i for i in range(len(individual))
                            if i != violator_idx
                        ]

                    if possible_positions:
                        other_idx = rd.choice(possible_positions)
                        individual[violator_idx], individual[other_idx] = (
                            individual[other_idx], individual[violator_idx]
                        )
                except (ValueError, IndexError):
                    pass

        # 基础变异：随机交换两个位置（添加范围限制）
        if rd.random() < 0.95:
            if use_range_limit:
                idx1 = rd.randint(0, len(individual) - 400)
                low_bound = max(0, idx1 - 400)
                high_bound = min(len(individual) - 1, idx1 + 400)
                possible_positions = [
                    i for i in range(low_bound, high_bound + 1)
                    if i != idx1
                ]
                if possible_positions:
                    idx2 = rd.choice(possible_positions)
                    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                else:
                    idx2 = rd.choice([i for i in range(len(individual)) if i != idx1])
                    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            else:
                idx1, idx2 = rd.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        if rd.random() < 0.5:
            individual = self.greedy_cluster_mutation(individual)

        return individual

    @profile_me # 新增 (来自 GPU_exp1)
    def greedy_cluster_mutation(self, individual):
        """(来自 V6，但添加了 @profile_me)"""
        if len(individual) < 50:
            return individual
        start = rd.randint(0, len(individual) - 50)
        end = start + rd.randint(20, 50)
        window = individual[start:end]
        exam_groups = defaultdict(list)
        for cid in window:
            exam_types = [clean_exam_name(e[1]) for e in self.patients[cid]['exams']]
            main_exam = max(set(exam_types), key=exam_types.count)
            exam_groups[main_exam].append(cid)
        sorted_groups = sorted(exam_groups.items(), key=lambda x: -len(x[1]))
        clustered = []
        for group in sorted_groups:
            clustered.extend(group[1])
        return individual[:start] + clustered + individual[end:]

    @profile_me # 新增 (来自 GPU_exp1)
    def crossover(self, parent1, parent2):
        """(来自 V6，但添加了 @profile_me)"""
        start, end = sorted(rd.sample(range(len(parent1)), 2))
        fragment = set(parent1[start:end + 1])
        child = [g for g in parent2 if g not in fragment]
        return child[:start] + parent1[start:end + 1] + child[start:]

    # ========== 核心替换：calculate_fitness (来自 GPU_exp1) ==========
    @profile_me
    def calculate_fitness(self, schedule):
        """
        (已替换为 GPU_exp1 版本) 计算适应度并返回违规统计：
        返回 (fitness, heart_cnt, angio_cnt, device_cnt, weekend_cnt, local_violations_set)
        """
        if schedule is None or not isinstance(schedule, list) or len(schedule) == 0:
            return -float('inf'), 0, 0, 0, 0, set()
        
        self._ensure_gpu_engine()
        
        # 将 [cid1, cid2] 转换为 [idx1, idx2] 的 [1,N] 张量
        perms = self._individual_to_perm(schedule).unsqueeze(0)  # [1,N]
        
        # 调用 GPU 引擎
        out = self._gpu_engine.fitness_batch(perms, return_assignment=False)
        
        # 解析结果
        fitness = float(out['fitness'][0].item())
        heart_cnt = int(out['heart_cnt'][0].item())
        angio_cnt = int(out['angio_cnt'][0].item())
        device_cnt = int(out['device_cnt'][0].item())
        weekend_cnt = int(out['weekend_cnt'][0].item())
        
        # 找出违规的 cids
        mask = out['any_violate_mask'][0].cpu().numpy()
        bad_idxs = np.where(mask)[0].tolist()
        bad_cids = {self._idx_to_cid[i] for i in bad_idxs}
        
        return fitness, heart_cnt, angio_cnt, device_cnt, weekend_cnt, bad_cids

    def save_intermediate_schedule(self, individual, gen, save_dir):
        """(来自 V6)"""
        system = self.generate_schedule(individual)
        filename = os.path.join(save_dir, f'排程_第{gen}代.xlsx')
        export_schedule(system, self.patients, filename)
        print(f"已保存第 {gen} 代排程结果至 {filename}")
        
    def generate_schedule(self, individual):
        system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
        for cid in individual:
            patient = self.patients.get(cid)
            if patient and not patient['scheduled']:
                for exam in patient['exams']:
                    exam_type = clean_exam_name(exam[1])
                    duration = exam[2]
                    try:
                        machine, start_time = system.find_available_slot(duration, exam_type, patient)
                        machine.add_exam(
                            system.current_date,
                            start_time,
                            duration,
                            exam_type,
                            patient
                        )
                    except Exception as e:
                        print(f"排程错误: {str(e)}")
        return system

def export_schedule(system, patients, filename):
    """(来自 V6)"""
    with pd.ExcelWriter(filename) as writer:
        all_records = []
        for machine in system.machines:
            for date in sorted(machine.timeline):
                sorted_slots = sorted(machine.timeline[date], key=lambda x: x[0])
                for slot in sorted_slots:
                    start, end, exam_type, pid, reg_date, is_self = slot
                    all_records.append({
                        '机器编号': machine.machine_id,
                        '日期': date.strftime('%Y-%m-%d'),
                        '开始时间': start.strftime('%H:%M:%S'),
                        '结束时间': end.strftime('%H:%M:%S'),
                        '检查项目': exam_type,
                        '患者ID': pid,
                        '登记日期': reg_date.strftime('%Y-%m-%d'),
                        '是否自选': '是' if is_self else '否'
                    })
        df = pd.DataFrame(all_records)
        if not df.empty:
            df_sorted = df.sort_values(by=['机器编号', '日期', '开始时间'])
            df_sorted.to_excel(writer, sheet_name='总排程', index=False)
        else:
            # 确保即使为空也写入表头
            pd.DataFrame(columns=[
                '机器编号','日期','开始时间','结束时间','检查项目',
                '患者ID','登记日期','是否自选'
            ]).to_excel(writer, sheet_name='总排程', index=False)


# ======================== 提取原始函数 (来自 V6) ========================
ORIGINAL_FUNCTIONS = {
    'evolve': inspect.getsource(BlockGeneticOptimizer.evolve),
    'mutate': inspect.getsource(BlockGeneticOptimizer.mutate),
    'crossover': inspect.getsource(BlockGeneticOptimizer.crossover),
    'greedy_cluster_mutation': inspect.getsource(BlockGeneticOptimizer.greedy_cluster_mutation)
}

import textwrap 

class MedicalLLM(sampler.LLM):
    """
    为医疗排程优化的LLM接口 (来自 V6)
    (V6.1 鲁棒性修复：已修改为使用 JSON 解析)
    """

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)

        optimizer_class_code = inspect.getsource(BlockGeneticOptimizer)

        self.additional_prompt = (
            "你需要优化BlockGeneticOptimizer类中的遗传算法函数。\n\n"
            "**完整的BlockGeneticOptimizer类定义如下**：\n"
            "```python\n"
            f"{optimizer_class_code}\n"
            "```\n\n"
            "**你的任务**：优化4个方法，返回一个包含代码字符串的JSON对象。\n\n"
            "**严格的输出格式要求**：\n"
            "1. 你的输出 **必须** 是一个单一、有效的JSON对象字符串。\n"
            "2. JSON对象必须包含4个键: `evolve`, `mutate`, `crossover`, `greedy_cluster_mutation`。\n"
            "3. 每个键的值 **必须** 是一个包含完整Python函数定义的字符串 (包括 `def` 关键字和正确的缩进)。\n"
            "4. 字符串必须使用 `\\n` 表示换行，并正确转义引号 (如 `\\\"` )。\n"
            "5. **不要** 在JSON对象之外添加任何文本、解释或Markdown标记 (如 ```json 或 ```python)。\n\n"
            "**示例格式 (你的输出应直接是这个JSON字符串)**：\n"
            "```\n"
            "{\n"
            "  \"evolve\": \"def evolve(self, generations=100, elite_size=5):\\n    try:\\n        # 你的优化代码...\\n    except KeyboardInterrupt:\\n        raise\",\n"
            "  \"mutate\": \"def mutate(self, individual, parent_violations=None, base_rate=0.3):\\n    # 你的优化代码...\\n    return individual\",\n"
            "  \"crossover\": \"def crossover(self, parent1, parent2):\\n    # 你的优化代码...\\n    return child\",\n"
            "  \"greedy_cluster_mutation\": \"def greedy_cluster_mutation(self, individual):\\n    # 你的优化代码...\\n    return individual\"\n"
            "}\n"
            "```\n\n"
            "**关键提醒**：\n"
            "- 确保 `evolve` 字符串中的函数签名是：`def evolve(self, generations=100, elite_size=5):`\n"
            "- 确保 `mutate` 字符串中的函数签名是：`def mutate(self, individual, parent_violations=None, base_rate=0.3):`\n"
            "- 确保 `crossover` 字符串中的函数签名是：`def crossover(self, parent1, parent2):`\n"
            "- 确保 `greedy_cluster_mutation` 字符串中的函数签名是：`def greedy_cluster_mutation(self, individual):`\n"
            "- 你的整个回复 **必须** 从 `{` 开始，到 `}` 结束。\n\n"

            "**优化方向**：\n"
            "- evolve: 调整精英比例、父代选择策略\n"
            "- mutate: 调整变异概率(0.7-0.9)、变异策略组合（插入、倒置）\n"
            "- crossover: 改进交叉点选择、片段大小、或尝试其他交叉算子（PMX, CX）\n"
            "- greedy_cluster_mutation: 优化窗口大小、聚类方式\n"

            "**禁止使用的错误模式（必须严格遵守以避免崩溃）**：\n"
            "1.  **类属性与方法 (AttributeError)**：\n"
            "    - ❌ 禁止：调用 **不存在** 的类方法。`BlockGeneticOptimizer` 类 **没有**没有tournament_selection `self.select_parents`, `self.tournament_selection`, `self.insert_mutation`, `self.inversion_mutation` 等方法。\n"
            "    - ❌ 禁止：访问 **不存在** 的类属性。`elite_size` 是 `evolve` 函数的参数，不是 `self.elite_size` 属性。\n"
            "    \n"
            "2.  **函数签名 (TypeError)**：\n"
            "    - ❌ 禁止：**绝对不能** 修改4个目标函数的签名（参数列表）。必须保持原样。\n"
            "\n"
            "3.  **变量作用域 (NameError)**：\n"
            "    - ❌ 禁止：使用未定义的变量（如 `name 'parent1' is not defined`）。\n"
            "    - ❌ 禁止：调用未定义的函数。必须在函数内部完整实现所有使用的函数。\n"
            "    - ❌ 禁止：在循环外使用循环变量（如 `for cid in ...` 循环结束后使用 `cid`）。\n"
            "\n"
            "4.  **随机函数参数 (ValueError / TypeError)**：\n"
            "    - ❌ 禁止：在 `rd.choices()` 中使用 `weights` 参数。\n"
            "    - ❌ 禁止：在 `rd.choices()` 中使用 `replace` 参数。\n"
            "    - ❌ 禁止：`rd.sample(population, k)` 其中 `k > len(population)` 或 `k` 为负数。\n"
            "    - ❌ 禁止：`a, b = rd.choice(my_list)`（应使用 `a, b = rd.choices(my_list, k=2)`）。\n"
            "\n"
            "5.  **数据类型与操作 (TypeError)**：\n"
            "    - ❌ 禁止：使用 `complex` 复数。所有计算必须使用 `int` 或 `float`。\n"
            "    - ❌ 禁止：将 `int` 当作列表使用（如 `my_int[0]`）。\n"
            "    - ❌ 禁止：将 `list` 用作 `set` 元素或 `dict` 的键。`cid` 必须是 `tuple`。\n"
            "    - ❌ 禁止：修改 `tuple`（元组是不可变的）。\n"
            "\n"
            "6.  **列表/索引操作 (IndexError / TypeError / ValueError)**：\n"
            "    - ❌ 禁止：使用 `tuple` 作为列表索引（如 `my_list[('a', 'b')]`）。\n"
            "    - ❌ 禁止：`list.index(item)` 而不检查 `item` 是否在列表中。\n"
            "    - ❌ 禁止：索引超出列表范围。\n"
            "    - ❌ 禁止：循环顺序错误（如 `for e in self.patients[cid]['exams'] for cid in window`）。\n"
        )
        self._trim = trim
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model = 'qwen'

    def draw_samples(self, prompt: str) -> Collection[str]:
        """生成多个样本"""
        samples = []
        for i in range(self._samples_per_prompt):
            sample = self._draw_sample(prompt)
            samples.append(sample)

        print("\n" + "=" * 60)
        print(f"LLM生成了 {len(samples)} 个样本 (已转换为FunSearch函数体)")
        print("=" * 60)
        for i, sample in enumerate(samples, 1):
            print(f"\n样本 {i} (插入到FunSearch的代码):")
            print("-" * 60)
            print(sample[:500] + "..." if len(sample) > 500 else sample)
            print("-" * 60)
        print("=" * 60 + "\n")

        return samples

    def cal_usage_LLM(self, prompt: str, response: str, encoding_name="cl100k_base"):
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            self.prompt_tokens += len(encoding.encode(prompt))
            self.completion_tokens += len(encoding.encode(response))
            print(f"Token使用: 提示={self.prompt_tokens}, 补全={self.completion_tokens}")
        except Exception as e:
            print(f"Token计算错误: {str(e)}")

    def _draw_sample(self, content: str) -> str:
        """调用LLM生成样本"""
        full_prompt = f"{content}\n\n{self.additional_prompt}"

        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                conn = http.client.HTTPSConnection("api.siliconflow.cn")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer sk-gsfslzsxuczuqyyhrbsjpmswvwepkbziwiltarhzenmflqaq",
                }
                payload = json.dumps({
                    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.7
                })

                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)

                if 'choices' not in data or len(data['choices']) == 0:
                    raise Exception(f"API返回格式错误")

                response = data['choices'][0]['message']['content']
                self.cal_usage_LLM(full_prompt, response)

                if self._trim:
                    parsed_body = self._parse_json_response(response)
                    if parsed_body:
                        return parsed_body
                    else:
                        return self._get_original_body()
                else:
                    print("❌ Trimming (JSON解析) 被禁用，但对 V6.1 是必需的。")
                    return self._get_original_body()

            except Exception as e:
                retry_count += 1
                print(f"API调用错误 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    time.sleep(2)
                else:
                    return self._get_original_body()

    def _parse_json_response(self, response_text: str) -> str | None:
        """
        (V6.1 鲁棒性修复)
        解析LLM的JSON输出，并将其转换为 get_evolution_functions 的函数体。
        """
        json_str = ""
        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
                print(f"❌ (JSON解析) 无法在回复中找到 {{...}} 块。")
                return None

            json_str = response_text[start_idx : end_idx + 1]
            
            # 2. 解析JSON
            data = json.loads(json_str)
            
            # 3. 验证键
            required_keys = {'evolve', 'mutate', 'crossover', 'greedy_cluster_mutation'}
            if not isinstance(data, dict):
                print(f"❌ (JSON解析) 解析出的不是字典。")
                return None
            
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                print(f"❌ (JSON解析) JSON中缺少键: {missing_keys}")
                return None
                
            # 4. 验证值 (必须是字符串)
            for key in required_keys:
                if not isinstance(data[key], str):
                    print(f"❌ (JSON解析) 键 '{key}' 的值不是字符串。")
                    return None
                if not data[key].strip().startswith("def"):
                    print(f"❌ (JSON解析) 键 '{key}' 的值不是以 'def' 开头的函数。")
                    return None

            formatted_dict_string = json.dumps(data, indent=4)
            
            # 构造 get_evolution_functions 的函数体
            function_body = f"""
    # LLM 生成的 JSON 已成功解析
    func_dict = {formatted_dict_string}
    return func_dict
"""
            print("✓ (JSON解析) 成功解析LLM的JSON输出。")
            return function_body

        except json.JSONDecodeError as e:
            print(f"❌ (JSON解析) JSON解码失败: {e}")
            print(f"    失败的文本 (前500字符): {json_str[:500]}...")
            return None
        except Exception as e:
            print(f"❌ (JSON解析) 解析时发生未知错误: {e}")
            traceback.print_exc()
            return None

    def _get_original_body(self) -> str:
        print("ℹ️ (JSON解析) LLM输出无效或解析失败。返回原始函数。")
        original_dict = {
            'evolve': ORIGINAL_FUNCTIONS['evolve'],
            'mutate': ORIGINAL_FUNCTIONS['mutate'],
            'crossover': ORIGINAL_FUNCTIONS['crossover'],
            'greedy_cluster_mutation': ORIGINAL_FUNCTIONS['greedy_cluster_mutation'],
        }
        formatted_dict_string = json.dumps(original_dict, indent=4).replace("{", "{{").replace("}", "}}")
        
        function_body = f"""
    # LLM输出无效，返回原始函数
    original_dict = {formatted_dict_string}
    return original_dict
"""
        return function_body

class MedicalSandbox(evaluator.Sandbox):
    """(来自 V6)"""
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            test_input: str,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """执行医疗排程评估"""
        try:
            print(f"\n{'=' * 60}")
            print(f"Sandbox.run 被调用")
            print(f"program长度: {len(program)} 字符")
            print(f"{'=' * 60}\n")

            if isinstance(inputs, dict) and test_input in inputs:
                data = inputs[test_input]
            else:
                print(f"警告: 无法找到测试输入")
                return None, False

            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._execute_medical,
                args=(program, data, result_queue) 
            )
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
                process.terminate()
                process.join()
                print("❌ 评估超时")
                return None, False

            if not result_queue.empty():
                result = result_queue.get()
                return result
            else:
                print("❌ 评估队列为空 (子进程可能已崩溃)")
                return None, False

        except Exception as e:
            print(f"Sandbox执行错误: {str(e)}")
            traceback.print_exc()
            return None, False

    def _setup_optimizer_with_functions(self, namespace, dataset, initial_population, queue):
        """
        (来自 V6)
        设置优化器并绑定进化函数
        """
        # ========== 1. 验证必需函数存在 ==========
        required_funcs = ['get_evolution_functions', 'evaluate_schedules']
        for func_name in required_funcs:
            if func_name not in namespace:
                print(f"❌ 未找到{func_name}函数")
                queue.put((None, False))
                return None, False
        print(f"✓ 找到所有必需函数: {required_funcs}")
        
        # ========== 2. 获取并验证进化函数字典 ==========
        try:
            func_dict = namespace['get_evolution_functions']()
            if not isinstance(func_dict, dict):
                print(f"❌ 返回类型错误: {type(func_dict)}")
                queue.put((None, False))
                return None, False
        except Exception as e:
            print(f"❌ 调用get_evolution_functions失败: {e}")
            traceback.print_exc()
            queue.put((None, False))
            return None, False
        
        required_keys = ['evolve', 'mutate', 'crossover', 'greedy_cluster_mutation']
        missing = set(required_keys) - set(func_dict.keys())
        if missing:
            print(f"❌ 缺少键: {missing}")
            queue.put((None, False))
            return None, False
        print(f"✓ 获得完整函数字典: {list(func_dict.keys())}")
        
        # ========== 3. 定义并绑定进化函数 ==========
        for func_name in required_keys:
            try:
                func_code = func_dict[func_name]
                if not isinstance(func_code, str):
                    print(f"❌ {func_name}的代码不是字符串")
                    queue.put((None, False))
                    return None, False
                
                try:
                    func_code = textwrap.dedent(func_code)
                except Exception as e:
                    print(f"❌ {func_name} de-indentation 失败: {e}")
                    queue.put((None, False))
                    return None, False
                      
                exec(func_code, namespace)
                if func_name not in namespace or not callable(namespace[func_name]):
                    print(f"❌ {func_name}定义失败或不可调用")
                    queue.put((None, False))
                    return None, False
                print(f"✓ 成功定义 {func_name}")
                
            except Exception as e:
                print(f"❌ 定义{func_name}失败: {e}")
                traceback.print_exc()
                queue.put((None, False))
                return None, False
        
        # ========== 4. 创建优化器并注入初始种群 ==========
        print("\n创建优化器实例...")
        optimizer = BlockGeneticOptimizer(
            dataset['patients'],
            dataset['machine_exam_map'],
            pop_size=len(initial_population)
        )
        optimizer.population = copy.deepcopy(initial_population)
        print(f"✓ 成功注入固定的初始种群 (大小: {len(optimizer.population)})")
        
        # ========== 5. 绑定新函数到优化器 ==========
        for func_name in required_keys:
            setattr(optimizer, func_name, types.MethodType(namespace[func_name], optimizer))
            print(f"✓ 绑定{func_name}")
        
        return optimizer, True

    def _execute_medical(self, program, dataset, queue):
        """(来自 V6)"""
        import hashlib
        start_time = time.time()
        
        try:
            initial_population = dataset.get('initial_population')
            if not initial_population or not isinstance(initial_population, list):
                print("❌ 错误：初始种群无效")
                queue.put((None, False)) # 修复：失败
                return
            print(f"✓ 成功加载固定的初始种群 (大小: {len(initial_population)})")
        

            allowed_names = [
                'rd', 'np', 're', 'os', 'time', 'traceback', 'types', 'pd', 'datetime', 'timedelta', 'defaultdict', 'copy', 
                'max', 'len', 'range', 'sorted', 'set', 'dict', 'str', 'list', 'int', 'enumerate', 'print', 'callable', 'clean_exam_name',
                'MachineSchedule', 'SchedulingSystem', 'BlockGeneticOptimizer', 'export_schedule','POPULATION_FILE', 'WEEKDAY_END_HOURS', 'WORK_START', 'START_DATE', 
                'MACHINE_COUNT', 'DEVICE_PENALTY', 'TRANSITION_PENALTY', 'LOGICAL', 'SELF_SELECTED_PENALTY', 'NON_SELF_PENALTY',
                'torch', 'functools', 'DEVICE', 'DTYPE_LONG', 'DTYPE_FLOAT',
                'TIMER_STORAGE', 'TIMER_COUNTS', 'profile_me', 'print_profiling_report', 'reset_profiler',
                '_weekday_minutes_matrix_from_end_hours', '_build_capacity_bins', '_assign_bins_batch_by_prefix',
                '_compute_order_in_bin_row', '_compute_order_in_bin_batch', '_GPUMatrixFitnessBatch',
                'funsearch'
            ]
            
            namespace = {name: globals()[name] for name in allowed_names if name in globals()}
            
            print(f"\n执行program (长度: {len(program)} 字符)...")
            try:
                exec(program, namespace)
                print("✓ Program执行成功")
            except Exception as e:
                print(f"❌ 执行program失败: {e}")
                traceback.print_exc()
                queue.put((None, False)) # 修复：失败
                return
            
            optimizer, success = self._setup_optimizer_with_functions(
                namespace, dataset, initial_population, queue
            )
            if not success:
                return
            
            print("\n开始进化测试...")
            evolution_start = time.time()
            final_population = optimizer.evolve(generations=10000, elite_size=5) #################################进化代数
            print(f"✓ 进化完成 (大小: {len(final_population)}, 耗时: {time.time()-evolution_start:.2f}s)")
            
            print("\n导出最佳排程...")
            try:
                if final_population:
                    scored_final = [(ind, optimizer.calculate_fitness(ind)[0]) 
                                for ind in final_population]
                    scored_final.sort(key=lambda x: x[1], reverse=True)
                    best_individual = scored_final[0][0]
                    
                    output_dir = 'output_schedules-v6big(id)'##############################################################excel
                    os.makedirs(output_dir, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3] 
                    program_hash = hashlib.md5(program.encode('utf-8')).hexdigest()[:10]
                    
                    # 文件名格式: [精确时间戳]-[HASH].xlsx
                    filename = os.path.join(output_dir, f'{timestamp}-{program_hash}.xlsx')
                    
                    # 使用 V6 的 generate_schedule (精排)
                    final_system = optimizer.generate_schedule(best_individual)
                    export_schedule(final_system, dataset['patients'], filename)
                    print(f"✓ 成功导出至 {filename}")
                else:
                    print("⚠️ 最终种群为空，无法导出")
            except Exception as e:
                print(f"❌ 导出失败: {e}")
                traceback.print_exc()

            # <--- 核心修改：在这里计算一次最终适应度 --->
            best_final_fitness = -float('inf')
            if final_population:
                try:
                    final_scores = [optimizer.calculate_fitness(ind)[0] for ind in final_population]
                    best_final_fitness = max(final_scores)
                    print(f"✓ (Sandbox) 最终最佳适应度计算完成: {best_final_fitness:.2f}")
                except Exception as e:
                    print(f"❌ (Sandbox) 计算最终适应度失败: {e}")
            # <--- 修改结束 --->
            
            print("\n评估整个种群...")
            evaluate_start = time.time()
            # evaluate_schedules 函数在 specification 中定义
            avg_score = namespace['evaluate_schedules'](
                best_final_fitness, # <--- 传递已计算的分数
                final_population,
                dataset['patients'],
                dataset['machine_exam_map']
            )
            
            total_time = time.time() - start_time
            print(f"✓ 评估完成 (得分: {avg_score:.2f}, 评估耗时: {time.time()-evaluate_start:.2f}s, 总耗时: {total_time:.2f}s)")
            
            queue.put((avg_score, True))
            
        except Exception as e:
            print(f"❌ 执行错误: {str(e)}")
            traceback.print_exc()
            queue.put((None, False)) # 修复：失败

def run_funsearch_optimization(patients, machine_exam_map, initial_population: List[List[Tuple]]):

    def escape_code(code):
        return code.replace('"""', r'\"\"\"').replace("'''", r"\'\'\'")

    # 注意：这里的 ORIGINAL_FUNCTIONS 已经包含了新的 GPU 版 evolve
    escaped_evolve = escape_code(ORIGINAL_FUNCTIONS['evolve'])
    escaped_mutate = escape_code(ORIGINAL_FUNCTIONS['mutate'])
    escaped_crossover = escape_code(ORIGINAL_FUNCTIONS['crossover'])
    escaped_greedy = escape_code(ORIGINAL_FUNCTIONS['greedy_cluster_mutation'])

    specification = f'''import random as rd
import numpy as np
from datetime import datetime, timedelta # 修复：缺少 datetime 导入
from collections import defaultdict
import torch # 新增 (保留，以防 LLM 生成的代码用到)
import functools # 新增 (保留，以防 LLM 生成的代码用到)

# ===== 评估函数 (超简化版：只接受值) =====
@funsearch.run
def evaluate_schedules(best_fitness_from_evolve, population, patients, machine_exam_map) -> float:
    """
    评估整个种群的排程质量 (精简版)
    本函数现在完全依赖于 Sandbox (_execute_medical) 传来的已计算分数。
    """

    # 导入必要的类和常量 (V6 原有)
    WEEKDAY_END_HOURS = {{
        1: 5.3, 2: 4.9, 3: 3.5, 4: 3.8, 5: 5.7, 6: 1.7, 7: 1.7
    }}
    WORK_START = datetime.strptime('07:00', '%H:%M').time()
    TRANSITION_PENALTY = 20000
    WAIT_PENALTY = 500
    LOGICAL = 10000
    SELF_SELECTED_PENALTY = 8000
    NON_SELF_PENALTY = 800
    DEVICE_PENALTY = 500000
    MACHINE_COUNT = 6
    START_DATE = datetime(2024, 12, 1, 7, 0)
    EXPECTED_PATIENT_COUNT = {len(patients)}


    # <--- 核心修改：只检查传入的分数 --->
    if isinstance(best_fitness_from_evolve, (float, int)) and best_fitness_from_evolve > -float('inf'):
        print(f"\\n{{'='*60}}")
        print(f"✓ (evaluate) 成功接收来自 Sandbox 的最佳适应度: {{best_fitness_from_evolve:.2f}}")
        print(f"{{'='*60}}\\n")
        return float(best_fitness_from_evolve)
    
    print(f"\\n{{'='*60}}")
    print(f" ❌ (evaluate) 未收到有效适应度 ({{best_fitness_from_evolve}})，评估失败！")
    print(f"{{'='*60}}")
    # 返回一个极差的分数，表示此次评估失败
    return -1e9
    # <--- 修改结束 --->

@funsearch.evolve
def get_evolution_functions():
    """返回优化后的进化函数代码 - 包含4个关键方法"""
    evolve_code = """{escaped_evolve}"""
    mutate_code = """{escaped_mutate}"""
    crossover_code = """{escaped_crossover}"""
    greedy_code = """{escaped_greedy}"""
    return {{
        'evolve': evolve_code,
        'mutate': mutate_code,
        'crossover': crossover_code,
        'greedy_cluster_mutation': greedy_code
    }}
'''

    inputs = {
        "medical_data": {
            "patients": patients,
            "machine_exam_map": machine_exam_map,
            "initial_population": initial_population
        }
    }

    class_config = config_lib.ClassConfig(
        llm_class=MedicalLLM,
        sandbox_class=MedicalSandbox
    )

    config_ = config_lib.Config(
        samples_per_prompt=4,
        evaluate_timeout_seconds=5000
    )

    log_dir = 'logs/v6big(id)' # ##############################################################################修改了log目录
    os.makedirs(log_dir, exist_ok=True)

    try:
        print("\n" + "=" * 60)
        print("启动FunSearch优化遗传算法函数 (使用 GPU Fitness)")
        print(f"使用固定的初始种群 (大小: {len(initial_population)})")
        print(f"GPU 设备: {DEVICE}")
        print("=" * 60)

        funsearch.main(
            specification=specification,
            inputs=inputs,
            config=config_,
            max_sample_nums=500,
            class_config=class_config,
            log_dir=log_dir,
        )

        print("FunSearch优化完成")

    except Exception as e:
        print(f"\nFunSearch优化过程中出错: {str(e)}")
        traceback.print_exc()


def main():
    """(来自 V6)"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        patient_file = os.path.join(current_dir, '实验数据6.1 - 副本.xlsx')
        duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
        device_constraint_file = os.path.join(current_dir, '设备限制4.xlsx')
        
        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        rd.seed(None)
        np.random.seed(None)
        
        pop_size = 50
        optimizer_for_init = BlockGeneticOptimizer(
            patients,
            machine_exam_map,
            pop_size=pop_size
        )
        optimizer_for_init.initialize_population(pop_size=pop_size)
        initial_population = optimizer_for_init.population
        
        print(f"✓ 已生成固定初始种群 (大小: {len(initial_population)})")

        print("\n===== 启动FunSearch优化遗传算法 (GPU版) =====")
        run_funsearch_optimization(patients, machine_exam_map, initial_population)

    except Exception as e:
        print(f"运行时错误: {str(e)}")
        traceback.print_exc()
    finally:
        print("优化运行结束。")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
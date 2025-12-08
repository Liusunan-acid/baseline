import pandas as pd
import numpy as np
import sys
import os
import json
import re

# ===================== 全局常量 (严格对齐 测量时间full-GPU实验.py) =====================
TRANSITION_PENALTY = 20000      # 换模罚分
LOGICAL_PENALTY = 10000         # 逻辑错误罚分 (负等待时间)
SELF_SELECTED_PENALTY = 8000    # 自选罚分系数
NON_SELF_PENALTY = 800          # 非自选罚分系数
DEVICE_PENALTY = 500000         # 设备不支持 & 特殊规则违规罚分 (统一为此值)

def clean_exam_name(name):
    """
    标准化检查项目名称 (完全一致的正则处理)
    """
    if pd.isnull(name):
        return ""
    s = str(name).strip().lower()
    s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
    s = re.sub(r'[^\w()-]', '', s)
    return s.replace('_', '-').replace(' ', '')

def safe_read_excel(file_path):
    """读取Excel，支持多种引擎"""
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 {file_path}")
        sys.exit(1)
    
    engines = ['openpyxl', 'odf', 'xlrd']
    for engine in engines:
        try:
            return pd.read_excel(file_path, engine=engine)
        except Exception:
            continue
    return pd.read_excel(file_path)

def load_device_constraints(constraint_file):
    """加载设备约束"""
    if not os.path.exists(constraint_file):
        return None
    try:
        # 尝试读取 Excel
        df = safe_read_excel(constraint_file)
        constraints = {}
        # 识别列
        mid_col = next((c for c in df.columns if '设备' in str(c) or 'machine' in str(c).lower()), None)
        exam_col = next((c for c in df.columns if '项目' in str(c) or 'exam' in str(c).lower()), None)
        
        if mid_col and exam_col:
            for _, row in df.iterrows():
                # 处理机器ID: 假设 Excel 是 1-based (1,2,3...) -> 转为 0-based (0,1,2...)
                raw_mid = row[mid_col]
                if str(raw_mid).isdigit():
                    mid = int(raw_mid) - 1 
                else:
                    mid = str(raw_mid) # 如果是字符串ID则保持
                
                exams = [clean_exam_name(x) for x in str(row[exam_col]).split(',')]
                if mid not in constraints:
                    constraints[mid] = set()
                constraints[mid].update(exams)
            return constraints
    except Exception as e:
        print(f"约束加载失败: {e}")
    return None

def calculate_schedule_score(schedule_file, constraint_file='device_constraints.xlsx'):
    print(f"正在读取排班表: {schedule_file}")
    df = safe_read_excel(schedule_file)
    
    # --- 1. 列名标准化 ---
    col_mapping = {
        'machine_id': ['机器编号', 'machine_id'],
        'date': ['日期', 'date'],
        'start_time': ['开始时间', 'start_time'],
        'exam_type': ['检查项目', 'exam_type'],
        'patient_id': ['患者ID', 'patient_id', 'id'],
        'reg_date': ['登记日期', 'reg_date'],
        'is_self_selected': ['是否自选', 'is_self_selected']
    }
    
    found_cols = {}
    for target, sources in col_mapping.items():
        for src in sources:
            if src in df.columns:
                found_cols[target] = src
                break
    
    # 检查必要列
    if len(found_cols) < len(col_mapping):
        missing = set(col_mapping.keys()) - set(found_cols.keys())
        print(f"警告: 缺少列 {missing}，可能会导致错误。")
    
    df = df.rename(columns={v: k for k, v in found_cols.items()})
    
    # --- 2. 数据清洗 ---
    df['cleaned_exam'] = df['exam_type'].apply(clean_exam_name)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['reg_date'] = pd.to_datetime(df['reg_date']).dt.date
    
    # 机器编号处理 (转为 int, 0-based)
    # 假设排班表里的机器编号如果全是数字，可能已经是 1-based 或 0-based
    # 为了保险，我们检测最小值。如果最小值是 1，则减 1；如果是 0，则不变。
    try:
        mids = pd.to_numeric(df['machine_id'], errors='coerce').dropna()
        if not mids.empty:
            if mids.min() >= 1:
                df['machine_idx'] = df['machine_id'].astype(int) - 1
            else:
                df['machine_idx'] = df['machine_id'].astype(int)
        else:
            df['machine_idx'] = df['machine_id']
    except:
        df['machine_idx'] = df['machine_id']
        
    df['weekday'] = pd.to_datetime(df['date']).dt.dayofweek # 0=Mon, 6=Sun

    constraints = load_device_constraints(constraint_file)

    # --- 3. 聚合为“患者”维度 (关键修正：对齐 GA 的 Patient 维度) ---
    # GA 视一个患者的一组检查为一个整体块。
    # 我们按 patient_id 聚合，提取关键信息。
    
    # 辅助函数：获取列表中的众数或第一个值
    def get_first(x): return x.iloc[0]
    
    patient_groups = df.groupby('patient_id')
    
    total_penalty = 0
    stats = {
        'waiting_penalty': 0,
        'special_rule_penalty': 0,
        'device_penalty': 0,
        'transition_penalty': 0,
        'logical_error_count': 0,
        'special_violation_count': 0,
        'transition_count': 0
    }
    
    # 准备用于计算换模的列表
    # 格式: (machine_idx, start_time, main_exam_type, patient_id)
    schedule_sequence = []

    print("\n开始按患者计算罚分...")

    for pid, group in patient_groups:
        # 提取患者级属性 (假设同一患者在同一天同一机器，这是GA的前提)
        # 如果Excel被手动修改导致分散，这里取第一条记录的信息作为基准
        first_row = group.sort_values('start_time').iloc[0]
        
        reg_date = first_row['reg_date']
        assigned_date = first_row['date']
        weekday = first_row['weekday']
        mid = first_row['machine_idx']
        
        # 处理是否自选
        raw_self = first_row['is_self_selected']
        is_self = str(raw_self).strip() in ['是', 'True', 'true', 'Yes', '1', '自选时间']
        
        # 提取该患者所有检查项目
        exams = group['cleaned_exam'].tolist()
        
        # === A. 等待时间罚分 (Per Patient) ===
        delta_days = (assigned_date - reg_date).days
        
        if delta_days >= 0:
            coef = SELF_SELECTED_PENALTY if is_self else NON_SELF_PENALTY
            p_wait = delta_days * coef
        else:
            p_wait = abs(delta_days) * LOGICAL_PENALTY
            stats['logical_error_count'] += 1
            
        total_penalty += p_wait
        stats['waiting_penalty'] += p_wait
        
        # === B. 设备支持 & 特殊规则 (Per Patient, 使用 OR 逻辑) ===
        # GA逻辑：p = (heart_v | angio_v | weekend_v | device_invalid) * DEVICE_PENALTY
        # 只要违反任意一项，就罚一次 DEVICE_PENALTY (500,000)
        
        violated = False
        
        # 1. 检查是否有任意项目设备不支持
        if constraints and (mid in constraints):
            supported_exams = constraints[mid]
            # 只要有一个项目不支持，就算违规
            if any(e not in supported_exams for e in exams):
                violated = True
                stats['device_penalty'] += DEVICE_PENALTY # 记录在 device 下以便区分，但在总分里是合并的
        
        # 2. 特殊规则标志位
        has_heart = any('心脏' in e for e in exams)
        has_angio = any('造影' in e for e in exams)
        has_contrast = any('增强' in e for e in exams)
        
        # Rule 1: 心脏 (周二/四, 机器4[idx=3])
        if has_heart:
            ok_wd = (weekday == 1) or (weekday == 3)
            ok_mc = (mid == 3)
            if not (ok_wd and ok_mc):
                violated = True
        
        # Rule 2: 造影 (周一/三/五, 机器2[idx=1])
        if has_angio:
            ok_wd = (weekday == 0) or (weekday == 2) or (weekday == 4)
            ok_mc = (mid == 1)
            if not (ok_wd and ok_mc):
                violated = True
                
        # Rule 3: 增强 (非周末)
        if has_contrast:
            is_weekend = (weekday == 5) or (weekday == 6)
            if is_weekend:
                violated = True
        
        # 结算特殊规则/设备罚分 (OR Logic)
        if violated:
            # 如果之前在 device 检查时还没加罚分，这里加上
            # 注意：我们在 device 检查时加了，为了避免重复，这里需要逻辑判断
            # 更好的方式是：先算状态，最后算分
            pass
        
        # 重置逻辑以严格匹配 GA 公式: penalty = (bool | bool ...) * 500000
        # 重新计算 Violated 状态
        is_device_invalid = False
        if constraints and (mid in constraints):
            if any(e not in constraints[mid] for e in exams):
                is_device_invalid = True
        
        is_special_invalid = False
        if has_heart and not ((weekday in [1, 3]) and (mid == 3)): is_special_invalid = True
        if has_angio and not ((weekday in [0, 2, 4]) and (mid == 1)): is_special_invalid = True
        if has_contrast and (weekday in [5, 6]): is_special_invalid = True
        
        if is_device_invalid or is_special_invalid:
            total_penalty += DEVICE_PENALTY
            stats['special_rule_penalty'] += DEVICE_PENALTY # 统称为特殊/设备罚分
            stats['special_violation_count'] += 1

        # === C. 准备换模数据 ===
        # GA 使用患者的"主要类型" (main type) 来判断换模。
        # 通常取第一个检查项目作为 Main Type (参考 GA import_data)
        main_type = exams[0] if exams else ""
        schedule_sequence.append({
            'mid': mid,
            'date': assigned_date,
            'time': first_row['start_time'], # 使用最早开始时间排序
            'type': main_type,
            'pid': pid
        })

    # === D. 换模罚分 (Transition Penalty) ===
    # 必须按 机器 -> 日期 -> 时间 排序，然后比较相邻患者
    seq_df = pd.DataFrame(schedule_sequence)
    
    # 确保 time 列是字符串或可排序格式
    seq_df['time_str'] = seq_df['time'].astype(str)
    seq_df = seq_df.sort_values(by=['mid', 'date', 'time_str'])
    
    print("\n开始计算换模罚分...")
    
    prev_row = None
    for idx, row in seq_df.iterrows():
        curr_mid = row['mid']
        curr_type = row['type']
        
        if prev_row is not None:
            prev_mid = prev_row['mid']
            prev_type = prev_row['type']
            
            # 逻辑：同一台机器 (GA逻辑通常同一天? 
            # 原代码 _penalty_machine_switching: same_bin & diff_type.
            # bin 隐含了 (day, machine)。所以跨天属于不同的 bin，不会产生换模罚分。
            # 所以条件是: 同机器 AND 同日期)
            
            if (curr_mid == prev_mid) and (row['date'] == prev_row['date']):
                if curr_type != prev_type:
                    total_penalty += TRANSITION_PENALTY
                    stats['transition_penalty'] += TRANSITION_PENALTY
                    stats['transition_count'] += 1
        
        prev_row = row

    # --- 4. 结果输出 ---
    fitness = -total_penalty
    
    print("=" * 40)
    print(f"最终评分结果 (Fitness): {fitness}")
    print("=" * 40)
    print(f"总罚分: {total_penalty}")
    print(f"  [1] 等待罚分: {stats['waiting_penalty']} (逻辑错误: {stats['logical_error_count']})")
    print(f"  [2] 违规罚分: {stats['special_rule_penalty']} (设备/心脏/造影/周末) - 共 {stats['special_violation_count']} 人次")
    print(f"  [3] 换模罚分: {stats['transition_penalty']} (切换次数: {stats['transition_count']})")
    print("-" * 40)
    print("说明：")
    print("  - 评分逻辑已对齐 '测量时间full-GPU实验.py'")
    print(f"  - 特殊违规罚分标准: {DEVICE_PENALTY}/人 (OR逻辑，不累加)")
    print(f"  - 换模罚分标准: {TRANSITION_PENALTY}/次 (基于患者主要检查项目)")
    
    return fitness, stats


if __name__ == "__main__":
    # --- 运行配置 ---
    # 默认查找 output_schedules 文件夹下的最新文件，或者当前目录的 schedule_result.xlsx
    target_file = '/home/preprocess/_funsearch/baseline/schedule_seconds_fifo_20251205_193816.xlsx'
    
    # 尝试自动寻找 output_schedules 中最新的 excel
    output_dir = 'output_schedules'
    if os.path.exists(output_dir):
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.xlsx')]
        if files:
            target_file = max(files, key=os.path.getmtime)
            print(f"自动检测到最新排班文件: {target_file}")

    constraint_file = '设备限制4.xlsx' 
    if not os.path.exists(constraint_file):
         # 尝试在上级目录找 (适应实验代码结构)
         constraint_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '设备限制4.xlsx')

    if os.path.exists(target_file):
        calculate_schedule_score(target_file, constraint_file)
    else:
        print(f"错误: 未找到排班文件 '{target_file}'。")
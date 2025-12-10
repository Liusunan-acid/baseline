import sys
import os
import importlib.util
import pandas as pd
import pulp
from datetime import datetime, timedelta
import time

# ==============================================================================
# 1. 动态加载现有模块 (复用数据读取函数)
# ==============================================================================
BASE_FILE_PATH = '/home/preprocess/_funsearch/baseline/测量时间full-GPU实验-Multi.py'
MODULE_NAME = 'baseline_module'

def load_source_module(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    spec = importlib.util.spec_from_file_location(MODULE_NAME, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module

print(f"正在加载模块: {BASE_FILE_PATH} ...")
base_mod = load_source_module(BASE_FILE_PATH)
print("模块加载成功！")

# ==============================================================================
# 2. MILP 求解器类 (基于 PDF 模型)
# ==============================================================================
class MILPSolver:
    def __init__(self, patients, machine_exam_map, work_start_time="08:00", work_end_time="18:00"):
        self.patients = patients
        self.machine_exam_map = machine_exam_map
        
        # 将机器限制转换为更易查阅的字典: {mid: [allowed_exams...]}
        # 注意：原代码机器ID是 0-based，Excel可能是 1-based，这里统一使用内部逻辑
        self.machines = list(machine_exam_map.keys())
        
        # 转换时间为分钟数以便计算 (以 00:00 为基准)
        self.work_start_min = int(pd.to_datetime(work_start_time).hour * 60 + pd.to_datetime(work_start_time).minute)
        self.work_end_min = int(pd.to_datetime(work_end_time).hour * 60 + pd.to_datetime(work_end_time).minute)
        
        # Big M (用于松弛约束，必须足够大，例如一天的总分钟数)
        self.BigM = 24 * 60 * 10
        
        # 换模时间 (PDF中的 T_k1_k2)，这里简化为固定值，可根据需求修改
        self.TRANSITION_TIME = 1 

    def solve_daily_batch(self, date_str, patient_batch, time_limit=300):
        """
        对某一天的病人进行 MILP 求解
        :param date_str: 当前排程日期
        :param patient_batch: 该日需要检查的病人列表 [(cid, exam_type, duration, arrival_time_obj), ...]
        :param time_limit: 求解器超时时间(秒)
        """
        if not patient_batch:
            return []

        # 创建问题实例：最小化等待时间
        # PDF Objective: min sum(t_i - t_in,i)
        prob = pulp.LpProblem(f"Schedule_{date_str}", pulp.LpMinimize)

        # === 集合 ===
        I = range(len(patient_batch)) # 病人集合索引
        J = self.machines             # 机器集合索引

        # 提取病人参数
        # p_data: index -> {'cid':..., 'duration':..., 'exam':..., 'arrival':...}
        p_data = {}
        for idx, (cid, exam_type, duration, arrival_dt) in enumerate(patient_batch):
            # 将到达时间转换为当天的分钟数 (如果是前一天登记的，默认从工作开始时间算)
            arrival_min = arrival_dt.hour * 60 + arrival_dt.minute
            # 如果到达时间早于工作开始，则视为工作开始时到达
            eff_arrival = max(arrival_min, self.work_start_min)
            
            p_data[idx] = {
                'cid': cid,
                'exam': exam_type,
                'duration': duration,
                'arrival': eff_arrival,
                'raw_data': patient_batch[idx]
            }

        # === 决策变量 ===
        # x_ij: 病人 i 是否分配给机器 j (Binary) [PDF Eq 17]
        x = pulp.LpVariable.dicts("x", ((i, j) for i in I for j in J), cat='Binary')
        
        # t_i: 病人 i 的开始时间 (Continuous/Integer) [PDF Eq 16]
        # 范围：工作开始 ~ 工作结束
        t = pulp.LpVariable.dicts("t", I, lowBound=self.work_start_min, upBound=self.work_end_min, cat='Continuous')
        
        # delta_ik: 顺序变量，如果 i 在 k 之前则为 1 (Binary) [PDF Eq 18]
        # 仅当 i < k 时定义，减少变量数量
        delta = pulp.LpVariable.dicts("delta", ((i, k) for i in I for k in I if i < k), cat='Binary')

        # === 目标函数 ===
        # Minimize sum(t_i - arrival_i) [PDF Eq 1/21]
        prob += pulp.lpSum([t[i] - p_data[i]['arrival'] for i in I])

        # === 约束条件 ===

        for i in I:
            # 1. 每个病人只能分配给 1 台机器 [PDF Eq 2/25]
            prob += pulp.lpSum([x[i, j] for j in J]) == 1

            # 2. 开始时间必须晚于到达时间 [PDF Eq 4/30]
            prob += t[i] >= p_data[i]['arrival']
            
            # 3. 机器能力约束 [PDF Eq 3/28]
            # 如果机器 j 不能做病人 i 的检查，则 x_ij = 0
            for j in J:
                allowed_exams = [base_mod.clean_exam_name(e) for e in self.machine_exam_map.get(j, [])]
                patient_exam = base_mod.clean_exam_name(p_data[i]['exam'])
                
                # 模糊匹配逻辑 (参考原 GA 代码逻辑)
                can_do = False
                if patient_exam in allowed_exams:
                    can_do = True
                else:
                    # 尝试处理包含关系，如 "头部CT" vs "CT"
                    for allowed in allowed_exams:
                        if allowed in patient_exam or patient_exam in allowed:
                            can_do = True
                            break
                
                if not can_do:
                    prob += x[i, j] == 0

            # 4. 时间窗约束 (在变量定义中已包含 lowBound/upBound，但也可是显式约束)
            # t_i + duration <= End_Time [PDF Eq 6/36]
            prob += t[i] + p_data[i]['duration'] <= self.work_end_min

        # 5. 不重叠约束 (Big-M) [PDF Eq 38-40]
        # 对于每一对病人 (i, k)，如果他们在同一台机器 j 上，必须有先后顺序
        for i in I:
            for k in I:
                if i < k: # 避免重复，只处理上三角
                    # 我们需要确保：
                    # 如果 x_ij=1 且 x_kj=1:
                    #   如果是 i -> k (delta_ik=1): t_k >= t_i + d_i + setup
                    #   如果是 k -> i (delta_ik=0): t_i >= t_k + d_k + setup
                    
                    # 为了简化 PDF 中的复杂求和形式，我们直接针对每台机器生成约束
                    # 这样更清晰且容易被 Solver 优化
                    for j in J:
                        # 转换 PDF 逻辑：
                        # t_k >= t_i + d_i + T - M * (3 - x_ij - x_kj - delta_ik)
                        # 当 x_ij=1, x_kj=1, delta_ik=1 (i在k前) 时，约束生效：t_k >= t_i + ...
                        prob += t[k] >= t[i] + p_data[i]['duration'] + self.TRANSITION_TIME - \
                                self.BigM * (3 - x[i, j] - x[k, j] - delta[i, k])
                        
                        # t_i >= t_k + d_k + T - M * (3 - x_ij - x_kj - (1-delta_ik))
                        # 当 x_ij=1, x_kj=1, delta_ik=0 (k在i前) 时，约束生效
                        prob += t[i] >= t[k] + p_data[k]['duration'] + self.TRANSITION_TIME - \
                                self.BigM * (3 - x[i, j] - x[k, j] - (1 - delta[i, k]))

        # === 求解 ===
        print(f"  - 正在求解日期 {date_str}, 病人数: {len(patient_batch)} ...")
        # 使用 CBC 求解器 (PuLP自带), 设置时间限制
        solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0) 
        prob.solve(solver)

        # === 结果解析 ===
        status = pulp.LpStatus[prob.status]
        print(f"  - 求解状态: {status}")

        results = []
        if status in ['Optimal', 'Feasible']:
            for i in I:
                assigned_machine = -1
                for j in J:
                    if pulp.value(x[i, j]) > 0.5:
                        assigned_machine = j
                        break
                
                start_min = pulp.value(t[i])
                
                # 将分钟数转回时间对象
                start_time_obj = datetime.strptime(f"{int(start_min)//60:02d}:{int(start_min)%60:02d}", "%H:%M")
                end_time_obj = start_time_obj + timedelta(minutes=p_data[i]['duration'])
                
                results.append({
                    'machine_id': assigned_machine,
                    'cid': p_data[i]['cid'],
                    'exam_type': p_data[i]['exam'],
                    'date': date_str,
                    'start_time': start_time_obj.strftime("%H:%M:%S"),
                    'end_time': end_time_obj.strftime("%H:%M:%S"),
                    'reg_date': p_data[i]['raw_data'][3] # 假设raw_data保留了登记日期
                })
        else:
            print(f"  ⚠️ 无法找到日期 {date_str} 的可行解 (可能是病人过多导致超时或约束冲突)")
        
        return results

# ==============================================================================
# 3. 主程序流程
# ==============================================================================
def main():
    # --- 配置路径 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设这些文件与您的原始脚本在同一位置，或者是您提供的路径
    patient_file = os.path.join(current_dir, '实验数据6.1small - 副本.xlsx')
    duration_file = os.path.join(current_dir, '程序使用实际平均耗时3 - 副本.xlsx')
    device_file = os.path.join(current_dir, '设备限制4.xlsx')

    # 检查文件
    if not all(os.path.exists(f) for f in [patient_file, duration_file, device_file]):
        print("错误: 找不到数据文件，请确保Excel文件在当前目录下。")
        return

    # --- 1. 使用原始模块导入数据 ---
    print("正在导入数据 (使用原始模块函数)...")
    # import_data 返回字典: {cid: {'exams': [[部位, 项目, 时长, 登记日期], ...], ...}}
    patients_data = base_mod.import_data(patient_file, duration_file)
    # import_device_constraints 返回字典: {machine_id: [exam_list]}
    machine_constraints = base_mod.import_device_constraints(device_file)

    print(f"导入完成。共 {len(patients_data)} 位患者。")

    # --- 2. 数据预处理：按登记日期分组 ---
    # MILP 无法一次性解决数千人的排程，必须按天分批处理 (Rolling Horizon)
    patients_by_date = {}
    
    for cid, info in patients_data.items():
        # 这里简化处理：取第一个检查项目进行排程
        # 如果一个病人有多个检查，这在 MILP 中需要更复杂的处理（通过 link 约束），
        # 这里为了演示 PDF 模型，假设每个条目是一个独立任务。
        for exam in info['exams']:
            # exam: [部位, 项目名称, 时长, 登记日期对象]
            reg_date = exam[3] # datetime.date
            exam_name = exam[1]
            duration = exam[2]
            
            date_str = reg_date.strftime('%Y-%m-%d')
            if date_str not in patients_by_date:
                patients_by_date[date_str] = []
            
            # 构造任务元组
            # 假设所有人都需要在登记当天做检查 (如果做不完顺延逻辑比较复杂，这里仅演示当天排程)
            # 实际排程中，通常取 max(登记时间, 当前排程日期)
            patients_by_date[date_str].append((cid, exam_name, duration, datetime.combine(reg_date, datetime.min.time())))

    # --- 3. 初始化求解器 ---
    # 假设工作时间 07:00 - 18:00 (参考原代码 WORK_START='07:00', 以及大致下班时间)
    milp_solver = MILPSolver(patients_data, machine_constraints, work_start_time="07:00", work_end_time="18:00")

    all_schedules = []

    # --- 4. 按日期循环求解 ---
    sorted_dates = sorted(patients_by_date.keys())
    
    # 【安全限制】如果数据量太大，建议先测试前几天
    # sorted_dates = sorted_dates[:3] 

    for date_str in sorted_dates:
        batch = patients_by_date[date_str]
        print(f"\n>>> 处理日期: {date_str}, 待排程任务数: {len(batch)}")
        
        # 如果单日人数超过 100，MILP 可能会非常慢。
        # 这里添加一个简单的分块逻辑，每 50 人算一次 (Heuristic batching)
        BATCH_SIZE =  30
        for i in range(0, len(batch), BATCH_SIZE):
            print(f"    正在求解子批次 {i//BATCH_SIZE + 1} ({len(sub_batch)} 人)...")
            
            results = milp_solver.solve_daily_batch(date_str, sub_batch)
            all_schedules.extend(results)

    # --- 5. 导出结果 ---
    if all_schedules:
        df_res = pd.DataFrame(all_schedules)
        output_file = 'MILP_Result_Schedule.xlsx'
        
        # 简单格式化导出
        with pd.ExcelWriter(output_file) as writer:
            df_res.sort_values(by=['date', 'machine_id', 'start_time']).to_excel(writer, sheet_name='MILP排程', index=False)
        
        print(f"\n全部完成！结果已保存至: {output_file}")
        print(f"总计排程任务数: {len(df_res)}")
    else:
        print("\n未生成任何排程结果。")

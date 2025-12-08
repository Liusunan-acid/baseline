import pandas as pd
import random as rd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback
import re
from concurrent.futures import ProcessPoolExecutor
import json
import signal
import sys

# 配置常量
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
BLOCK_SIZE_DAYS = 7  # 每个块的天数


def clean_exam_name(name):
    """标准化检查项目名称"""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', cleaned)
    cleaned = re.sub(r'[^\w$$-]', '', cleaned)
    return cleaned.replace('_', '-').replace(' ', '')


def safe_read_excel(file_path, sheet_name=0):
    """安全读取Excel文件，自动尝试不同引擎"""
    # 根据文件扩展名选择引擎
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

    # 如果所有引擎都失败，尝试不指定引擎
    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        raise ValueError(f"无法读取文件 {file_path}: {str(e)}")


def import_data(patient_file, duration_file):
    """数据导入函数"""
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
    """导入设备限制数据"""
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
    """单台机器的排程管理"""

    def __init__(self, machine_id, allowed_exams):
        self.machine_id = machine_id
        self.allowed_exams = allowed_exams
        self.timeline = defaultdict(list)

    def reset_timeline(self):
        """重置时间线"""
        self.timeline = defaultdict(list)

    def get_work_end(self, date):
        """获取当日工作结束时间"""
        weekday = date.isoweekday()
        base_time = datetime.combine(date, WORK_START)
        work_duration = 15.0 - WEEKDAY_END_HOURS[weekday]
        return base_time + timedelta(hours=work_duration)

    def add_exam(self, date, start_time, duration_minutes, exam_type, patient_info):
        """添加检查到时间线"""
        duration = timedelta(minutes=duration_minutes)
        end_time = start_time + duration
        self.timeline[date].append((
            start_time,
            end_time,
            exam_type,
            patient_info['compound_id'][0],
            patient_info['reg_date'],
            patient_info['is_self_selected']
        ))
        self.timeline[date].sort(key=lambda x: x[0])
        return end_time


class SchedulingSystem:
    """多机器排程系统"""

    def __init__(self, machine_exam_map, start_date=None):
        self.machines = []
        for machine_id in range(MACHINE_COUNT):
            allowed_exams = machine_exam_map.get(machine_id, [])
            self.machines.append(MachineSchedule(machine_id, allowed_exams))

        # 修复：直接使用日期对象，不需要调用.date()
        if start_date is None:
            self.current_date = START_DATE.date()
        else:
            self.current_date = start_date

        self.current_machine = 0
        self.start_date = self.current_date

    def reset(self):
        """重置排程系统状态"""
        for machine in self.machines:
            machine.reset_timeline()
        self.current_date = self.start_date
        self.current_machine = 0

    def find_available_slot(self, duration_minutes, exam_type, patient_info):
        """寻找可用时间段"""
        duration = timedelta(minutes=duration_minutes)
        for _ in range(365):
            machine = self.machines[self.current_machine]
            work_end = machine.get_work_end(self.current_date)
            scheduled = machine.timeline.get(self.current_date, [])

            if not scheduled:
                start_time = datetime.combine(self.current_date, WORK_START)
                end_time = start_time + duration
                if end_time <= work_end:
                    return machine, start_time
                else:
                    self.move_to_next()
                    continue

            prev_end = datetime.combine(self.current_date, WORK_START)
            for slot in scheduled:
                available_time = slot[0] - prev_end
                if available_time >= duration:
                    return machine, prev_end
                prev_end = slot[1]

            last_slot = scheduled[-1][1]
            end_time = last_slot + duration
            if end_time <= work_end:
                return machine, last_slot

            self.move_to_next()
        raise TimeoutError("无法在365天内找到可用时段")

    def move_to_next(self):
        """移动到下一可用位置"""
        self.current_machine += 1
        if self.current_machine >= MACHINE_COUNT:
            self.current_machine = 0
            self.current_date += timedelta(days=1)

    def get_exam_dates(self, individual, patients):
        """为个体生成排程并获取所有患者的检查日期"""
        self.reset()
        exam_dates = {}

        for cid in individual:
            patient = patients.get(cid)
            if patient and not patient['scheduled']:
                # 获取首次检查日期
                exam_types = [clean_exam_name(e[1]) for e in patient['exams']]
                exam_type = exam_types[0]  # 只需要安排第一个检查来获取日期
                duration = patient['exams'][0][2]

                try:
                    machine, start_time = self.find_available_slot(duration, exam_type, patient)
                    exam_date = start_time.date()
                    exam_dates[cid] = exam_date

                    # 模拟添加检查到时间线
                    machine.add_exam(
                        self.current_date,
                        start_time,
                        duration,
                        exam_type,
                        patient
                    )
                except Exception as e:
                    print(f"获取检查日期错误: {str(e)}")
                    exam_dates[cid] = patient['reg_date']  # 如果出错，使用登记日期

        return exam_dates


class BlockGeneticOptimizer:
    """基于实际检查日期分块的遗传算法优化器"""

    def __init__(self, patients, machine_exam_map, pop_size=50, block_start_date=None):
        self.patients = patients
        self.machine_exam_map = machine_exam_map
        self.scheduling_system = SchedulingSystem(machine_exam_map, start_date=block_start_date)
        self.population = []
        self.fitness_history = []
        self.sorted_patients = sorted(patients.keys(), key=lambda cid: patients[cid]['reg_date'])
        self.current_generation = 0
        self.individual_violations = defaultdict(set)
        self.pop_size = pop_size
        self.block_start_date = block_start_date  # 块的起始日期

        # 初始化种群
        self.initialize_population(pop_size)

    def save_state(self, filename):
        """保存优化器状态"""
        state = {
            'current_generation': self.current_generation,
            'population': [[list(cid) for cid in ind] for ind in self.population],
            'fitness_history': self.fitness_history,
            'block_start_date': self.block_start_date.strftime('%Y-%m-%d') if self.block_start_date else None
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"✅ 已保存第{self.current_generation}代状态")

    @classmethod
    def load_state(cls, filename, patients, machine_exam_map, pop_size=50):
        """加载优化器状态"""
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
            except:
                continue

        # 解析块起始日期
        block_start_date = None
        if 'block_start_date' in state and state['block_start_date']:
            block_start_date = datetime.strptime(state['block_start_date'], '%Y-%m-%d').date()

        optimizer = cls(patients, machine_exam_map, pop_size, block_start_date)
        optimizer.population = valid_population
        optimizer.current_generation = state['current_generation']
        optimizer.fitness_history = state['fitness_history']

        if len(valid_population) < pop_size:
            print(f"⚠️ 补充{pop_size - len(valid_population)}个新个体")
            optimizer.initialize_population(pop_size - len(valid_population))
            optimizer.population = valid_population + optimizer.population

        print(f"成功加载第{optimizer.current_generation}代状态")
        return optimizer

    def initialize_population(self, pop_size=None):
        """初始化种群"""
        if pop_size is None:
            pop_size = self.pop_size

        block_size = max(30, len(self.sorted_patients) // 20)  # 动态计算块大小

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

    def split_population_into_blocks(self):
        """根据实际检查日期将种群划分为块"""
        # 使用种群中第一个个体获取日期范围
        exam_dates = self.scheduling_system.get_exam_dates(self.population[0], self.patients)

        # 找到最小和最大检查日期
        min_date = min(exam_dates.values())
        max_date = max(exam_dates.values())

        # 创建日期块范围
        date_blocks = []
        current_start = min_date
        while current_start <= max_date:
            current_end = current_start + timedelta(days=BLOCK_SIZE_DAYS - 1)
            date_blocks.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        # 分配患者到日期块
        patient_blocks = defaultdict(list)
        for cid, exam_date in exam_dates.items():
            for idx, (start, end) in enumerate(date_blocks):
                if start <= exam_date <= end:
                    patient_blocks[idx].append(cid)
                    break

        # 构建每个块的子种群
        block_populations = defaultdict(list)
        for individual in self.population:
            # 根据检查日期将个体划分为多个块
            individual_blocks = defaultdict(list)
            for cid in individual:
                exam_date = exam_dates.get(cid, min_date)  # 使用实际或预估日期
                for block_idx, (start, end) in enumerate(date_blocks):
                    if start <= exam_date <= end:
                        individual_blocks[block_idx].append(cid)
                        break
                else:
                    # 如果日期超出范围，放入最后一个块
                    individual_blocks[len(date_blocks) - 1].append(cid)

            # 为每个块添加子个体
            for block_idx, patient_ids in individual_blocks.items():
                block_populations[block_idx].append(patient_ids)

        return block_populations, date_blocks

    def create_block_optimizers(self, block_populations, date_blocks):
        """为每个时间块创建优化器（带正确的起始日期）"""
        block_optimizers = {}

        for block_idx, population_block in block_populations.items():
            # 获取该块的病人数据
            block_patients = {}
            for cid in {pid for ind in population_block for pid in ind}:
                if cid in self.patients:
                    block_patients[cid] = self.patients[cid]

            if not block_patients:
                continue

            # 设置块的起始日期
            block_start_date = date_blocks[block_idx][0]

            # 创建块优化器（带起始日期）
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
        # 确保块按顺序排列
        sorted_blocks = sorted(block_optimizers.keys())

        # 重新构建完整种群
        merged_population = []
        num_individuals = len(block_optimizers[sorted_blocks[0]].population) if sorted_blocks else 0

        for i in range(num_individuals):
            full_individual = []
            for block_idx in sorted_blocks:
                if i < len(block_optimizers[block_idx].population):
                    full_individual.extend(block_optimizers[block_idx].population[i])
            merged_population.append(full_individual)

        return merged_population

    def block_evolution(self, block_generations=50, full_generations=50):
        """分块进化过程"""
        print("\n=== 开始分块进化 ===")

        # 将每个个体划分为时间块
        block_populations, date_blocks = self.split_population_into_blocks()
        print(f"已将种群划分为 {len(block_populations)} 个时间块")

        # 为每个时间块创建优化器（带正确的起始日期）
        block_optimizers = self.create_block_optimizers(block_populations, date_blocks)

        # 每个块独立进化
        for block_idx, block_opt in block_optimizers.items():
            block_start = date_blocks[block_idx][0]
            print(f"时间块 #{block_idx + 1} ({block_start} 至 {date_blocks[block_idx][1]}) 开始进化...")
            try:
                # 保存当前代数为后续日志使用
                start_gen = block_opt.current_generation

                # 执行进化
                best_ind, best_fitness = block_opt.evolve(block_generations)

                # 记录进化代数
                end_gen = block_opt.current_generation
                print(
                    f"时间块 #{block_idx + 1} 进化完成: 第{start_gen + 1}-{end_gen}代, 最佳适应度: {best_fitness:.2f}")
            except Exception as e:
                print(f"时间块 #{block_idx + 1} 进化出错: {str(e)}")
                traceback.print_exc()

        # 合并进化后的块
        self.population = self.merge_blocks(block_optimizers)
        print("已合并所有子种群")

        # 整体进化
        print("开始整体进化...")
        start_gen = self.current_generation
        best_ind, best_fitness = self.evolve(full_generations)
        end_gen = self.current_generation
        print(f"整体进化完成: 第{start_gen + 1}-{end_gen}代, 最佳适应度: {best_fitness:.2f}")

        return best_ind, best_fitness

    def evolve(self, generations=50, elite_size=5):
        """执行进化过程"""
        try:
            start_gen = self.current_generation
            end_gen = start_gen + generations
            print(f"开始优化，当前代数: {start_gen}，目标代数: {end_gen}")

            for gen in range(start_gen, end_gen):
                self.current_generation = gen + 1
                individual_violations = defaultdict(set)
                all_violation_stats = []

                # 并行评估适应度
                with ProcessPoolExecutor() as executor:
                    future_map = {
                        executor.submit(self.calculate_fitness, ind): idx
                        for idx, ind in enumerate(self.population)
                    }

                    scored = []
                    for future in future_map:
                        idx = future_map[future]
                        try:
                            fitness, h_viol, a_viol, total_viol, weekend_viol, local_viol = future.result()
                            scored.append((self.population[idx], fitness))
                            all_violation_stats.append((h_viol, a_viol, total_viol, weekend_viol, len(local_viol)))
                            individual_violations[idx] = local_viol
                        except Exception as e:
                            print(f"计算适应度失败: {str(e)}")
                            scored.append((self.population[idx], -float('inf')))
                            all_violation_stats.append((0, 0, 0, 0, 0))

                # 排序
                sorted_pairs = sorted(enumerate(scored), key=lambda x: x[1][1], reverse=True)
                sorted_indices = [i[0] for i in sorted_pairs]
                sorted_pop = [i[1][0] for i in sorted_pairs]
                sorted_fitness = [i[1][1] for i in sorted_pairs]
                sorted_viol_stats = [all_violation_stats[i] for i in sorted_indices]
                best_fitness = sorted_fitness[0]
                self.fitness_history.append(best_fitness)

                # 日志输出
                if (gen + 1) % 50 == 0 or gen == end_gen - 1:
                    best_h, best_a, best_total, best_week, best_patients = sorted_viol_stats[0]
                    print(f"\n=== 第 {gen + 1} 代关键指标 ===")
                    print(f"当前最佳适应度: {best_fitness:.2f}")
                    print(f"心脏检查违规: {best_h}")
                    print(f"造影检查违规: {best_a}")
                    print(f"总设备违规: {best_total}")
                    print(f"周末增强违规: {best_week}")
                    print(f"影响患者数: {best_patients}")

                # 进化操作
                new_population = []
                elites = sorted_pop[:elite_size]
                new_population.extend(elites)

                parents = sorted_pop[:int(0.2 * len(sorted_pop))]
                while len(new_population) < len(self.population):
                    parent1, parent2 = rd.choices(parents, k=2)
                    child = self.crossover(parent1, parent2)

                    selected_parent = rd.choice([parent1, parent2])
                    parent_orig_idx = self.population.index(selected_parent)
                    parent_violations = individual_violations.get(parent_orig_idx, set())

                    mutated_child = self.mutate(child.copy(), parent_violations)
                    new_population.append(mutated_child)

                self.population = new_population
                print(f"Generation {gen + 1} | Best Fitness: {best_fitness:.2f}")

            # 返回最佳个体
            fitness_scores = [self.calculate_fitness(ind)[0] for ind in self.population]
            best_idx = np.argmax(fitness_scores)
            return self.population[best_idx], fitness_scores[best_idx]
        except KeyboardInterrupt:
            self.save_state(POPULATION_FILE)
            print("用户中断，已保存当前状态")
            raise

    def mutate(self, individual, parent_violations=None, base_rate=0.3):
        """改进后的变异策略：前1000代限制变异范围，1000代后无限制"""
        current_gen = self.current_generation
        use_range_limit = current_gen <= 1000000000

        # 优先处理违规患者（100%概率触发）
        if parent_violations and rd.random() < 1:
            violating = [cid for cid in individual if cid in parent_violations]

            if len(violating) >= 1:
                try:
                    violator = rd.choice(violating)
                    violator_idx = individual.index(violator)

                    if use_range_limit:
                        low_bound = max(0, violator_idx - 0)
                        high_bound = min(len(individual) - 1, violator_idx + 400)

                        # 选择范围内的位置
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
            # 添加范围限制逻辑
            if use_range_limit:
                # 随机选择第一个位置
                idx1 = rd.randint(0, len(individual) - 1)

                # 在idx1周围400范围内选择第二个位置
                low_bound = max(0, idx1 - 400)
                high_bound = min(len(individual) - 1, idx1 + 400)

                # 确保第二个位置与第一个不同
                possible_positions = [
                    i for i in range(low_bound, high_bound + 1)
                    if i != idx1
                ]

                if possible_positions:
                    idx2 = rd.choice(possible_positions)
                    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
                else:
                    # 如果没有可用位置，随机选择另一个位置
                    idx2 = rd.choice([i for i in range(len(individual)) if i != idx1])
                    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
            else:
                # 无范围限制时，随机选择两个位置
                idx1, idx2 = rd.sample(range(len(individual)), 2)
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        if rd.random() < 0.5:
            individual = self.greedy_cluster_mutation(individual)

        return individual

    def greedy_cluster_mutation(self, individual):
        """贪心聚类变异"""
        start = rd.randint(0, len(individual) - 100)
        end = start + rd.randint(50, 100)
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

    def crossover(self, parent1, parent2):
        """交叉操作"""
        start, end = sorted(rd.sample(range(len(parent1)), 2))
        fragment = set(parent1[start:end + 1])
        child = [g for g in parent2 if g not in fragment]
        return child[:start] + parent1[start:end + 1] + child[start:]

    def calculate_fitness(self, schedule):
        """计算适应度并返回违规统计"""
        system = SchedulingSystem(self.machine_exam_map, self.block_start_date)
        total_penalty = 0
        heart_violations = 0
        angiogram_violations = 0
        total_violations = 0
        weekend_violations = 0
        local_violations = set()

        for cid in schedule:
            patient = self.patients.get(cid)
            if not patient:
                continue

            first_exam_date = None
            prev_exam_type = None

            for exam in patient['exams']:
                exam_type = clean_exam_name(exam[1])
                duration = exam[2]

                try:
                    # 获取排程信息
                    machine, start_time = system.find_available_slot(duration, exam_type, patient)
                    scheduled_weekday = start_time.date().isoweekday()
                    machine_id = machine.machine_id

                    # 设备兼容性检查
                    allowed_exams = self.machine_exam_map.get(machine_id, [])
                    if exam_type not in machine.allowed_exams:
                        total_penalty += DEVICE_PENALTY
                        total_violations += 1
                        local_violations.add(cid)

                    # 心脏检查限制 (周二/周四 + 机器4)
                    if '心脏' in exam_type:
                        if scheduled_weekday not in [2, 4] or machine_id != 3:
                            total_penalty += DEVICE_PENALTY
                            heart_violations += 1
                            local_violations.add(cid)

                    # 造影检查限制 (周一/三/五 + 机器2)
                    if '造影' in exam_type:
                        if scheduled_weekday not in [1, 3, 5] or machine_id != 1:
                            total_penalty += DEVICE_PENALTY
                            angiogram_violations += 1
                            local_violations.add(cid)

                    # 增强检查限制 (不在周末)
                    if '增强' in exam_type and scheduled_weekday in [6, 7]:
                        total_penalty += DEVICE_PENALTY
                        weekend_violations += 1
                        local_violations.add(cid)

                    # 添加检查到时间线
                    end_time = machine.add_exam(
                        system.current_date,
                        start_time,
                        duration,
                        exam_type,
                        patient
                    )

                    # 记录首次检查日期
                    if not first_exam_date:
                        first_exam_date = start_time.date()

                    # 计算转换惩罚
                    if prev_exam_type and prev_exam_type != exam_type:
                        total_penalty += TRANSITION_PENALTY
                    prev_exam_type = exam_type

                except TimeoutError:
                    total_penalty += 1e6
                    break

            # 计算时间差惩罚
            if first_exam_date:
                reg_date = patient['reg_date']
                delta_days = (first_exam_date - reg_date).days

                if delta_days < 0:
                    total_penalty += LOGICAL * abs(delta_days)
                else:
                    penalty_type = SELF_SELECTED_PENALTY if patient['is_self_selected'] else NON_SELF_PENALTY
                    total_penalty += penalty_type * delta_days

        return (
            -total_penalty, heart_violations, angiogram_violations, total_violations, weekend_violations,
            local_violations)

    def save_intermediate_schedule(self, individual, gen, save_dir):
        """保存中间结果"""
        system = self.generate_schedule(individual)
        filename = os.path.join(save_dir, f'排程_第{gen}代.xlsx')
        export_schedule(system, self.patients, filename)
        print(f"已保存第 {gen} 代排程结果至 {filename}")

    def generate_schedule(self, individual):
        """生成排程系统（带起始日期）"""
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
    """导出排程结果"""
    with pd.ExcelWriter(filename) as writer:
        all_records = []
        for machine in system.machines:
            for date in sorted(machine.timeline):
                for slot in machine.timeline[date]:
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
        pd.DataFrame(all_records).to_excel(writer, sheet_name='总排程', index=False)


def main():
    def signal_handler(sig, frame):
        if 'optimizer' in locals():
            optimizer.save_state(POPULATION_FILE)
        print("\n已保存当前状态后退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    try:
        patient_file = os.path.join(BASE_DIR, '实验数据6.1small - 副本.xlsx')
        duration_file = os.path.join(BASE_DIR, '程序使用实际平均耗时3 - 副本.xlsx')
        output_file = os.path.join(os.path.expanduser('~'), 'Desktop', '优化算法结果.xlsx')
        output_dir = os.path.join(BASE_DIR, 'output')
        intermediate_dir = os.path.join(output_dir, '按周分块，贪心聚类 发现问题后再次实验0论文')
        device_constraint_file = os.path.join(BASE_DIR, '设备限制4.xlsx')
        os.makedirs(intermediate_dir, exist_ok=True)

        print("正在导入数据...")
        patients = import_data(patient_file, duration_file)
        machine_exam_map = import_device_constraints(device_constraint_file)

        print("\n尝试加载已有状态...")
        optimizer = BlockGeneticOptimizer.load_state(POPULATION_FILE, patients, machine_exam_map, pop_size=50)

        if not optimizer:
            print("初始化新优化器...")
            optimizer = BlockGeneticOptimizer(patients, machine_exam_map, pop_size=50)
            optimizer.initialize_population()

        total_cycles = 1000000  # 总循环次数（每个循环包含分块进化和整体进化）
        block_generations = 50  # 每个块进化的代数
        full_generations = 50  # 整体进化的代数

        best_fitness = -float('inf')
        best_schedule = None

        for cycle in range(total_cycles):
            print(f"\n===== 开始第 {cycle + 1}/{total_cycles} 轮分块优化 =====")

            # 执行分块进化
            best_ind, fitness = optimizer.block_evolution(
                block_generations=block_generations,
                full_generations=full_generations
            )

            # 更新最佳方案
            if fitness > best_fitness:
                best_fitness = fitness
                best_schedule = best_ind
                print(f"发现新的全局最佳适应度: {best_fitness:.2f}")

                # 保存中间结果
                optimizer.save_intermediate_schedule(best_schedule, optimizer.current_generation, intermediate_dir)

            # 保存状态
            optimizer.save_state(POPULATION_FILE)
            print(f"✅ 已保存第{optimizer.current_generation}代状态")

        # 最终处理
        print("\n优化完成! 生成最终排程...")
        final_system = optimizer.generate_schedule(best_schedule)
        export_schedule(final_system, patients, output_file)
        print(f"已保存最终排程至: {output_file}")

        # 绘制适应度变化图
        plt.figure(figsize=(12, 6))
        plt.plot(optimizer.fitness_history)
        plt.title('分块优化过程适应度变化')
        plt.xlabel('代数')
        plt.ylabel('适应度')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fitness_progress_block.png'))
        plt.show()

    except Exception as e:
        print(f"运行时错误: {str(e)}")
        traceback.print_exc()
    finally:
        if 'optimizer' in locals():
            optimizer.save_state(POPULATION_FILE)
        input("按回车退出...")


if __name__ == "__main__":
    main()
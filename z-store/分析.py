# import pandas as pd
# import os
# import warnings
# import re  # 1. 新增正则模块

# # 忽略一些pandas的切片警告
# warnings.filterwarnings('ignore')

# # ================= 配置区域 =================
# # 规则3的关键词
# KEYWORD_RULE_3 = "关节造影" 
# # ===========================================

# # 2. 新增：清洗函数（你提供的代码）
# def clean_exam_name(name):
#     s = str(name).strip().lower()
#     s = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', s)
#     s = re.sub(r'[^\w()-]', '', s)
#     return s.replace('_', '-').replace(' ', '')

# def load_device_limits(limit_file_path):
#     """
#     读取《设备限制4》表格，构建允许的检查项目字典。
#     """
#     print(f"正在读取设备限制表: {limit_file_path} ...")
#     try:
#         if limit_file_path.endswith('.csv'):
#             df = pd.read_csv(limit_file_path)
#         else:
#             df = pd.read_excel(limit_file_path)
#     except Exception as e:
#         print(f"❌ 无法读取设备限制表: {e}")
#         return None

#     # 假设第一列是设备，第二列是检查项目
#     allowed_map = {}
    
#     col_dev = df.columns[0]
#     col_item = df.columns[1]

#     for index, row in df.iterrows():
#         try:
#             # 读取设备号并减1
#             raw_dev = int(row[col_dev])
#             dev_idx = raw_dev - 1
            
#             # 3. 修改：加载规则时直接清洗，确保规则库是标准化的
#             item = clean_exam_name(row[col_item])
            
#             if dev_idx not in allowed_map:
#                 allowed_map[dev_idx] = set()
#             allowed_map[dev_idx].add(item)
#         except ValueError:
#             continue # 跳过非数字的设备行

#     print(f"✅ 设备限制表加载完成，包含 {len(allowed_map)} 个设备规则。")
#     return allowed_map

# def analyze_medical_data(schedule_path, limit_path):
#     """
#     主分析函数
#     """
#     # 1. 加载设备限制表
#     allowed_rules = load_device_limits(limit_path)
#     if not allowed_rules:
#         return

#     # 2. 加载排班表
#     try:
#         if schedule_path.endswith('.csv'):
#             df = pd.read_csv(schedule_path)
#         else:
#             df = pd.read_excel(schedule_path)
#     except Exception as e:
#         print(f"❌ 读取排班文件失败: {e}")
#         return

#     print(f"--- 成功读取排班文件，共 {len(df)} 行数据 ---")

#     # ---------------------------------------------------------
#     # 列名识别
#     # ---------------------------------------------------------
#     column_mappings = {
#         '日期': ['日期', 'date', 'Date'],
#         '登记日期': ['登记日期', 'reg_date'],
#         '检查项目': ['检查项目', 'exam_type', '项目'],
#         '设备': ['机器编号', 'device', '设备编号', '机号','machine_id']
#     }

#     actual_cols = {}
#     for key, possible_names in column_mappings.items():
#         found_col = next((col for col in possible_names if col in df.columns), None)
#         if found_col:
#             actual_cols[key] = found_col
#         else:
#             print(f"❌ 错误: 未能在表格中找到代表 '{key}' 的列。")
#             return

#     col_date = actual_cols['日期']
#     col_reg_date = actual_cols['登记日期']
#     col_item = actual_cols['检查项目']
#     col_dev = actual_cols['设备']

#     # ---------------------------------------------------------
#     # 数据预处理
#     # ---------------------------------------------------------
#     # 1. 处理设备编号：转数字 -> 减1
#     df['_calc_device_id'] = pd.to_numeric(df[col_dev], errors='coerce').fillna(-999).astype(int) - 1

#     # 2. 处理日期
#     df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
#     df['_calc_weekday'] = df[col_date].dt.dayofweek

#     # 3. 处理登记日期
#     df[col_reg_date] = pd.to_datetime(df[col_reg_date], errors='coerce')

#     # 4. 修改：处理检查项目 -> 使用 clean_exam_name 清洗
#     # 原代码：df['_calc_item'] = df[col_item].astype(str).fillna('').str.strip()
#     # 既然 _calc_item 是用于后续判断的，我们在这里直接把它变成“清洗后的标准名”
#     df['_calc_item'] = df[col_item].apply(clean_exam_name)
    
#     # ---------------------------------------------------------
#     # 任务 A: 计算等待时间
#     # ---------------------------------------------------------
#     df['计算_等待时间'] = (df[col_date] - df[col_reg_date]).dt.days
    
    
#     violation_list = [] 
    
#     count_device_mismatch = 0 
#     count_heart_rule = 0      
#     count_rule3 = 0           
#     count_contrast = 0        

#     # 下面的循环逻辑完全保持原样
#     # 因为 item_name 现在已经是清洗后的数据，allowed_rules 也是清洗后的数据
#     # 所以 if item_name not in allowed_rules[dev_id] 自动变成了“清洗后的严格匹配”
#     for index, row in df.iterrows():
#         item_name = row['_calc_item'] # 这里拿到的是清洗后的名称
#         dev_id = row['_calc_device_id']
#         weekday = row['_calc_weekday'] 
        
#         reasons = []

#         # --- 检查 1: 常规违规 (设备能力不匹配) ---
#         if dev_id in allowed_rules:
#             if item_name not in allowed_rules[dev_id]:
#                 reasons.append(f"设备能力不符(设备{dev_id}无法做{item_name})")
#                 count_device_mismatch += 1
#         else:
#             pass 

#         # --- 检查 2: “心脏” 规则 ---
#         if "心脏" in item_name:
#             is_time_ok = weekday in [1, 3] 
#             is_machine_ok = (dev_id == 3)  
            
#             if not (is_time_ok and is_machine_ok):
#                 reasons.append("违反心脏规则(非周二周四或非3号机)")
#                 count_heart_rule += 1

#         # --- 检查 3 & 4: “造影” 与 “关节造影” 规则 ---
#         if KEYWORD_RULE_3 in item_name: 
#             is_time_ok = weekday in [0, 2, 4]
#             is_machine_ok = (dev_id == 1)
            
#             if not (is_time_ok and is_machine_ok):
#                 reasons.append(f"违反{KEYWORD_RULE_3}规则(非135或非1号机)")
#                 count_rule3 += 1
        
#         elif "增强" in item_name:
#             if weekday in [5, 6]:
#                 reasons.append("违反造影规则(非周末)")
#                 count_contrast += 1

#         if len(reasons) > 0:
#             violation_list.append("; ".join(reasons))
#         else:
#             violation_list.append("正常")

#     df['违规情况'] = violation_list
#     df['是否违规'] = df['违规情况'] != "正常"

#     # ---------------------------------------------------------
#     # 统计计算区域 (保持原样)
#     # ---------------------------------------------------------
#     total_count = len(df)
#     violation_count = df['是否违规'].sum()
#     violation_rate = (violation_count / total_count * 100) if total_count > 0 else 0

#     clean_items = df['_calc_item'] # 这里的 _calc_item 已经是清洗过的了，直接用来算差异更准确
#     prev_items = clean_items.shift(1)
    
#     diff_mask = (clean_items != prev_items) & (prev_items.notna())
#     diff_count = diff_mask.sum()
#     same_mask = (clean_items == prev_items) & (prev_items.notna())
#     same_count = same_mask.sum()

#     valid_wait = df['计算_等待时间'].dropna()
#     wait_count_all = len(valid_wait)
#     wait_mean_all = valid_wait.mean() if wait_count_all > 0 else 0
#     wait_non_negative = valid_wait[valid_wait >= 0]
#     wait_count_pos = len(wait_non_negative)
#     wait_mean_pos = wait_non_negative.mean() if wait_count_pos > 0 else 0

#     # ---------------------------------------------------------
#     # 输出报告
#     # ---------------------------------------------------------
#     print(f"\n{'='*30}")
#     print(f"【详细分析报告】")
#     print(f"{'='*30}")

#     print(f"1. 违规检查统计 (分门别类):")
#     print(f"   - 总样本数: {total_count}")
#     print(f"   - 总体违规样本: {violation_count} (占比 {violation_rate:.2f}%)")
#     print(f"   ---------------------------")
    
#     rate_dev = (count_device_mismatch / total_count * 100) if total_count > 0 else 0
#     print(f"   - 常规违规 (设备能力不符): {count_device_mismatch} 个 ({rate_dev:.2f}%)")
    
#     rate_heart = (count_heart_rule / total_count * 100) if total_count > 0 else 0
#     print(f"   - 心脏规则违规: {count_heart_rule} 个 ({rate_heart:.2f}%)")
    
#     rate_rule3 = (count_rule3 / total_count * 100) if total_count > 0 else 0
#     print(f"   - {KEYWORD_RULE_3}规则违规: {count_rule3} 个 ({rate_rule3:.2f}%)")
    
#     rate_contrast = (count_contrast / total_count * 100) if total_count > 0 else 0
#     print(f"   - 增强违规: {count_contrast} 个 ({rate_contrast:.2f}%)")

#     print(f"\n2. 相邻项目变动统计:")
#     print(f"   - 差异个数 (换项目): {diff_count}")
#     print(f"   - 相同个数 (没换项目): {same_count}")

#     print(f"\n3. 等待时间统计 (日期 - 登记日期):")
#     print(f"   [包含负值]")
#     print(f"   - 总个数: {wait_count_all}")
#     print(f"   - 平均等待: {wait_mean_all:.2f} 天")
#     print(f"   [剔除负值 (>=0)]")
#     print(f"   - 有效个数: {wait_count_pos}")
#     print(f"   - 平均等待: {wait_mean_pos:.2f} 天")


# if __name__ == "__main__":
#     schedule_file = '/home/preprocess/_funsearch/baseline/schedule_seconds_fifo_20251211_213515.xlsx'
#     limit_file = '/home/preprocess/_funsearch/baseline/设备限制4.xlsx'
#     analyze_medical_data(schedule_file, limit_file)





import pandas as pd
import os
import warnings
import re

# 忽略一些pandas的切片警告
warnings.filterwarnings('ignore')

# ================= 配置区域 =================
# 规则3的关键词
KEYWORD_RULE_3 = "关节造影"
# ===========================================


# ✅ 1. 修改：清洗函数严格对齐你上面的参考代码
def clean_exam_name(name):
    """标准化检查项目名称（对齐参考实现）"""
    cleaned = str(name).strip().lower()
    cleaned = re.sub(r'[（）]', lambda x: '(' if x.group() == '（' else ')', cleaned)
    # 注意：这里完全沿用你参考代码的规则
    cleaned = re.sub(r'[^\w$$-]', '', cleaned)
    return cleaned.replace('_', '-').replace(' ', '')


def load_device_limits(limit_file_path):
    """
    读取《设备限制4》表格，构建允许的检查项目字典。
    """
    print(f"正在读取设备限制表: {limit_file_path} ...")
    try:
        if limit_file_path.endswith('.csv'):
            df = pd.read_csv(limit_file_path)
        else:
            df = pd.read_excel(limit_file_path)
    except Exception as e:
        print(f"❌ 无法读取设备限制表: {e}")
        return None

    # 假设第一列是设备，第二列是检查项目
    allowed_map = {}

    col_dev = df.columns[0]
    col_item = df.columns[1]

    for index, row in df.iterrows():
        try:
            # 读取设备号并减1（对齐参考代码：规则表 1-based -> 内部 0-based）
            raw_dev = int(row[col_dev])
            dev_idx = raw_dev - 1

            # ✅ 2. 修改：加载规则时直接清洗
            item = clean_exam_name(row[col_item])

            if dev_idx not in allowed_map:
                allowed_map[dev_idx] = set()
            allowed_map[dev_idx].add(item)
        except ValueError:
            continue  # 跳过非数字的设备行

    print(f"✅ 设备限制表加载完成，包含 {len(allowed_map)} 个设备规则。")
    return allowed_map


def analyze_medical_data(schedule_path, limit_path):
    """
    主分析函数
    """
    # 1. 加载设备限制表
    allowed_rules = load_device_limits(limit_path)
    if not allowed_rules:
        return

    # 2. 加载排班表
    try:
        if schedule_path.endswith('.csv'):
            df = pd.read_csv(schedule_path)
        else:
            df = pd.read_excel(schedule_path)
    except Exception as e:
        print(f"❌ 读取排班文件失败: {e}")
        return

    print(f"--- 成功读取排班文件，共 {len(df)} 行数据 ---")

    # ---------------------------------------------------------
    # 列名识别
    # ---------------------------------------------------------
    column_mappings = {
        '日期': ['日期', 'date', 'Date'],
        '登记日期': ['登记日期', 'reg_date'],
        '检查项目': ['检查项目', 'exam_type', '项目'],
        '设备': ['机器编号', 'device', '设备编号', '机号', 'machine_id']
    }

    actual_cols = {}
    for key, possible_names in column_mappings.items():
        found_col = next((col for col in possible_names if col in df.columns), None)
        if found_col:
            actual_cols[key] = found_col
        else:
            print(f"❌ 错误: 未能在表格中找到代表 '{key}' 的列。")
            return

    col_date = actual_cols['日期']
    col_reg_date = actual_cols['登记日期']
    col_item = actual_cols['检查项目']
    col_dev = actual_cols['设备']

    # ---------------------------------------------------------
    # 数据预处理
    # ---------------------------------------------------------
    # ✅ 3. 关键修改：排班表设备编号不再 -1（严格对齐参考代码）
    # 参考代码：schedule_df['设备ID'] = schedule_df['机器编号'].astype(int)
    df['_calc_device_id'] = pd.to_numeric(df[col_dev], errors='coerce').fillna(-999).astype(int)

    # 2. 处理日期
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df['_calc_weekday'] = df[col_date].dt.dayofweek

    # 3. 处理登记日期
    df[col_reg_date] = pd.to_datetime(df[col_reg_date], errors='coerce')

    # ✅ 4. 修改：检查项目清洗函数对齐参考版
    df['_calc_item'] = df[col_item].apply(clean_exam_name)

    # ---------------------------------------------------------
    # 任务 A: 计算等待时间
    # ---------------------------------------------------------
    df['计算_等待时间'] = (df[col_date] - df[col_reg_date]).dt.days

    violation_list = []

    count_device_mismatch = 0
    count_heart_rule = 0
    count_rule3 = 0
    count_contrast = 0

    # 下面的循环逻辑保持原样（只修“基础对齐”错误）
    for index, row in df.iterrows():
        item_name = row['_calc_item']
        dev_id = row['_calc_device_id']
        weekday = row['_calc_weekday']

        reasons = []

        # --- 检查 1: 常规违规 (设备能力不匹配) ---
        if dev_id in allowed_rules:
            if item_name not in allowed_rules[dev_id]:
                reasons.append(f"设备能力不符(设备{dev_id}无法做{item_name})")
                count_device_mismatch += 1
        else:
            # 设备不在规则表中：保持原逻辑不强行判违规
            pass

        # --- 检查 2: “心脏” 规则 ---
        if "心脏" in item_name:
            is_time_ok = weekday in [1, 3]
            is_machine_ok = (dev_id == 3)

            if not (is_time_ok and is_machine_ok):
                reasons.append("违反心脏规则(非周二周四或非3号机)")
                count_heart_rule += 1

        # --- 检查 3 & 4: “造影” 与 “关节造影” 规则 ---
        if KEYWORD_RULE_3 in item_name:
            is_time_ok = weekday in [0, 2, 4]
            is_machine_ok = (dev_id == 1)

            if not (is_time_ok and is_machine_ok):
                reasons.append(f"违反{KEYWORD_RULE_3}规则(非135或非1号机)")
                count_rule3 += 1

        elif "增强" in item_name:
            # 保持原逻辑（你只要求参照上面的设备对齐修逻辑错误）
            if weekday in [5, 6]:
                reasons.append("违反造影规则(非周末)")
                count_contrast += 1

        if len(reasons) > 0:
            violation_list.append("; ".join(reasons))
        else:
            violation_list.append("正常")

    df['违规情况'] = violation_list
    df['是否违规'] = df['违规情况'] != "正常"

    # ---------------------------------------------------------
    # 统计计算区域 (保持原样)
    # ---------------------------------------------------------
    total_count = len(df)
    violation_count = df['是否违规'].sum()
    violation_rate = (violation_count / total_count * 100) if total_count > 0 else 0

    clean_items = df['_calc_item']
    prev_items = clean_items.shift(1)

    diff_mask = (clean_items != prev_items) & (prev_items.notna())
    diff_count = diff_mask.sum()
    same_mask = (clean_items == prev_items) & (prev_items.notna())
    same_count = same_mask.sum()

    valid_wait = df['计算_等待时间'].dropna()
    wait_count_all = len(valid_wait)
    wait_mean_all = valid_wait.mean() if wait_count_all > 0 else 0
    wait_non_negative = valid_wait[valid_wait >= 0]
    wait_count_pos = len(wait_non_negative)
    wait_mean_pos = wait_non_negative.mean() if wait_count_pos > 0 else 0

    # ---------------------------------------------------------
    # 输出报告
    # ---------------------------------------------------------
    print(f"\n{'='*30}")
    print(f"【详细分析报告】")
    print(f"{'='*30}")

    print(f"1. 违规检查统计 (分门别类):")
    print(f"   - 总样本数: {total_count}")
    print(f"   - 总体违规样本: {violation_count} (占比 {violation_rate:.2f}%)")
    print(f"   ---------------------------")

    rate_dev = (count_device_mismatch / total_count * 100) if total_count > 0 else 0
    print(f"   - 常规违规 (设备能力不符): {count_device_mismatch} 个 ({rate_dev:.2f}%)")

    rate_heart = (count_heart_rule / total_count * 100) if total_count > 0 else 0
    print(f"   - 心脏规则违规: {count_heart_rule} 个 ({rate_heart:.2f}%)")

    rate_rule3 = (count_rule3 / total_count * 100) if total_count > 0 else 0
    print(f"   - {KEYWORD_RULE_3}规则违规: {count_rule3} 个 ({rate_rule3:.2f}%)")

    rate_contrast = (count_contrast / total_count * 100) if total_count > 0 else 0
    print(f"   - 增强违规: {count_contrast} 个 ({rate_contrast:.2f}%)")

    print(f"\n2. 相邻项目变动统计:")
    print(f"   - 差异个数 (换项目): {diff_count}")
    print(f"   - 相同个数 (没换项目): {same_count}")

    print(f"\n3. 等待时间统计 (日期 - 登记日期):")
    print(f"   [包含负值]")
    print(f"   - 总个数: {wait_count_all}")
    print(f"   - 平均等待: {wait_mean_all:.2f} 天")
    print(f"   [剔除负值 (>=0)]")
    print(f"   - 有效个数: {wait_count_pos}")
    print(f"   - 平均等待: {wait_mean_pos:.2f} 天")


if __name__ == "__main__":
    schedule_file = "/home/preprocess/_funsearch/baseline/output_schedules/final_schedule_RUN2_20251211_222119_fit_-108529600.xlsx"
    limit_file = '/home/preprocess/_funsearch/baseline/设备限制4.xlsx'
    analyze_medical_data(schedule_file, limit_file)

# # import re
# # import matplotlib
# # # 服务器环境强制使用 Agg
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt

# # def process_log(file_path, output_image="trace_4curves.png"):
# #     ep = []
# #     avg_best = []
# #     gbest = []
# #     feas = []
# #     viols = []
    
# #     # 修正后的正则表达式
# #     # 匹配 [gen (10000/5000] action=6 ...
# #     # 注意：\[ 和 \( 是对特殊字符的转义
# #     pattern = re.compile(
# #         r"\[gen\s*\(10000/5000\]\s*"  # 精确匹配你要求的起始标志
# #         r"action=(?P<action>\d+)\s+"
# #         r"mask=(?P<mask>\d+)\s+"
# #         r"avg_best_fit=(?P<avg>[-+]?\d*\.?\d+)\s+"
# #         r"gbest_fit=(?P<gbest>[-+]?\d*\.?\d+)\s+"
# #         r"mean_viols=(?P<viols>[-+]?\d*\.?\d+)\s+"
# #         r"feas_rate=(?P<feas>[-+]?\d*\.?\d+)\s+"
# #         r"stg=(?P<stg>\d+)"
# #     )

# #     count = 0
# #     with open(file_path, "r", encoding="utf-8") as f:
# #         for line in f:
# #             m = pattern.search(line)
# #             if not m:
# #                 continue

# #             # 每匹配到一行符合 [gen (10000/5000] 的数据，序号递增 1
# #             count += 1
# #             ep.append(count)
# #             avg_best.append(float(m.group("avg")))
# #             gbest.append(float(m.group("gbest")))
# #             viols.append(float(m.group("viols")))
# #             feas.append(float(m.group("feas")))

# #     if not ep:
# #         print("未匹配到任何记录，请检查 drl.log 是否包含 '[gen (10000/5000]'。")
# #         return

# #     print(f"成功提取到 {len(ep)} 条记录。")

# #     # --- 绘图逻辑 ---
# #     fig, ax1 = plt.subplots(figsize=(12, 6))

# #     # 左轴：Fitness (目标函数值)
# #     lns1 = ax1.plot(ep, gbest, '-', color='tab:blue', linewidth=1.5, label="Global Best Fit")
# #     lns2 = ax1.plot(ep, avg_best, '--', color='tab:cyan', linewidth=1, label="Avg Best Fit")
# #     ax1.set_xlabel("Occurrence Index (Timeline)")
# #     ax1.set_ylabel("Fitness Value", color='tab:blue')
# #     ax1.tick_params(axis='y', labelcolor='tab:blue')
# #     ax1.grid(True, linestyle="--", alpha=0.3)
# #     ax1.ticklabel_format(style="plain", axis="y") # 禁止科学计数法

# #     # 右轴：Feasible Rate & Violations (约束违背情况)
# #     ax2 = ax1.twinx()
# #     lns3 = ax2.plot(ep, feas, '-', color='tab:green', linewidth=1.5, label="Feasible Rate")
# #     lns4 = ax2.plot(ep, viols, ':', color='tab:red', linewidth=1, label="Mean Violations")
# #     ax2.set_ylabel("Rate / Violations", color='tab:green')
# #     ax2.tick_params(axis='y', labelcolor='tab:green')

# #     # 合并图例
# #     lns = lns1 + lns2 + lns3 + lns4
# #     labs = [l.get_label() for l in lns]
# #     ax1.legend(lns, labs, loc="upper left", bbox_to_anchor=(0.1, 1.15), ncol=4)

# #     plt.title(f"DRL Training Trace (Extracted from 7000-Patient Task Log)", pad=20)
# #     plt.tight_layout()
# #     plt.savefig(output_image, dpi=300)
# #     print(f"图像已保存至: {output_image}")

# # if __name__ == "__main__":
# #     process_log("/home/preprocess/_funsearch/baseline/0EMBC/drl.log")


# import re
# import matplotlib
# # 服务器环境强制使用 Agg
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np

# def process_log(file_path, output_image="trace_mask_rounds.png", max_plot_rounds=10):
#     """
#     参数:
#     file_path: 日志路径
#     output_image: 输出图片路径
#     max_plot_rounds: 
#         - 设置为 整数 (如 10, 50): 限制只绘制最后 N 轮数据，防止图片过于混乱。
#         - 设置为 None: 绘制所有轮次（如果不介意太密的话）。
#     """
    
#     # 用于存储所有轮次的数据
#     all_rounds_gen = []   # 存每一轮的 X 轴
#     all_rounds_mask = []  # 存每一轮的 Y 轴

#     # 临时存储当前轮次的数据
#     current_gen_list = []
#     current_mask_list = []
    
#     # 用于检测轮次重置
#     last_gen_idx = -1

#     # --- 1. 正则表达式 (保持不变) ---
#     pattern = re.compile(
#         r"\[gen\s*\((?P<gen_idx>\d+)/(?P<total_gen>\d+)\]\s*" 
#         r"action=(?P<action>\d+)\s+"
#         r"mask=(?P<mask>\w+)\s+" 
#         r".*?" 
#     )

#     count_valid_lines = 0

#     print(f"开始处理日志: {file_path}")
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             m = pattern.search(line)
#             if not m:
#                 continue

#             count_valid_lines += 1
            
#             gen_idx = int(m.group("gen_idx"))
#             raw_mask = m.group("mask")
#             try:
#                 # 尝试二进制转换
#                 mask_val = int(raw_mask, 2) 
#             except ValueError:
#                 mask_val = float(raw_mask)

#             # --- 2. 多轮次检测逻辑 ---
#             if gen_idx < last_gen_idx:
#                 if current_gen_list:
#                     all_rounds_gen.append(current_gen_list)
#                     all_rounds_mask.append(current_mask_list)
#                 current_gen_list = []
#                 current_mask_list = []

#             current_gen_list.append(gen_idx)
#             current_mask_list.append(mask_val)
#             last_gen_idx = gen_idx

#     # 保存最后一轮
#     if current_gen_list:
#         all_rounds_gen.append(current_gen_list)
#         all_rounds_mask.append(current_mask_list)

#     total_rounds = len(all_rounds_gen)
#     if total_rounds == 0:
#         print("未匹配到任何有效记录，请检查日志格式或正则表达式。")
#         return

#     print(f"提取完成: 共找到 {total_rounds} 轮数据 (Total Lines: {count_valid_lines})")

#     # --- 3. 筛选绘图轮次 (新增逻辑) ---
    
#     # 决定要画哪些轮次
#     if max_plot_rounds is not None and total_rounds > max_plot_rounds:
#         print(f"注意: 轮数 ({total_rounds}) 超过限制 ({max_plot_rounds})，将只绘制最后 {max_plot_rounds} 轮。")
#         # 切片截取最后 max_plot_rounds 轮
#         plot_rounds_gen = all_rounds_gen[-max_plot_rounds:]
#         plot_rounds_mask = all_rounds_mask[-max_plot_rounds:]
#         # 计算起始轮次的编号，以便图例显示正确的 Round ID
#         start_round_idx = total_rounds - max_plot_rounds + 1
#     else:
#         # 绘制所有
#         plot_rounds_gen = all_rounds_gen
#         plot_rounds_mask = all_rounds_mask
#         start_round_idx = 1

#     actual_plot_count = len(plot_rounds_gen)

#     # --- 4. 绘图逻辑 ---
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # 根据实际要画的线条数量生成颜色
#     colors = cm.rainbow(np.linspace(0, 1, actual_plot_count))

#     for i in range(actual_plot_count):
#         x_data = plot_rounds_gen[i]
#         y_data = plot_rounds_mask[i]
        
#         # 计算真实的 Round 编号
#         real_round_num = start_round_idx + i
        
#         ax.plot(x_data, y_data, 
#                 linestyle='-', 
#                 linewidth=1.5, 
#                 color=colors[i], 
#                 alpha=0.8, 
#                 label=f"Round {real_round_num}")

#     ax.set_xlabel("Generation")
#     ax.set_ylabel("Mask Value")
    
#     title_str = f"Mask Change (Showing {actual_plot_count} rounds"
#     if total_rounds > actual_plot_count:
#         title_str += f", Total History: {total_rounds}"
#     title_str += ")"
#     ax.set_title(title_str)
    
#     # 智能图例控制：如果要画的线条还是很多 (>20)，就不显示图例或者缩小字体
#     if actual_plot_count <= 15:
#         ax.legend(loc="upper right", framealpha=0.7)
#     elif actual_plot_count <= 30:
#         # 线条稍多时，分列显示，字体变小
#         ax.legend(loc="upper right", ncol=2, fontsize='x-small', framealpha=0.5)
#     else:
#         print("绘图轮次依然过多 (>30)，已自动隐藏图例。")

#     ax.grid(True, linestyle="--", alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_image, dpi=300)
#     print(f"图像已保存至: {output_image}")

# if __name__ == "__main__":
#     # 请确保路径正确
#     log_file_path = "/home/preprocess/_funsearch/baseline/0EMBC/drl.log"
    
#     # 修改这里来控制画线的数量：
#     # max_plot_rounds=50  -> 只画最后50轮
#     # max_plot_rounds=None -> 画所有轮(如果轮次成千上万会很慢且看不清)
#     process_log(log_file_path, max_plot_rounds=10)


import re
import pandas as pd
import matplotlib
# Force Agg backend if running on a server without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def process_log_loss(file_path, output_csv="loss_table.csv", output_image="loss_curve.png"):
    steps = []
    losses = []
    
    # Define Regex pattern for the new format:
    # Example: [step  75000] ... loss=-87945.921875 ...
    # \s+ matches one or more spaces
    # .*? non-greedy match for content in between
    pattern = re.compile(
        r"\[step\s+(?P<step>\d+)\]"       # Match [step  123] and capture digits
        r".*?"                            # Match anything in between (non-greedy)
        r"loss=(?P<loss>[-+]?\d*\.?\d+)"  # Match loss=123.456 (handles negative and float)
    )

    print(f"Reading file: {file_path} ...")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    # Append extracted data to lists
                    steps.append(int(m.group("step")))
                    losses.append(float(m.group("loss")))
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    # Check if data was found
    if not steps:
        print("No matching data found. Please check the log file content and regex.")
        return

    print(f"Successfully extracted {len(steps)} records.")

    # --- 1. Create and Save Table (DataFrame) ---
    df = pd.DataFrame({
        "Step": steps,
        "Loss": losses
    })
  
    # Display the first few rows
    print("Head of the extracted table:")
    print(df.head())

    # --- 2. Plotting Logic (Draw Chart) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Loss Curve
    ax.plot(steps, losses, '-', color='tab:red', linewidth=1.5, label="Loss")
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss per Step")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    
    # Handle scientific notation for large step numbers if needed
    ax.ticklabel_format(style="plain", axis="x") 

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Loss curve image saved to: {output_image}")

if __name__ == "__main__":
    # Replace with your actual log file path
    # process_log_loss("path/to/your/drl.log")
    
    # For testing with a dummy file (Uncomment below to test)
    with open("dummy_drl.log", "w") as f:
        f.write("[step  75000] reward_mean=-33710.078125 reward_std=559141.062500 baseline_ema=+93611.014787 loss=-87945.921875 time=1091.9s\n")
        f.write("[step  75500] reward_mean=-33000.000000 reward_std=550000.000000 baseline_ema=+93000.000000 loss=-85000.123456 time=1100.0s\n")
    
    process_log_loss("/home/preprocess/_funsearch/baseline/trans.log")

'''

/home/preprocess/.conda/envs/fastsurfer_gpu/bin/python /home/preprocess/_funsearch/baseline/draw.py

'''
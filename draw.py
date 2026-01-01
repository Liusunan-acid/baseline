import re
import matplotlib
# 服务器环境强制使用 Agg
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def process_log(file_path, output_image="trace_4curves.png"):
    ep = []
    avg_best = []
    gbest = []
    feas = []
    viols = []
    
    # 修正后的正则表达式
    # 匹配 [gen (10000/5000] action=6 ...
    # 注意：\[ 和 \( 是对特殊字符的转义
    pattern = re.compile(
        r"\[gen\s*\(10000/5000\]\s*"  # 精确匹配你要求的起始标志
        r"action=(?P<action>\d+)\s+"
        r"mask=(?P<mask>\d+)\s+"
        r"avg_best_fit=(?P<avg>[-+]?\d*\.?\d+)\s+"
        r"gbest_fit=(?P<gbest>[-+]?\d*\.?\d+)\s+"
        r"mean_viols=(?P<viols>[-+]?\d*\.?\d+)\s+"
        r"feas_rate=(?P<feas>[-+]?\d*\.?\d+)\s+"
        r"stg=(?P<stg>\d+)"
    )

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            # 每匹配到一行符合 [gen (10000/5000] 的数据，序号递增 1
            count += 1
            ep.append(count)
            avg_best.append(float(m.group("avg")))
            gbest.append(float(m.group("gbest")))
            viols.append(float(m.group("viols")))
            feas.append(float(m.group("feas")))

    if not ep:
        print("未匹配到任何记录，请检查 drl.log 是否包含 '[gen (10000/5000]'。")
        return

    print(f"成功提取到 {len(ep)} 条记录。")

    # --- 绘图逻辑 ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：Fitness (目标函数值)
    lns1 = ax1.plot(ep, gbest, '-', color='tab:blue', linewidth=1.5, label="Global Best Fit")
    lns2 = ax1.plot(ep, avg_best, '--', color='tab:cyan', linewidth=1, label="Avg Best Fit")
    ax1.set_xlabel("Occurrence Index (Timeline)")
    ax1.set_ylabel("Fitness Value", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.ticklabel_format(style="plain", axis="y") # 禁止科学计数法

    # 右轴：Feasible Rate & Violations (约束违背情况)
    ax2 = ax1.twinx()
    lns3 = ax2.plot(ep, feas, '-', color='tab:green', linewidth=1.5, label="Feasible Rate")
    lns4 = ax2.plot(ep, viols, ':', color='tab:red', linewidth=1, label="Mean Violations")
    ax2.set_ylabel("Rate / Violations", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 合并图例
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", bbox_to_anchor=(0.1, 1.15), ncol=4)

    plt.title(f"DRL Training Trace (Extracted from 7000-Patient Task Log)", pad=20)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"图像已保存至: {output_image}")

if __name__ == "__main__":
    process_log("/home/preprocess/_funsearch/baseline/0EMBC/drl.log")
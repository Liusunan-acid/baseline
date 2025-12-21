

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 配置区域 ======
LOG_FILES = [
    r"/home/preprocess/_funsearch/baseline/0PSO/PSO-2700.log",
    r"/home/preprocess/_funsearch/baseline/0ACO/ACO.log",
    r"/home/preprocess/_funsearch/baseline/ga-2000.log",
    r"", # 空字符串会被忽略

]

LABELS = [
    "PSO",
    "ACO",
    "GA-co-evolve",
]


MAX_GEN = 10000

# ====== 修复后的正则匹配模式 (关键修改) ======
# 使用 [^:]*: 替代 .*? 
# 强制匹配到关键字后的【第一个冒号】即停止，防止跳过数值匹配到后面的 Time: 或其他字段
LOG_PATTERNS = [
    # 模式 1: [PSO] Iter ... Avg gbest(K=10): -659...
    re.compile(r"\[PSO\]\s+Iter\s+(?P<gen>\d+)/\d+\s*\|.*?Avg gbest[^:]*:\s*(?P<fit>[-+]?\d+(?:\.\d+)?)"),
    
    # 模式 2: Iter ... Best: -191... | Time: ... (关键修复点)
    # (?<!\[PSO\]\s) 确保不匹配带 [PSO] 前缀的行
    re.compile(r"(?<!\[PSO\]\s)Iter\s+(?P<gen>\d+)/\d+\s*\|.*?Best[^:]*:\s*(?P<fit>[-+]?\d+(?:\.\d+)?)"),
    
    # 模式 3: Generation ... Fitness ...
    re.compile(r"Generation\s+(?P<gen>\d+)\s*\|.*?Fitness[^:]*:\s*(?P<fit>[-+]?\d+(?:\.\d+)?)")
]

def parse_log(path: str):
    data = [] 
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                for pat in LOG_PATTERNS:
                    m = pat.search(line)
                    if m:
                        try:
                            gen = int(m.group("gen"))
                            fit = float(m.group("fit"))
                            data.append((gen, fit))
                            break 
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件: {path}")
        return [], []

    data.sort(key=lambda x: x[0])
    
    if not data: return [], []
    return [x[0] for x in data], [x[1] for x in data]

def main():
    series = []
    print(f"开始解析日志 (Max Gen: {MAX_GEN if MAX_GEN else 'All'})...")
    
    for i, p in enumerate(LOG_FILES):
        if not p: continue
        
        path_obj = Path(p).expanduser()
        gens, fits = parse_log(str(path_obj))
        
        if not gens:
            print(f"[WARN] 未匹配到数据：{path_obj.name}")
            continue
        
        if MAX_GEN is not None:
            filtered = [(g, f) for g, f in zip(gens, fits) if g <= MAX_GEN]
            if not filtered:
                print(f"[WARN] {path_obj.name} 在前 {MAX_GEN} 代没有数据。")
                continue
            gens, fits = zip(*filtered)

        label = LABELS[i] if i < len(LABELS) and LABELS[i] else path_obj.name
        series.append((label, gens, fits))
        print(f" -> {label}: 加载 {len(gens)} 点 (Last Gen: {gens[-1]}, Last Fit: {fits[-1]})")

    # ====== 新增 ALNS 单点数据 ======
    alns_gen = [10000]
    alns_fit = [-99936400.0000]
    series.append(("ALNS", alns_gen, alns_fit))
    print(f" -> ALNS: 手动添加点 (Gen: {alns_gen[0]}, Fit: {alns_fit[0]})")

    if len(series) < 1:
        print("[ERROR] 没有有效数据可供绘图。")
        return

    # ====== 绘图 ======
    plt.figure(figsize=(10, 6))
    
    for label, gens, fits in series:
        if label == "ALNS":
            # ALNS 只有一个点，用散点(scatter)或星号标注，否则 plot 连线不可见
            plt.scatter(gens, fits, label=label, color='red', marker='*', s=100, zorder=5)
        else:
            plt.plot(gens, fits, label=label, linewidth=1.5, alpha=0.8)

    plt.xlabel("Generation / Iteration")
    plt.ylabel("Best Fitness")
    plt.title(f"Optimization Convergence")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 因为数值很大，y轴可能需要科学计数法或调整格式
    # plt.ticklabel_format(style='plain', axis='y') # 如果不想用科学计数法可保留此行

    if MAX_GEN is not None:
        plt.xlim(left=0, right=MAX_GEN)

    plt.tight_layout()
    output_filename = "avg_best_fitness.png"
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    print(f"\n绘图完成！已保存为: {output_filename}")

if __name__ == "__main__":
    main()
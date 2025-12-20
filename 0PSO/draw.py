#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 你在这里填 3~4 个 log 文件路径（不用的留空字符串 "" 即可）======
LOG_FILES = [
    r"/home/preprocess/_funsearch/baseline/0PSO/PSO-2700.log",
    r"/home/preprocess/_funsearch/baseline/0PSO/PSO-tuning.log",
    r"",
    r"",
]

# 可选：自定义每条曲线的名字（不填则用文件名）
LABELS = [
    "PSO Baseline",
    "Run B",
    # "Run C",
]

# 匹配示例：
# [PSO] Iter   100/10000 | Avg gbest(K=10): -775092544.0000
LINE_RE = re.compile(
    r"\[PSO\]\s+Iter\s+(?P<gen>\d+)/\d+\s*\|.*?"
    r"Avg gbest.*?:\s*(?P<fit>[-+]?\d+(?:\.\d+)?)"
)

def parse_log(path: str):
    gens, fits = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            gens.append(int(m.group("gen")))
            fits.append(float(m.group("fit")))
    return gens, fits

def main():
    series = []
    for i, p in enumerate(LOG_FILES):
        if not p:
            continue
        p = str(Path(p).expanduser())
        
        # 解析日志（移除了倍率参数）
        gens, fits = parse_log(p)
        
        if not gens:
            print(f"[WARN] 未解析到数据：{p}")
            continue
        
        label = LABELS[i] if i < len(LABELS) and LABELS[i] else Path(p).name
        series.append((label, gens, fits))

    if len(series) < 1:
        print("[ERROR] 至少需要1个有效log文件用于绘图。")
        return

    # ====== 图1：Avg Best Fitness ======
    plt.figure()
    for label, gens, fits in series:
        plt.plot(gens, fits, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Avg Best Fitness")
    plt.title("Avg Best Fitness vs Generation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("avg_best_fitness.png", dpi=200, bbox_inches="tight")
    print("已保存：avg_best_fitness.png")

if __name__ == "__main__":
    main()
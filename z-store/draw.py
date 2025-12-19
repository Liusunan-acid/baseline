#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 你在这里填 3~4 个 log 文件路径（不用的留空字符串 "" 即可）======
LOG_FILES = [
    r"/home/preprocess/_funsearch/baseline/z-store/0单种群-患者级.txt",
    r"/home/preprocess/_funsearch/baseline/z-store/0单种群-位置级.txt",
    r"/home/preprocess/_funsearch/baseline/z-store/0多种群-患者级.txt",
    # r"",  # 预留第4个
]

# 可选：自定义每条曲线的名字（不填则用文件名）
LABELS = [
    "Run A",
    "Run B",
    "Run C",
    # "Run D",
]

# ✅ 只对指定文件做 Generation 倍率修正（这里让“0多种群-患者级.txt”翻倍）
GEN_MULTIPLIERS = {
    r"/home/preprocess/_funsearch/baseline/z-store/0多种群-患者级.txt": 2
}

# 匹配示例：
# Generation 1400 | Avg Best Fitness (K=4): -205136400.00 | Avg Violations: 91.25
LINE_RE = re.compile(
    r"Generation\s+(?P<gen>\d+)\s*\|.*?"
    r"Avg Best Fitness.*?:\s*(?P<fit>[-+]?\d+(?:\.\d+)?)\s*\|.*?"
    r"Avg Violations:\s*(?P<vio>[-+]?\d+(?:\.\d+)?)"
)

def parse_log(path: str, gen_multiplier: int = 1):
    gens, fits, vios = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            gens.append(int(m.group("gen")) * gen_multiplier)  # ✅ 这里应用倍率
            fits.append(float(m.group("fit")))
            vios.append(float(m.group("vio")))
    return gens, fits, vios

def main():
    series = []
    for i, p in enumerate(LOG_FILES):
        if not p:
            continue
        p = str(Path(p).expanduser())
        mult = GEN_MULTIPLIERS.get(p, 1)  # ✅ 没配置就默认 1
        gens, fits, vios = parse_log(p, gen_multiplier=mult)
        if not gens:
            print(f"[WARN] 未解析到数据：{p}")
            continue
        label = LABELS[i] if i < len(LABELS) and LABELS[i] else Path(p).name
        series.append((label, gens, fits, vios))

    if len(series) < 2:
        print("[ERROR] 至少需要2个有效log文件用于对比绘图。")
        return

    # ====== 图1：Avg Best Fitness ======
    plt.figure()
    for label, gens, fits, _ in series:
        plt.plot(gens, fits, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Avg Best Fitness")
    plt.title("Avg Best Fitness vs Generation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ====== 图2：Avg Violations ======
    plt.figure()
    for label, gens, _, vios in series:
        plt.plot(gens, vios, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Avg Violations")
    plt.title("Avg Violations vs Generation")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.figure(1)
    plt.savefig("avg_best_fitness.png", dpi=200, bbox_inches="tight")
    plt.figure(2)
    plt.savefig("avg_violations.png", dpi=200, bbox_inches="tight")
    print("已保存：avg_best_fitness.png, avg_violations.png")

if __name__ == "__main__":
    main()

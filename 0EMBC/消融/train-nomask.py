#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_ablate_nomask_attention.py

训练 Step2 的“位置选择注意力策略（idx2 选择）”，但**去掉所有 hard mask**。

用途：消融实验 —— 验证“没有 mask 时，注意力机制是否仍能学到有效指导”。

与原 train.py 的区别（最小化）：
1) 默认 --coe_py 指向 trans_ablate_nomask_attention.py（也可手动指定任意 coe 模块）
2) 保存的 ckpt 文件名加上 _NOMASK_ 前缀，避免和原版混淆

注意：
- 这个脚本本身不改算法逻辑；“去 mask”发生在你传入的 coe_py（即 trans_ablate_nomask_attention.py）里。
- 如果你把 --coe_py 指回原 trans.py，那就变回原版训练。
"""

from __future__ import annotations

import argparse
import os
import time
import random
from pathlib import Path

import torch

import importlib.util


def load_module_from_path(py_path: str):
    py_path = os.path.abspath(py_path)
    spec = importlib.util.spec_from_file_location("coe_step2_attn_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def eval_batch_fitness(optimizer, perms: torch.Tensor) -> torch.Tensor:
    out = optimizer._gpu_engine.fitness_batch(perms, return_assignment=False)
    return out["fitness"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coe_py",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "trans_ablate_nomask_attention.py"),
        help="Path to COE module .py (default: trans_ablate_nomask_attention.py)",
    )
    ap.add_argument("--patient_file", type=str, required=True)
    ap.add_argument("--duration_file", type=str, required=True)
    ap.add_argument("--device_constraint_file", type=str, required=True)

    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ema_baseline", type=float, default=0.99)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", type=str, default="./step2_ckpts_nomask")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--resume", type=str, default="", help="resume ckpt path")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    mod = load_module_from_path(args.coe_py)
    DEVICE = mod.DEVICE
    print(f"[INFO] Using DEVICE={DEVICE}")
    print(f"[INFO] COE module: {os.path.abspath(args.coe_py)}")

    # --- load data using module helpers ---
    patients = mod.import_data(args.patient_file, args.duration_file)
    machine_exam_map = mod.import_device_constraints(args.device_constraint_file)

    optimizer = mod.MultiRunOptimizer(
        patients=patients,
        machine_exam_map=machine_exam_map,
        num_parallel_runs=args.K,
        pop_size_per_run=args.B,
        block_start_date=None,
    )
    optimizer._ensure_gpu_engine()
    optimizer.initialize_population()

    # Enable Step2 policy/value
    optimizer.enable_step2_position_policy(ckpt_path=None, deterministic=False)
    assert optimizer.step2_pos_policy is not None
    optimizer.step2_pos_window = int(args.window)
    optimizer.step2_pos_deterministic = False

    policy = optimizer.step2_pos_policy
    value = optimizer.step2_pos_value  # may be None
    policy.train()
    if value is not None:
        value.train()

    params = list(policy.parameters())
    if value is not None:
        params += list(value.parameters())

    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # optional resume
    start_step = 0
    baseline_ema = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            policy.load_state_dict(ckpt["policy"], strict=False)
            if value is not None and "value" in ckpt and ckpt["value"] is not None:
                value.load_state_dict(ckpt["value"], strict=False)
            start_step = int(ckpt.get("step", 0))
            baseline_ema = float(ckpt.get("baseline_ema", 0.0))
        else:
            policy.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Resumed from {args.resume} (start_step={start_step})")

    # Build a pool of perms to sample from (flatten initial pop)
    pop = optimizer.population_tensor  # [K,B,N]
    if pop is None:
        raise RuntimeError("population_tensor is None after initialize_population()")
    K, B, N = pop.shape
    pool = pop.reshape(K * B, N)
    pool_size = pool.shape[0]
    bs = min(args.batch_size, pool_size)

    # training loop
    t0 = time.time()
    for step in range(start_step, start_step + args.steps):
        # sample parents from pool
        idx = torch.randint(0, pool_size, (bs,), device=DEVICE)
        parents = pool[idx].clone()  # [bs,N]

        # fitness before
        with torch.no_grad():
            fit_parent = eval_batch_fitness(optimizer, parents)

        # mutate ONLY step2; force all rows to swap to get logprob for each row
        children = parents.clone()
        optimizer._mutate_step2_base_swap(children, current_gen=0, base_swap_prob=1.0)

        logprob = optimizer.last_step2_logprob
        if logprob is None:
            raise RuntimeError(
                "last_step2_logprob is None; ensure deterministic=False and step2 policy enabled."
            )
        if logprob.numel() != bs:
            # 理论上不该发生；发生了就先截断，避免训练直接中断
            logprob = logprob[:bs]

        # fitness after
        with torch.no_grad():
            fit_child = eval_batch_fitness(optimizer, children)

        reward = (fit_child - fit_parent)  # maximize
        r_mean = float(reward.mean().item())
        r_std = float(reward.std(unbiased=False).item())

        baseline_ema = args.ema_baseline * baseline_ema + (1.0 - args.ema_baseline) * r_mean
        adv = reward - baseline_ema

        # entropy proxy
        entropy_approx = (-logprob).mean()
        loss_actor = -(logprob * adv.detach()).mean() - args.entropy_coef * entropy_approx
        loss = loss_actor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            print(
                f"[step {step+1:>6}] "
                f"reward_mean={r_mean:+.6f} reward_std={r_std:.6f} "
                f"baseline_ema={baseline_ema:+.6f} "
                f"loss={float(loss.item()):.6f} "
                f"time={dt:.1f}s"
            )

        if (step + 1) % args.save_every == 0:
            ckpt_path = save_dir / f"step2_pos_attn_NOMASK_step{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "baseline_ema": baseline_ema,
                    "policy": policy.state_dict(),
                    "value": value.state_dict() if value is not None else None,
                    "window": int(args.window),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                    "coe_py": os.path.abspath(args.coe_py),
                },
                ckpt_path,
            )
            print(f"[SAVE] {ckpt_path}")

    # final save
    final_path = save_dir / "step2_pos_attn_NOMASK_final.pt"
    torch.save(
        {
            "step": start_step + args.steps,
            "baseline_ema": baseline_ema,
            "policy": policy.state_dict(),
            "value": value.state_dict() if value is not None else None,
            "window": int(args.window),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "coe_py": os.path.abspath(args.coe_py),
        },
        final_path,
    )
    print(f"[DONE] saved: {final_path}")


if __name__ == "__main__":
    main()

    

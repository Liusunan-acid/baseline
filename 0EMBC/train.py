
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Train Step2 position-selection attention policy (paper-style) for COE.

# This script trains ONLY the idx2 selection in `_mutate_step2_base_swap`:
# - reference: idx1 (randomly chosen swap position)
# - candidates: idx1 ± window
# - action: pick idx2 among candidates via reference-conditioned attention
# - reward: Δfitness = fitness(child) - fitness(parent)  (maximize)

# It does NOT use Step1 violations expert mutation or other operators.
# """

# from __future__ import annotations
# import argparse
# import os
# import time
# import math
# import random
# from pathlib import Path

# import torch

# # ---- dynamic import of your COE module (Chinese filename supported) ----
# import importlib.util


# def load_module_from_path(py_path: str):
#     py_path = os.path.abspath(py_path)
#     spec = importlib.util.spec_from_file_location("coe_step2_attn_mod", py_path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError(f"Cannot import module from: {py_path}")
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     return mod


# @torch.no_grad()
# def eval_batch_fitness(optimizer, perms: torch.Tensor) -> torch.Tensor:
#     out = optimizer._gpu_engine.fitness_batch(perms, return_assignment=False)
#     fit = out["fitness"]
#     return fit


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--coe_py", type=str, required=True,
#                     help="Path to coe-位置级-子种群对齐_step2_attn.py")
#     ap.add_argument("--patient_file", type=str, required=True)
#     ap.add_argument("--duration_file", type=str, required=True)
#     ap.add_argument("--device_constraint_file", type=str, required=True)

#     ap.add_argument("--K", type=int, default=1, help="num parallel runs (unused for training but needed to init)")
#     ap.add_argument("--B", type=int, default=64, help="pop size per run (controls initial population pool)")
#     ap.add_argument("--batch_size", type=int, default=256, help="training batch size (num perms per step)")
#     ap.add_argument("--steps", type=int, default=20000, help="training steps (iterations)")
#     ap.add_argument("--window", type=int, default=400, help="candidate window radius (±window)")
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--weight_decay", type=float, default=0.0)
#     ap.add_argument("--ema_baseline", type=float, default=0.99, help="EMA factor for scalar baseline")
#     ap.add_argument("--entropy_coef", type=float, default=0.01, help="entropy bonus coef")
#     ap.add_argument("--grad_clip", type=float, default=1.0)
#     ap.add_argument("--seed", type=int, default=42)

#     ap.add_argument("--save_dir", type=str, default="./step2_ckpts")
#     ap.add_argument("--save_every", type=int, default=1000)
#     ap.add_argument("--log_every", type=int, default=50)
#     ap.add_argument("--resume", type=str, default="", help="resume ckpt path")

#     args = ap.parse_args()

#     random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     mod = load_module_from_path(args.coe_py)

#     DEVICE = mod.DEVICE
#     print(f"[INFO] Using DEVICE={DEVICE}")

#     # --- load data using module helpers ---
#     patients = mod.import_data(args.patient_file, args.duration_file)
#     machine_exam_map = mod.import_device_constraints(args.device_constraint_file)

#     optimizer = mod.MultiRunOptimizer(
#         patients=patients,
#         machine_exam_map=machine_exam_map,
#         num_parallel_runs=args.K,
#         pop_size_per_run=args.B,
#         block_start_date=None,
#     )
#     optimizer._ensure_gpu_engine()
#     optimizer.initialize_population()

#     # Enable Step2 policy/value
#     optimizer.enable_step2_position_policy(ckpt_path=None, deterministic=False)
#     assert optimizer.step2_pos_policy is not None
#     optimizer.step2_pos_window = int(args.window)
#     optimizer.step2_pos_deterministic = False

#     policy = optimizer.step2_pos_policy
#     value = optimizer.step2_pos_value  # may be None
#     policy.train()
#     if value is not None:
#         value.train()

#     params = list(policy.parameters())
#     if value is not None:
#         params += list(value.parameters())

#     opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     # optional resume
#     start_step = 0
#     baseline_ema = 0.0
#     if args.resume:
#         ckpt = torch.load(args.resume, map_location=DEVICE)
#         if isinstance(ckpt, dict) and "policy" in ckpt:
#             policy.load_state_dict(ckpt["policy"], strict=False)
#             if value is not None and "value" in ckpt and ckpt["value"] is not None:
#                 value.load_state_dict(ckpt["value"], strict=False)
#             start_step = int(ckpt.get("step", 0))
#             baseline_ema = float(ckpt.get("baseline_ema", 0.0))
#         else:
#             policy.load_state_dict(ckpt, strict=False)
#         print(f"[INFO] Resumed from {args.resume} (start_step={start_step})")

#     # Build a pool of perms to sample from (flatten initial pop)
#     pop = optimizer.population_tensor  # [K,B,N]
#     if pop is None:
#         raise RuntimeError("population_tensor is None after initialize_population()")
#     K, B, N = pop.shape
#     pool = pop.reshape(K * B, N)
#     pool_size = pool.shape[0]
#     bs = min(args.batch_size, pool_size)

#     # training loop
#     t0 = time.time()
#     for step in range(start_step, start_step + args.steps):
#         # sample parents from pool
#         idx = torch.randint(0, pool_size, (bs,), device=DEVICE)
#         parents = pool[idx].clone()  # [bs,N]

#         # fitness before
#         with torch.no_grad():
#             fit_parent = eval_batch_fitness(optimizer, parents)  # [bs]

#         # mutate ONLY step2; force all rows to swap to get logprob for each row
#         children = parents.clone()
#         optimizer._mutate_step2_base_swap(children, current_gen=0, base_swap_prob=1.0)
#         logprob = optimizer.last_step2_logprob
#         if logprob is None:
#             raise RuntimeError("last_step2_logprob is None; ensure deterministic=False and step2 policy enabled.")
#         # Ensure shape [bs]
#         if logprob.numel() != bs:
#             # In rare cases, some rows might have been skipped; align by truncation/pad (better to debug)
#             logprob = logprob[:bs]

#         # fitness after
#         with torch.no_grad():
#             fit_child = eval_batch_fitness(optimizer, children)

#         reward = (fit_child - fit_parent)  # [bs], maximize
#         r_mean = float(reward.mean().item())
#         r_std = float(reward.std(unbiased=False).item())

#         # scalar baseline (EMA)
#         baseline_ema = args.ema_baseline * baseline_ema + (1.0 - args.ema_baseline) * r_mean
#         adv = reward - baseline_ema

#         # entropy bonus (encourage exploration)
#         # We don't store full dist, but can estimate entropy from logprob only poorly.
#         # Instead, we recompute logits for entropy (cheap relative to fitness? no, fitness is heavy).
#         # For minimal, set entropy bonus = 0 or approximate using -logprob mean.
#         entropy_approx = (-logprob).mean()  # rough proxy (higher => more random)
#         loss_actor = -(logprob * adv.detach()).mean() - args.entropy_coef * entropy_approx

#         loss = loss_actor

#         opt.zero_grad(set_to_none=True)
#         loss.backward()
#         if args.grad_clip > 0:
#             torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
#         opt.step()

#         if (step + 1) % args.log_every == 0:
#             dt = time.time() - t0
#             print(
#                 f"[step {step+1:>6}] "
#                 f"reward_mean={r_mean:+.6f} reward_std={r_std:.6f} "
#                 f"baseline_ema={baseline_ema:+.6f} "
#                 f"loss={float(loss.item()):.6f} "
#                 f"time={dt:.1f}s"
#             )

#         if (step + 1) % args.save_every == 0:
#             ckpt_path = save_dir / f"step2_pos_attn_step{step+1}.pt"
#             torch.save(
#                 {
#                     "step": step + 1,
#                     "baseline_ema": baseline_ema,
#                     "policy": policy.state_dict(),
#                     "value": value.state_dict() if value is not None else None,
#                     "window": int(args.window),
#                     "lr": float(args.lr),
#                     "seed": int(args.seed),
#                 },
#                 ckpt_path,
#             )
#             print(f"[SAVE] {ckpt_path}")

#     # final save
#     final_path = save_dir / f"step2_pos_attn_final.pt"
#     torch.save(
#         {
#             "step": start_step + args.steps,
#             "baseline_ema": baseline_ema,
#             "policy": policy.state_dict(),
#             "value": value.state_dict() if value is not None else None,
#             "window": int(args.window),
#             "lr": float(args.lr),
#             "seed": int(args.seed),
#         },
#         final_path,
#     )
#     print(f"[DONE] saved: {final_path}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Step2 position-selection attention policy (paper-style) for COE.

This script trains ONLY the idx2 selection in `_mutate_step2_base_swap`:
- reference: idx1 (randomly chosen swap position)
- candidates: idx1 ± window
- action: pick idx2 among candidates via reference-conditioned attention
- reward: Δfitness = fitness(child) - fitness(parent)  (maximize)

It does NOT use Step1 violations expert mutation or other operators.
"""

from __future__ import annotations
import argparse
import os
import time
import math
import random
from pathlib import Path

import torch

# ---- dynamic import of your COE module (Chinese filename supported) ----
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
    fit = out["fitness"]
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coe_py", type=str, required=True,
                    help="Path to coe-位置级-子种群对齐_step2_attn.py")
    ap.add_argument("--patient_file", type=str, required=True)
    ap.add_argument("--duration_file", type=str, required=True)
    ap.add_argument("--device_constraint_file", type=str, required=True)

    ap.add_argument("--K", type=int, default=1, help="num parallel runs (unused for training but needed to init)")
    ap.add_argument("--B", type=int, default=64, help="pop size per run (controls initial population pool)")
    ap.add_argument("--batch_size", type=int, default=256, help="training batch size (num perms per step)")
    ap.add_argument("--steps", type=int, default=20000, help="training steps (iterations)")
    ap.add_argument("--window", type=int, default=400, help="candidate window radius (±window)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--ema_baseline", type=float, default=0.99, help="EMA factor for scalar baseline")
    ap.add_argument("--entropy_coef", type=float, default=0.01, help="entropy bonus coef")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", type=str, default="./step2_ckpts")
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
            fit_parent = eval_batch_fitness(optimizer, parents)  # [bs]

        # mutate ONLY step2; force all rows to swap to get logprob for each row
        children = parents.clone()
        optimizer._mutate_step2_base_swap(children, current_gen=0, base_swap_prob=1.0)
        logprob = optimizer.last_step2_logprob
        if logprob is None:
            raise RuntimeError("last_step2_logprob is None; ensure deterministic=False and step2 policy enabled.")
        # Ensure shape [bs]
        if logprob.numel() != bs:
            # In rare cases, some rows might have been skipped; align by truncation/pad (better to debug)
            logprob = logprob[:bs]

        # fitness after
        with torch.no_grad():
            fit_child = eval_batch_fitness(optimizer, children)

        reward = (fit_child - fit_parent)  # [bs], maximize
        r_mean = float(reward.mean().item())
        r_std = float(reward.std(unbiased=False).item())

        # scalar baseline (EMA)
        baseline_ema = args.ema_baseline * baseline_ema + (1.0 - args.ema_baseline) * r_mean
        adv = reward - baseline_ema

        # entropy bonus (encourage exploration)
        # We don't store full dist, but can estimate entropy from logprob only poorly.
        # Instead, we recompute logits for entropy (cheap relative to fitness? no, fitness is heavy).
        # For minimal, set entropy bonus = 0 or approximate using -logprob mean.
        entropy_approx = (-logprob).mean()  # rough proxy (higher => more random)
        loss_actor = -(logprob * adv.detach()).mean() - args.entropy_coef * entropy_approx

        loss = loss_actor

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
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
            ckpt_path = save_dir / f"step2_pos_attn_step{step+1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "baseline_ema": baseline_ema,
                    "policy": policy.state_dict(),
                    "value": value.state_dict() if value is not None else None,
                    "window": int(args.window),
                    "lr": float(args.lr),
                    "seed": int(args.seed),
                },
                ckpt_path,
            )
            print(f"[SAVE] {ckpt_path}")

    # final save
    final_path = save_dir / f"step2_pos_attn_final.pt"
    torch.save(
        {
            "step": start_step + args.steps,
            "baseline_ema": baseline_ema,
            "policy": policy.state_dict(),
            "value": value.state_dict() if value is not None else None,
            "window": int(args.window),
            "lr": float(args.lr),
            "seed": int(args.seed),
        },
        final_path,
    )
    print(f"[DONE] saved: {final_path}")


if __name__ == "__main__":
    main()


'''

训练
CUDA_VISIBLE_DEVICES=3 python /home/preprocess/_funsearch/baseline/0EMBC/train.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/trans.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 40000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir ./step2_ckpts \
  --save_every 1000 \
  --log_every 50

CUDA_VISIBLE_DEVICES=1  nohup python /home/preprocess/_funsearch/baseline/0EMBC/train.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/trans.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 1000000000000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir ./step2_ckpts \
  --save_every 1000 \
  --log_every 50 > maskandattention.log 2>&1 &

继续训练
CUDA_VISIBLE_DEVICES=3 python /home/preprocess/_funsearch/baseline/0EMBC/train.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/trans.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 40000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir /home/preprocess/_funsearch/baseline/step2_ckpts \
  --save_every 1000 \
  --log_every 50 \
  --resume /home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_final.pt

export COE_STEP2_POS_CKPT=/home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_final.pt
export COE_STEP2_POS_CKPT=/home/preprocess/_funsearch/baseline/step2_ckpts/step2_pos_attn_step50000.pt
export COE_STEP2_POS_DETERMINISTIC=1

CUDA_VISIBLE_DEVICES=3 python /home/preprocess/_funsearch/baseline/0EMBC/trans.py


'''
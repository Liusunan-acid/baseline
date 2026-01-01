#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COE with Step2 mask-only (no attention) using trans_maskonly_full.py.

Usage:
  COE_STEP2_MASK_ONLY=1 COE_STEP2_POS_WINDOW=400 python run_mask_only.py
or edit the env vars below.
"""

import os
import importlib.util

# Default switches (can be overridden by environment variables)
os.environ.setdefault("COE_STEP2_MASK_ONLY", "1")
os.environ.setdefault("COE_STEP2_POS_WINDOW", "400")
os.environ.setdefault("COE_STEP2_POS_DETERMINISTIC", "0")

# TODO: if your trans main reads other env vars / args, set them here as well.

def load_module_from_path(py_path: str):
    spec = importlib.util.spec_from_file_location("coe_mod", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    mod = load_module_from_path(os.path.join(os.path.dirname(__file__), "/home/preprocess/_funsearch/baseline/0EMBC/消融/trans-noattention.py"))
    mod.main()

if __name__ == "__main__":
    main()


'''

训练
CUDA_VISIBLE_DEVICES=1 python /home/preprocess/_funsearch/baseline/0EMBC/消融/train-noattention.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/消融/trans-noattention.py \
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

CUDA_VISIBLE_DEVICES=3  nohup python /home/preprocess/_funsearch/baseline/0EMBC/train.py \
  --coe_py /home/preprocess/_funsearch/baseline/0EMBC/trans.py \
  --patient_file "/home/preprocess/_funsearch/baseline/data/实验数据6.1 - 副本.xlsx" \
  --duration_file "/home/preprocess/_funsearch/baseline/data/程序使用实际平均耗时3 - 副本.xlsx" \
  --device_constraint_file "/home/preprocess/_funsearch/baseline/data/设备限制4.xlsx" \
  --K 1 --B 50 \
  --batch_size 256 \
  --steps 100000000 \
  --window 400 \
  --lr 1e-4 \
  --save_dir ./step2_ckpts \
  --save_every 1000 \
  --log_every 50 > trans.log 2>&1 &

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
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/hard_negative
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
cd "$ROOT"
date -Is > start.txt
hostname >> start.txt
echo "physical_gpu=0 wrapper_pid=$$" >> start.txt
CUDA_VISIBLE_DEVICES=0 "$PY" -u train_weighted.py \
  --data "$ROOT/hand_features.parquet" \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output "$ROOT" \
  --folds fold_01,fold_02 \
  --variants hard2,hard4,random2,random4 \
  --coverage 0.10 --max-len 224 --epochs 2 --bs 192 --lr 2e-5 --seed 20260814
date -Is > end.txt

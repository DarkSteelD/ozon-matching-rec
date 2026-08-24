#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/unit_normalization
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
cd "$ROOT"
date -Is > start.txt; hostname >> start.txt; echo "physical_gpu=1 wrapper_pid=$$" >> start.txt
CUDA_VISIBLE_DEVICES=1 "$PY" -u train.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output "$ROOT" --folds fold_01,fold_02 --variants baseline,normalized,corrupt \
  --max-len 224 --epochs 2 --bs 192 --lr 2e-5 --seed 20260814
date -Is > end.txt

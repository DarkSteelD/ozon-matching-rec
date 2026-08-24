#!/usr/bin/env bash
set -euo pipefail
base=/home/dzkhomidov/matching-work/rescue_20260824/positive_composition_devbox02
out=$base/e2_len384_fold4_gpu3
test -f /home/dzkhomidov/matching-work/locks/gpu3_positive_composition.lock/OWNER
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=true
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python "$base/train.py" \
  --variant e2_len384 --data "$base/hand_pairs_pd_v3cal.parquet" \
  --init "$base/rubase_llmfull_e2" --output "$out" --max-len 384 \
  --folds fold_04 --epochs 2 --effective-bs 256 --micro-bs 128 --eval-bs 256 \
  --lr 2e-5 --seed 20260814 > "$out/train.log" 2>&1
date -Is > "$out/COMPLETE"

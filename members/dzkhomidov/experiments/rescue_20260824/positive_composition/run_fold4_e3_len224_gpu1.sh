#!/usr/bin/env bash
set -euo pipefail
root=/home/dzkhomidov/matching-work/rescue_20260824/positive_composition
out=$root/parallel_gpu1_e3_len224_fold4
test -f /home/dzkhomidov/matching-work/locks/gpu1_positive_composition.lock/OWNER
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=true
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/rescue_20260824/student_long_context/train.py \
  --variant e3_len224 \
  --data /home/dzkhomidov/matching-work/rescue_20260824/macro_balance/inputs/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/rescue_20260824/third_pretrain/ckpt/rubase_llmfull_e3 \
  --output "$out" --max-len 224 --folds fold_04 \
  --epochs 2 --effective-bs 256 --micro-bs 128 --eval-bs 256 \
  --lr 2e-5 --seed 20260814 > "$out/train.log" 2>&1
date -Is > "$out/COMPLETE"

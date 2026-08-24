#!/usr/bin/env bash
set -euo pipefail

root=/home/dzkhomidov/matching-work/rescue_20260824/positive_composition
trainer=/home/dzkhomidov/matching-work/rescue_20260824/student_long_context/train.py
python_bin=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
data=/home/dzkhomidov/matching-work/rescue_20260824/macro_balance/inputs/hand_pairs_pd_v3cal.parquet
e2=/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2
e3=/home/dzkhomidov/matching-work/rescue_20260824/third_pretrain/ckpt/rubase_llmfull_e3
lock=/home/dzkhomidov/matching-work/locks/gpu0_student_composition_migrate.lock

test -f "$lock/OWNER"
test -f "$trainer"
test -f "$data"
test -f "$e2/model.safetensors"
test -f "$e3/model.safetensors"

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

run_cell() {
  variant=$1
  max_len=$2
  init=$3
  "$python_bin" "$trainer" \
    --variant "$variant" \
    --data "$data" \
    --init "$init" \
    --output "$root" \
    --max-len "$max_len" \
    --folds fold_03,fold_04 \
    --epochs 2 \
    --effective-bs 256 \
    --micro-bs 128 \
    --eval-bs 256 \
    --lr 2e-5 \
    --seed 20260814 \
    > "$root/${variant}_folds34.log" 2>&1
}

run_cell e2_len224 224 "$e2"
run_cell e3_len224 224 "$e3"
run_cell e2_len384 384 "$e2"
run_cell e3_len384 384 "$e3"

"$python_bin" "$root/score_4fold.py" > "$root/score_4fold.log" 2>&1
date -Is > "$root/COMPLETE"

#!/usr/bin/env bash
set -uo pipefail

root=/home/dzkhomidov/matching-work/rescue_20260824/macro_balance
mkdir -p "$root/job" "$root/logs" "$root/predictions/baseline" "$root/predictions/category_balanced"
hostname > "$root/job/host.txt"
echo 3 > "$root/job/physical_gpu.txt"
date --iso-8601=seconds > "$root/job/start.txt"
echo $$ > "$root/job/pid.txt"
cd "$root" || exit 2

CUDA_VISIBLE_DEVICES=3 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train_macro_balance.py \
  --variant baseline --data inputs/hand_pairs_pd_v3cal.parquet \
  --init inputs/rubase_llmfull_e2 --tokenizer inputs/rubase_llmfull_e2 \
  --output predictions/baseline --folds fold_01,fold_02 \
  --epochs 2 --batch-size 256 --lr 2e-5 --max-len 224 --seed 20260814 \
  > logs/baseline.log 2> logs/baseline.err
baseline_status=$?
echo "$baseline_status" > job/baseline.exit_code

candidate_status=99
if [[ "$baseline_status" -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES=3 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train_macro_balance.py \
    --variant category_balanced --data inputs/hand_pairs_pd_v3cal.parquet \
    --init inputs/rubase_llmfull_e2 --tokenizer inputs/rubase_llmfull_e2 \
    --output predictions/category_balanced --folds fold_01,fold_02 \
    --epochs 2 --batch-size 256 --lr 2e-5 --max-len 224 --seed 20260814 \
    > logs/category_balanced.log 2> logs/category_balanced.err
  candidate_status=$?
fi
echo "$candidate_status" > job/category_balanced.exit_code
date --iso-8601=seconds > job/end.txt
exit "$candidate_status"

#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/category_experts
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
for V in shared random category; do
  echo "START $V $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=3 "$PY" -u "$ROOT/train.py" --variant "$V" \
    --data "$ROOT/input/hand_pairs_pd_v3cal.parquet" \
    --init "$ROOT/input/rubase_llmfull_e2" --output "$ROOT" \
    --folds fold_01,fold_02 --max-len 224 --bs 256 --epochs 2 --lr 2e-5 --seed 20260814
  echo "DONE $V $(date --iso-8601=seconds)"
done
"$PY" "$ROOT/score.py"

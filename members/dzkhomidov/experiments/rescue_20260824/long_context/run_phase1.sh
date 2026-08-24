#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/long_context
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
MODEL="$ROOT/input/rubase_llmfull_e2"
DATA="$ROOT/input/hand_pairs.parquet"
for LEN in 224 384 448 512; do
  START=$(date +%s)
  echo "START len${LEN} $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=3 "$PY" -u "$ROOT/train_hand_long.py" \
    --exp "len${LEN}" --model "$MODEL" --init "$MODEL" --data "$DATA" \
    --output-root "$ROOT/preds" --max-len "$LEN" --folds fold_01,fold_02 \
    --epochs 2 --effective-bs 256 --micro-bs 128 --eval-bs 128 --lr 2e-5 --seed 20260814
  END=$(date +%s)
  echo "DONE len${LEN} runtime_seconds=$((END-START)) $(date --iso-8601=seconds)"
done
"$PY" "$ROOT/score_partial.py" --data "$DATA" --pred-root "$ROOT/preds" \
  --output "$ROOT/metrics_phase1.json" len224 len384 len448 len512

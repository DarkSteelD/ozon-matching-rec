#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/field_rdrop
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
GPU=${GPU:-4}
test -f "$ROOT/PHASE2_COMPLETE" || { echo "phase2 incomplete" >&2; exit 1; }
LOCK=/dev/shm/codex_field_rdrop_gpu${GPU}.lock
mkdir "$LOCK" 2>/dev/null || { echo "GPU lock exists" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
for _ in 1 2; do
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$apps" ]] || { echo "GPU $GPU has compute apps: $apps" >&2; exit 1; }
  sleep 2
done
echo "$(date --iso-8601=seconds) controls wrapper_pid=$$ gpu=$GPU" | tee "$ROOT/logs/controls_preflight.log"
run_control() {
  local arm=$1 mode=$2
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train_field_rdrop.py" \
    --exp "$arm" --mode "$mode" --drop-rate 0.05 --consistency-weight 0.1 \
    --model "$ROOT/inputs/rubase_llmfull_e2" --init "$ROOT/inputs/rubase_llmfull_e2" \
    --data "$ROOT/inputs/hand_pairs.parquet" --outdir "$ROOT" \
    --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
    --seed 20260814 --folds fold_01,fold_02 \
    >"$ROOT/logs/${arm}.log" 2>"$ROOT/logs/${arm}.err"
}
run_control span05 span
run_control mismatch05 mismatch
"$PY" "$ROOT/code/audit_and_score.py" --data "$ROOT/inputs/hand_pairs.parquet" \
  --pred-root "$ROOT/preds" --variants bce2view,field05,span05,mismatch05 \
  --folds fold_01,fold_02 --output "$ROOT/metrics/controls.json" \
  >"$ROOT/logs/score_controls.log"
"$PY" "$ROOT/code/score_slices.py" --data "$ROOT/inputs/hand_pairs.parquet" \
  --pred-root "$ROOT/preds" --variants bce2view,field05,span05,mismatch05 \
  --folds fold_01,fold_02 --output "$ROOT/metrics/control_slices.json" \
  >"$ROOT/logs/control_slices.log"
date --iso-8601=seconds | tee "$ROOT/CONTROLS_COMPLETE"

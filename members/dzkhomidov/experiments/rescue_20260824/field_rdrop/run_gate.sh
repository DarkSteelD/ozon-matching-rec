#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/field_rdrop
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
GPU=${GPU:-4}
LOCK=/dev/shm/codex_field_rdrop_gpu${GPU}.lock
mkdir "$LOCK" 2>/dev/null || { echo "GPU lock exists: $LOCK" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
for _ in 1 2; do
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$apps" ]] || { echo "GPU $GPU has compute apps: $apps" >&2; exit 1; }
  sleep 2
done
{
  date --iso-8601=seconds
  hostname
  echo "wrapper_pid=$$ physical_gpu=$GPU"
  nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
} | tee "$ROOT/logs/gpu_preflight.log"

run_arm() {
  local arm=$1 mode=$2 rate=$3
  echo "$(date --iso-8601=seconds) arm=$arm wrapper_pid=$$" | tee -a "$ROOT/logs/pids.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train_field_rdrop.py" \
    --exp "$arm" --mode "$mode" --drop-rate "$rate" --consistency-weight 0.1 \
    --model "$ROOT/inputs/rubase_llmfull_e2" --init "$ROOT/inputs/rubase_llmfull_e2" \
    --data "$ROOT/inputs/hand_pairs.parquet" --outdir "$ROOT" \
    --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
    --seed 20260814 --folds fold_01,fold_02 \
    >"$ROOT/logs/${arm}.log" 2>"$ROOT/logs/${arm}.err"
}

score_gate() {
  local variants=$1 suffix=$2
  "$PY" "$ROOT/code/audit_and_score.py" --data "$ROOT/inputs/hand_pairs.parquet" \
    --pred-root "$ROOT/preds" --variants "$variants" --folds fold_01,fold_02 \
    --output "$ROOT/metrics/gate${suffix}.json" >"$ROOT/logs/score${suffix}.log"
  "$PY" "$ROOT/code/gate_decision.py" "$ROOT/metrics/gate${suffix}.json" \
    "$ROOT/metrics/gate_decision${suffix}.json" | tee "$ROOT/logs/gate_decision${suffix}.log"
}

run_arm bce2view baseline 0
run_arm field05 field 0.05
run_arm negative05 negative 0.05
score_gate bce2view,field05,negative05 _phase1

variants=bce2view,field05,negative05
if "$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if next(d for d in x["decisions"] if d["variant"]=="field05")["gate"]=="GO" else 1)' "$ROOT/metrics/gate_decision_phase1.json"; then
  run_arm field10 field 0.10
  variants=$variants,field10
fi
score_gate "$variants" _final
"$PY" "$ROOT/code/score_slices.py" --data "$ROOT/inputs/hand_pairs.parquet" \
  --pred-root "$ROOT/preds" --variants "$variants" --folds fold_01,fold_02 \
  --output "$ROOT/metrics/slices.json" >"$ROOT/logs/slices.log"
date --iso-8601=seconds | tee "$ROOT/COMPLETE" -a "$ROOT/logs/pids.log"

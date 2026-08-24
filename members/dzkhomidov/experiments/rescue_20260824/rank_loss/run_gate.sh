#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/rank_loss
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
GPU=${GPU:-3}
LOCK=/dev/shm/codex_rank_loss_gpu${GPU}.lock
mkdir "$LOCK" 2>/dev/null || { echo "GPU lock exists: $LOCK" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if [[ -n "$apps" ]]; then
  echo "GPU $GPU has compute apps: $apps" >&2
  exit 1
fi
sleep 2
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if [[ -n "$apps" ]]; then
  echo "GPU $GPU acquired by another task during preflight: $apps" >&2
  exit 1
fi

{
  date --iso-8601=seconds
  hostname
  echo "launcher_pid=$$ physical_gpu=$GPU"
  nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
} | tee "$ROOT/logs/gpu_preflight.log"

run_arm() {
  local arm=$1 mode=$2 weight=$3
  echo "$(date --iso-8601=seconds) arm=$arm launcher_pid=$$" | tee -a "$ROOT/logs/pids.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train_rank_fast.py" \
    --exp "$arm" --model "$ROOT/inputs/rubase_llmfull_e2" \
    --init "$ROOT/inputs/rubase_llmfull_e2" \
    --data "$ROOT/inputs/hand_pairs.parquet" --outdir "$ROOT" \
    --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
    --seed 20260814 --folds fold_01,fold_02 \
    --rank-mode "$mode" --rank-lambda "$weight" \
    >"$ROOT/logs/${arm}.log" 2>"$ROOT/logs/${arm}.err"
}

run_arm bce within 0
run_arm rank01 within 0.1
run_arm rank03 within 0.3
run_arm random03 random 0.3

"$PY" "$ROOT/code/audit_and_score.py" \
  --data "$ROOT/inputs/hand_pairs.parquet" --pred-root "$ROOT/preds" \
  --variants bce,rank01,rank03,random03 --folds fold_01,fold_02 \
  --output "$ROOT/metrics/gate.json" >"$ROOT/logs/score_gate.log"
"$PY" "$ROOT/code/gate_decision.py" "$ROOT/metrics/gate.json" \
  "$ROOT/metrics/gate_decision.json" | tee "$ROOT/logs/gate_decision.log"
date --iso-8601=seconds | tee -a "$ROOT/logs/pids.log"

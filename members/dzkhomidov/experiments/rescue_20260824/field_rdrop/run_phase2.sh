#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/field_rdrop
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
GPU=${GPU:-4}
test -f "$ROOT/COMPLETE" || { echo "phase1 incomplete" >&2; exit 1; }
winner=$($PY -c 'import json,sys; x=json.load(open(sys.argv[1])); a=[d for d in x["decisions"] if d["variant"].startswith("field") and d["gate"]=="GO"]; print(max(a,key=lambda d:d["pooled_macro_category_ap_delta"])["variant"])' "$ROOT/metrics/gate_decision_final.json")
rate=0.05; [[ "$winner" == field10 ]] && rate=0.10
LOCK=/dev/shm/codex_field_rdrop_gpu${GPU}.lock
mkdir "$LOCK" 2>/dev/null || { echo "GPU lock exists" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
for _ in 1 2; do
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  [[ -z "$apps" ]] || { echo "GPU $GPU has compute apps: $apps" >&2; exit 1; }
  sleep 2
done
echo "$(date --iso-8601=seconds) phase2 winner=$winner wrapper_pid=$$ gpu=$GPU" | tee "$ROOT/logs/phase2_preflight.log"

run_f34() {
  local arm=$1 mode=$2 dose=$3
  cp "$ROOT/metrics/${arm}_training.json" "$ROOT/metrics/${arm}_training_f12.json"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train_field_rdrop.py" \
    --exp "$arm" --mode "$mode" --drop-rate "$dose" --consistency-weight 0.1 \
    --model "$ROOT/inputs/rubase_llmfull_e2" --init "$ROOT/inputs/rubase_llmfull_e2" \
    --data "$ROOT/inputs/hand_pairs.parquet" --outdir "$ROOT" \
    --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
    --seed 20260814 --folds fold_03,fold_04 \
    >"$ROOT/logs/${arm}_f34.log" 2>"$ROOT/logs/${arm}_f34.err"
  mv "$ROOT/metrics/${arm}_training.json" "$ROOT/metrics/${arm}_training_f34.json"
}

run_f34 bce2view baseline 0
run_f34 "$winner" field "$rate"
"$PY" "$ROOT/code/audit_and_score.py" --data "$ROOT/inputs/hand_pairs.parquet" \
  --pred-root "$ROOT/preds" --variants "bce2view,$winner" \
  --folds fold_01,fold_02,fold_03,fold_04 --output "$ROOT/metrics/full4.json" \
  >"$ROOT/logs/score_full4.log"
"$PY" "$ROOT/code/score_slices.py" --data "$ROOT/inputs/hand_pairs.parquet" \
  --pred-root "$ROOT/preds" --variants "bce2view,$winner" \
  --folds fold_01,fold_02,fold_03,fold_04 --output "$ROOT/metrics/slices_full4.json" \
  >"$ROOT/logs/slices_full4.log"
echo "$(date --iso-8601=seconds) winner=$winner" | tee "$ROOT/PHASE2_COMPLETE"

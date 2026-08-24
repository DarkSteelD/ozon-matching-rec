#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/field_rdrop
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
GPU=2
LOCK=/dev/shm/codex_field_rdrop_gpu2.lock

test "$(hostname)" = avi-ix-devbox03
mkdir "$LOCK" 2>/dev/null || { echo "experiment lock exists: $LOCK" >&2; exit 1; }
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
for _ in 1 2; do
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  test -z "$apps" || { echo "GPU $GPU has compute apps: $apps" >&2; exit 1; }
  sleep 2
done

sha256sum -c "$ROOT/migration/input_sha256.txt"
echo "$(date --iso-8601=seconds) host=$(hostname) physical_gpu=$GPU wrapper_pid=$$" \
  | tee "$ROOT/logs/migration_preflight.log"

run_arm() {
  local arm=$1 mode=$2 folds=$3 suffix=$4
  for fold in ${folds//,/ }; do
    test ! -e "$ROOT/preds/$arm/$fold.csv" || {
      echo "refusing to rerun finished artifact: $arm/$fold" >&2
      exit 1
    }
  done
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train_field_rdrop.py" \
    --exp "$arm" --mode "$mode" --drop-rate 0.05 --consistency-weight 0.1 \
    --model "$ROOT/inputs/rubase_llmfull_e2" --init "$ROOT/inputs/rubase_llmfull_e2" \
    --data "$ROOT/inputs/hand_pairs.parquet" --outdir "$ROOT" \
    --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
    --seed 20260814 --folds "$folds" \
    >"$ROOT/logs/${arm}_${suffix}.log" 2>"$ROOT/logs/${arm}_${suffix}.err"
  mv "$ROOT/metrics/${arm}_training.json" "$ROOT/metrics/${arm}_training_${suffix}.json"
}

# Mechanism controls were not run on fsk35.  Keep them separate: equal-rate
# unstructured span corruption, then field corruption with mismatched R-Drop.
run_arm span05 span fold_01,fold_02 f12
run_arm mismatch05 mismatch fold_01,fold_02 f12

"$PY" "$ROOT/code/audit_and_score.py" \
  --data "$ROOT/inputs/hand_pairs.parquet" --pred-root "$ROOT/preds" \
  --variants bce2view,field05,span05,mismatch05 --folds fold_01,fold_02 \
  --output "$ROOT/metrics/controls_f12.json" >"$ROOT/logs/score_controls_f12.log"
"$PY" "$ROOT/code/score_slices.py" \
  --data "$ROOT/inputs/hand_pairs.parquet" --pred-root "$ROOT/preds" \
  --variants bce2view,field05,span05,mismatch05 --folds fold_01,fold_02 \
  --output "$ROOT/metrics/control_slices_f12.json" >"$ROOT/logs/control_slices_f12.log"
date --iso-8601=seconds > "$ROOT/CONTROLS_COMPLETE"

# field05 already passed the predeclared two-fold gate; complete only its and
# its matched baseline's missing folds.  Existing fold 01/02 files are guarded.
cp "$ROOT/metrics/bce2view_training.json" "$ROOT/metrics/bce2view_training_f12.json"
cp "$ROOT/metrics/field05_training.json" "$ROOT/metrics/field05_training_f12.json"
run_arm bce2view baseline fold_03,fold_04 f34
run_arm field05 field fold_03,fold_04 f34

"$PY" "$ROOT/code/audit_and_score.py" \
  --data "$ROOT/inputs/hand_pairs.parquet" --pred-root "$ROOT/preds" \
  --variants bce2view,field05 --folds fold_01,fold_02,fold_03,fold_04 \
  --output "$ROOT/metrics/full4.json" >"$ROOT/logs/score_full4.log"
"$PY" "$ROOT/code/score_slices.py" \
  --data "$ROOT/inputs/hand_pairs.parquet" --pred-root "$ROOT/preds" \
  --variants bce2view,field05 --folds fold_01,fold_02,fold_03,fold_04 \
  --output "$ROOT/metrics/slices_full4.json" >"$ROOT/logs/slices_full4.log"
date --iso-8601=seconds > "$ROOT/PHASE2_COMPLETE"

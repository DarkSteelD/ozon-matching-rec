#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dzkhomidov/matching-work/rescue_20260824/minilm_early_diversity
ARCH=/home/dzkhomidov/matching-work/rescue_20260824/architecture_scout
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
MODEL="$ARCH/models/mmarco-mMiniLMv2-L12-H384-v1"
DATA=/home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet
HARD=/home/dzkhomidov/matching-work/data/hand_pairs.parquet
BASE="$ARCH/preds/full_baseline"
LOCK=/tmp/dzkhomidov_gpu0_minilm_early_diversity.lock

if [[ -f "$ROOT/selection_fold01.json" ]]; then
  echo "fold_01 is already complete; refusing to restart or auto-launch fold_02" >&2
  exit 75
fi

test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)"
set -o noclobber
printf 'owner=dzkhomidov task=minilm_early_diversity pid=%s started=%s\n' "$$" "$(date -Is)" > "$LOCK"
set +o noclobber
trap 'rm -f "$LOCK"' EXIT

mkdir -p "$ROOT/logs" "$ROOT/preds" "$ROOT/preds_head_only"
printf 'host=%s physical_gpu=0 wrapper_pid=%s started=%s\n' "$(hostname)" "$$" "$(date -Is)" > "$ROOT/run_start.txt"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/train_trajectory.py" \
  --data "$DATA" --model "$MODEL" --output "$ROOT/preds" --fold fold_01 \
  --checkpoints 250,500,1000,full --stop-step full \
  2>&1 | tee "$ROOT/logs/fold01_trajectory.log"

"$PY" "$ROOT/score_select.py" --hard "$HARD" --baseline "$BASE" \
  --predictions "$ROOT/preds" --fold fold_01 --labels 250,500,1000,full \
  --output "$ROOT/selection_fold01.json" --select \
  2>&1 | tee "$ROOT/logs/fold01_score.log"

SELECTED=$("$PY" -c 'import json; print(json.load(open("/home/dzkhomidov/matching-work/rescue_20260824/minilm_early_diversity/selection_fold01.json"))["selection"]["label"])')
printf '%s\n' "$SELECTED" > "$ROOT/selected_label.txt"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/train_trajectory.py" \
  --data "$DATA" --model "$MODEL" --output "$ROOT/preds" --fold fold_02 \
  --checkpoints "$SELECTED" --stop-step "$SELECTED" \
  2>&1 | tee "$ROOT/logs/fold02_selected.log"

"$PY" "$ROOT/score_select.py" --hard "$HARD" --baseline "$BASE" \
  --predictions "$ROOT/preds" --fold fold_02 --labels "$SELECTED" \
  --output "$ROOT/confirmation_fold02.json" \
  2>&1 | tee "$ROOT/logs/fold02_score.log"

CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/train_trajectory.py" \
  --data "$DATA" --model "$MODEL" --output "$ROOT/preds_head_only" --fold fold_01 \
  --checkpoints 500 --stop-step 500 --head-only \
  2>&1 | tee "$ROOT/logs/fold01_head_only.log"

"$PY" "$ROOT/score_select.py" --hard "$HARD" --baseline "$BASE" \
  --predictions "$ROOT/preds_head_only" --fold fold_01 --labels 500 \
  --output "$ROOT/head_only_fold01.json" \
  2>&1 | tee "$ROOT/logs/fold01_head_only_score.log"

printf 'ended=%s selected=%s\n' "$(date -Is)" "$SELECTED" > "$ROOT/run_end.txt"

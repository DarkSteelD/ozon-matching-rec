#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/architecture_scout
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
cd "$ROOT";date -Is > full_start.txt;hostname >> full_start.txt;echo "physical_gpu=1 wrapper_pid=$$" >> full_start.txt
CUDA_VISIBLE_DEVICES=1 "$PY" -u run_hand_local.py \
 --exp full_minilm --model "$ROOT/models/mmarco-mMiniLMv2-L12-H384-v1" \
 --init "$ROOT/models/mmarco-mMiniLMv2-L12-H384-v1" \
 --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
 --max-len 224 --cat --attrs --sym --folds fold_01,fold_02 \
 --epochs 2 --bs 256 --lr 2e-5 --seed 20260814 &
TRAIN_PID=$!;echo "train_pid=$TRAIN_PID" >> full_start.txt
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  date -Is
  nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
  sleep 5
done > logs/gpu_full.csv &
MON_PID=$!;wait "$TRAIN_PID";wait "$MON_PID"
"$PY" score_full.py --hard /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
 --baseline "$ROOT/preds/full_baseline" --candidate "$ROOT/preds/full_minilm" \
 --output "$ROOT/metrics_full.json" | tee logs/full_score.log
date -Is > full_end.txt

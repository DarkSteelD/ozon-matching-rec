#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/dzkhomidov/matching-work/rescue_20260824/architecture_scout
PY=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
cd "$ROOT"; date -Is > start.txt; hostname >> start.txt; echo "physical_gpu=1 wrapper_pid=$$" >> start.txt
CUDA_VISIBLE_DEVICES=1 "$PY" -u train_screen.py \
 --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet --output "$ROOT" \
 --rubase /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
 --minilm "$ROOT/models/mmarco-mMiniLMv2-L12-H384-v1" \
 --steps 500 --bs 128 --max-len 160 --seed 20260814
"$PY" score_screen.py --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
 --pred-root "$ROOT/preds" --output "$ROOT/metrics_screen.json" | tee logs/score.log
date -Is > end.txt

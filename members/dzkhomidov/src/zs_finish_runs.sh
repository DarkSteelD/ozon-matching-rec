#!/usr/bin/env bash
# Orchestrate the tail of the zero-shot full runs on GPUs 2 and 3 only.
set -uo pipefail
W=~/matching-work
Z=$W/zs
PY=~/ozon-hack/.venv-ml/bin/python
cd $W/scripts

# wait for gemma (gpu2) and qwen first half (gpu3)
while [ ! -f $Z/full_gemma4_e4b_a800.csv ] || [ ! -f $Z/fq_p1.csv ]; do sleep 60; done
echo "stage1 done: gemma + qwen p1"

# qwen second half (rows 182827..365653) split across gpu2/gpu3: 91413 + 91414
CUDA_VISIBLE_DEVICES=2 $PY zs_llm_hf.py \
  --model ~/ozon-hack/shared_models/Qwen/Qwen3.5-4B \
  --data $W/data/hand_pairs.parquet --out $Z/fq_p2a.csv \
  --attrs-limit 800 --batch-tokens 48000 --offset 182827 --limit 91413 \
  > $Z/fq_p2a.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY zs_llm_hf.py \
  --model ~/ozon-hack/shared_models/Qwen/Qwen3.5-4B \
  --data $W/data/hand_pairs.parquet --out $Z/fq_p2b.csv \
  --attrs-limit 800 --batch-tokens 48000 --offset 274240 --limit 91414 \
  > $Z/fq_p2b.log 2>&1 &
wait
echo "stage2 done: qwen p2a + p2b"
ls -la $Z/fq_p1.csv $Z/fq_p2a.csv $Z/fq_p2b.csv $Z/full_gemma4_e4b_a800.csv

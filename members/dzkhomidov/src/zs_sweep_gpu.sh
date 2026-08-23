#!/usr/bin/env bash
# Run zero-shot sweep models sequentially on one GPU.
# Usage: zs_sweep_gpu.sh <gpu_id> <gpu_mem_frac> <name:path> [<name:path> ...]
set -uo pipefail
GPU="$1"; MEM="$2"; shift 2
WORK=~/matching-work
DATA="$WORK/data/zs_sample_3000.parquet"
OUT="$WORK/zs"
mkdir -p "$OUT"
for spec in "$@"; do
  name="${spec%%:*}"; path="${spec#*:}"
  echo "=== [gpu$GPU] $name ($path)"
  (cd "$WORK/scripts" && CUDA_VISIBLE_DEVICES=$GPU \
    ~/ozon-hack/.venv-ml/bin/python zs_llm_hf.py \
    --model "$path" --data "$DATA" --out "$OUT/${name}.csv" \
    > "$OUT/${name}.log" 2>&1)
  rc=$?
  echo "=== [gpu$GPU] $name rc=$rc"
  tail -3 "$OUT/${name}.log"
done

#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
gpu=3
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
  awk -F, -v u="$uuid" '{gsub(/ /,"",$1)} $1 == u {found=1} END {exit !found}'; then
  echo "physical GPU $gpu has a live compute app" >&2
  exit 2
fi

nvidia-smi -i "$gpu" \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader > "$here/preflight_stage2.log"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader >> "$here/preflight_stage2.log"

export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20260814
py=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
data=/home/dzkhomidov/matching-work/data/hand_pairs.parquet
oof=/home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224
init=/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2

"$py" "$here/train_controlled.py" \
  --data "$data" --oof-root "$oof" --init "$init" --output "$here" \
  --folds fold_03,fold_04 --variants baseline,hard050 \
  2>&1 | tee "$here/train_stage2.log"
mv "$here/run_manifest.json" "$here/run_manifest_stage2.json"

"$py" "$here/score.py" \
  --data "$data" --root "$here/preds" --archived-baseline "$oof" \
  --folds fold_01,fold_02,fold_03,fold_04 \
  --variants baseline_rerun,hard050 \
  --output "$here/metrics_all.json" 2>&1 | tee "$here/score_all.log"

sha256sum "$here"/preds/{baseline,hard050}/fold_*.csv | sort > "$here/predictions.sha256"

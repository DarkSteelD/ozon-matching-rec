#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); gpu=5
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v u="$uuid" '$1 == u {found=1} END {exit !found}'; then
  echo "physical GPU $gpu has a live compute app" >&2; exit 2
fi
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$here/preflight.log"
nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$here/preflight.log" || true
export CUDA_VISIBLE_DEVICES="$gpu"
py=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
data=/home/dzkhomidov/matching-work/data/hand_pairs.parquet
oof=/home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224
init=/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2
"$py" "$here/train.py" --data "$data" --oof-root "$oof" --init "$init" --output "$here" 2>&1 | tee "$here/train.log"
"$py" "$here/score.py" --data "$data" --root "$here/preds" --archived-baseline "$oof" \
  --output "$here/metrics_stage1.json" 2>&1 | tee "$here/score.log"

#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
gpu=5
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v u="$uuid" '$1 == u {found=1} END {exit !found}'; then
  echo "GPU $gpu has a live compute app; refusing to launch" >&2
  exit 2
fi
nvidia-smi -i "$gpu" --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$here/preflight.log"
nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$here/preflight.log" || true
export CUDA_VISIBLE_DEVICES="$gpu"
py=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
data=/home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet
truth=/home/dzkhomidov/matching-work/data/hand_pairs.parquet
init=/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2
for variant in baseline structured shuffled; do
  "$py" "$here/train.py" --variant "$variant" --data "$data" --init "$init" \
    --output "$here/preds/$variant" --folds fold_01,fold_02 2>&1 | tee "$here/${variant}.log"
done
"$py" "$here/score.py" --data "$data" --truth "$truth" --root "$here/preds" --output "$here/metrics_stage1.json" 2>&1 | tee "$here/score.log"
if "$py" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["gate_pass"])' "$here/metrics_stage1.json"; then
  for variant in baseline structured shuffled; do
    "$py" "$here/train.py" --variant "$variant" --data "$data" --init "$init" \
      --output "$here/preds/$variant" --folds fold_03,fold_04 2>&1 | tee "$here/${variant}_stage2.log"
  done
  "$py" "$here/score.py" --data "$data" --truth "$truth" --root "$here/preds" --folds fold_01,fold_02,fold_03,fold_04 \
    --output "$here/metrics_all.json" 2>&1 | tee "$here/score_all.log"
fi

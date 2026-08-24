#!/bin/bash
set -euo pipefail

HERE=/home/dzkhomidov/matching-work/rescue_20260824/rularge
HOST=avi-gn-fsk35
GPU=4
LOCK=/home/dzkhomidov/ozon-hack/scratch-q2/gpu_registry/${HOST}_gpu${GPU}
mkdir -p "$(dirname "$LOCK")"

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')
test "$used" -lt 100 && test -z "$apps" || { echo "GPU occupied before lock" >&2; exit 75; }
mkdir "$LOCK" 2>/dev/null || { echo "GPU lock exists: $LOCK" >&2; exit 73; }
cleanup() { rm -f "$LOCK/OWNER"; rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')
test "$used" -lt 100 && test -z "$apps" || { echo "GPU occupied after lock" >&2; exit 75; }
created=$(date --iso-8601=seconds)
{
  echo "task=matching rularge hand-only folds 01-02 scale screen"
  echo "owner=dzkhomidov"
  echo "host=$HOST"
  echo "gpu=$GPU"
  echo "pid=$$"
  echo "created_msk=$created"
} > "$LOCK/OWNER"
printf '{"event":"lock_acquired","time":"%s","host":"%s","gpu":%s,"pid":%s}\n' "$created" "$HOST" "$GPU" "$$" >> "$HERE/resource_registry.jsonl"
nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$HERE/preflight_${HOST}_gpu${GPU}.log"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$HERE/preflight_${HOST}_gpu${GPU}.log" || true

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python "$HERE/train_hand_screen.py" \
  --model "$HERE/model" \
  --data "$HERE/data/hand_pairs.parquet" \
  --output "$HERE/preds/rularge_hand" \
  --folds fold_01,fold_02 \
  --epochs 2 --micro-batch 64 --effective-batch 256 --lr 1e-5 --max-len 128 --seed 20260814
finished=$(date --iso-8601=seconds)
printf '{"event":"run_finished","time":"%s","host":"%s","gpu":%s,"pid":%s}\n' "$finished" "$HOST" "$GPU" "$$" >> "$HERE/resource_registry.jsonl"

#!/usr/bin/env bash
set -euo pipefail
root=/home/dzkhomidov/matching-work/rescue_20260824/container_candidate_e3_len384
pkg=$root/package
smoke=$root/primary_full_run
lock=/home/dzkhomidov/matching-work/locks/gpu1_primary_e3_len384_full.lock
test -f "$lock/OWNER"
mkdir -p "$smoke"
release_lock() {
  stamp=$(date +%Y%m%dT%H%M%S)
  if test -d "$lock"; then mv "$lock" "${lock}.released_${stamp}"; fi
}
trap release_lock EXIT
export CUDA_VISIBLE_DEVICES=1
printf 'timestamp,memory_used_mib,utilization_gpu_pct\n' > "$smoke/board_samples.csv"
(
  cd "$pkg"
  /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u run.py \
    --items_path /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/items.parquet \
    --matches_path /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/matches.parquet \
    --output_path "$smoke/predictions.csv"
) > "$smoke/run.log" 2>&1 &
bench_pid=$!
echo "$bench_pid" > "$smoke/run.pid"
while kill -0 "$bench_pid" 2>/dev/null; do
  values=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 1)
  printf '%s,%s\n' "$(date -Is)" "$values" >> "$smoke/board_samples.csv"
  sleep 1
done
wait "$bench_pid"

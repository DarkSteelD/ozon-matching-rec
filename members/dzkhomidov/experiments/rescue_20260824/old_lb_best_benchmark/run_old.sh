#!/usr/bin/env bash
set -euo pipefail
root=/home/dzkhomidov/matching-work/rescue_20260824/old_lb_best_benchmark
pkg=$root/pkg
lock=/home/dzkhomidov/matching-work/locks/gpu1_old_lb_best_benchmark.lock
test -f "$lock/OWNER"
release_lock() {
  stamp=$(date +%Y%m%dT%H%M%S)
  if test -d "$lock"; then mv "$lock" "${lock}.released_${stamp}"; fi
}
trap release_lock EXIT
export CUDA_VISIBLE_DEVICES=1
printf 'timestamp,memory_used_mib,utilization_gpu_pct\n' > "$root/board_samples.csv"
(
  cd "$pkg"
  /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u run.py \
    --items_path /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/items.parquet \
    --matches_path /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/matches.parquet \
    --output_path "$root/predictions.csv"
) > "$root/run.log" 2>&1 &
bench_pid=$!
echo "$bench_pid" > "$root/benchmark.pid"
while kill -0 "$bench_pid" 2>/dev/null; do
  values=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 1)
  printf '%s,%s\n' "$(date -Is)" "$values" >> "$root/board_samples.csv"
  sleep 1
done
wait "$bench_pid"

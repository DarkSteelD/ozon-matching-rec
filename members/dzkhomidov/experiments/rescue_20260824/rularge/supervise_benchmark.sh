#!/bin/bash
set -euo pipefail

HERE=/home/dzkhomidov/matching-work/rescue_20260824/rularge
echo "$$" > "$HERE/benchmark_supervisor.pid"
for pid_file in hand_screen.pid token_cache.pid; do
  pid=$(cat "$HERE/$pid_file")
  while ps -p "$pid" -o pid= >/dev/null; do sleep 30; done
done
grep -q 'fold_02: written' "$HERE/hand_screen.log"
grep -q '^cache complete$' "$HERE/token_cache.log"
bash "$HERE/run_benchmark_locked.sh" > "$HERE/benchmark.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
pid=$1
artifact=$2
expected=$3
ledger=$4
while kill -0 "$pid" 2>/dev/null; do
  if test -f "$artifact"; then
    cmd=$(ps -p "$pid" -o cmd= || true)
    if [[ "$cmd" != *"$expected"* ]]; then
      printf 'refused: pid %s cmd mismatch: %s\n' "$pid" "$cmd" > "$ledger"
      exit 3
    fi
    kill -TERM "$pid"
    printf 'stopped_pid=%s\nartifact=%s\ntime=%s\ncmd=%s\n' \
      "$pid" "$artifact" "$(date -Is)" "$cmd" > "$ledger"
    exit 0
  fi
  sleep 0.2
done
printf 'pid_exited_before_artifact=%s\ntime=%s\n' "$pid" "$(date -Is)" > "$ledger"
exit 4

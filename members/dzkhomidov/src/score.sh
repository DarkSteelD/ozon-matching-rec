#!/usr/bin/env bash
# Score an experiment's fold predictions WITHOUT touching validation/ (which auto-publishes).
# Usage: score.sh <experiment> [notes]
set -euo pipefail
EXP="$1"; NOTES="${2:-}"
REPO=~/ozon-hack/repos/ozon-matching-rec
WORK=~/matching-work
cd "$REPO"
.venv/bin/python -m validation.evaluate \
  --member local --experiment "$EXP" \
  --predictions-dir "$WORK/preds/$EXP" \
  --results-dir "$WORK/results" \
  --leaderboard "$WORK/leaderboard.csv" \
  --notes "$NOTES"

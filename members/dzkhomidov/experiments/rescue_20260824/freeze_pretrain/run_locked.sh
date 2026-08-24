#!/usr/bin/env bash
set -euo pipefail

root=/home/dzkhomidov/matching-work/rescue_20260824/freeze_pretrain
python_bin=/home/dzkhomidov/ozon-hack/.venv-ml/bin/python
lock=/home/dzkhomidov/ozon-hack/scratch-q2/gpu_registry/avi-ix-devbox02_gpu3
gpu=3

test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
mkdir "$lock"
cleanup() { rm -f "$lock/OWNER"; rmdir "$lock" 2>/dev/null || true; }
trap cleanup EXIT
printf 'task=matching freeze_pretrain\nhost=avi-ix-devbox02\ngpu=3\npid=%s\nstarted=%s\n' \
  "$$" "$(date --iso-8601=seconds)" > "$lock/OWNER"

export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=true

run_variant() {
  variant=$1
  freeze=$2
  folds=$3
  "$python_bin" -u "$root/train.py" \
    --variant "$variant" --freeze "$freeze" \
    --data "$root/input/hand_pairs_pd_v3cal.parquet" \
    --init "$root/input/rubase_llmfull_e3" --output "$root" \
    --max-len 224 --folds "$folds" --epochs 2 \
    --effective-bs 256 --micro-bs 128 --eval-bs 256 \
    --lr 2e-5 --seed 20260814
}

run_variant bottom6 bottom6 fold_01,fold_02
run_variant top6 top6 fold_01,fold_02
"$python_bin" "$root/score.py"

if "$python_bin" -c 'import json; raise SystemExit(not json.load(open("'$root'/metrics.json"))["gate_pass"])'; then
  run_variant full none fold_03,fold_04
  run_variant bottom6 bottom6 fold_03,fold_04
  run_variant top6 top6 fold_03,fold_04
  "$python_bin" "$root/score.py"
fi
date --iso-8601=seconds > "$root/COMPLETE"

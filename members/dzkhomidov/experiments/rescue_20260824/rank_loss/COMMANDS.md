# Reproduction commands

Environment: `/home/dzkhomidov/ozon-hack/.venv-ml/bin/python`.

Each training command uses:

```text
CUDA_VISIBLE_DEVICES=3 python code/train_rank_fast.py \
  --exp ARM --model inputs/rubase_llmfull_e2 \
  --init inputs/rubase_llmfull_e2 \
  --data inputs/hand_pairs.parquet \
  --outdir /home/dzkhomidov/matching-work/rescue_20260824/rank_loss \
  --attrs --cat --max-len 224 --bs 256 --epochs 2 --lr 2e-5 \
  --seed 20260814 --folds fold_01,fold_02 \
  --rank-mode MODE --rank-lambda LAMBDA
```

Arms: `bce/within/0`, `rank01/within/0.1`, `rank03/within/0.3`,
`random03/random/0.3`.

The assigned queue moved to `avi-ix-devbox02` physical GPU 3, after
`macro_balance` and `hard_negative`. `run_gate.sh` uses an atomic task lock and
two live compute-app checks. If either is non-empty, it exits without training.

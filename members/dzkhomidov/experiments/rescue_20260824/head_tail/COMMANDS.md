# Executed commands and jobs

Host: `avi-gn-fsk35`; physical GPU: `0`.

## Token preparation

PID `1869943`, completed successfully in 63.2 s.

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u prepare_tokens.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --model /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/head_tail/tokens
```

## Stage 1 training

Shared arguments for every mode:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train.py \
  --mode MODE \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --tokens /home/dzkhomidov/matching-work/rescue_20260824/head_tail/tokens \
  --model /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/head_tail/preds \
  --folds fold_01,fold_02 --epochs 2 --bs 256 --lr 2e-5 --seed 20260814
```

| mode | PID | log | status |
|---|---:|---|---|
| prefix | 1876055 | `prefix_folds12.log` | completed |
| headtail | 1928381 | `headtail_folds12.log` | completed |
| middle | 1981829 | `middle_folds12.log` | completed |

Stage 2 was not launched because the preregistered stage-1 gate failed.

## Scoring

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --coverage tokens/coverage.parquet --preds preds --output metrics \
  --folds fold_01,fold_02
```

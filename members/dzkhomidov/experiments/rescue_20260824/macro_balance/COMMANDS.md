# Commands and resources

## Live accelerator check

Checked on `avi-ix-devbox03` before launch on 2026-08-24 (Europe/Moscow).
Requested GPU 1 was occupied by PID 1947295 (`dzkhomidov`, quality-track Qwen
scoring). All other GPUs became occupied before launch; no training process was
started there. The coordinator redirected this short screening to
`avi-ix-devbox02`. A live check there showed GPU 3 at 1 MiB, 0% utilization,
with no compute application. The experiment therefore uses physical GPU 3 on
`avi-ix-devbox02` (`CUDA_VISIBLE_DEVICES=3`).

## Training commands

Run from the isolated experiment directory on `avi-ix-devbox02`:

```bash
CUDA_VISIBLE_DEVICES=3 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train_macro_balance.py \
  --variant baseline --data inputs/hand_pairs_pd_v3cal.parquet \
  --init inputs/rubase_llmfull_e2 --tokenizer inputs/rubase_llmfull_e2 \
  --output predictions/baseline --folds fold_01,fold_02 \
  --epochs 2 --batch-size 256 --lr 2e-5 --max-len 224 --seed 20260814

CUDA_VISIBLE_DEVICES=3 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train_macro_balance.py \
  --variant category_balanced --data inputs/hand_pairs_pd_v3cal.parquet \
  --init inputs/rubase_llmfull_e2 --tokenizer inputs/rubase_llmfull_e2 \
  --output predictions/category_balanced --folds fold_01,fold_02 \
  --epochs 2 --batch-size 256 --lr 2e-5 --max-len 224 --seed 20260814
```

The wrapper runs the two commands sequentially and writes separate stdout and
stderr logs. PID, start/end times, exit codes, host, and GPU are persisted in
`job/`.

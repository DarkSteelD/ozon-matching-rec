# Commands

```bash
CUDA_VISIBLE_DEVICES=6 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  train_checkpoint_average.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/checkpoint_average \
  --folds fold_01,fold_02 --max-len 224 --epochs 2 --batch-size 256 \
  --lr 2e-5 --seed 20260814 --ema-decay 0.995
```

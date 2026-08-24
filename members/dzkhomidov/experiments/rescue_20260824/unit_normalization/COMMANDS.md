# Exact commands

Self-check:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python normalize_units.py
```

Stage 1 on `avi-gn-fsk35`, physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/unit_normalization \
  --folds fold_01,fold_02 --variants baseline,normalized,corrupt \
  --max-len 224 --epochs 2 --bs 192 --lr 2e-5 --seed 20260814
```

Scoring:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --masks slice_masks.parquet --pred-root preds --folds fold_01,fold_02 \
  --output metrics_stage1.json
```

# Exact commands

Feature preparation (run on `avi-ling-gpu03`):

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u prepare_features.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --preds /home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/hard_negative/hand_features.parquet
```

Stage 1 training (run on `avi-gn-fsk35`, physical GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train_weighted.py \
  --data /home/dzkhomidov/matching-work/rescue_20260824/hard_negative/hand_features.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/hard_negative \
  --folds fold_01,fold_02 --variants hard2,hard4,random2,random4 \
  --coverage 0.10 --max-len 224 --epochs 2 --bs 192 --lr 2e-5 --seed 20260814
```

Scoring:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py \
  --data hand_features.parquet \
  --baseline /home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224 \
  --pred-root preds --folds fold_01,fold_02 --output metrics_stage1.json
```

# Reproduction commands

Host: `avi-gn-fsk35`. Physical GPUs were selected only after a live
`nvidia-smi --query-compute-apps` check.

```bash
CUDA_VISIBLE_DEVICES=5 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python train.py \
  --variant baseline --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output preds/baseline --folds fold_01,fold_02

CUDA_VISIBLE_DEVICES=5 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python train.py \
  --variant structured --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output preds/structured --folds fold_01,fold_02

CUDA_VISIBLE_DEVICES=7 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python train.py \
  --variant shuffled --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output preds/shuffled --folds fold_01,fold_02

/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --truth /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --root preds --output metrics_stage1.json
```

All omitted training arguments use the recorded defaults: seed 20260814, two
epochs, max length 224, batch size 192, learning rate 2e-5, symmetric pair-order
augmentation during training and two-direction averaging during evaluation.

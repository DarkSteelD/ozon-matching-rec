# Commands

Historical completed run: host `avi-gn-fsk35`, physical GPU0, PID `2057398`.
The host was released by user order on 2026-08-24 and must not be used for new
processes. This command is retained only for exact provenance:

```bash
CUDA_VISIBLE_DEVICES=0 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train.py \
 --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
 --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
 --output /home/dzkhomidov/matching-work/rescue_20260824/poly1_loss \
 --variants bce,poly05,polyneg05 --folds fold_01,fold_02 \
 --max-len 224 --epochs 2 --bs 256 --lr 2e-5 --seed 20260814
```

Stage-1 status: completed and recovered read-only. Log: `train_folds12.log`.
The preregistered gate failed, so no promotion command exists and no further
training is authorized for this hypothesis.

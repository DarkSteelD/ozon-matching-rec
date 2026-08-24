# Commands

Commands run on `avi-gn-fsk35` with `CUDA_VISIBLE_DEVICES=2` after the parent
reassigned the experiment from occupied physical GPU 3. The GPU-2 UUID must
have no row in `nvidia-smi --query-compute-apps` immediately before launch.

## Epoch 3

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/scripts/train_ce_fast.py \
  --exp rubase_llmfull_e3 --model DeepPavlov/rubert-base-cased \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --llm-file /home/dzkhomidov/matching-work/data/llm_pairs_full.parquet \
  --cache /home/dzkhomidov/matching-work/data/tok_rubase_len128 \
  --epochs 1 --bs 512 --lr 3e-5 --max-len 128 --attrs --cat \
  --save /home/dzkhomidov/matching-work/rescue_20260824/third_pretrain/ckpt/rubase_llmfull_e3
```

## Identical hand FT gate

Run once with epoch-2 init and once with epoch-3 init, changing only `--exp`
and `--init`:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/rescue_20260824/third_pretrain/run_hand_local.py \
  --exp hand_e2_gate --model DeepPavlov/rubert-base-cased \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --epochs 2 --bs 256 --lr 2e-5 --max-len 160 --attrs --cat \
  --seed 20260814 --folds fold_01,fold_02
```

The candidate command is identical except `--exp hand_e3_gate` and epoch-3
`--init`. Score with `score_folds.py`. Continue folds 03-04 only if each gate
fold delta is positive and greater than 0.001.

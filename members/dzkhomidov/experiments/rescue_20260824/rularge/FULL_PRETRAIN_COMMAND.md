# Deferred full ruRoBERTa-large pretrain command

This is deliberately not launched. `avi-ix-devbox01` GPU 0/1 are reserved by
services per the root coordinator, even though GPU 1 exposed no compute-app PID
at the 01:26 MSK snapshot. Empty `nvidia-smi --query-compute-apps` is therefore
not sufficient ownership evidence.

After a fresh ownership clearance, stage the immutable inputs and the minimal
model files in this task directory on the chosen host. Then:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/rescue_20260824/rularge/tokenize_llm.py \
  --model /home/dzkhomidov/matching-work/rescue_20260824/rularge/model \
  --file /home/dzkhomidov/matching-work/rescue_20260824/rularge/data/llm_pairs_full.parquet \
  --cache /home/dzkhomidov/matching-work/rescue_20260824/rularge/cache/tok_rularge_len128 \
  --max-len 128 --attrs --cat --workers 12

CUDA_VISIBLE_DEVICES=GPU_ID PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/rescue_20260824/rularge/train_pretrain_resumable.py \
  --model /home/dzkhomidov/matching-work/rescue_20260824/rularge/model \
  --cache /home/dzkhomidov/matching-work/rescue_20260824/rularge/cache/tok_rularge_len128 \
  --save /home/dzkhomidov/matching-work/rescue_20260824/rularge/pretrain \
  --epochs 1 --micro-batch 64 --effective-batch 256 --lr 1e-5 \
  --seed 20260814 --save-every 2000 --max-updates 200
```

Tokenization is resumable by parquet-row-group done markers. Training resumes
exactly from the atomically replaced `pretrain/training_state.pt`, including
model, optimizer, scheduler, epoch and update. Before the real run, perform a
200-update benchmark with the same micro/effective batch. Proceed only if the
measured one-epoch ETA is below 12 hours; otherwise defer rather than silently
changing the registered batch or schedule.

If and only if the printed projection is below 12 hours and the hand-only
positive control passes, rerun the identical training command without
`--max-updates`. It resumes exactly at update 200; it does not restart or alter
the OneCycle schedule.

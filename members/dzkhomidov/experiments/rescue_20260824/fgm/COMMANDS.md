# Commands and provenance

- repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- git SHA: `5099db5df398e6aa4fec9eccdaf6959f50cfbf29`
- worktree: clean
- host: `avi-ix-devbox02`
- physical accelerator: GPU2, H100 PCIe 80 GB
- experiment lock: `/tmp/dzkhomidov_gpu2.lock`
- soft-data SHA256: `b9ebd015f1881c1ac58b5966233b74390a25f13bf751af9a72dafc803c106af9`
- hard-data SHA256: `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`
- train-script SHA256: `70dfce33b6aeb54dc2dff30bc40117583df66fa8bdad6a29b73673e47dbe3148`

Stage 1:

```bash
CUDA_VISIBLE_DEVICES=2 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u \
  /home/dzkhomidov/matching-work/rescue_20260824/fgm/train.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/fgm \
  --variants bce,bce2x,fgm05,random05 --folds fold_01,fold_02 \
  --max-len 224 --epochs 2 --batch-size 256 --lr 2e-5 \
  --seed 20260814 --random-seed 20260815
```

Promotion commands are intentionally absent until the frozen stage-1 gate is
evaluated. No command targets fsk35 or a validation directory.

Run status: interrupted by submission priority at the clean variant boundary
after BCE folds 1–2. `bce2x` loaded initial weights but completed zero update
steps and produced no artifact. Exact owned PIDs 857300/857299 were terminated;
the lock was released and GPU2 verified at 1 MiB, 0% on 2026-08-24 10:39:56 MSK.

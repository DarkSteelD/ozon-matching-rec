# Commands / process ledger

## Original phase 1 (completed before fsk35 release policy)

```bash
cd /home/dzkhomidov/matching-work/rescue_20260824/category_experts
nohup ./run.sh > logs/phase1.log 2>&1 &
# runner PID 1956487; initial shared child PID 1956492
```

The full shared -> random -> category sequence ran from
`2026-08-24T03:22:36+03:00` through `2026-08-24T04:09:05+03:00` on
`avi-gn-fsk35`, physical GPU3. This predates
`RESOURCE_POLICY_20260824.md`. No new fsk35 process was launched during
recovery.

## Migration / gate decision

At `2026-08-24T10:10:27+03:00` and again at
`2026-08-24T10:10:43+03:00`, `avi-ix-devbox02` physical GPU3 reported 1 MiB,
0% utilization, no compute process. GPU1's resident VLLM PID 4140935 was left
untouched.

Finished `preds/`, `logs/phase1.log`, and `metrics.json` were copied read-only
from fsk35 and then copied to devbox02. Relative SHA256 hashes are recorded in
`SHA256SUMS`; all prediction and metric files matched after migration.

The gate failed on existing complete folds 01-02, so the planned devbox02 launch
was cancelled before claiming a lock or starting a GPU process. Folds 03-04
remain intentionally unchecked.

Exact training command retained in `run.sh`:

```bash
CUDA_VISIBLE_DEVICES=3 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u train.py \
  --variant VARIANT --data input/hand_pairs_pd_v3cal.parquet \
  --init input/rubase_llmfull_e2 --output . \
  --folds fold_01,fold_02 --max-len 224 --bs 256 --epochs 2 \
  --lr 2e-5 --seed 20260814
```

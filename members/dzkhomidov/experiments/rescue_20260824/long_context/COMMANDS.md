# Reproduction commands

Source evidence:

```bash
git -C /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec rev-parse HEAD
# 2da459984a1207677ff9eb863ca28589027a4bc3
sha256sum input/hand_pairs.parquet input/rubase_llmfull_e2/model.safetensors
# d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31  hand_pairs.parquet
# 0a90825fbeb584fda7dfb3faded702b302b338aa3b0d8e4dc8217be77d0399f6  model.safetensors
```

CPU truncation audit:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python measure_lengths.py \
  --data input/hand_pairs.parquet \
  --tokenizer input/rubase_llmfull_e2 \
  --output truncation_coverage.json
```

Phase 1 was launched on `avi-gn-fsk35`, physical GPU 3, after an idle/co-tenant
check (all GPUs reported 4 MiB and 0% at check time):

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
ps -eo pid,user,cmd --sort=-%cpu | head -20
nohup ./run_phase1.sh > logs/phase1.log 2>&1 &
# runner PID 1686744; training child PID 1686747
```

`run_phase1.sh` contains the exact fixed command for len224, len384, len448,
and len512, folds 1-2, seed 20260814. No write is made to any repository or
`validation/` directory.

The coordinator originally assigned physical GPU 2. The staged runner still
specified `CUDA_VISIBLE_DEVICES=3`, so the exact process began on idle physical
GPU 3. It was not killed or migrated; the coordinator explicitly approved
continuing PID 1686747 there.

Long/short subset diagnostic:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python analyze_subsets.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --tokenizer /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --pred-root preds --output subset_metrics.json
```

Key artifact hashes:

```text
72b256aa86aa039080267e6767ba8d8bc96dd006507133e81fcaf4b261928279  metrics_phase1.json
aaee535067a5dc6fdcf1c313a7843b9e311d08de26d384a408da1addd20cb79c  subset_metrics.json
9caceb52b2cf5e3014da270d138a207839f9afcc247c9c3731bb1f9b6d4be7b5  logs/phase1.log
```

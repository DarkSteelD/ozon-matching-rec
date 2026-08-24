# Distill category-blend teacher signal into rubase student

## Claim and gate

Replacing 10 percentage points of the existing isotonic stack-v3 teacher
component with the calibrated five-model category-shrink75 OOF teacher improves
rubase student hard-label macro category PR-AUC by more than 0.001 on both
fold_01 and fold_02 versus an exact rerun on `hand_pairs_pd_v3cal.parquet`.

- Stage 1: baseline, global-teacher negative control, and category teacher on
  folds 01–02 only.
- Gate: category minus exact baseline `> +0.001` on each fold (same sign).
- Stage 2: folds 03–04 only if the gate passes. No tuning after fold metrics.

## Frozen target construction

Existing target is reconstructed as
`v3cal = 0.3*hard + 0.7*p_old`, where `p_old` is the foldwise isotonic stack-v3
teacher. Global/category rank teachers are independently isotonic-calibrated on
each OOF fold to the hard-label probability scale (`[0.001, 0.999]`).

- exact baseline: `0.3*hard + 0.7*p_old` (existing file, no rewrite)
- negative control: `0.3*hard + 0.6*p_old + 0.1*p_global_nested`
- candidate: `0.3*hard + 0.6*p_old + 0.1*p_category_shrink75`

This preserves the 30% hard-label component and 90% of the prior total target;
only one fixed weight, 0.10, is tested.

## Frozen training config

- model `DeepPavlov/rubert-base-cased`
- init `/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2`
- max_len 224, category + attrs, symmetrization enabled
- epochs 2, batch 256, lr 2e-5, seed 20260814
- host `avi-gn-fsk35`, GPU 6 after live ownership check
- source trainer `/home/dzkhomidov/matching-work/scripts/train_hand_fast.py`

## Commands

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python build_targets.py

# On avi-gn-fsk35, after explicit output symlinks are created:
CUDA_VISIBLE_DEVICES=6 /home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/scripts/train_hand_fast.py \
  --exp <experiment> --model DeepPavlov/rubert-base-cased \
  --init /home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2 \
  --max-len 224 --cat --attrs --sym --folds fold_01,fold_02 \
  --data <dataset> --seed 20260814

/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py --stage folds12
```

## Matrix

| variant | fold_01 | fold_02 | fold_03 | fold_04 |
|---|---|---|---|---|
| exact v3cal baseline | checked | checked | not run (gate failed) | not run (gate failed) |
| global nested control 10% | checked | checked | not run (gate failed) | not run (gate failed) |
| category shrink75 candidate 10% | checked | checked | not run (gate failed) | not run (gate failed) |

Stage-1 gate result: **failed**. Candidate deltas versus exact baseline are
`-0.000146` on fold_01 and `+0.000380` on fold_02.

# Category-balanced hand fine-tuning

## Claim

Inverse-frequency category weighting during hand-pair fine-tuning improves the
mean over category AP (`macro_cat_prauc`) by more than 0.001 on folds 1 and 2,
with the same delta sign on both folds, while not reducing pooled fold AP
(`mean_prauc`). The proposed mechanism is equal training influence per category,
matching the hidden metric's category-macro aggregation.

## Fixed controls

- Repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- Git SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`
- Dataset: read-only `hand_pairs_pd_v3cal.parquet`, 365654 rows
- Initialization: read-only `ckpt_disk/rubase_llmfull_e2`
- Seed: `20260814`
- Folds: gate on `fold_01,fold_02`; folds 3–4 only if gate passes
- Common training: 2 epochs, batch 256, lr 2e-5, max length 224,
  category + attributes, symmetric pair-order augmentation/evaluation
- Baseline: ordinary example-average BCE
- Candidate: BCE weighted by `N / (K * category_count)` on each fold's training
  rows, normalized to mean one

The baseline and candidate differ only in loss weights. Predictions, logs,
commands, and metrics are written under this experiment directory. Source data
and checkpoints are read-only. Nothing is written to `validation/`.

## Before launch

- [x] Check repository SHA/status
- [x] Check free disk
- [x] Check live GPU processes and ownership
- [x] Persist remote host/GPU/PID and exact commands

## Status matrix

| variant | fold | status |
|---|---|---|
| baseline | fold_01 | checked |
| baseline | fold_02 | checked |
| category_balanced | fold_01 | checked |
| category_balanced | fold_02 | checked |
| baseline | fold_03 | unchecked: gate failed |
| baseline | fold_04 | unchecked: gate failed |
| category_balanced | fold_03 | unchecked: gate failed |
| category_balanced | fold_04 | unchecked: gate failed |

## Acceptance gate

Continue to folds 3–4 only if candidate-minus-baseline `macro_cat_prauc` is
positive on both folds and the two-fold aggregate delta is greater than 0.001.
Report fold/category AP, pooled AP, macro-category AP, changed rankings,
runtime, and GPU cost. A result smaller than fold/seed noise is inconclusive.

## Pre-run risk

The dataset is already nearly category-balanced: most categories contain
17.3k–19.0k rows, although `Одежда` has 23.4k. Therefore inverse-frequency
weighting is a deliberately minimal and low-amplitude intervention; a null
result would test this precise weighting, not every possible macro-optimization
scheme.

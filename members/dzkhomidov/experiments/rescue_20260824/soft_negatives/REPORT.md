# Ambiguous-negative downweighting — finalized 2026-08-24

## Decision

**INCONCLUSIVE for adoption; positive same-run mechanism signal.** The preregistered
`hard050` variant beats its same-run baseline on all four folds by +0.002004 mean
macro-category PR-AUC and beats the matched-random control on all four folds by
+0.001800. However, the rerun baseline is 0.001177 below the archived diagnostic
baseline. Against that stronger archived baseline, `hard050` is only +0.000827,
below the +0.001 adoption gate, and the category bootstrap interval crosses zero.
Do not search another dose from these results and do not replace the current model
without a clean same-code repeated-seed confirmation.

No validation directory, submission, push, or commit was touched.

## Claim and fixed design

Claim: for each held-out fold, downweighting the top 10% highest-OOF-score
training negatives within each category to BCE weight 0.50 improves held-out
macro-category PR-AUC by more than 0.001, with the same sign on every fold.

- Data: `data/hand_pairs.parquet`, 365,654 rows, component-disjoint folds 01–04.
- Init: `ckpt_disk/rubase_llmfull_e2`.
- Text/max length: `name | category | attrs`, 224 tokens.
- Training: two epochs, batch 192, LR 2e-5, seed 20260814.
- Selection: training negatives only; per-category top 10% by the archived
  `ce_rubase_e2_len224` OOF prediction. Validation rows used in selection: zero.
- Negative control: the same number of training negatives per category sampled
  randomly with a fixed fold-specific seed, also weighted 0.50.
- Selected negative counts: 20,388 / 20,389 / 20,361 / 20,394 on folds 01–04.

## Primary results

| fold | rerun baseline | archived baseline | hard050 | random050 | hard − rerun | hard − archived | hard − random |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.779865 | 0.781200 | 0.781382 | 0.779731 | +0.001517 | +0.000182 | +0.001651 |
| 02 | 0.787180 | 0.788423 | 0.789620 | 0.788124 | +0.002439 | +0.001196 | +0.001496 |
| 03 | 0.781367 | 0.782716 | 0.784109 | 0.780947 | +0.002742 | +0.001393 | +0.003162 |
| 04 | 0.784776 | 0.785559 | 0.786094 | 0.785204 | +0.001317 | +0.000535 | +0.000890 |
| mean | 0.783297 | 0.784475 | 0.785301 | 0.783502 | **+0.002004** | **+0.000827** | **+0.001800** |

The random control is +0.000204 versus the rerun baseline and has mixed fold
signs. Thus generic removal of negative loss mass does not explain the same-run
gain. `hard050` is positive versus both the rerun and random control on every
fold.

The originally registered folds-01/02 gate passed: +0.001978 versus the rerun
baseline, positive on both folds. Folds 03/04 independently preserved the sign
and magnitude (+0.002030 mean versus rerun).

## Reproduction gap and uncertainty

The archived diagnostic baseline beats the in-experiment rerun on every fold,
mean +0.001177. This is a pipeline/reproduction gap, not a clean seed-noise
estimate: the archived predictions were produced by the historical training
path, while treatment and controls use the experiment script. Consequently the
same-run comparison is the valid causal ablation, but it is insufficient to
claim an absolute deployable gain over the archived model.

A deterministic 20,000-sample bootstrap resampled the 20 category effects after
averaging each category across four folds. This is a category-heterogeneity
interval, not a duplicate-component bootstrap:

| comparison | mean | positive categories | bootstrap 95% interval |
|---|---:|---:|---:|
| hard050 − rerun | +0.002004 | 17/20 | [+0.000938, +0.003359] |
| hard050 − random050 | +0.001800 | 14/20 | [+0.000681, +0.003173] |
| hard050 − archived | +0.000827 | 11/20 | [−0.000153, +0.001917] |
| random050 − rerun | +0.000204 | 13/20 | [−0.000519, +0.000880] |

No component identifier was available in the scored artifact, so a true grouped
bootstrap at hidden-test size was not claimed.

## Secondary metrics and mechanism

Four-fold mean pooled AP is 0.840013 rerun, 0.840647 hard050, and 0.840232
random050. The much larger macro gain indicates category redistribution, not a
large pooled improvement. The largest category mean deltas versus rerun are
clothing +0.012015, shoes +0.005375, beauty/hygiene +0.004823, musical
instruments +0.002739, and electronics +0.002713. Seventeen of twenty category
means improve.

The treatment raises mean prediction from 0.242041 to 0.260194, worsens Brier
from 0.087931 to 0.089284, and worsens ECE20 from 0.027363 to 0.037790. This is
consistent with reducing suppression from model-hard negatives, but it also
causes overprediction. At the fixed diagnostic threshold 0.5:

| variant | TP | FP | FN | TN | F1 |
|---|---:|---:|---:|---:|---:|
| rerun baseline | 65,210 | 15,122 | 28,680 | 256,642 | 0.748585 |
| hard050 | 69,661 | 19,541 | 24,229 | 252,223 | 0.760940 |
| random050 | 66,091 | 15,716 | 27,799 | 256,048 | 0.752329 |

Relative to rerun, hard050 adds 4,451 TP and 4,419 FP while removing 4,451 FN.
These threshold counts are diagnostic only because 0.5 was not calibrated for
the competition metric.

## Coverage, runtime, and checked status

- Checked: baseline/hard050/random050, folds 01–04, identical treatment script,
  seed and folds; archived baseline diagnostic; matched coverage random control;
  row-count and one-to-one ID joins; all prediction hashes recorded.
- Checked earlier in stage 1 only: weight 0.75 on folds 01–02. It was below the
  registered gate (+0.000828) and was not promoted.
- Unchecked: a second same-code seed; a true component-group bootstrap; another
  dose; hidden-test transfer; calibration repair after hard050.
- GPU time for baseline/hard050/random050 across four folds: about 4,376 seconds
  (1.21 H100-hours). Including stage-1 0.75 variants: about 5,835 seconds
  (1.62 H100-hours).

## Recovery and independent verification

The completed prediction artifacts were copied read-only from the released
`avi-gn-fsk35` storage. They were rescored independently on `avi-ix-devbox03`;
the regenerated JSON is byte-identical to the recovered `metrics_all.json`:

- metrics SHA-256: `dc9aa34e2c73394f1a3cfd07d6cac3f54c03b38a007840656f924dd8cf2eca05`
- stage-1 manifest SHA-256: `870846b87ede9308fa2b62a6953fe6885eae51af72e190e314dd3c7f6e66cffe`
- stage-2 manifest SHA-256: `2fae0d88179cdc4db179a41900ee6869ebdecd02d2d28552df3a656105f0cdf5`
- prediction hashes: `finalized_devbox03/predictions.sha256`
- recovered artifacts: `recovered_fsk35/`
- independent score: `finalized_devbox03/metrics_rescored_devbox03.json`

Reproduction command used on devbox03:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python \
  /home/dzkhomidov/matching-work/rescue_20260824/soft_negatives_finalize/score.py \
  --data /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --root /home/dzkhomidov/matching-work/rescue_20260824/soft_negatives_finalize/preds \
  --archived-baseline /home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224 \
  --folds fold_01,fold_02,fold_03,fold_04 \
  --variants baseline_rerun,baseline_archived,hard050,random050 \
  --output /home/dzkhomidov/matching-work/rescue_20260824/soft_negatives_finalize/metrics_rescored_devbox03.json
```

Because all twelve required predictions were already complete and hash-valid,
no retraining was necessary on devbox03 and physical GPU 3 remained idle.

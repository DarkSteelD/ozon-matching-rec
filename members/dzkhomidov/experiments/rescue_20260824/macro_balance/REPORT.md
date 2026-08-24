# Category-balanced fine-tuning report

## Outcome

**Negative for the tested hypothesis.** Inverse-frequency category weighting
did not improve the target macro-category PR-AUC. It reduced the two-fold pooled
macro score by **0.000376** (`0.801946` to `0.801570`), with a negative delta on
both folds. The required gain was greater than 0.001, so the gate failed and
folds 3–4 were not run.

The ordinary fold-average pooled AP was effectively unchanged: **+0.000035**
(`0.854023` to `0.854058`). This is smaller than rerun variability.

## Setup and controls

- Data: `hand_pairs_pd_v3cal.parquet` for training targets; original binary
  `hand_pairs.parquet` labels for scoring.
- Init: `rubase_llmfull_e2`; seed 20260814; folds 1–2.
- Both variants: 2 epochs, batch 256, lr 2e-5, max length 224, category and
  attributes, symmetric pair-order augmentation and two-direction evaluation.
- Only difference: baseline BCE uses weight 1; candidate uses per-fold
  `N / (K * category_count)`, normalized to mean 1.
- Coverage: 100% of training examples. Candidate weight ranges were
  0.7883–1.0565 on fold 1 and 0.7798–1.0516 on fold 2.
- Host/resource: `avi-ix-devbox02`, physical GPU 3, peak observed allocation
  about 25.9 GiB. Wrapper PID 232492, baseline PID 232496, candidate PID 261931.
  Wall time 39m35s; four fold runs totalled 2124.1s (about 0.66 GPU-hours),
  excluding tokenization in the per-fold runtimes.

## Required result table

| variant | fold | pooled AP | delta | macro-cat AP | macro delta | mean fashion | delta | runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 01 | 0.849689 | — | 0.799682 | — | 0.632600 | — | 530.4s |
| balanced | 01 | 0.849858 | +0.000169 | 0.799648 | **-0.000034** | 0.631681 | -0.000919 | 531.4s |
| baseline | 02 | 0.858358 | — | 0.804719 | — | 0.619277 | — | 531.8s |
| balanced | 02 | 0.858259 | -0.000099 | 0.804091 | **-0.000628** | 0.615746 | -0.003532 | 530.5s |
| baseline | 2-fold | 0.854023 | — | 0.801946 | — | 0.625287 | — | 531.1s/fold |
| balanced | 2-fold | 0.854058 | +0.000035 | 0.801570 | **-0.000376** | 0.623172 | -0.002115 | 530.9s/fold |

`2-fold macro-cat AP` is computed after pooling folds 1–2 within each category.
The fold-level worst-fashion AP fell by 0.000185 and 0.004705. The pooled
minimum happens to rise by 0.001867 because footwear improves, but this is not
stable: clothing falls sharply and becomes the worst category on fold 2.

## Fashion mechanism

Pooled category deltas show why the intervention failed:

| category | baseline | balanced | delta |
|---|---:|---:|---:|
| Одежда | 0.588524 | 0.579737 | **-0.008787** |
| Ювелирные изделия | 0.597999 | 0.595761 | -0.002238 |
| Галантерея и аксессуары | 0.739699 | 0.740399 | +0.000699 |
| Обувь | 0.574924 | 0.576791 | +0.001867 |

The source data was already close to category-balanced. Clothing was the one
large category (23.4k rows versus roughly 17.3k–19.0k for most categories), so
inverse-frequency weighting mainly downweighted clothing. Its AP then fell on
both folds (-0.00684 and -0.01151). Small gains in several lighter categories
did not compensate for that loss in the macro average. This supports a concrete
mechanism for the negative result, not the proposed improvement mechanism.

## Changed rankings

The loss change was not a numerical no-op. Across pooled folds, baseline and
candidate predictions had Spearman 0.99835, mean absolute rank movement 2098
positions, 41.3% of rows moved by at least 1% of the dataset, and top-decile
Jaccard 0.9342. Fold-level details are in `metrics/ranking_fold_01.json` and
`metrics/ranking_fold_02.json`.

## Noise and interpretation

The baseline was rerun as required. Against the prior same-config
`ce_rubase_v3cal_sym` predictions, its macro-category AP moved by +0.000114 on
fold 1 and +0.000143 on fold 2; pooled AP moved by +0.000075 and -0.000127.
Thus the fold-1 macro delta (-0.000034) is below rerun noise, while the fold-2
delta (-0.000628) is materially larger and negative. The two candidate deltas
have the same negative sign and the aggregate misses the +0.001 gate by a wide
margin. No positive claim is supported.

## Checked and unchecked

- Checked: baseline and inverse-frequency candidate on folds 1–2; identical
  seed, init, data, augmentation, evaluation, batch schedule and hyperparameters;
  prediction completeness; per-category AP; pooled/fold AP; ranking changes;
  runtimes and exit codes (both zero).
- Unchecked by design: folds 3–4, additional seeds, balanced sampler, stronger
  temperature weighting, class-by-category weighting, and direct macro/ranking
  objectives. They are not ruled out by this result.
- This result closes only the precise minimal inverse-frequency weighting
  hypothesis under the stated +0.001/two-positive-fold gate.

## Artifacts and reproduction

- Commands and GPU record: `COMMANDS.md`
- Source: `train_macro_balance.py`, `score_experiment.py`
- Raw logs: `logs/`
- Predictions and run metadata: `predictions/`
- Machine-readable scores: `metrics/metrics.csv`, `metrics/metrics.json`,
  `metrics/comparison_metrics.csv`, `metrics/per_category_metrics.csv`
- Exact training commands are in `COMMANDS.md`; both commands exited 0.

No validation directory, submission, branch, commit, push, or external service
was touched.

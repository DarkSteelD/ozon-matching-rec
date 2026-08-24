# Poly1 BCE screen — final report

## Decision

**NEGATIVE / NO-GO.** Poly1 with `epsilon=+0.5` moves macro category AP by
only `+0.000290` on fold 1 and `+0.000132` on fold 2. It fails the frozen gate
of more than `+0.001` on every screening fold. Per protocol, `epsilon=1.0` and
folds 3–4 were not run.

No job was launched on `avi-ix-devbox02`: all three stage-1 arms had finished
before `avi-gn-fsk35` was released, were recovered read-only, and were sufficient
to make the preregistered decision. `avi-gn-fsk35` was not written to and no new
process was launched there during recovery.

## Frozen claim and setup

Claim: on RuBERT epoch-2 init, v3cal soft targets, symmetric max_len 224 hand FT,
`BCE + epsilon*(1-p_t)` with `epsilon=0.5` improves hard-label macro category AP
by more than `+0.001` on both folds 1 and 2 against a matched fresh BCE run.

- repo SHA: `5099db5df398e6aa4fec9eccdaf6959f50cfbf29`
- repo status: clean
- dataset: 365,654 rows, component-disjoint folds; soft v3cal targets for train,
  row-aligned hard labels for scoring
- init: `rubase_llmfull_e2`
- seed: `20260814`, reset for every arm/fold
- two epochs, batch 256, lr `2e-5`, AdamW, OneCycle, pair-order swap augmentation,
  two-direction evaluation averaging
- historical same-architecture noise floor: approximately `0.0005`; frozen gate:
  `+0.001` on each screen fold

## Primary and secondary results

| variant | fold | macro category AP | delta | pooled AP delta | Brier delta | log-loss delta | ECE15 delta | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCE | 1 | 0.800000 | 0 | 0 | 0 | 0 | 0 | 370.9 s |
| BCE | 2 | 0.804469 | 0 | 0 | 0 | 0 | 0 | 369.9 s |
| Poly1 +0.5 | 1 | 0.800290 | +0.000290 | +0.000110 | +0.000839 | +0.005722 | +0.013785 | 370.1 s |
| Poly1 +0.5 | 2 | 0.804601 | +0.000132 | +0.000028 | +0.000756 | +0.005254 | +0.011731 | 369.0 s |
| Poly1 -0.5 | 1 | 0.799544 | -0.000456 | -0.000157 | +0.000317 | +0.000457 | +0.003126 | 369.6 s |
| Poly1 -0.5 | 2 | 0.804224 | -0.000245 | -0.000083 | +0.000500 | +0.001322 | +0.005596 | 369.1 s |

Across the two folds, `epsilon=+0.5` has mean macro delta `+0.000211`
(sample SD `0.000112`), below both the frozen gate and the historical noise
floor. Mean pooled AP delta is `+0.000069`. Mean calibration changes are harmful:
Brier `+0.000797`, log loss `+0.005488`, and ECE15 `+0.012758` (lower is better).
The six recovered runs used about 37.0 accelerator-minutes total; mean runtime was
369.8 seconds per arm/fold.

## Mechanism and controls

The sign control behaves directionally as expected. `epsilon=+0.5` makes scores
more extreme: relative to BCE, mean prediction moves about `-0.0101` for hard
negatives and `+0.0091` for hard positives. `epsilon=-0.5` reverses that
separation and loses macro AP on both folds. This supports the intended margin-
sharpening mechanism, but the ranking gain is too small and calibration becomes
materially worse.

At the fixed 0.5 threshold, Poly1 +0.5 changes outcomes as follows:

| fold | delta TP | delta FP | delta FN | delta TN |
|---|---:|---:|---:|---:|
| 1 | +24 | -19 | -24 | +19 |
| 2 | +36 | -10 | -36 | +10 |

Category effects are concentrated. Clothing improves `+0.001977` and shoes
`+0.001580` on average, with the same sign on both folds. Only 8/20 categories
are positive on both folds; four are negative on both. A category-resampling
diagnostic gives a nominal 95% interval `[+0.000004, +0.000475]`, but categories
are the fixed scoring population and there are only two folds, so this is not
used to override the preregistered gate.

## Checked and unchecked

Checked:

- matched fresh BCE, folds 1–2;
- Poly1 `epsilon=+0.5`, folds 1–2;
- sign/mechanism control `epsilon=-0.5`, folds 1–2;
- primary macro AP, pooled AP, category AP, Brier, log loss, ECE15;
- fixed-threshold TP/FP/FN/TN and prediction-shift diagnostics;
- row order and IDs against hard-label data;
- recovered artifact hashes.

Intentionally unchecked after gate failure:

- `epsilon=1.0` on folds 1–2;
- all variants on folds 3–4;
- additional seeds and post-hoc epsilon/category tuning.

This result closes global Poly1 `epsilon=+0.5` for the frozen recipe. It does not
establish that a category-specific loss would fail; that is a different,
unregistered hypothesis and was not tested here.

## Evidence and reproduction

- raw log: `train_folds12.log`
- recovered predictions and metadata: `preds/{bce,poly05,polyneg05}/`
- hashes: `RECOVERED_SHA256.txt`
- metrics: `stage1_score/metrics.csv`, `metrics.json`
- category diagnostics: `stage1_score/category_deltas.csv`,
  `category_summary.csv`
- calibration/prediction diagnostics: `stage1_score/prediction_diagnostics.csv`
- threshold outcomes: `stage1_score/threshold05_counts.csv`

Exact historical command is preserved in `COMMANDS.md`. To reproduce on an
allowed, freshly checked GPU, change only the host and physical CUDA index; keep
the data, init, variants, folds, hyperparameters, and seed unchanged. No
validation write, submission, push, or commit was performed.

Input SHA256:

- hard labels: `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`
- soft targets: `b9ebd015f1881c1ac58b5966233b74390a25f13bf751af9a72dafc803c106af9`
- init weights: `0a90825fbeb584fda7dfb3faded702b302b338aa3b0d8e4dc8217be77d0399f6`

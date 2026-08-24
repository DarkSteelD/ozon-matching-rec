# Controlled paired soft-negative rerun — 2026-08-24

## Decision

**POSITIVE / GO for incorporation testing.** With the RNG state reset identically
before every fresh arm, `hard050` improves macro-category PR-AUC by **+0.002161**
over a fresh baseline across four folds. Every fold is positive and every fold
exceeds the preregistered +0.001 gate. A duplicate fresh baseline on folds 01/02
is byte-identical to the first baseline, so no residual run-order or CUDA
nondeterminism was observed under the controlled settings.

This resolves the reproduction blocker in the previous soft-negative report.
It authorizes testing `hard050` in the current student/container composition; it
does not itself authorize a leaderboard submission.

No validation directory, submission, push, or commit was touched. `avi-gn-fsk35`
was not accessed for this rerun.

## Measurable claim and controls

Claim: downweighting the per-category top 10% OOF-scored training negatives to
BCE weight 0.50 improves fresh macro-category PR-AUC by more than 0.001 on both
folds 01 and 02. Only after this stage-1 gate passed were folds 03 and 04 run.

Fresh baseline and treatment shared:

- data and component-disjoint folds: `hand_pairs.parquet`, 365,654 rows;
- init: `rubase_llmfull_e2`;
- text/max length: `name | category | attrs`, 224 tokens;
- two epochs, batch 192, LR 2e-5;
- physical H100 GPU 3 on `avi-ix-devbox03`;
- fold seeds 20260815, 20260816, 20260817, and 20260818;
- Python, NumPy, torch CPU, and every CUDA RNG reset before each arm;
- deterministic torch algorithms enabled; cuDNN deterministic enabled;
  cuDNN benchmark and TF32 disabled; `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- identical initial model weights and minibatch permutations within each fold.

The selection uses only the three training folds: within each category, labelled
negatives are ranked by their row's archived OOF CE prediction and the top 10%
are selected. Held-out rows used in selection: zero. Selected counts are 20,388,
20,389, 20,361, and 20,394 on folds 01–04.

No fresh random arm was run, as preregistered. The duplicate baseline is the
negative control for run-order and residual nondeterminism. The earlier separate
experiment already showed matched-random downweighting was near zero and mixed
sign, but those historical numbers are not used in the fresh paired estimate.

## Primary result

| fold | fresh baseline | fresh hard050 | paired delta | gate |
|---|---:|---:|---:|---|
| 01 | 0.780269 | 0.782568 | **+0.002299** | pass |
| 02 | 0.787448 | 0.789395 | **+0.001947** | pass |
| 03 | 0.781158 | 0.783596 | **+0.002438** | pass |
| 04 | 0.785248 | 0.787207 | **+0.001959** | pass |
| mean | 0.783531 | 0.785691 | **+0.002161** | pass |

The stage-1 mean delta was +0.002123. The untouched confirmation folds 03/04
average +0.002199, preserving both sign and magnitude.

Mean pooled PR-AUC moves from 0.839993 to 0.840903 (+0.000910). The larger macro
gain is the relevant competition objective and indicates useful redistribution
across categories rather than only a pooled-class effect.

## Determinism audit and noise floor

The fresh baseline was trained twice, in different run-order positions, on folds
01 and 02. Predictions are byte-identical:

| fold | baseline SHA-256 | repeat SHA-256 | metric delta |
|---|---|---|---:|
| 01 | `af2eeb60ff63a34f91ed3d38b6b693704b478e5b1a15277f1411bcedab9bcbfb` | same | 0.000000 |
| 02 | `8258fe955d5e9211da718c693e9d30166f71a6a39c365c64fbac2be532d73bea` | same | 0.000000 |

Thus the measured residual nondeterminism/noise floor for this exact software,
hardware, fold, and seed audit is zero at CSV float precision. This does not
claim cross-driver or cross-GPU bitwise reproducibility. The strict deterministic
path ran at about 5.2 updates/s, versus roughly 8.2 updates/s in the earlier
non-strict run.

## Category stability and mechanism

Nineteen of twenty categories have a positive four-fold mean effect. Largest
mean gains are clothing +0.007280, shoes +0.006059, accessories +0.004476,
electronics +0.003502, and furniture +0.003412. The only negative category mean
is hobby/creative goods at −0.000096.

A deterministic 20,000-sample bootstrap resampling the 20 category effects gives
a 95% interval **[+0.001407, +0.003037]**. This measures category heterogeneity;
it is not presented as a component-group or hidden-test-size bootstrap.

The treatment raises mean prediction from 0.239374 to 0.258557, Brier from
0.088008 to 0.089174, and ECE20 from 0.028084 to 0.038485. The mechanism is
consistent with relieving excessive suppression by model-hard labelled
negatives. It improves ranking, especially in fashion, while worsening raw
calibration; any thresholded use should recalibrate after training.

At the fixed diagnostic threshold 0.5:

| variant | TP | FP | FN | TN | F1 |
|---|---:|---:|---:|---:|---:|
| fresh baseline | 65,120 | 14,708 | 28,770 | 257,056 | 0.749721 |
| fresh hard050 | 69,511 | 19,395 | 24,379 | 252,369 | 0.760531 |

Hard050 adds 4,391 TP and 4,687 FP while removing 4,391 FN. These counts are
diagnostic only; macro PR-AUC is threshold-free.

## Coverage, runtime, and checked status

- Checked: fresh baseline and hard050 on folds 01–04; stage gate before folds
  03/04; explicit seed metadata; identical initialization/minibatch order;
  duplicate baseline on folds 01/02; one-to-one prediction/truth joins; category
  bootstrap; fixed-threshold outcome counts.
- Unchecked here: fresh matched-random arm by instruction; another controlled
  seed; cross-host determinism; component-group bootstrap; container inference
  cost; ensemble marginal gain; hidden-test transfer and calibration repair.
- Primary eight runs: 4,602.93 seconds, 1.28 H100-hours.
- Including the two duplicate-baseline audits: 5,754.27 seconds, 1.60 H100-hours.

## Artifacts and reproduction

- Metrics: `metrics_all.json`, SHA-256
  `b3ae3180ad2610f9f1d09e076b9fac1ee5e1298b40b1f24f29ad80cbc12fee68`.
- Complete secondary evidence: `evidence.json`.
- Prediction hashes: `predictions.sha256`.
- Stage manifests: `run_manifest_stage1.json` and `run_manifest_stage2.json`.
- Per-arm seed/hardware metadata: `preds/<variant>/fold_*.meta.json`.
- Training code: `train_controlled.py`, SHA-256
  `d58057d9ce4e9333cfe83fe995d59192fbe5502258194e641a892b5ace14bdeb`.
- Logs: `train_stage1.log`, `train_stage2.log`, `score_stage1.log`,
  `score_all.log`, and both preflight logs.

Exact launch commands are persisted in `run_stage1.sh` and `run_stage2.sh`.
Both scripts refuse to start if physical GPU 3 has a compute application.

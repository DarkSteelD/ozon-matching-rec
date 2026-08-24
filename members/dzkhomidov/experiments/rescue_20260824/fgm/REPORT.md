# FGM screen — paused for submission priority

## Status

**INCONCLUSIVE / PAUSED.** The exact BCE baseline completed on folds 1–2. Before
the first compute-matched or perturbation update, the experiment was stopped at
a variant boundary and GPU2 was released for the submission pool. No FGM result
exists yet, so no claim about FGM is supported or rejected.

The process was stopped only by exact owned PIDs. At 2026-08-24 10:39:56 MSK,
`avi-ix-devbox02` physical GPU2 showed 1 MiB, 0% utilization, no compute app, and
`/tmp/dzkhomidov_gpu2.lock` was absent. fsk35 was never used.

## Completed evidence

| variant | fold | macro category AP | pooled AP | Brier | log loss | ECE15 | runtime | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BCE | 1 | 0.799400 | 0.849525 | 0.083577 | 0.276207 | 0.013371 | 535.1 s | checked |
| BCE | 2 | 0.804786 | 0.858217 | 0.081985 | 0.271699 | 0.015161 | 536.2 s | checked |

These values are baseline provenance only. There is no candidate delta and the
frozen `>+0.001` per-fold promotion gate has not been evaluated.

## Checked and unchecked

Checked:

- exact v3cal + sym len224 BCE, seed 20260814, folds 1–2;
- row-aligned hard-label scoring and calibration metrics;
- complete CSV/JSON artifacts and SHA256 hashes;
- clean process shutdown, lock release and free-GPU state.

Unchecked:

- compute/dropout-matched `bce2x`, both folds;
- FGM norm 0.5, both folds;
- equal-active-norm fixed random direction control, both folds;
- FGM norm 1.0 and folds 3–4, which remain gate-conditional;
- category effects, threshold TP/FP/FN and mechanism controls for FGM.

## Resume contract

Do not rerun the completed BCE folds. Resume on a newly double-checked allowed
GPU with variants `bce2x,fgm05,random05` and folds `fold_01,fold_02`, keeping all
other arguments in `COMMANDS.md` identical. The train script tokenizes once and
then writes one complete CSV/JSON pair per variant/fold. Inspect metrics only
after all remaining stage-1 arms complete.

Evidence:

- `preds/bce/fold_01.{csv,json}` and `fold_02.{csv,json}`
- `train_stage1.log`
- `RECOVERED_SHA256.txt`
- `stage1_score/metrics.{csv,json}` and `category_metrics.csv`
- frozen protocol `PLAN.md`; command provenance `COMMANDS.md`

No validation write, submission, push, or commit was performed.

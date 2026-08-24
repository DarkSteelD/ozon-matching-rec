# FGM embedding perturbation screen

## Frozen claim

On the exact RuBERT epoch-2 init + v3cal soft-target + symmetric max_len224
recipe, FGM with global active-embedding perturbation norm 0.5 improves hard-label
macro category AP by more than +0.001 on each of folds 1 and 2 versus exact
single-pass BCE.

## Stage 1 (inspect only after all arms finish)

| variant | fold 1 | fold 2 | role |
|---|---|---|---|
| bce | unchecked | unchecked | exact recipe baseline, one forward/backward |
| bce2x | unchecked | unchecked | compute/dropout matched, two clean passes at 0.5 loss each |
| fgm05 | unchecked | unchecked | clean + gradient-aligned perturbed pass, 0.5 loss each |
| random05 | unchecked | unchecked | clean + equal-norm fixed random-direction pass, 0.5 loss each |

FGM and controls retain the same examples, update order, optimizer steps,
scheduler steps and effective averaged loss scale. Perturbation is applied only
to word-embedding rows active in the clean gradient. Random direction is fixed
by seed 20260815 per vocabulary row, restricted to the same active rows, and
renormalized to exactly 0.5 each step. It contains no label-gradient direction.

Promotion requires FGM-0.5 minus exact BCE >+0.001 on both folds. To claim the
FGM mechanism rather than extra compute/dropout, FGM must also beat bce2x and
random05 with the same positive sign on both folds. Only then test FGM-1.0 on
folds 1–2; only a candidate satisfying the same primary gate is promoted with
fresh matched controls to folds 3–4. No post-hoc epsilon/category/seed selection.

## Fixed recipe

- train: `hand_pairs_pd_v3cal.parquet`; score: row-aligned `hand_pairs.parquet`
- init: `ckpt_disk/rubase_llmfull_e2`
- input `name | category | attrs`, prefix max_len224
- pair-order swap train augmentation; two-direction eval average
- 2 epochs, batch 256, lr 2e-5, AdamW wd .01, OneCycle
- seed 20260814 reset before every variant/fold
- primary macro category AP; secondary pooled AP, per-category AP, Brier,
  log loss, ECE15, fixed-0.5 TP/FP/FN/TN
- historical noise approximately 0.0005; gate +0.001 each fold
- isolated output `/home/dzkhomidov/matching-work/rescue_20260824/fgm`
- host `avi-ix-devbox02`, physical GPU2 only after two live checks and lock

No fsk35 use, validation write, submission, push or commit.

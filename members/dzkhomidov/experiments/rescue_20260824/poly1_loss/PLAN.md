# Poly1 BCE screen

## Frozen claim and ladder

On the strongest reproducible deployable recipe (RuBERT epoch-2 init, v3cal
soft targets, symmetric max_len224 hand FT), `BCE + eps*(1-p_t)` with eps=0.5
improves hard-label macro category AP by more than +0.001 on both fold_01 and
fold_02 against a fresh matched eps=0 BCE run.

For soft target y and p=sigmoid(logit), `p_t = y*p + (1-y)*(1-p)`. This reduces
to standard binary Poly1 for y in {0,1}.

Stage 1 runs, without metric inspection between arms:

| variant | eps | fold_01 | fold_02 | role |
|---|---:|---|---|---|
| bce | 0.0 | unchecked | unchecked | exact matched baseline |
| poly05 | 0.5 | unchecked | unchecked | candidate |
| polyneg05 | -0.5 | unchecked | unchecked | sign/mechanism control |

Only if poly05 beats BCE by >+0.001 on each fold is eps=1.0 run on folds 1-2.
Among eps 0.5/1.0, the highest mean candidate satisfying the same per-fold gate
is promoted with a fresh matched BCE to folds 3-4. Otherwise stop. No epsilon,
category, calibration or seed is selected post-hoc.

## Fixed recipe

- train targets: `hand_pairs_pd_v3cal.parquet` (soft, verified [0.0007,0.9993]);
  scoring labels: row-aligned `hand_pairs.parquet` (binary)
- init/model: `ckpt_disk/rubase_llmfull_e2`
- category + attrs, prefix max_len224, pair-order swap train augmentation and
  two-direction eval average
- epochs 2, batch 256, AdamW, LR 2e-5, weight decay .01, OneCycle schedule,
  seed 20260814 reset before every fold/variant, identical update order/count
- primary: mean category AP; secondary pooled AP, category AP, Brier, log loss,
  ECE15. Historical same-architecture noise is ~0.0005; gate is +0.001.
- host `avi-gn-fsk35`, physical GPU0 after live ownership check

No validation writes, submission, push or commit.

# Category expert routing

## Outcome

**Negative for the preregistered residual-head recipe; gate failed.** A true
category-routed residual classifier does not improve hard-label macro-category
PR-AUC by more than 0.001 on both gate folds. The fold deltas versus the exact
shared-head baseline are mixed-sign (`+0.000195`, `-0.000117`), and the mean
delta is only `+0.000039` (fold sample standard deviation `0.000221`). Folds
03-04 were therefore not run.

| variant | fold | macro AP | delta vs shared | delta vs random | status |
|---|---:|---:|---:|---:|---|
| shared | 01 | 0.799967 | - | - | checked |
| random router | 01 | 0.799923 | -0.000044 | - | checked control |
| category router | 01 | 0.800162 | +0.000195 | +0.000238 | checked |
| shared | 02 | 0.804526 | - | - | checked |
| random router | 02 | 0.804498 | -0.000028 | - | checked control |
| category router | 02 | 0.804409 | -0.000117 | -0.000089 | checked |
| shared mean | 01-02 | 0.802247 | - | - | checked |
| random mean | 01-02 | 0.802211 | -0.000036 | - | checked control |
| category mean | 01-02 | 0.802285 | +0.000039 | +0.000074 | gate failed |

The random router behaves as intended: its two deltas are small and negative,
with a mean of `-0.000036`. The category router does not consistently beat it:
category-minus-random is `+0.000238` on fold 01 and `-0.000089` on fold 02.

## Protocol and resource record

- 365,654 rows from `hand_pairs_pd_v3cal.parquet`; hard labels from the exact
  row-aligned `hand_pairs.parquet` are used only for scoring.
- `rubase_llmfull_e2`, max length 224, swap augmentation and two-direction
  evaluation, 2 epochs, batch 256, lr `2e-5`, seed `20260814`.
- `shared`: ordinary shared classifier. `category`: shared logit plus 0.75 times
  one of 20 zero-initialized category residual heads. `random`: identical
  capacity routed by a stable pair hash independent of category.
- Every head had at least 5,000 training rows on both folds; the fallback did
  not mask any category.
- Runtime was 16m14s for shared, 15m24s for random, and 14m51s for category,
  including tokenization; total wall time 46m29s on one H100.
- Phase 1 originally completed on `avi-gn-fsk35` physical GPU3 before the new
  release policy. No new work was launched there. Finished predictions were
  copied read-only to `avi-ix-devbox02`, and their hashes were verified. Since
  the gate failed, no GPU process was launched on devbox02 either; physical GPU3
  remained free after two live checks.

## Mechanism and scope

There is no stable residual-head mechanism in the tested recipe. Category beats
shared in only 23/40 category-fold cells. Even its largest mean category gain,
Shoes `+0.000500`, changes sign across folds (`+0.001304`, `-0.000304`). The
largest stable negatives include Stationery `-0.000404` mean and Furniture
`-0.000207` mean. These effects are too small and inconsistent to distinguish
from training variation with only two gate folds.

This result does **not** contradict the positive direct five-model category
blend. That experiment changes post-hoc cross-architecture weights and achieved
4-fold `+0.002614` with all fold deltas positive and bootstrap CI
`[+0.001414,+0.004070]`. The present experiment instead asks whether one RuBERT
can learn category-specific logit corrections while fine-tuning; it cannot
under the tested residual-head design.

It is also separate from the failed category-teacher target distillation. That
10% target perturbation produced `-0.000146/+0.000380` on folds 01-02 versus its
exact baseline (mean `+0.000117`, zero-crossing CI) and likewise failed its
gate. Together the two negative compression experiments say that the positive
five-model routing signal has not yet survived conversion into one RuBERT, not
that direct category blending is false.

TP/FP/FN are not defined because this is a rank-only PR-AUC experiment with no
classification threshold.

## Checked and unchecked

Checked: exact shared baseline, stable random-router capacity control, true
category router, folds 01-02, pooled and per-category AP, row/key alignment,
artifact hashes, and the predeclared gate.

Not run by design: folds 03-04 after gate failure. Unchecked: other residual
strengths, seeds, partial pooling/regularization, larger students, hidden-test
transfer. The conclusion is negative only for the preregistered zero-initialized
0.75 residual-head recipe.

## Artifacts

- `PLAN.md`: claim and gate.
- `train.py`, `score.py`, `run.sh`: exact implementation and command.
- `logs/phase1.log`: complete stdout/stderr.
- `preds/{shared,random,category}/fold_0{1,2}.csv`: row predictions.
- `metrics.json`: pooled, macro, and per-category metrics.
- `SHA256SUMS`: hashes checked after migration.
- `COMMANDS.md`: process and migration ledger.

No validation output, container, repository source, commit, push, or submission
was changed.

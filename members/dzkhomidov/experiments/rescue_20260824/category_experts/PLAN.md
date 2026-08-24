# Category expert routing

Claim: a shared RuBERT encoder with zero-initialized category residual heads
improves hard-label macro-category PR-AUC by >0.001 on both fold_01 and fold_02
over an exact shared-head baseline.

Fixed: `rubase_llmfull_e2`, `hand_pairs_pd_v3cal`, len224, sym, 2 epochs,
effective batch 256, lr 2e-5, seed 20260814. Hard labels from
`hand_pairs.parquet` are used only for scoring.

Variants:

- `shared`: recovered shared classifier only; expert parameters present but unused.
- `category`: shared logit + 0.75 * zero-initialized category residual head.
- `random`: identical residual-head capacity, but stable pair-hash routing
  independent of category (permutation/capacity negative control).

Safeguards: global shared fallback for unknown or train count <5000; expert
weights start at zero; only 20 tiny linear residual heads; no category loss
weights, category prefix, adapters, or target changes.

Gate: category minus shared >0.001 macro AP on both folds, and category must
beat the random-router control. Only then run folds 3-4.

All writes stay under this scratch directory. No validation, submission, push,
commit, or container work.

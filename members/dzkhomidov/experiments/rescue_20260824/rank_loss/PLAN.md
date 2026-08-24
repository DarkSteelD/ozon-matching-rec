# Rank-loss gate

Claim: adding within-category RankNet loss to the same hand fine-tune improves
macro category AP by more than 0.001 on both gate folds relative to BCE.

Fixed inputs: `hand_pairs.parquet`, ruBERT LLM-pretrained epoch-2 checkpoint,
seed 20260814, batch 256, two epochs, max length 224, category + attrs. Arms:

- `bce`: BCE only.
- `rank01`: BCE + 0.1 RankNet within category.
- `rank03`: BCE + 0.3 RankNet within category.
- `random03`: BCE + 0.3 RankNet with positive/negative pairs sampled without
  category restriction; negative control for the within-category mechanism.

Gate: run folds 01–02 first. A candidate advances only if its delta versus BCE
has the same positive sign on both folds and pooled macro category AP improves by
more than 0.001. Only passing arms run folds 03–04.

No prediction is written below this task directory. No validation, submission,
commit, push, or external publication is part of this experiment.

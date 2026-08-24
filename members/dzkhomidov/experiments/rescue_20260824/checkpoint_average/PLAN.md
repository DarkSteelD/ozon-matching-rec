# Late-checkpoint averaging / EMA

## Claim and gate

A single deployable RuBERT student obtained by averaging late hand-FT weights or
by EMA improves hard-label macro category PR-AUC over the final checkpoint from
the exact same trajectory. Screen folds 01-02; advance only if the same candidate
is positive on both and mean delta exceeds +0.001.

## Frozen protocol

- Host/GPU: `avi-gn-fsk35`, physical GPU 6 (live-check required).
- Data: `hand_pairs_pd_v3cal.parquet`; evaluation labels from `hand_pairs.parquet`.
- Init: `rubase_llmfull_e2`; RuBERT base, max_len 224, attrs+category.
- Seed 20260814, 2 epochs, batch 256, lr 2e-5, pair swap train augmentation,
  two-direction evaluation.
- The final baseline and all candidates come from one matched trajectory per fold.

## Variants

- `final`: step 100%, exact final checkpoint (baseline).
- `late_avg`: arithmetic weight mean at 75%, 87.5%, 100% (deployable).
- `ema`: per-update EMA after 50%, decay 0.995 (deployable).
- `early_avg`: equal weight mean at 25%, 50%, 75% (negative control).
- `late_pred_avg`: mean predictions of the three late checkpoints (positive
  diagnostic only, not a deployable claim).

All stdout/stderr, checkpoints, predictions, metrics and runtime records stay in
this directory. No repository/validation/submission/commit/push writes.

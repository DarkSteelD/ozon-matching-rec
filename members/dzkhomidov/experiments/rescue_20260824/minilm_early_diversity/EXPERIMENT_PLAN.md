# MiniLM early-diversity trajectory

## Measurable claim

The fixed 10% within-category MiniLM rank blend has a useful early-stopping
window before standard hand fine-tuning erases its mMARCO diversity. A checkpoint
is useful only if it improves macro-category PR-AUC by more than `+0.001` on fold
01 and then by more than `+0.001` on untouched confirmation fold 02.

## Frozen protocol

- Host: `avi-ix-devbox02`, physical GPU0 only, after two live compute-app checks
  and under `/tmp/dzkhomidov_gpu0_minilm_early_diversity.lock`.
- Never launch on fsk35.
- Train data: `hand_pairs_pd_v3cal.parquet`; evaluation labels/categories:
  `hand_pairs.parquet`; component-separated folds.
- Model: recovered `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Same standard recipe as the negative full experiment: category + attrs,
  max length 224, swap augmentation, two-direction eval, batch 256, LR `2e-5`,
  seed `20260814`, two-epoch OneCycle schedule.
- Fold 01 ladder along one trajectory: updates 250, 500, 1000, and symbolic
  `full` (2144 on fold 01). Evaluate only the fixed blend
  `0.9*category_rank(exact_rubase_baseline) + 0.1*category_rank(MiniLM)`.
- Select exactly one checkpoint by highest fold-01 blend delta. Ties choose the
  earliest checkpoint. Only that selected update is evaluated on fold 02. A
  numeric early checkpoint on fold 02 uses the fold-02 full-trajectory OneCycle
  schedule and identical prefix of the seeded training order; `full` means the
  fold-specific endpoint (2142 updates).
- The exact baseline is the recovered fixed-seed v3cal/sym rubase prediction in
  `architecture_scout/preds/full_baseline`; it is never reselected or altered.
- Mechanism control: fold-01 classifier-head-only training for 500 updates with
  encoder frozen. It is reported but is not eligible for checkpoint selection.
- Primary gate: selected fixed blend delta strictly greater than `+0.001` on
  both folds. No weight tuning, category selection, or standalone-model claim.
- Diagnostics: within-category mean Spearman correlation to baseline at each
  checkpoint, fold-01 gain/correlation curve, wall time and peak memory.

## Controls and stopping

- Exact zero-update MiniLM prediction is recovered from the architecture scout
  only as contextual evidence; this experiment's causal control is the frozen
  head-only arm.
- The recovered full checkpoint is expected to reproduce the prior near-zero
  blend delta and acts as the negative endpoint control.
- If the single fold-01-selected checkpoint fails `+0.001` on fold 02, stop.
  Do not inspect alternative checkpoints on fold 02.
- If it passes both folds, folds 03-04 require a separate explicit continuation;
  they are not part of this selection/confirmation run.

No validation directory, submission, push, or commit is allowed.

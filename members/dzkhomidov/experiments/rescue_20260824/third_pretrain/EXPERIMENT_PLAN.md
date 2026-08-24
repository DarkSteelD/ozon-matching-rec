# Third rubase LLM-pretrain epoch

## Claim

Continuing `rubase_llmfull_e2` for exactly one more pass over the same
11,187,780 soft-label pairs improves identical hand fine-tuning PR-AUC by
more than 0.001 on both gate folds (`fold_01`, `fold_02`). Only if both deltas
have the same positive sign and each is >0.001 will folds 03-04 be run.

## Fixed recipe

- LLM continuation: `train_ce_fast.py`, init epoch-2 checkpoint, rubert-base
  tokenizer, cache `tok_rubase_len128`, 1 epoch, batch 512, LR 3e-5,
  OneCycleLR freshly constructed for this pass (`pct_start=0.06`, linear),
  BCE soft labels, seed 20260814, attrs + category.
- Hand FT: `train_hand_fast.py`, 2 epochs, batch 256, LR 2e-5, max length
  160, attrs + category, seed 20260814. Baseline and candidate differ only in
  init checkpoint.
- Primary metric: pooled average precision per fold, then arithmetic mean.
- Gate: candidate-baseline >0.001 separately on both folds 01 and 02.

## Controls

- Negative/control baseline: epoch-2 init under the identical hand recipe.
- Positive pipeline control: archived epoch-2 OOF predictions score 0.83473663
  globally and are usable only after exact recipe/row alignment is verified.
- Candidate: epoch-3 continuation.
- Fold/variant statuses start `unchecked`.

## Safety and paths

- No writes to repository `validation/`; all artifacts stay in this folder.
- No submission, push, commit, or process termination.
- Accelerator: inputs were staged on `avi-gn-fsk35`. Physical GPU 3 became
  occupied between check and atomic launch, so no job was started there.
  Parent explicitly reassigned this run to physical GPU 2; launch still
  requires a fresh compute-app ownership check on its UUID.
- Repository SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`.

## Known noise

No same-recipe seed-repeat is archived. The acceptance margin of 0.001 per
fold is therefore treated as the minimum effect gate, not as a measured
confidence interval. Fold spread and all checked/unchecked cases will be
reported honestly.

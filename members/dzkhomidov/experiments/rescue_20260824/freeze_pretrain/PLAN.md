# Preserving pair-pretraining by partial encoder freezing

Claim: with epoch-3 pair-pretrained RuBERT, freezing the bottom six transformer
blocks during v3cal symmetric hand fine-tuning improves hard-label macro
category PR-AUC by more than 0.001 versus a fully unfrozen baseline separately
on fold_01 and fold_02.

Fixed recipe: `rubase_llmfull_e3`, `hand_pairs_pd_v3cal`, max length 224,
symmetric pair-order train/eval, 2 epochs, effective batch 256 (microbatch 128),
LR 2e-5, OneCycle linear schedule, seed 20260814, identical 2144/2142 updates.

Variants:

- `full`: all parameters trainable; exact existing positive-composition baseline.
- `bottom6`: freeze only BERT encoder blocks 0-5; embeddings and blocks 6-11 train.
- `top6`: matched-parameter-count control freezing blocks 6-11 instead.

The top-six control distinguishes preservation of low-level pair-pretrained
representations from generic capacity reduction/regularization. Existing full
predictions may be reused only after trainer, checkpoint, data, row ordering,
seed, folds, and recipe are verified exactly.

Gate: `bottom6 - full > 0.001` on both folds 01 and 02. Only then run all three
variants on folds 03-04. Primary metric is macro mean of 20 category AP values;
pooled AP is secondary. All outputs stay in this scratch directory. No
validation, submission, container, push, or commit.

# Hard-negative reweighting experiment

## Claim

Upweighting genuinely difficult hand-labelled negatives during the hand fine-tune of
the LLM-pretrained rubase cross-encoder improves pooled PR-AUC or macro-category
PR-AUC by more than 0.001 on folds 1 and 2, with the same sign on both folds.

## Fixed recipe

- Data: `/home/dzkhomidov/matching-work/data/hand_pairs.parquet` (365,654 rows).
- Initial checkpoint: `rubase_llmfull_e2` (two epochs of LLM-pair pretraining).
- Hand FT: names + category + attrs, max length 224, two epochs, batch 192,
  AdamW, LR 2e-5, weight decay 0.01, OneCycle schedule, seed 20260814.
- Baseline predictions: existing `ce_rubase_e2_len224`, produced by this same
  checkpoint/recipe. They are reused rather than spending GPU time reproducing a
  deterministic reference already saved on disk.

## Leakage control and hard-negative definition

The CE hardness signal for every row is its saved OOF prediction: that row and its
component fold were absent from the model which produced the score. For a target
fold, hard-negative selection is computed only among the other three (training)
folds. No validation row participates in thresholds, ranks, sampling, or training.

Within each category in the target fold's training portion, negatives receive:

`hardness = percentile(CE OOF score) + percentile(name token Jaccard) + 0.5 * attribute_conflict`

`attribute_conflict` means at least one normalized attribute key appears on both
sides with different non-empty normalized values. The top 10% of negatives per
category are hard. The random control selects exactly the same count per category
with a fixed seed, independently for every target fold.

## Variants and gate

| variant | folds 1-2 | folds 3-4 |
|---|---|---|
| baseline (weight 1) | checked | unchecked: gate failed |
| hard 2x | checked | unchecked: gate failed |
| hard 4x | checked | unchecked: gate failed |
| random 2x | checked | diagnostic only |
| random 4x | checked | diagnostic only |

Continue the winning hard weight to folds 3-4 only if folds 1 and 2 have the same
delta sign and mean pooled PR-AUC delta >0.001, or mean macro-category PR-AUC delta
>0.001 without a catastrophic category regression (>0.02 absolute in a category).

## Noise and controls

Baseline fold spread and per-category fold spread are reported. Because only one
seed is affordable, effects near 0.001 remain provisional. Random-negative
reweighting at identical coverage is the negative control for generic loss-scale
or extra-negative-weight effects.

## Outputs

All scripts, logs, predictions, metrics, and the final report remain under this
directory. Nothing is written to a repository `validation/` directory, submitted,
committed, pushed, or published.

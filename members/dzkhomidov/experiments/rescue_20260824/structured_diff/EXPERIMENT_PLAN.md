# Structured attribute-difference gate

Claim: prepending label-free pair-level states for shared structured attributes
improves fixed-seed `rubase_llmfull_e2` hand fine-tuning by more than `0.001`
mean macro-category PR-AUC on both folds 01 and 02.

Inputs are read-only:

- `/home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet`
- `/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2`

Variants, initially unchecked:

1. `baseline`: unchanged `name | category | attrs` text.
2. `structured`: the same text with a prepended, pair-level label-free token block.
3. `shuffled`: identical token coverage but token blocks deterministically permuted
   between rows (negative control).

The block reports `совпал`, `различен`, or `неизвестно` for brand, model/article,
colour, material, quantity, volume, weight, plus conflicts for any other shared
numeric attribute key. It is derived from names/attributes only; target is never
passed to extraction.

Stage 1: folds 01-02, seed 20260814, two epochs, max length 224, symmetric pair
order augmentation/evaluation. Gate to folds 03-04 only if the structured delta
has the same positive sign on both stage-1 folds and mean delta is greater than
0.001 macro-category PR-AUC. Report pooled PR-AUC, macro-category PR-AUC,
per-category values, state coverage, runtime, host/GPU/PID, and the shuffled
negative control.

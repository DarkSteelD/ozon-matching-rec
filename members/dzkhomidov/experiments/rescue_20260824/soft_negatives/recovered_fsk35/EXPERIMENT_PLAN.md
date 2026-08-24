# OOF-only ambiguous-negative downweighting

Claim: mildly reducing the BCE contribution of train-only ambiguous negatives
improves mean macro-category PR-AUC by more than 0.001 on folds 01 and 02, with a
positive delta on each fold.

Fixed inputs:

- hard-label text data: `/home/dzkhomidov/matching-work/data/hand_pairs.parquet`
- label-free hardness signal: OOF predictions from `ce_rubase_e2_len224`
- initial checkpoint: `rubase_llmfull_e2`
- seed 20260814, max length 224, two epochs, batch 192, LR 2e-5

For each held-out fold independently, selection sees only the other three training
folds. Within every category, labelled negatives are ranked by their single OOF CE
score; the top 10% form the ambiguous zone. No held-out score, label, prevalence,
threshold, or metric is used. The random control samples exactly the same negative
count in every training category with a fixed seed.

Stage-1 variants, initially unchecked:

| variant | selected-negative loss weight |
|---|---:|
| exact rerun baseline | 1.00 |
| OOF-hard mild | 0.75 |
| matched random mild | 0.75 |
| OOF-hard stronger | 0.50 |
| matched random stronger | 0.50 |

Primary metric: mean macro-category PR-AUC against hard labels. Secondary:
pooled PR-AUC, per-category AP, prediction prevalence/mean, Brier score and ECE.
The archived exact-recipe baseline is also scored to audit rerun drift. Promote the
best OOF-hard dose to folds 03-04 only if its macro delta exceeds +0.001 and is
positive on both folds; the matched-random result must not explain the gain.

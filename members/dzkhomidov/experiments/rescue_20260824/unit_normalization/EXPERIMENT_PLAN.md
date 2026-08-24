# Deterministic in-place unit normalization

## Measurable claim

Replacing numeric quantity/unit spans in names and attributes with deterministic,
semantically equivalent canonical spans improves pooled or macro-category PR-AUC by
more than 0.001 on folds 1-2, with the same delta sign on both folds. The change
must particularly improve rows containing edited unit spans.

## Controlled recipe

- Data: original `hand_pairs.parquet`, 365,654 rows and unchanged four-fold split.
- Initial model: two-epoch LLM-pretrained `rubase_llmfull_e2`.
- All variants: names + category + attrs, max_len 224, two epochs, batch 192,
  AdamW LR 2e-5, seed 20260814, identical row ordering and updates.
- `baseline`: byte-for-byte original name/attribute text.
- `normalized`: only matched spans are replaced; every output string has exactly
  the original character length. Nothing is prepended and no text grows.
- `corrupt`: edits the exact same spans and same number of characters as
  `normalized`, but deterministically changes the canonical numeric value. This is
  the negative control for token-boundary/edit-position effects.

Covered families: g/kg, ml/l, dimension separators and units, counts, and fashion
size notation. Folds 3-4 run only if normalized exceeds baseline by >0.001 in
pooled or macro-category PR-AUC with the same positive sign on folds 1 and 2 and no
category regression below -0.02.

## Evidence

Persist raw logs, exact commands, prediction files, normalization coverage and
examples, per-fold/per-category metrics, and unit-bearing slice metrics under this
directory. No output enters a repository validation directory.

## Status

Baseline, normalized, and corrupted-control are checked on folds 1-2. The gate
failed with the same negative normalized delta on both folds; folds 3-4 remain
unchecked by design.

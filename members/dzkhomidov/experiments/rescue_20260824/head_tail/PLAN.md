# Head+tail packing at max_len=384

## Claim and gate

At identical 384-token compute, preserving both the head and tail of each item
text improves hard-label macro category PR-AUC over ordinary prefix truncation
by more than `+0.001` on both fold_01 and fold_02.

- Stage 1: prefix, headtail, and equal-length middle negative control on folds 1–2.
- Promote all three to folds 3–4 only if `headtail - prefix > +0.001` on each
  gate fold (same positive sign).
- No packing ratio, seed, or slice definition is tuned after fold metrics.

## Fixed packing

Rubert pair format has 381 content tokens plus `[CLS] A [SEP] B [SEP]`.
Content budget allocation exactly matches HuggingFace `longest_first`: preserve
the short side when possible; otherwise 190/191 tokens, with the longer side
receiving the odd token.

- `prefix`: first `k` tokens from each side (exact standard truncation baseline).
- `headtail`: `ceil(k/2)` first and remaining last tokens from each side.
- `middle`: centered contiguous `k` tokens from each side. This is the matched
  equal-length negative control.
- Rows fitting the budget are byte-identical across all modes.

## Frozen training config

- data `/home/dzkhomidov/matching-work/data/hand_pairs.parquet`
- init/model `/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2`
- max_len 384, epochs 2, batch 256, lr 2e-5, seed 20260814
- category and attrs included, no symmetrization
- host `avi-gn-fsk35`, physical GPU0 only after live ownership check
- primary macro mean of category AP; secondary pooled and truncated-slice AP
- noise floor: historical same-architecture variation about 0.0005; gate 0.001

## Slices fixed before scoring

`no_trunc`, `any_trunc`, `both_trunc`, total token lengths `382–512`, `513–768`,
and `>768`; per-category AP with explicit focus on Обувь and Галантерея.

## Matrix

| mode | fold_01 | fold_02 | fold_03 | fold_04 |
|---|---|---|---|---|
| prefix | checked | checked | not run: gate failed | not run: gate failed |
| headtail | checked | checked | not run: gate failed | not run: gate failed |
| middle control | checked | checked | not run: gate failed | not run: gate failed |

Exact commands are in `COMMANDS.md` after launch, with PIDs and logs.

Final gate result: **failed**. Headtail deltas were `+0.000002` and
`+0.000023`, both below the preregistered `+0.001` threshold.

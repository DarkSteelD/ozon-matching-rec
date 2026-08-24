# Head+tail packing at max_len=384 — stage-1 report

## Decision

**Reject / do not promote.** Head+tail packing is indistinguishable from the
ordinary prefix baseline on the primary metric. It improved macro category AP
by only `+0.000002` on fold_01 and `+0.000023` on fold_02, far below the frozen
`+0.001`-per-fold gate and the historical `~0.0005` noise scale. Folds 3–4 were
therefore not run.

The matched middle-segment negative control lost about `-0.011` macro AP on
both folds. The setup can detect a material packing effect; it did not detect a
useful head+tail effect.

## Primary held-out results

Mean category AP is primary; pooled AP is secondary.

| fold | mode | macro category AP | delta vs prefix | pooled AP | pooled delta |
|---|---|---:|---:|---:|---:|
| fold_01 | prefix | 0.788354 | — | 0.839346 | — |
| fold_01 | headtail | 0.788356 | +0.000002 | 0.839719 | +0.000373 |
| fold_01 | middle | 0.777685 | -0.010669 | 0.830124 | -0.009222 |
| fold_02 | prefix | 0.795888 | — | 0.850193 | — |
| fold_02 | headtail | 0.795912 | +0.000023 | 0.850027 | -0.000167 |
| fold_02 | middle | 0.784422 | -0.011466 | 0.839624 | -0.010569 |
| mean | prefix | 0.792121 | — | 0.844770 | — |
| mean | headtail | 0.792134 | +0.000013 | 0.844873 | +0.000103 |
| mean | middle | 0.781053 | -0.011068 | 0.834874 | -0.009896 |

Category-resampling bootstrap for the mean headtail-minus-prefix macro delta:
95% interval `[-0.000389, +0.000446]`. It crosses zero and remains well below
the promotion threshold.

## Truncation coverage and slices

Of 365,654 rows, 48,092 (`13.15%`) require any truncation and 22,402 (`6.13%`)
truncate both sides. Total-token quantiles before special tokens are median 232,
p90 405, p95 452, p99 572. Thus 86.85% of rows are token-identical across all
three packings; only the truncated minority can carry a direct packing effect.

Headtail-minus-prefix macro category AP:

| slice | fold_01 | fold_02 | note |
|---|---:|---:|---|
| all | +0.000002 | +0.000023 | primary; null |
| any truncation | +0.001502 | +0.010256 | positive, but diluted and category-unstable |
| both sides truncated | -0.007934 | +0.016106 | sign reversal |
| total length 382–512 | +0.001666 | +0.009950 | positive within truncated region |
| total length 513–768 | -0.013221 | +0.006651 | sign reversal |
| total length >768 | +0.002273 | +0.283333 | only 21/14 rows; not interpretable |
| no truncation | -0.000321 | -0.000390 | weights differ because training rows differ |

The truncated-slice signal is not stable enough to override the preregistered
all-row gate. Pooled AP on `any_trunc` was `+0.002048` on fold_01 but
`-0.000514` on fold_02, reinforcing that conclusion.

## Requested categories

Only 2.58% of Обувь rows and 2.95% of Галантерея и аксессуары rows truncate,
so these categories have little direct exposure to the intervention.

| category / slice | fold_01 delta | fold_02 delta |
|---|---:|---:|
| Обувь, all | -0.002698 | +0.001784 |
| Обувь, any truncation | -0.021016 (133 rows) | +0.076261 (115 rows) |
| Галантерея и аксессуары, all | -0.000427 | -0.001330 |
| Галантерея и аксессуары, any truncation | -0.001683 (137 rows) | +0.033498 (124 rows) |

The small truncated samples swing strongly across folds. On the stable all-row
view, shoes are mixed and accessories are negative on both folds.

## Design and controls

- Data: read-only `hand_pairs.parquet`; hard target; fold_01 and fold_02 only.
- Init: read-only `rubase_llmfull_e2`; max_len 384, epochs 2, batch 256,
  learning rate 2e-5, seed 20260814.
- Pair budget: 381 content tokens plus `[CLS] A [SEP] B [SEP]`; identical
  longest-first side allocation for every arm.
- Prefix retains the first allocated tokens; headtail retains half head and half
  tail; middle retains a centered contiguous segment of identical length.
- Native HuggingFace prefix packing matched on 128 checked rows. The first 1,000
  fitting rows were byte-identical across packings.
- No result, slice, ratio, or seed was inspected or tuned before all three
  stage-1 arms completed.
- Runtime was matched: prefix 1257.5 s, headtail 1255.3 s, middle 1256.2 s on
  physical GPU0 (about 42.7–45.0 GiB, ~3.58 train steps/s).

## Interpretation

At max_len=384, preserving tails is not a deployable global improvement for
this Rubase hard-pair training recipe. The likely reason is exposure: only 13%
of examples change at all, while the categories explicitly requested here
truncate in under 3% of rows. There may be a real effect inside moderately long
rows, but it is inconsistent across folds/categories and does not move the
competition metric beyond noise. A future test would need a separately
preregistered long-text routing policy or a shorter token budget; selecting one
now from these slices would be post-hoc leakage.

## Evidence and reproducibility

- Frozen plan/config: `PLAN.md`, `config.json`
- Exact commands/PIDs: `COMMANDS.md`
- Minimal code: `prepare_tokens.py`, `train.py`, `score.py`
- Logs: `logs/prepare.log`, `logs/{prefix,headtail,middle}_folds12.log`
- OOF predictions: `preds/{prefix,headtail,middle}/fold_0{1,2}.csv`
- Metrics: `metrics/primary_metrics.csv`, `metrics/slice_metrics.csv`,
  `metrics/category_metrics.csv`, `metrics/metrics.json`
- Coverage: `tokens/coverage.parquet`, `tokens/coverage_summary.json`

Scoring initially hit a pandas attribute/name collision (`mode`); only bracket
column access was corrected before rerunning. Predictions and metric formulas
were unchanged. Final `score.py` SHA256 is
`56ddcd108c7be001c36b8f1bc1a77ea8c847ca64557c6bf850944347c384386a`.

GPU0 was verified free at the end (`4 MiB`, `0%`). No validation artifact,
submission, commit, push, or tracked project file was written.

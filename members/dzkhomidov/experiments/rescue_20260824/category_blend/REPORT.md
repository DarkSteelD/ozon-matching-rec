# Category-specific cross-architecture blending — LOFO report

## Outcome

**Positive for the five-channel OOF blend.** Category-specific coarse weights
with fixed 75% shrinkage toward a nested global blend improve aggregate held-out
macro category PR-AUC by **+0.002614** (`0.803123 -> 0.805736`). All four outer
fold deltas are positive and the paired category-bootstrap 95% interval is
`[+0.001414, +0.004070]`, so this passes the predeclared gate.

**Inconclusive for the deployable two-model proxy.** On `ce_priodistill +
ce_mdeb_len224`, the same category shrinkage improves the nested global blend by
`+0.000529` (`0.801065 -> 0.801595`) and is positive on all folds, but its 95%
interval is `[-0.000013, +0.001331]`. It does not clear the noise gate.

No repository, `validation/`, container, commit, or submission was changed.

## Protocol

- 365,654 OOF rows, four folds, 20 categories; input SHA256
  `2dd369a0032891246c9dd0181414b7506a5c33e93e622f6bed2112ba1bf84083`.
- Five arms: rubase384, e5-288, mdeb224, zero-shot blend, priodistill.
- Predictions converted to percentile ranks separately inside each
  fold/category; no labels are used by this normalization.
- For every held-out fold, model/weight selection uses only the other three.
- Coarse simplex grid step 0.25 (70 five-model vectors). Selection maximizes the
  mean category AP over the three training folds.
- Primary metric: mean of 20 category APs after concatenating the four held-out
  predictions. Secondary: mean of the four held-out fold macro APs.
- Noise: held-out fold spread plus 10,000 paired category-bootstrap draws.
- Controls: 10 label-permutation selections and 25 random category-weight runs.

## Five-model results

| Variant | Aggregate macro AP | Delta vs equal | Delta vs nested global | 95% CI vs nested global |
|---|---:|---:|---:|---:|
| equal global rank blend | 0.797443 | — | -0.005680 | — |
| nested global coarse blend | 0.803123 | +0.005680 | — | — |
| category best single | 0.799303 | +0.001860 | -0.003819 | [-0.005664, -0.002087] |
| category coarse, raw | 0.804182 | +0.006739 | +0.001059 | [+0.000049, +0.002538] |
| category coarse, shrink 25% | 0.802016 | +0.004573 | -0.001107 | [-0.002194, -0.000089] |
| category coarse, shrink 50% | 0.804818 | +0.007375 | +0.001695 | [+0.000820, +0.002626] |
| **category coarse, shrink 75%** | **0.805736** | **+0.008293** | **+0.002614** | **[+0.001414, +0.004070]** |

Held-out deltas of shrink75 versus the strong nested global blend:

| Fold | Global | Category shrink75 | Delta |
|---|---:|---:|---:|
| fold_01 | 0.800681 | 0.804157 | +0.003476 |
| fold_02 | 0.807674 | 0.809577 | +0.001903 |
| fold_03 | 0.801541 | 0.805021 | +0.003479 |
| fold_04 | 0.804154 | 0.806042 | +0.001888 |

The gain is broad: 18/20 categories improve. The largest deltas are Одежда
`+0.012034`, Обувь `+0.007766`, Мебель `+0.006549`, Галантерея `+0.005177`, and
Ювелирные изделия `+0.004573`. The top two categories contribute only 37.2% of
the total positive category delta, so the aggregate is not a one-category
accident. Бытовая химия (`-0.000617`) and Хобби (`-0.000269`) are the two small
negatives.

The global coarse selector uses rubase/mdeb/distill weights
`0.25/0.25/0.50` on three outer rounds; fold_01 substitutes e5 for mdeb.
Zero-shot receives zero global weight. Category weights are stable (identical
on all outer rounds) in 7/20 categories and vary among two or three nearby grid
points elsewhere. Shrinkage is therefore doing useful variance control: raw
category weights help, but 75% category / 25% global is better.

Controls behave as expected: random category weights average `0.789351 ±
0.002298`; selection after within-fold/category label permutation averages
`0.742962 ± 0.017091`. Neither creates a false gain.

## Two-model container proxy

This follow-up uses only `ce_priodistill` and `ce_mdeb_len224`. Fixed 2:1 means
distill weight 2/3 and mdeb weight 1/3. The nested global selector independently
chooses distill 0.75 / mdeb 0.25 on every outer round.

| Variant | Aggregate macro AP | Delta vs nested global |
|---|---:|---:|
| fixed distill:mdeb 2:1 | 0.800176 | -0.000890 |
| nested global distill:mdeb 3:1 | 0.801065 | — |
| category coarse shrink75 | 0.801595 | +0.000529 |

Per-fold category-shrink deltas are `+0.000624`, `+0.000295`, `+0.000681`, and
`+0.000517`. Raw category weights differ from the global weight in only 16/80
category-fold choices:

- Одежда and Обувь: all four rounds choose mdeb 0 instead of 0.25; after
  shrinkage the deployed proxy weight is mdeb 0.0625 / distill 0.9375.
- Красота и гигиена: all four choose mdeb 0.50; shrunk weight is 0.4375.
- Галантерея and Строительство: change on two of four rounds each.

The aggregate proxy gain is driven mainly by Одежда (`+0.006412`) and Обувь
(`+0.003946`); 15 categories are exactly unchanged. This is a clear mechanism
(mdeb hurts the two local fashion categories relative to distill), but it is
also the part of local CV known to transfer poorly to the hidden test.

The proxy is **not exact**: the OOF column is mdeb length 224, while the current
container uses mdeb length 160; its student/checkpoint and raw-score blending
also need not equal `ce_priodistill` and rank blending. Therefore this result
does not authorize a container change or submission.

## Interpretation and limits

The supported mechanism is architecture usefulness varying by category, with
partial pooling toward one global blend reducing selection variance. The
five-model result is large enough locally and survives strict LOFO and controls.
The available two-model deployment path is much weaker and statistically
inconclusive, so the actionable next check is to generate exact full-fold OOF
for the actual container student plus mdeb160, then rerun this unchanged proxy
script before touching inference code.

TP/FP/FN changes are not defined here: this is a rank-only PR-AUC experiment
with no binary threshold or row-level rule. No claim about classification counts
is made.

Unchecked: exact container models, raw-score rather than rank blending,
duplicate-group bootstrap (group identifiers are absent), and hidden-test
transfer. Checked: every planned five-model arm/fold, strong nested global
baseline, random/permutation controls, fixed shrinkages, two-model proxy, manual
independent AP recomputation (maximum absolute discrepancy `3.33e-16`).

## Artifacts and reproduction

- `experiment.py`: five-model LOFO experiment and controls.
- `two_model_proxy.py`: deployable-proxy LOFO experiment.
- `analyze.py`: per-category analysis and independent AP check.
- `artifacts_v2/metrics.csv`, `fold_comparison.csv`, `category_metrics.csv`.
- `artifacts_v2/selected_weights.csv`, `controls.csv`.
- `artifacts_v2/two_model_metrics.csv`, `two_model_category_metrics.csv`,
  `two_model_weights.csv`, `two_model_summary.json`.
- `artifacts_v2/heldout_predictions.parquet` and
  `two_model_heldout_predictions.parquet`.
- Exact commands are in `PLAN.md`; stdout is in `run_v2.log`,
  `analysis_v2.log`, and `two_model.log`.

Runtime: 469.8 seconds for the full five-model run (CPU only); 18.6 seconds for
the two-model proxy. No new dependency was installed.

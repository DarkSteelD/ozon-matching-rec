# Fashion hidden-like stress audit

## Outcome

The audit found reproducible local failure mechanisms, but **no saved independent
channel repairs any collapsed slice in all four folds**. The useful result is a
stress panel, not a new rule or blend.

- **Positive:** several pre-defined mechanisms consistently reduce local AP.
- **Negative:** neither zero-shot LLM nor name KNRM adds positive marginal rank
  utility on a collapsed slice in all four folds at the frozen 90/10 blend.
- **Negative:** no hand-written rule meets the acceptance contract; none is proposed.
- **Inconclusive:** these local mechanisms do not identify the hidden candidate
  mixture because hidden rows are unavailable.

All 78,398 fashion OOF pairs, four folds, 9,896 positives, and 68,502 negatives
were checked. The frozen baseline is `final_stack_all`; no slice was defined from
a mixed score. Runtime was about 2m19s CPU-only after parser/bootstrap self-checks.

## Strong local collapse mechanisms

Numbers below are medians over four folds. Every listed slice has both classes and
at least 100 rows per fold.

| Category | Mechanism | Coverage | Baseline AP | AP delta vs category |
|---|---|---:|---:|---:|
| Обувь | text length asymmetric (>=3x) | 12.7% | 0.389 | -0.187 |
| Обувь | exact normalized names | 5.4% | 0.436 | -0.138 |
| Галантерея | comparable size mismatch | 3.1% | 0.566 | -0.170 |
| Галантерея | long text, 801–1600 chars | 12.7% | 0.620 | -0.124 |
| Ювелирка | article present on only one side | 70.8% | 0.519 | -0.092 |
| Ювелирка | comparable size mismatch | 37.7% | 0.535 | -0.088 |
| Ювелирка | text length asymmetric (>=3x) | 39.6% | 0.525 | -0.086 |

Одежда has no four-fold slice below the frozen -0.05 median-collapse criterion.
Its worst stable mechanisms are exact names (-0.045) and comparable-size mismatch
(-0.031). Exact names are therefore a hard slice, not a positive match rule: in
Обувь and Галантерея they rank substantially worse than the category as a whole.

These are correlations with named mechanisms. Extraction itself has not been
human-validated, so this does not establish that size/article parsing is causal.

## Marginal channel utility

The fixed test ranks each score inside category-fold and evaluates
`0.9 * baseline_rank + 0.1 * alternative_rank`. There was no weight sweep.

| Category | zero-shot LLM mean delta | KNRM mean delta | positive folds |
|---|---:|---:|---:|
| Обувь | -0.0109 | -0.0248 | 0/4, 0/4 |
| Одежда | -0.0102 | -0.0049 | 0/4, 1/4 |
| Галантерея | -0.0077 | -0.0038 | 0/4, 0/4 |
| Ювелирка | -0.0047 | -0.0250 | 1/4, 0/4 |

On the complete category slices, the 1,000-replicate connected-component
bootstrap confirms negative median deltas. The intervals closest to zero are KNRM
for Одежда [-0.0129, +0.0016], KNRM for Галантерея [-0.0069, +0.0003], and LLM
for Ювелирка [-0.0078, +0.0017]; all remain inconclusive-to-negative, never
positive. The other five intervals are wholly below zero.

The non-independent CE diagnostics do show small local gains:

- `ce_priodistill`: +0.00315 Обувь, +0.00174 Одежда, +0.00102 Галантерея,
  all positive in 4/4 folds; Ювелирка +0.00031 and 2/4 folds.
- fashion specialist: +0.00279 Обувь in 4/4 folds, but mixed elsewhere.

These channels share training lineage with the baseline and therefore do not solve
the requested independent-signal problem. Their deltas are also on the scale of
fold variation, so they are correlation-level evidence only.

The positive control (`target` as score) reaches AP 1.0. Permuting an alternative
before the same 90/10 blend degrades AP by about -0.034 on average, confirming that
the pipeline detects dilution of a strong baseline; it is a negative control, not
a zero-centred noise estimate.

## Prevalence-shift control

Public priors come from the constant/all-ones probe, not from fitting to model LB
outcomes. Positives and negatives were reweighted while preserving local baseline
ranking.

| Category | Local AP | Expected at public prior | Public baseline AP | Public - expected |
|---|---:|---:|---:|---:|
| Обувь | 0.576 | 0.402 | 0.097 | -0.306 |
| Одежда | 0.596 | 0.425 | 0.111 | -0.314 |
| Галантерея | 0.740 | 0.472 | 0.242 | -0.229 |
| Ювелирка | 0.615 | 0.239 | 0.287 | +0.047 |

Thus prevalence alone cannot explain Обувь, Одежда, or Галантерея. Ювелирка is
different: its public AP is slightly above the prevalence-shift expectation, so it
does not show the same transfer catastrophe after this control.

This is a weighted local counterfactual, not an estimator with hidden candidate
features; no causal or calibrated hidden-performance claim is made.

## Human audit

203 fashion audit records join to OOF: 173 decisive labels (`0/1`) and 30 unsure
labels (`-1`). Among decisive labels, 139 differ from the original target. Counts
by category are:

| Category | records | decisive | unsure | flips |
|---|---:|---:|---:|---:|
| Обувь | 74 | 63 | 11 | 46 |
| Одежда | 58 | 51 | 7 | 45 |
| Галантерея | 62 | 51 | 11 | 44 |
| Ювелирка | 9 | 8 | 1 | 4 |

The audited set was selected from disputed/model-error queues. Its very high flip
rate is strong evidence that these queues contain label problems, but it cannot
estimate population label noise or hidden composition. AP on the selected audit
rows is saved only as a selection-biased diagnostic.

## Grouping, controls, and limitations

The local pair graph has 73,611 connected components; 3,770 have multiple edges
and the largest has 15. Bootstrap resamples these components, so repeated related
pairs do not receive independent weight.

Checked:

- all four categories and folds;
- every frozen exact/near-name, missing-attribute, brand/model/article,
  comparable-size, numeric-conflict, text-length, and baseline-only score slice;
- four saved alternative channels at one frozen blend weight;
- positive/permutation controls;
- 1,000-replicate grouped bootstrap on the mandatory category slices and the
  frozen candidate-confirmation gate;
- public-prior reweighting and decisive human audit labels.

Unchecked or intrinsically unavailable:

- hidden row features, hidden groups, and hidden mechanism coverage;
- manual validation of brand/model/article/size extraction quality;
- alternative blend weights, learned gates, new models, images, or seller signals;
- confidence intervals at the actual hidden sample composition/size;
- population conclusions from the actively selected human audit queue.

Because zero collapse slices passed the four-fold independent-channel gate, no
post-hoc slice bootstrap was run beyond the mandatory category controls. This is a
negative result for the saved channels, not proof that no future independent data
source can help.

## Reproduction and artifacts

Exact command:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python -u \
  /home/dzkhomidov/matching-work/rescue_20260824/fashion_stress/audit.py \
  --output /home/dzkhomidov/matching-work/rescue_20260824/fashion_stress \
  --bootstrap 1000 --seed 20260824
```

Primary evidence:

- `slice_metrics.csv`: coverage, class counts, and AP per fold/category/slice.
- `blend_metrics.csv`: alternative and fixed-blend marginal utility.
- `bootstrap_metrics.csv` and `bootstrap_candidates.csv`: grouped confirmation.
- `prevalence_shift.csv`, `label_audit_metrics.csv`, `control_metrics.csv`.
- `row_features.parquet`: auditable row-level mechanisms without raw text copies.
- `summary.json`, `conclusions.json`, and `run.log`.

Repository SHA was `2da459984a1207677ff9eb863ca28589027a4bc3`; repository status remained clean.
No `validation/`, commit, submit, network submission, or GPU was used.

The exact full run was repeated with the same seed. The combined hash over the
nine generated metric/row artifacts matched:
`0cf03768fe3d12b3746298b4395901c75bad19f338940d5a3fa8d59195c6115f`.

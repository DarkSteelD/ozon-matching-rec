# Long-context teacher hypothesis — report

## Conclusion

**Negative for the preregistered gate; small sub-gate signal remains
inconclusive.** Raising max length above 384 improved pooled PR-AUC on both
initial folds, but only by +0.000713/+0.000411 at len448 and
+0.000586/+0.000785 at len512. Neither candidate reached the required +0.001
on both folds. Macro-category gains were also too small/mixed, and the worst
category did not improve consistently. Folds 3-4 were therefore not run.

This is a teacher/upper-bound test only. No container or submission was built.

## Setup and audit trail

- Repo evidence SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`
- Init: recovered `rubase_llmfull_e2`, model SHA256
  `0a90825fbeb584fda7dfb3faded702b302b338aa3b0d8e4dc8217be77d0399f6`
- Data: recovered `hand_pairs.parquet`, SHA256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`
- Text: exact recovered `name | category | attrs` construction
- Fixed: two epochs, seed 20260814, effective batch 256, LR 2e-5,
  AdamW + OneCycleLR, frozen fold definitions
- Memory-safe implementation: micro-batch 128 with two-step gradient
  accumulation for every variant; optimizer steps and example order fixed
- Host/resource: `avi-gn-fsk35`, H100 physical GPU 3; runner PID 1686744,
  initial child PID 1686747
- GPU 3 was idle/no co-tenant at launch. It was used instead of initially
  assigned GPU 2 because the staged runner retained index 3; the coordinator
  explicitly approved continuing the exact process.
- Scratch only; repository, checkpoints, data, and all `validation/` paths were
  not modified. No submit, push, commit, or process kill.

## Truncation coverage

| scope | rows | >224 | >384 | >448 | >512 |
|---|---:|---:|---:|---:|---:|
| all | 365,654 | 52.99% | 13.15% | 5.53% | 1.97% |
| fold 1 | 91,157 | 53.12% | 13.27% | 5.51% | 1.94% |
| fold 2 | 91,474 | 52.90% | 13.09% | 5.60% | 2.03% |

Coverage beyond 384 is largest for Аптека (43.08%), Мебель (35.33%),
Электроника (34.02%), and Бытовая техника (32.19%). It is small for the hidden
fashion problem: Обувь 2.58%, Одежда 2.38%, Галантерея 2.95%, Ювелирные
изделия 8.41%. Long context cannot plausibly be the direct fashion rescue.
Full category/fold counts are in `truncation_coverage.json`.

## Primary results

PR-AUC is pooled within each fold. Macro is the mean of the 20 category PR-AUCs.

| variant | fold | PR-AUC | delta vs fresh 384 | macro | macro delta | status |
|---|---|---:|---:|---:|---:|---|
| len224 | 1 | 0.835688 | -0.004410 | 0.779333 | -0.009976 | checked negative control |
| len224 | 2 | 0.845603 | -0.004371 | 0.786460 | -0.009302 | checked negative control |
| len384 | 1 | 0.840098 | 0 | 0.789309 | 0 | checked baseline |
| len384 | 2 | 0.849975 | 0 | 0.795762 | 0 | checked baseline |
| len448 | 1 | 0.840811 | +0.000713 | 0.789642 | +0.000333 | checked, gate failed |
| len448 | 2 | 0.850386 | +0.000411 | 0.796044 | +0.000282 | checked, gate failed |
| len512 | 1 | 0.840684 | +0.000586 | 0.789581 | +0.000272 | checked, gate failed |
| len512 | 2 | 0.850760 | +0.000785 | 0.796924 | +0.001162 | checked, gate failed |

| variant | two-fold mean ± population SD | mean macro | runtime | observed peak GPU memory |
|---|---:|---:|---:|---:|
| len224 | 0.840646 ± 0.004958 | 0.782896 | 858 s | 15.4 GiB |
| len384 | 0.845036 ± 0.004938 | 0.792535 | 1,405 s | 23.6 GiB |
| len448 | 0.845598 ± 0.004787 | 0.792843 | 1,668 s | 27.5 GiB |
| len512 | 0.845722 ± 0.005038 | 0.793252 | 1,961 s | 30.2 GiB |

The negative control reproduces the known ordering cleanly: fresh len384 beats
len224 by +0.004410/+0.004371. Absolute fresh scores are about 0.0011-0.0015
below the historical direct-batch run, consistently with the changed numerical
path; all gate comparisons use the paired fresh runs.

## Mechanism check

The small candidate improvement is larger on rows actually truncated at 384:

| variant | fold | delta on <=384 rows | delta on >384 rows |
|---|---|---:|---:|
| len448 | 1 | +0.000343 | +0.003929 |
| len448 | 2 | +0.000233 | +0.001684 |
| len512 | 1 | +0.000349 | +0.002513 |
| len512 | 2 | +0.000248 | +0.005201 |

This supports a real long-text mechanism, but it affects only about 13% of the
data and does not lift the aggregate enough. Finer 385-448 / >448 slices are
mixed across folds, so this is correlation-level evidence, not a robust
category mechanism. Category-level coverage vs gain is weak: for len512,
Pearson r=0.424 (p=0.063), Spearman r=0.272 (p=0.246); len448 has essentially
no relationship. Электроника is the clearest repeated beneficiary
(+0.00570/+0.00302 at len448; +0.00134/+0.00962 at len512), while the worst
category Обувь is mixed or worse.

## Checked / unchecked and limitations

Checked: all four preregistered variants on folds 1-2, exact truncation coverage,
fold/category metrics, negative control, long-vs-short subset diagnostic,
runtimes and GPU memory.

Unchecked by gate: folds 3-4 for len384/448/512, additional seeds, downstream
ensemble marginal value, and container latency. No seed-level noise estimate
was run; the observed candidate deltas are below the predeclared practical
threshold and far below the between-fold SD, so they are not promoted as an
improvement.

## Artifacts

- `metrics_phase1.json`: complete fold and per-category scores
- `subset_metrics.json`: <=384, >384, 385-448, and >448 diagnostics
- `truncation_coverage.json`: exact full/fold/category length coverage
- `preds/`: eight fold prediction CSVs
- `logs/phase1.log`: complete training log
- `PLAN.md`, `COMMANDS.md`, scripts: exact plan and reproduction

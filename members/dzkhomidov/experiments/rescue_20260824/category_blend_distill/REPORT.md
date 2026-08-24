# Category-blend teacher distillation into rubase

## Outcome

**Negative / gate failed.** Replacing 10 percentage points of the existing
v3cal teacher component with the calibrated category-shrink75 teacher does not
improve the rubase student consistently:

| Fold | exact v3cal baseline | global10 control | category10 | category − baseline | category − global |
|---|---:|---:|---:|---:|---:|
| fold_01 | 0.799520 | 0.799360 | 0.799374 | **-0.000146** | +0.000014 |
| fold_02 | 0.804345 | 0.804444 | 0.804725 | **+0.000380** | +0.000281 |
| mean | 0.801932 | 0.801902 | 0.802050 | +0.000117 | +0.000147 |

The preregistered gate required `> +0.001` with the same sign on both folds.
It fails on both magnitude and sign, so folds 03–04 were not run.

The paired category-bootstrap 95% interval is `[-0.000328, +0.000580]` versus
baseline and `[-0.000361, +0.000602]` versus global10. Secondary pooled PR-AUC
also decreases on both folds: `-0.000699` and `-0.000833` versus baseline.

No container, checkpoint, source data, repository, `validation/`, commit, or
submission was changed.

## Target construction and diagnostics

The held-out teacher file has 365,654 rows and columns `fold`, hard `target`,
`category`, plus rank-valued global/category predictions. Its values are
percentile ranks (mean approximately 0.5), so directly mixing them into v3cal
probabilities would have broken calibration.

The existing `hand_pairs_pd_v3cal.target` was verified as
`0.3*hard + 0.7*p_old`, range `0.0007..0.9993`, mean `0.256804`. Reconstruction
error was below `1e-12`. New rank teachers were isotonic-calibrated independently
inside each OOF fold, then the frozen targets were:

- baseline: existing `0.3*hard + 0.7*p_old`;
- global10: `0.3*hard + 0.6*p_old + 0.1*p_global`;
- category10: `0.3*hard + 0.6*p_old + 0.1*p_category`.

Thus the strong 30% hard-label component is unchanged, and no raw rank is mixed
with a calibrated probability. Category10 mean is `0.256799`, correlation with
baseline `0.998924`; its delta from global10 has mean `6.3e-8` and std `0.003057`.
All row keys matched exactly by `fold/id1/id2/category` before writing.

Input SHA256:

- baseline target parquet: `b9ebd015f1881c1ac58b5966233b74390a25f13bf751af9a72dafc803c106af9`;
- source OOF parquet: `2dd369a0032891246c9dd0181414b7506a5c33e93e622f6bed2112ba1bf84083`;
- held-out blend predictions: `beb578ea928fb35253022190ff180d7239369e3e2aea445391e095e471668deb`.

Full diagnostics and output hashes are in `target_diagnostics.json`.

## Training protocol

All three variants used the same:

- `DeepPavlov/rubert-base-cased` initialized from
  `/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2`;
- max length 224, category + attrs, two epochs, batch 256, lr 2e-5;
- pair-order swap augmentation and two-direction evaluation (`--sym`);
- seed `20260814`, folds 01–02, fresh initialization per fold;
- `avi-gn-fsk35`, GPU6 (H100 80 GB), approximately 26 GB peak, 100% train utilization;
- identical trainer SHA256
  `c56eba21077d54ca0a62e3b6d50e5917172ab22bdbae282771e1e2c4985e6138`.

GPU6 was live-checked at 4 MiB / 0% with no compute process before launch.
Each two-fold variant took about 15 minutes, including 129–134 seconds of
two-direction CPU tokenization. Baseline and candidate/control differ only by
the target parquet.

## Controls and mechanism

The global10 negative control is essentially tied with category10. The marginal
category signal over the global teacher is only `+0.000014/+0.000281`, with a
zero-crossing CI. This rejects the claim that category-specific teacher weights
survive the student bottleneck at the fixed 10% mix.

Across categories, category10 versus baseline is positive in only 17/40
category-fold cells. Mean gains are concentrated in Ювелирные изделия
`+0.002679`, Одежда `+0.002219`, and Бытовая техника `+0.001013`. Обувь loses
`-0.002394`, even though the direct five-model category teacher previously
improved Обувь by `+0.007766`. This inversion is useful mechanistic evidence:
the direct rank ensemble's fashion gain is not faithfully learnable by this
single rubase text student under the tested target perturbation.

TP/FP/FN are not defined because this is a rank-only PR-AUC experiment without
a classification threshold.

## Checked and unchecked

Checked: exact baseline rerun, candidate on both gate folds, global nested
negative control, hard-label macro AP, pooled AP, per-category effects, target
scale/row alignment/hashes, paired category bootstrap.

Not run by design: folds 03–04 after gate failure. Unchecked: other mix weights,
other seeds, a larger/different student, raw-score blending, hidden-test
transfer. The negative result applies to the preregistered 10% rubase recipe;
it does not prove that every possible distillation method is impossible.

## Artifacts

- `PLAN.md`: preregistration and exact commands.
- `build_targets.py`, `target_diagnostics.json`, `data/*.parquet`.
- `score.py`, `metrics_folds12.json`, `category_metrics_folds12.csv`.
- `preds/cbd_v3cal_{baseline,global10,category10}_s20260814/fold_0{1,2}.csv`.
- `baseline_folds12.log`, `global_folds12.log`, `category_folds12.log`.
- `build_targets_v2.log`, `score_folds12_v2.log`.

Source repository `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec` was clean
at final check, HEAD `5099db5df398e6aa4fec9eccdaf6959f50cfbf29`; this experiment made no repository
writes. (The shared repository HEAD advanced externally since the preceding
blend audit.)

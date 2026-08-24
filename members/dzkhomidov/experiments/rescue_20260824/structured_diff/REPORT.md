# Structured attribute-difference tokens: stage-1 report

## Conclusion

**Negative; gate failed.** Prepending explicit pair-level structured states reduced
mean macro-category PR-AUC by **0.009335** versus the unchanged-text baseline on
folds 01-02. Both fold deltas were negative. Folds 03-04 were therefore not run.

The shuffled-token negative control fell by essentially the same amount. Relative
to shuffled tokens, correct tokens changed macro PR-AUC by only **+0.000205 mean**,
with mixed fold signs (`-0.000185`, `+0.000595`). The evidence supports a text
budget/placement mechanism: a long prepended block displaces useful product text,
while the content of these exact-match states does not recover the loss.

## Provenance and controls

- Repository reference (read-only): `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- Git SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`
- Worktree status at reporting: clean.
- Training data/text: `hand_pairs_pd_v3cal.parquet`, 365,654 rows.
- Hard evaluation labels: `hand_pairs.parquet`, joined 1:1 by `(id1,id2)`.
- Initial checkpoint: `rubase_llmfull_e2`.
- Fixed configuration: seed 20260814, folds 01-02, 2 epochs, max length 224,
  batch size 192, LR 2e-5, pair-order swap augmentation and two-direction eval.
- Baseline: exact unchanged `name | category | attrs` text.
- Negative control: the same candidate block coverage, deterministically permuted
  across rows without using labels.
- Extraction positive checks in `train.py`: Unicode/`ё` normalization, missing
  state, colour conflict, and `1.5 кг == 1500 г`.
- Feature extraction receives only `attrs1/attrs2`; target is not read by it.
- No repository, validation, submission, push, or commit action was performed.

An initial candidate invocation stopped before training because its extraction
self-check exposed that numeric states compared raw strings. Baseline predictions
were already complete and unchanged. Numeric units were canonicalized, the check
passed locally, and structured/shuffled were then run from scratch with the fixed
extractor. The failed preflight is preserved in `run.log`.

## Results

Primary metric is macro mean of per-category PR-AUC. Pooled PR-AUC is secondary.

| variant | fold | macro PR-AUC | delta vs baseline | pooled PR-AUC | pooled delta | runtime |
|---|---|---:|---:|---:|---:|---:|
| baseline | 01 | 0.800875 | 0 | 0.850483 | 0 | 380.4 s |
| baseline | 02 | 0.806219 | 0 | 0.859493 | 0 | 379.1 s |
| structured | 01 | 0.791658 | -0.009217 | 0.844189 | -0.006294 | 380.7 s |
| structured | 02 | 0.796766 | -0.009453 | 0.853635 | -0.005858 | 380.2 s |
| shuffled | 01 | 0.791843 | -0.009031 | 0.844149 | -0.006334 | 379.7 s |
| shuffled | 02 | 0.796172 | -0.010047 | 0.853347 | -0.006146 | 379.8 s |
| **baseline mean** | 01-02 | **0.803547** | **0** | **0.854988** | **0** | **759.5 s** |
| **structured mean** | 01-02 | **0.794212** | **-0.009335** | **0.848912** | **-0.006076** | **760.9 s** |
| **shuffled mean** | 01-02 | **0.794008** | **-0.009539** | **0.848748** | **-0.006240** | **759.5 s** |

The structured fold-delta sample standard deviation is 0.000167; the effect is
about 56 times its cross-fold delta spread and has the same negative sign. Seed
spread was not measured, so this is not a seed-noise estimate. For the important
content-only contrast (structured minus shuffled), the mean is +0.000205, fold
signs are mixed, and sample standard deviation is 0.000552: no stable signal.

TP/FP/FN counts are not applicable because this task and gate use ranking PR-AUC
without a decision threshold. No threshold-dependent claim is made.

## Per-category structured delta

Values are means over folds 01-02; exact fold values are in
`metrics_stage1.json`.

| category | delta |
|---|---:|
| Автотовары | -0.003689 |
| Аптека | -0.011923 |
| Бытовая техника | -0.000011 |
| Бытовая химия | -0.001638 |
| Галантерея и аксессуары | -0.011252 |
| Детские товары | -0.003926 |
| Дом и сад | -0.012588 |
| Канцелярские товары | -0.003131 |
| Красота и гигиена | -0.004913 |
| Мебель | -0.022821 |
| Музыкальные инструменты | -0.000916 |
| Обувь | -0.020652 |
| Одежда | -0.022401 |
| Продукты питания | -0.007370 |
| Спорт и отдых | -0.004582 |
| Строительство и ремонт | -0.014531 |
| Товары для животных | -0.008440 |
| Хобби и творчество | -0.002010 |
| Электроника | -0.011146 |
| Ювелирные изделия | -0.018756 |

Only one of 40 category-fold deltas was positive (Бытовая техника, fold 02,
+0.001425). Fashion categories were among the most damaged, so the variant does
not support the intended rescue mechanism.

## Extraction coverage

Known coverage means both sides exposed the canonical group and a state could be
assigned. Counts and category-specific coverage are in `coverage_analysis.json`.

| group | known coverage | matched | different | unknown |
|---|---:|---:|---:|---:|
| brand | 17.34% | 30,182 | 33,213 | 302,259 |
| model/article/part number | 9.88% | 2,731 | 33,394 | 329,529 |
| colour | 38.27% | 51,236 | 88,705 | 225,713 |
| material | 30.46% | 60,896 | 50,488 | 254,270 |
| quantity | 31.98% | 39,023 | 77,908 | 248,723 |
| volume | 7.08% | 2,941 | 22,932 | 339,781 |
| weight | 22.17% | 2,546 | 78,505 | 284,603 |
| other shared numeric keys | 15.85% | 28,967 | 28,973 | 307,714 |

Coverage is sufficient to reject “the extractor never fired” as an explanation,
but most fields are unknown on most rows. Exact string matching also remains a
quality ceiling for synonym-rich material/model values.

## Resource record

- Host: `avi-gn-fsk35`.
- Baseline: physical GPU 5, PID 1698959.
- Structured: physical GPU 5, PID 1745146.
- Shuffled: physical GPU 7, PID 1745147.
- Each active model used about 20.4 GiB and held 97-100% GPU utilization.
- GPU 6 was found occupied by foreign PID 1731668 and was not touched.
- All experiment PIDs exited normally; GPUs 5 and 7 returned to 4 MiB/0%.

## Checked and unchecked

Checked: baseline, structured candidate, shuffled negative control, folds 01-02,
all 20 category metrics, hard-label 1:1 scoring, overall/category coverage, numeric
and Unicode extraction self-checks.

Unchecked by gate: folds 03-04. Also unchecked: other placements (short suffix,
separate model inputs), shorter subsets of tokens, fuzzy semantic value matching,
and repeated seeds. Therefore this closes only the tested **long prepended exact
state block**, not every possible structured-attribute representation.

Exact commands are in `COMMANDS.md`; stdout/stderr are in `baseline.log`,
`structured.log`, `shuffled.log`, `score.log`, and `run.log`.

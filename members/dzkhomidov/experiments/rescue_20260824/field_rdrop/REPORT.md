# Field dropout + matched consistency: migrated four-fold result

## Conclusion

**Positive under the predeclared fold gate, but not yet seed-replicated.**  Five
percent semantic-field dropout with matched two-view Bernoulli-KL consistency
improves pooled four-fold macro-category AP from **0.782914 to 0.783974**
(**+0.001060**).  All four fold deltas are positive.  The mean paired fold delta
is +0.001120, fold SD 0.000188, with a descriptive t interval of
[+0.000821, +0.001419].

The controls support the proposed mechanism.  An equal-character-rate contiguous
span deletion is mixed-sign and only +0.000146 pooled on folds 01–02.  Using the
same semantic field dropout but matching each first view to another row's second
view is negative (-0.000876).  The result therefore is not explained by generic
text deletion or simply adding the consistency term.

## Experimental identity

- Matching repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- Git SHA: `a1c1c58a000db9b26b4963862374cd8b8961d133` (clean at report time)
- Data: `hand_pairs.parquet`, 365,654 rows, SHA256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`
- Initialization: `rubase_llmfull_e2/model.safetensors`, SHA256
  `0a90825fbeb584fda7dfb3faded702b302b338aa3b0d8e4dc8217be77d0399f6`
- Fixed configuration: max length 224, batch 256 rows / 512 encoded views,
  two epochs, LR 2e-5, seed 20260814, component folds 01–04.
- Candidate: independent whole-field dropout at rate 0.05 in each view,
  symmetric Bernoulli-KL weight 0.1; evaluation is always clean text.
- Matched baseline: two identical clean views, same update count and batch order,
  with consistency weight forced to zero.

## Four-fold macro-category AP

| Variant | Fold 01 | Fold 02 | Fold 03 | Fold 04 | Pooled category macro | Fold mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| bce2view | 0.780343 | 0.787939 | 0.780965 | 0.784939 | 0.782914 | 0.783547 ± 0.003567 |
| field05 | 0.781550 | 0.789233 | 0.781824 | 0.786060 | 0.783974 | 0.784667 ± 0.003678 |
| Delta | +0.001207 | +0.001293 | +0.000858 | +0.001121 | **+0.001060** | +0.001120 ± 0.000188 |

The pooled number is the competition-relevant macro mean of AP computed after
concatenating OOF predictions within each of the 20 categories.  The fold mean
is shown only as a stability diagnostic.

## Mechanism controls, folds 01–02

| Variant | Fold 01 delta | Fold 02 delta | Pooled macro | Pooled delta | Status |
|---|---:|---:|---:|---:|---|
| field05 | +0.001207 | +0.001293 | 0.784955 | +0.001266 | GO |
| span05 | -0.000102 | +0.000605 | 0.783834 | +0.000146 | mixed / below gate |
| mismatch05 | -0.000002 | -0.001379 | 0.782812 | -0.000876 | negative control behaves as expected |

`span05` deletes a single contiguous span with the same 5% character budget;
`mismatch05` keeps field corruption but rolls the second-view logits by one row
inside the consistency loss.

## Category result

18 of 20 categories improve.  The two negative categories are small in magnitude.

| Category | Baseline AP | field05 AP | Delta |
|---|---:|---:|---:|
| Автотовары | 0.715814 | 0.717612 | +0.001798 |
| Аптека | 0.870560 | 0.869620 | -0.000940 |
| Бытовая техника | 0.838159 | 0.839884 | +0.001725 |
| Бытовая химия | 0.906355 | 0.906929 | +0.000574 |
| Галантерея и аксессуары | 0.712940 | 0.715452 | +0.002512 |
| Детские товары | 0.912124 | 0.913059 | +0.000934 |
| Дом и сад | 0.834548 | 0.834318 | -0.000231 |
| Канцелярские товары | 0.806242 | 0.806482 | +0.000240 |
| Красота и гигиена | 0.825303 | 0.826301 | +0.000997 |
| Мебель | 0.710657 | 0.712139 | +0.001482 |
| Музыкальные инструменты | 0.862003 | 0.863172 | +0.001169 |
| Обувь | 0.520549 | 0.521739 | +0.001189 |
| Одежда | 0.542311 | 0.543915 | +0.001604 |
| Продукты питания | 0.886985 | 0.888208 | +0.001223 |
| Спорт и отдых | 0.733615 | 0.734626 | +0.001011 |
| Строительство и ремонт | 0.803919 | 0.804061 | +0.000142 |
| Товары для животных | 0.898040 | 0.898788 | +0.000748 |
| Хобби и творчество | 0.930660 | 0.931094 | +0.000434 |
| Электроника | 0.761522 | 0.762190 | +0.000669 |
| Ювелирные изделия | 0.585978 | 0.589898 | +0.003919 |

## Calibration and fixed-threshold diagnostics

Lower is better for Brier, log loss, and ECE.  ECE uses 20 fixed equal-width
probability bins.

| Metric | Baseline | field05 | Delta |
|---|---:|---:|---:|
| Pooled Brier | 0.087933 | 0.087478 | -0.000455 |
| Pooled log loss | 0.293206 | 0.291151 | -0.002055 |
| Pooled ECE20 | 0.028538 | 0.026265 | -0.002273 |
| Category-macro Brier | 0.088587 | 0.088129 | -0.000458 |
| Category-macro log loss | 0.295217 | 0.293154 | -0.002064 |
| Category-macro ECE20 | 0.030581 | 0.028202 | -0.002379 |

At the diagnostic threshold 0.5, field05 changes TP by +4, FP by -196, FN by
-4 and F1 by +0.000872.  This threshold was not tuned and is not the target
metric, but the direction is consistent with improved calibration.

## Predeclared slices

- Field-key overlap below 0.25: +0.001061 pooled AP (354,586 rows).
- Field-count asymmetry at least 2x: +0.000970 (130,487 rows).
- Total-text length asymmetry at least 2x: +0.001161 (121,249 rows).
- Either-side empty attributes: -0.004237, but only 74 rows / 30 positives;
  this slice is too small for a stable conclusion.

## Runtime and resource accounting

The migration ran only on `avi-ix-devbox03`, physical GPU 2.  The migrated
unchecked work used about 2.16 GPU-hours of measured training time (span and
mismatch folds 01–02 plus baseline and field05 folds 03–04), with 49.4 GB peak
resident GPU memory during training.  Wrapper wall time was about 2 h 22 min,
including repeated tokenization and scoring.  GPU 2 had no compute applications
in two checks before launch and was back to 0 MiB / 0% with no compute
applications after completion.

No process was launched on `avi-gn-fsk35`.  It was used read-only to recover the
finished fold 01–02 artifacts.  All 55 recovered files passed their pre-migration
SHA256 manifest after the devbox run; finished arms were not rerun.

## Checked and unchecked

Checked:

- exact input and initialization hashes;
- same fold, seed, update count, preprocessing and evaluation for each paired arm;
- no train/eval row overlap within each fold;
- all four baseline and candidate folds, complete row coverage;
- separated span and mismatched-view controls;
- per-category, slice, calibration and threshold diagnostics;
- fsk35 release and devbox03 GPU ownership before/after the run.

Unchecked:

- a second training seed / host-level determinism repeat;
- grouped bootstrap by connected product component (component IDs are not stored
  in the experiment parquet); the interval above uses only four paired folds;
- interaction with the independently promising max-length-384 and epoch-3 arms;
- hidden-test/container transfer;
- a full-data deployable checkpoint (this experiment persisted OOF predictions,
  not final model weights).

## Evidence and reproduction

- Four-fold scores: `metrics/full4.json`
- Controls: `metrics/controls_f12.json`
- Categories/slices: `metrics/slices_full4.json`
- Calibration: `metrics/calibration_full4.json`
- Input manifest: `migration/input_sha256.txt`
- Recovered fsk35 manifest: `migration/fsk35_sha256_before.txt`
- Guarded migration command: `./run_migrated_devbox03.sh`
- Exact per-fold predictions: `preds/{bce2view,field05}/fold_0{1,2,3,4}.csv`

No validation directory, submission, commit, push, or publication was touched.

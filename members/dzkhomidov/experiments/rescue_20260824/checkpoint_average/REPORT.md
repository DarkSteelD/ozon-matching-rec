# Late-checkpoint averaging / EMA — report

## Outcome

**Negative; the folds 01–02 gate failed.** Neither deployable single-model
candidate improved hard-label macro category PR-AUC. Both candidates were
negative on both folds, so folds 03–04 were not run.

| variant | fold | macro AP | delta | pooled AP | pooled delta | status |
|---|---|---:|---:|---:|---:|---|
| final checkpoint | 01 | 0.799313 | 0 | 0.849410 | 0 | checked baseline |
| late weight average | 01 | 0.798835 | -0.000478 | 0.848934 | -0.000476 | checked |
| EMA | 01 | 0.799017 | -0.000296 | 0.849138 | -0.000272 | checked |
| early weight average | 01 | 0.794288 | -0.005025 | 0.845315 | -0.004095 | negative control |
| late prediction average | 01 | 0.798884 | -0.000429 | 0.848973 | -0.000437 | diagnostic only |
| final checkpoint | 02 | 0.804624 | 0 | 0.858432 | 0 | checked baseline |
| late weight average | 02 | 0.804240 | -0.000384 | 0.858304 | -0.000128 | checked |
| EMA | 02 | 0.804325 | -0.000299 | 0.858344 | -0.000088 | checked |
| early weight average | 02 | 0.799349 | -0.005275 | 0.854270 | -0.004162 | negative control |
| late prediction average | 02 | 0.804293 | -0.000332 | 0.858331 | -0.000101 | diagnostic only |
| **final mean** | 01–02 | **0.801969** | **0** | **0.853921** | **0** | baseline |
| **late weight mean** | 01–02 | **0.801538** | **-0.000431** | **0.853619** | **-0.000302** | negative |
| **EMA mean** | 01–02 | **0.801671** | **-0.000298** | **0.853741** | **-0.000180** | negative |
| **early average mean** | 01–02 | **0.796819** | **-0.005150** | **0.849793** | **-0.004128** | control works |
| **late prediction mean** | 01–02 | **0.801588** | **-0.000380** | **0.853652** | **-0.000269** | diagnostic negative |

The screening gate required the same positive sign on both folds and mean delta
above +0.001. The first condition already fails for every candidate. The measured
neural/cheap-pipeline noise reference is about 0.00031; late-average is somewhat
larger than it, while EMA is approximately at that floor. EMA nevertheless gives
nearly identical negative fold deltas (`-0.000296/-0.000299`, delta SD 0.000002),
so there is no evidence of an improvement hidden by opposite fold noise.

## Exact matched trajectory

Each fold was trained exactly once. The final checkpoint is the baseline; all
other states came from that same optimizer trajectory, so no seed, initialization,
sample order, dropout trajectory, target, or preprocessing difference is mixed into
the comparison.

- init: `rubase_llmfull_e2`, SHA256
  `0a90825fbeb584fda7dfb3faded702b302b338aa3b0d8e4dc8217be77d0399f6`;
- target data: `hand_pairs_pd_v3cal.parquet`, SHA256
  `b9ebd015f1881c1ac58b5966233b74390a25f13bf751af9a72dafc803c106af9`;
- hard scoring data: `hand_pairs.parquet`, SHA256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`;
- RuBERT base, max length 224, category+attrs, seed 20260814, two epochs,
  batch 256, lr 2e-5, pair-order swap augmentation and two-direction eval;
- fold 01/02 update counts: 2144/2142;
- checkpoints: 25%, 50%, 75%, 87.5%, 100%;
- late weights: arithmetic mean of 75%, 87.5%, 100%;
- EMA: initialized at 50%, per-update decay 0.995;
- early negative control: arithmetic mean of 25%, 50%, 75%;
- prediction-average diagnostic: mean predictions from 75%, 87.5%, 100%.

Trainer SHA256:
`1d41a8995d5f492d38dc33d7f9cf1cb063ee1b133a4f735caadab6dda9f92348`.
Repository reference was clean at HEAD
`5099db5df398e6aa4fec9eccdaf6959f50cfbf29`; it was not modified.

## Controls, calibration and mechanism

The early average loses `-0.00515` macro AP and is negative in all 40
category-fold cells. This is a successful sensitivity control: the pipeline can
detect the expected harm from mixing under-trained states.

Late checkpoints are already highly correlated (`0.9985–0.9996` fold 01 and
`0.9989–0.9996` fold 02), but their row-level prediction standard deviation is
still 0.00519/0.00449. Averaging reduces that variance without improving ranking.
Late weight averaging is positive in only 7/40 category-fold cells; EMA in 5/40.
The largest mean late-average losses are Electronics `-0.00152`, Shoes
`-0.00093`, and Furniture `-0.00090`.

Calibration also does not improve. Against final, late averaging changes mean
Brier `+0.000150`, log loss `+0.000515`, ECE15 `+0.000551`; EMA changes them
`+0.000083`, `+0.000310`, `+0.000217` (lower is better). The supported mechanism
is that the OneCycle hand fine-tune is still converging useful ordering/calibration
near its endpoint; mixing earlier states pulls the model modestly backwards.
Prediction averaging is negative too, which argues against weight-space mismatch
as the sole explanation.

TP/FP/FN are not defined for this threshold-free PR-AUC claim; no classification
threshold claim is made.

## Runtime and artifacts

- Host: `avi-gn-fsk35`; physical GPU6, runner PID 1867969.
- GPU6 was live-checked at 4 MiB/0% before launch; peak observed memory about
  33.8 GiB, training/evaluation 97–100% utilization.
- Fold runtimes including all state evaluations: 556.9 s and 555.5 s; total
  runner wall time including shared tokenization was about 21.5 minutes.
- Local report/predictions/metrics/logs:
  `/home/dzkhomidov/matching-work/rescue_20260824/checkpoint_average`.
- Full checkpoint states remain on `avi-gn-fsk35` under the identical absolute
  path in `checkpoints/` (18 files; baseline, candidates, controls and raw points).
- Exact command: `COMMANDS.md`; stdout/stderr: `train.log`, `score.log`; metrics:
  `metrics_folds12.json`, `category_metrics_folds12.csv`; run metadata and raw
  late predictions: `diagnostics/`.

## Checked and unchecked

Checked: final baseline, late weight average, EMA, early/unstable negative control,
late prediction-average diagnostic, folds 01–02, macro/pooled/category AP,
calibration, checkpoint correlations and prediction variance.

Not run by gate: folds 03–04. Unchecked: other EMA decays/start points, SWA with a
different LR schedule, repeated seeds, hidden-test transfer. The result closes the
specified late-window/EMA recipe, not every possible averaging schedule. No
validation, submission, push, commit, or external publication was performed.

# Hard-negative reweighting: negative result

## Outcome

The proposed hard-negative weighting fails the stage-1 gate. On both checked
folds, both 2x and 4x weighting reduce pooled and macro-category PR-AUC. The harm
is dose-dependent and substantially larger than equal-coverage random-negative
weighting. Folds 3-4 were therefore not run, as pre-registered.

| variant | fold | PR-AUC | delta | macro-cat PR-AUC | delta macro |
|---|---|---:|---:|---:|---:|
| baseline | 01 | 0.836714 | — | 0.781200 | — |
| hard 2x | 01 | 0.832350 | -0.004364 | 0.774917 | -0.006283 |
| hard 4x | 01 | 0.822934 | -0.013780 | 0.762861 | -0.018339 |
| random 2x | 01 | 0.836590 | -0.000124 | 0.780612 | -0.000588 |
| random 4x | 01 | 0.834698 | -0.002017 | 0.778173 | -0.003027 |
| baseline | 02 | 0.846621 | — | 0.788423 | — |
| hard 2x | 02 | 0.843664 | -0.002957 | 0.784238 | -0.004185 |
| hard 4x | 02 | 0.834811 | -0.011811 | 0.774024 | -0.014400 |
| random 2x | 02 | 0.845482 | -0.001139 | 0.786641 | -0.001782 |
| random 4x | 02 | 0.844095 | -0.002526 | 0.784765 | -0.003659 |

| variant | mean PR-AUC | mean delta | mean macro | mean delta macro |
|---|---:|---:|---:|---:|
| baseline | 0.841668 | — | 0.784812 | — |
| hard 2x | 0.838007 | -0.003661 | 0.779578 | -0.005234 |
| hard 4x | 0.828872 | -0.012795 | 0.768442 | -0.016370 |
| random 2x | 0.841036 | -0.000632 | 0.783627 | -0.001185 |
| random 4x | 0.839396 | -0.002271 | 0.781469 | -0.003343 |

The 2x hard result has the same negative sign on both folds and misses the gate by
0.00466 pooled and 0.00623 macro relative to the required +0.001. The 4x result is
worse. The worst single category changes are -0.0313 (hard 2x) and -0.0561 (hard
4x), both on fold 1, so the no-catastrophe condition also fails.

## Coverage and signal distribution

Selection was recomputed using only the training portion of each target fold. It
covered exactly 10% of negative rows in every one of the 20 categories: 20,388
rows for fold 1 and 20,389 for fold 2. Random controls use exactly the same counts
per category.

| held-out fold | set | mean OOF CE | mean name Jaccard | attribute-conflict rate |
|---|---|---:|---:|---:|
| 01 | hard | 0.2565 | 0.6295 | 0.9651 |
| 01 | random | 0.0932 | 0.3842 | 0.6949 |
| 02 | hard | 0.2601 | 0.6308 | 0.9629 |
| 02 | random | 0.0944 | 0.3808 | 0.6911 |

Thus the selector did identify the intended region: negatives that the baseline
scores much higher, whose names are much more alike, and whose attributes usually
conflict. This is not a null selector or a coverage failure.

The OOF CE-score quantiles `[min, p10, p25, p50, p75, p90, max]` are
`[0.0008, 0.0203, 0.0481, 0.1268, 0.3775, 0.7681, 0.9933]` for fold-1 hard versus
`[0.0007, 0.0011, 0.0036, 0.0200, 0.0770, 0.2894, 0.9906]` for its random control.
Fold 2 is nearly identical: hard median 0.1277 and p90 0.7827 versus random median
0.0191 and p90 0.3050. Name-Jaccard hard medians are 0.625 on both folds versus
0.360/0.357 for random, with hard p90 0.875/0.870 versus 0.714/0.700.

## Category behaviour and mechanism

Mean category deltas for hard 2x are most negative in Обувь (-0.0182),
Галантерея (-0.0160), Ювелирные изделия (-0.0108), and Одежда (-0.0091). Hard 4x
deepens these to -0.0383, -0.0336, -0.0333, and -0.0251 respectively. Only 3 of
20 categories have a positive hard-2x mean delta, all at <=0.00124.

The evidence supports a mechanism in which emphasizing high-similarity labelled
negatives moves the decision boundary too aggressively against positives,
especially in low-positive-rate fashion categories. Equal-coverage random
negatives also hurt at 4x, but far less: the extra negative class weight itself is
harmful, and targeted hardness amplifies it. Because the audit queue previously
found many disputed labels to be wrong, label noise in this selected ambiguous
region is a plausible additional cause, but this experiment did not relabel the
selected rows and cannot prove that part.

## Noise, resource cost, and limitations

- Baseline fold PR-AUC standard deviation across folds 1-2 is 0.0070; paired
  hard-2x deltas are -0.00436 and -0.00296. Both signs agree and the dose response
  is much larger than the random 2x control, but only one training seed was run.
- Each model/fold took 364-366 seconds on one H100; eight runs used about 49 GPU
  minutes plus 77 seconds of tokenization. Host `avi-gn-fsk35`, physical GPU 0,
  wrapper PID 1685561, Python PID 1685564.
- The baseline was reused from the saved deterministic same-recipe OOF artifact,
  not retrained in this stage.
- Only the pre-registered 10%-per-category composite selector was checked. Lower
  coverage, CE-only, name-only, conflict-only, focal loss, and confidence-aware
  label smoothing remain unchecked. This result rejects the tested recipe; it does
  not prove every possible hard-negative method is harmful.
- Folds 3-4 are explicitly unchecked because both improvement gates and the
  category-safety gate failed on folds 1-2.

## Reproduction and artifacts

Exact commands are in `COMMANDS.md`. Raw training output is
`logs/stage1.log`; scoring output is `logs/score_stage1.log`; row-level metrics
and all category scores are in `metrics_stage1.json`; coverage, signal
distributions, runtime, arguments, and run status are in `run_manifest.json`.
Predictions are under `preds/{hard2x,hard4x,random2x,random4x}`.

Status: **negative** for the tested hard-negative reweighting recipe. Do not carry
it to folds 3-4 or a submission.

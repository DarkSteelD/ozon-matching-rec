# In-place unit normalization: negative result

## Outcome

Deterministic unit normalization does not improve this cross-encoder recipe. It
reduces pooled PR-AUC on both checked folds, misses the +0.001 gate, and also
reduces the macro-category metric. Folds 3-4 were not run, as pre-registered.

| variant | fold | PR-AUC | delta | macro-cat PR-AUC | delta macro |
|---|---|---:|---:|---:|---:|
| baseline | 01 | 0.836403 | — | 0.779536 | — |
| normalized | 01 | 0.835464 | -0.000939 | 0.779010 | -0.000526 |
| corrupt control | 01 | 0.827160 | -0.009244 | 0.769443 | -0.010093 |
| baseline | 02 | 0.846015 | — | 0.788011 | — |
| normalized | 02 | 0.845106 | -0.000908 | 0.786649 | -0.001362 |
| corrupt control | 02 | 0.838430 | -0.007585 | 0.780481 | -0.007530 |

| variant | mean PR-AUC | mean delta | mean macro | mean delta macro |
|---|---:|---:|---:|---:|
| baseline | 0.841209 | — | 0.783774 | — |
| normalized | 0.840285 | -0.000924 | 0.782830 | -0.000944 |
| corrupt control | 0.832795 | -0.008414 | 0.774962 | -0.008811 |

The normalization gate fails: both pooled deltas have the same negative sign and
the mean is 0.001924 below the required improvement. The worst normalized
category delta is -0.00880, so there is no >0.02 category catastrophe; lack of
benefit, rather than one catastrophic category, is the reason for rejection.

## Edit invariants and coverage

The baseline uses the original strings unchanged. Normalized and corrupted text
edit exactly the same spans, and runtime assertions verify for every field that
`len(original) == len(normalized) == len(corrupted)`. Replacements are padded
inside their original spans; no text is prepended or lengthened. The corrupted
control deterministically rotates the first canonical numeric digit while keeping
positions and edit count equal.

- 282,394 of 365,654 rows (77.2%) contain at least one editable span.
- 829,585 spans were edited across the four name/attribute fields.
- Row coverage: mass 142,253; volume 35,832; dimensions 27,087; counts 156,897;
  fashion sizes 60,458.
- The saved audit includes 100 source/normalized/corrupted examples. Self-checks
  cover equivalent `1 кг`/`1000 г`, litres/millilitres, dimensions, counts, and
  fashion sizes.

The high coverage gives enough power for the global claim. The corrupted control
loses 0.0084 pooled and 0.0088 macro, so the pipeline is sensitive to the edited
numeric semantics; the normalized null is not caused by ignored edits or a dead
code path.

## Unit-bearing slices

| slice | fold 1 rows | norm delta | fold 2 rows | norm delta |
|---|---:|---:|---:|---:|
| any edited unit | 70,671 | -0.000942 | 70,560 | -0.000955 |
| mass | 35,679 | -0.001131 | 35,513 | -0.001016 |
| volume | 8,990 | -0.003744 | 8,911 | -0.001084 |
| dimension | 6,880 | -0.000170 | 6,878 | -0.002995 |
| count | 39,193 | -0.001493 | 39,170 | -0.000846 |
| fashion size | 15,182 | +0.003633 | 15,165 | -0.004483 |

No unit family has a repeatable positive sign. The tempting fold-1 fashion-size
gain reverses on fold 2. The same instability appears by category: Обувь changes
+0.00505 on fold 1 but not consistently enough to rescue the metric; Одежда
changes +0.00312 then -0.00880. Full per-category and per-slice values are in
`metrics_stage1.json`.

## Mechanism and controls

The result is consistent with the LLM-pretrained encoder already learning common
unit aliases from raw text. Canonicalization discards familiar surface forms and
introduces compact forms such as `1e3g`; that distribution shift offsets any gain
from equivalence. The much worse semantically corrupted control confirms that
quantities do carry useful information. This supports the correlation and a
plausible mechanism, but does not separately measure tokenizer vocabulary effects.

## Noise, cost, and limitations

- Paired normalized deltas are -0.000939 and -0.000908. They agree closely, but
  only one seed was run; this is enough to reject the required +0.001 gain, not to
  estimate a tiny negative effect precisely.
- Baseline fold spread is 0.00680 PR-AUC. The paired design is more informative
  than unpaired fold spread because seed, rows, schedule, and updates are fixed.
- Six model/fold runs took 364-366 seconds each: about 36.5 H100 GPU-minutes, plus
  text preparation and 229 seconds total tokenization. Host `avi-gn-fsk35`, GPU 1,
  wrapper PID 1866219, Python PID 1866222.
- Checked: combined normalizer and equal-edit corrupted control on folds 1-2.
  Unchecked: folds 3-4 (gate failed), individual-family-only normalizers, alternate
  canonical vocabularies, and a second seed. Therefore the combined tested recipe
  is negative; every possible normalization vocabulary is not proven negative.
- PR-AUC has no fixed decision threshold, so TP/FP/FN counts are not naturally
  defined here. Row counts, positives, and marginal PR-AUC are reported for every
  unit slice instead of inventing a threshold after observing results.

## Artifacts

- `normalization_audit.json`: coverage and examples.
- `slice_masks.parquet`: exact unit-family masks.
- `metrics_stage1.json`: fold/category/slice scores and deltas.
- `run_manifest.json`: arguments and runtimes.
- `logs/stage1.log`, `logs/score_stage1.log`: raw logs.
- `preds/{baseline,normalized,corrupt}`: six prediction files.
- `COMMANDS.md`: exact reproduction commands.

Status: **negative** for the tested combined in-place unit normalizer. Do not carry
it to folds 3-4 or a submission.

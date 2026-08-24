# Fashion hidden-like stress audit — frozen plan

Status at plan freeze: all cases **unchecked**.

## Claim

Within the four fashion categories, pre-defined observable mechanisms identify
OOF slices where the single frozen baseline (`final_stack_all`) loses substantial
rank utility, and at least one independent saved channel adds positive marginal
rank utility consistently across folds. No mechanism or blend weight will be
selected using public-leaderboard outcomes.

Primary metric: average precision (AP / PR-AUC), per category and fold. Secondary:
coverage, positive/negative counts, prevalence, baseline-vs-alternative fixed rank
blend delta, grouped-bootstrap interval, and audited-label agreement.

## Frozen inputs

- Repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`
- Git SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`
- OOF predictions: `members/dzkhomidov/preds/all_model_predictions_oof.parquet`
- Pair text/folds: `/home/dzkhomidov/matching-work/data/hand_pairs.parquet`
- Raw item cards: `data/raw/items_human.parquet`
- Human audit: `label_audit.jsonl`
- Folds: `fold_01` through `fold_04`, all required.
- Seed: 20260824 for grouped bootstrap and permutation controls.
- Output: this directory only; repository, data, predictions, and `validation/`
  remain read-only.

## Frozen baseline and alternatives

- Sole slice/score-band baseline: `final_stack_all`.
- Independent alternatives: `zs_llm_blend`, `knrm_name_v2`.
- Diagnostic non-independent alternatives: `ce_priodistill`,
  `ce_fashion_specialist`.
- Marginal test is fixed before results: within each category-fold rank scores,
  then evaluate `0.9 * rank(baseline) + 0.1 * rank(alternative)`. No weight sweep.
- Negative control: the same fixed blend with the alternative permuted within
  category-fold.
- Positive pipeline control: perfect target score.

## Frozen mechanisms

Every row may belong to multiple mechanisms. Definitions are deterministic:

1. `name_exact`: normalized names identical.
2. `name_near`: not exact; normalized character SequenceMatcher ratio >= 0.85.
3. `name_other`: neither exact nor near.
4. `attrs_both_missing`, `attrs_one_missing`, `attrs_present`: based on parsed
   non-empty attribute dictionaries.
5. For brand, model, and article separately: `*_agree`, `*_conflict`,
   `*_one_missing`, `*_both_missing`; agreement means normalized extracted value
   sets intersect, conflict means both non-empty with empty intersection.
6. `size_comparable_agree`, `size_comparable_mismatch`, `size_not_comparable`:
   use explicit size attributes only; normalize numeric sizes, and compare only
   when the detected systems are the same (RU, maker/plain, or letter).
7. `numeric_conflict`: normalized name-number token sets are both non-empty and
   disjoint; `numeric_overlap` if they intersect; otherwise `numeric_missing`.
8. Text length uses full `name + attrs` characters per side:
   `text_short_both` (max <= 160), `text_medium` (160 < max <= 800),
   `text_long` (800 < max <= 1600), `text_very_long` (max > 1600), plus
   `text_asymmetric` when max/min >= 3 (minimum denominator 1).
9. Baseline-only fixed score bands:
   `[0,.01)`, `[.01,.05)`, `[.05,.2)`, `[.2,.5)`, `[.5,.8)`,
   `[.8,.95)`, `[.95,.99)`, `[.99,1]`.

Attribute key families are frozen as casefolded substring matches:

- brand: `бренд`, `brand`, `марка`
- model: `модель`, `model`, `линейка`, `коллекция`
- article: `артикул`, `партномер`, `sku`, `код товара`, `код производителя`
- size: keys containing `размер` but not packaging/transport dimensions.

## Grouping and bootstrap

Pairs form an undirected item graph. The connected-component ID is the bootstrap
group; if all components are single-edge, report that limitation. Bootstrap
resamples pooled groups 1,000 times and reports the 2.5/50/97.5 percentiles of
fixed-blend AP delta when AP is defined. Candidate confirmation is frozen as:
`all`, plus non-score-band slices with >=100 rows and both classes in every fold,
positive fixed-blend delta in all four folds, and median baseline collapse <=-0.05.

Runtime amendment before any result file was written or inspected: the first
implementation that bootstrapped every descriptive slice was stopped because it
rebuilt pandas frames per replicate. The candidate-confirmation definition above
preserves the full descriptive fold tables and avoids presenting thousands of
post-hoc intervals as confirmatory evidence.

## Prevalence-shift control

Public constant-probe priors are frozen from the saved ODS history evidence:

- Обувь 0.045808781600456185
- Одежда 0.057251184834123225
- Галантерея и аксессуары 0.043035306516774986
- Ювелирные изделия 0.015708741452596563

For each category, preserve baseline ranking and reweight positives/negatives so
the pooled local sample has the public prior. Report weighted AP and compare it
with the public baseline category AP recorded in the same saved evidence. This is
a control, not a fitted estimator of hidden performance.

## Planned outputs

- `audit.py`, exact reproduction script
- `row_features.parquet`
- `slice_metrics.csv`
- `blend_metrics.csv`
- `bootstrap_metrics.csv`
- `prevalence_shift.csv`
- `label_audit_metrics.csv`
- `summary.json`
- `run.log`
- `REPORT.md`

## Acceptance and statuses

A useful slice must have coverage and class support, baseline collapse relative
to its category-fold parent, positive fixed-blend delta in all evaluable folds,
and a grouped-bootstrap 95% interval above zero. A manual rule is not proposed
unless its firing-zone purity is >= 0.95 with support across folds/groups.
Anything not executed is listed as **unchecked**; weak or mixed evidence is
**inconclusive**, not negative.

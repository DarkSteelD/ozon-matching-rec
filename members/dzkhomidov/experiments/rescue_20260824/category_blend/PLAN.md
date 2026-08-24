# Category-specific cross-architecture rank blending

## Claim

Category-specific model selection or coarse convex weights, chosen only on the
other three folds, improve held-out `macro_category_prauc` over one global
equal-rank blend of rubase/e5/mdeb/zs/distill.

## Fixed protocol

- Input (read-only): `all_model_predictions_oof.parquet`.
- Source repository: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`,
  git SHA `2da459984a1207677ff9eb863ca28589027a4bc3`, clean worktree at launch.
- Runtime: `/home/dzkhomidov/ozon-hack/.venv-ml`, seed `20260824`, CPU only.
- Arms: `ce_rubase_len384`, `ce_e5_len288`, `ce_mdeb_len224`,
  `zs_llm_blend`, `ce_priodistill`.
- Normalize each arm to percentile ranks independently within fold/category.
- Four outer rounds: choose each category's model/weights on three folds and
  evaluate only on the fourth.
- Baseline: equal global weights `(0.2, 0.2, 0.2, 0.2, 0.2)`.
- Strong baseline control: one coarse global weight vector, selected on the same
  three training folds and evaluated on the same held-out fold.
- Candidates: best single model; all simplex weights at step 0.25; fixed
  25%/50%/75% shrinkage of the chosen category weight toward the baseline.
- Selection objective: mean category AP over the three training folds.
- Controls: permuted-label selection and random category weights.
- Primary: AP per category after concatenating all four held-out predictions,
  then mean over categories. Secondary: mean of the four held-out fold macro APs.
- Noise check: fold delta spread and paired category bootstrap of aggregate AP
  deltas. Gate requires non-negative aggregate delta, no negative held-out fold,
  and a 95% bootstrap lower bound above zero.

## Reproduction

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python experiment.py \
  --input /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet \
  --output artifacts_v2 --seed 20260824 2>&1 | tee run_v2.log

/home/dzkhomidov/ozon-hack/.venv-ml/bin/python analyze.py \
  --artifacts artifacts_v2 2>&1 | tee analysis.log

/home/dzkhomidov/ozon-hack/.venv-ml/bin/python two_model_proxy.py \
  --input /home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet \
  --output artifacts_v2 2>&1 | tee two_model.log
```

## Matrix

| variant | folds | status |
|---|---|---|
| global_equal | 1,2,3,4 | checked |
| global_grid_nested | 1,2,3,4 | checked |
| category_best_single | 1,2,3,4 | checked |
| category_grid_raw | 1,2,3,4 | checked |
| category_grid_shrink25 | 1,2,3,4 | checked |
| category_grid_shrink50 | 1,2,3,4 | checked |
| category_grid_shrink75 | 1,2,3,4 | checked |
| permuted_selection (10 seeds) | 1,2,3,4 | checked |
| random_weights (25 seeds) | 1,2,3,4 | checked |
| two-model fixed 2:1 / nested global / category shrink75 proxy | 1,2,3,4 | checked |

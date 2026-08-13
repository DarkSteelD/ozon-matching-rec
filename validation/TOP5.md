# TOP-5 — e-cup-2026-matching

Обновлено бенчем 2026-08-13 08:42 UTC. Первичная метрика `mean_prauc` (больше = лучше). Источник: `validation/leaderboard.csv`. Не редактировать руками.

| # | member | experiment | mean_prauc | public | notes |
|---:|---|---|---:|---:|---|
| 1 | darksteeld | lgbm_cheap_v1 | 0.63786621 | — | LightGBM OOF over frozen folds, 21 cheap pair features (tfidf cosine,  |
| 2 | darksteeld | name_tfidf_cos | 0.32913087 | — | char_wb 3-5gram TF-IDF cosine of names; fit on items_human names only  |
| 3 | darksteeld | name_tfidf_attr_blend | 0.32655209 | — | 0.5 * name TF-IDF cosine + 0.5 * attributes key=value Jaccard |
| 4 | darksteeld | name_exact | 0.25992352 | — | 1.0 if normalized names are equal else 0.0 (lowercase, yo->ye, non-aln |
| 5 | darksteeld | const_prior | 0.25677178 | — | Constant prediction = global hand-label prior 0.2568; PR-AUC of a cons |

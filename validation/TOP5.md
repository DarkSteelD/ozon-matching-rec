# TOP-5 — e-cup-2026-matching (внутренняя валидация / CV)

Обновлено бенчем 2026-08-23 23:34 UTC. Первичная метрика `macro_cat_prauc` (больше = лучше). Источник: `validation/leaderboard.csv`. Не редактировать руками.

> Это **наша валидация на CV-фолдах**, не реальный ЛБ соревнования. Реальные места всех команд — в `PUBLIC_LEADERBOARD.md`.

| # | member | experiment | macro_cat_prauc | public | notes |
|---:|---|---|---:|---:|---|
| 1 | dzkhomidov | final_stack_v3 | 0.80713002 | — | v2 + 0.3*len288 student (wp 0.2); +0.00012 all folds |
| 2 | dzkhomidov | final_stack_v2 | 0.80677528 | — | stack v2: 0.94*final_combo + 0.06*zs_llm_blend + 0.4*priodistill_stude |
| 3 | dzkhomidov | final_stack_all | 0.80498184 | — | final stack of ALL solutions: rank 0.94*final_combo(7 CE + LGBM) + 0.0 |
| 4 | dzkhomidov | ce_priodistill_single | 0.79916906 | — | single rubase224: distilled on 0.7*final_stack soft labels + prio-attr |
| 5 | dzkhomidov | zs_llm_blend | 0.63327116 | — | zero-shot LLM rank-blend: gemma-4-E4B-it + Qwen3.5-4B, no finetune, P( |

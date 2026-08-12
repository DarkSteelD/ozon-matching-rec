# Общие данные задачи

Единая локальная копия данных задачи «Матчинг товаров» для всех участников:
файлы кладутся в `raw/` в том виде, в котором выданы на странице задачи
(ODS: `e-cup-2026-matching`), считаются неизменяемыми и не хранятся в Git.

Ожидаемый состав `raw/` (со страницы задачи, 2026-08-13):

```text
matches.parquet        # 4.1 MB   — пары с ручной разметкой (id1, id2, target 0/1)
matches_llm.parquet    # 104.7 MB — пары с LLM-разметкой (target 0..1)
items.parquet          # 4.1 GB   — полные данные товаров (id, name, attributes, category)
items_human.parquet    # 214.2 MB — товары, встречающиеся только в ручной разметке
```

Архивы базовых решений (`matching-baseline-submit.zip` 1.2 GB,
`matching-baseline-lightweight.zip` 19.6 KB) складывать в `raw/baselines/`.

Производные датасеты и признаки — только внутри `members/<github-name>/data/`.

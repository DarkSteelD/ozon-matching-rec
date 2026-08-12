# Prior E-CUP editions — metrics, data scale, container envelopes

`[V]` = read off a primary source. `[INF]` = inference. Three editions exist: 2024 (2 tracks),
2025 (3 tracks), 2026 Students (3 tracks, live). The domain `e-cup-ozon.ru` is reused every year,
so past editions survive only in the Wayback Machine.

**Why this file matters:** 2026 publishes no metrics and no container limits before 10.08. The
2024/2025 editions are the only calibration we have for what Ozon asks for and what hardware it
gives you.

---

## Cross-edition table `[V]`

| | 2024 | 2025 | 2026 Students |
|---|---|---|---|
| Tracks | 2 | 3 | 3 |
| Eligibility | open, 18+ | open, 18+ | **students only**, 18+ |
| Platform | Codenrock | Codenrock | **ODS.ai** |
| Prize pool | 1 200 000 ₽ | 7 200 000 ₽ | 7 200 000 ₽ |
| Per-track | 350/150/100k | 1 100/800/500k | 1 100/800/500k |
| Submissions/day | 3 | 5 (site) / 10 (rules, T1) — conflict | 5 |
| Registered | 1 781 | 2 984 | **941 as of 09.08** (in progress) |
| Teams | 682 entries | 1 221 | — |
| Solutions | 209 | 453 | — |
| Solving window | 3 weeks | 2 weeks | 3 weeks |
| Container limits published pre-launch | no | **yes** | **no** |

Field size context: 2025 had 1 221 teams and 453 solutions → roughly **1 in 3 registered teams
actually submits**. Finalists = top-5 per task, so the real competition for a finalist slot in 2025
was ~150 submitting teams per track `[INF from V counts]`.

---

## E-CUP 2024 — metrics and framing `[V]`

**Track 1 «Matching товаров».** *«Разработать ML-модель, которая по названиям, атрибутам и картинкам
сможет ответить на вопрос являются ли два товара… одинаковыми. Модель должна найти среди
предложенных пар-кандидатов товаров как можно больше одинаковых товаров с долей ошибок меньше 25%.»*
- Data: candidate **pairs**; per item — название, **готовые эмбеддинги картинок**, атрибуты;
  20 товарных категорий; crowd labels via Ozon.Profit; "1 000 000+ строк" pre-event claim.
- **Metric: PR AUC.** Top-5 by PR AUC → finals.
- Note the framing: *candidate pairs are given*. Candidate generation was **not** the participant's
  job in 2024. Whether that holds in 2026 is unknown and is a first-day question.

**Track 2 «Поиск товаров с запрещёнными изображениями».** Detect tobacco/nicotine/hookah content in
card images *«вне зависимости от стилистики изображения (фото, рисунок)»*.
- **Metric: precision + recall, summed**, with hard floors **precision ≥ 0.95, recall ≥ 0.6.**
- Business context mentioned 1 000 RPS `[INF — not established as a scored constraint]`.

**Winners 2024** (spellings approximate — came through transliterated): T1 — Kuper / MISIS Neychev
Loss / zvezdochka. T2 — «Саша и балласт» / «Движ-радар» / Oleg-Solo.
**No winning or baseline scores were ever published, for either track.**

---

## E-CUP 2025 — the richest precedent `[V]`

Submitted as **repositories**; infrastructure ran each team's **container**, recomputed the metric
and updated a live leaderboard.

**Track 1 «Рекомендации: предсказание следующей покупки»** — next-item in apparel, personal top-k.
- Data scale, verbatim: *«История заказов — более 19 миллионов записей»*, *«Каталог товаров с
  атрибутами и CLIP-эмбеддингами 6,4 миллиона позиций»*, *«примерно 1,6 млрд событий»*,
  *«Иерархия 7 тысяч категорий»*, overall *«датасеты на десятки гигабайт»*.
- **Metric: NDCG@k** (the *k* is nowhere stated). **Public/private = 40 / 60.**
- Rules: Python ≥3.10, any open-source PyPI/Conda libs, **manual annotation of test and external
  API calls banned in the final run**, 10 submissions/day.

**Track 2 «Логистика: автопланирование курьеров»** — assign 20 000 orders to 200 couriers,
minimise total working time; micro-polygons, per-courier per-polygon service times.
- Post-mortem gives 20 160 orders / 240 couriers and a precomputed **400 million point-pair**
  distance matrix — conflicts with the 20 000/200 in rules and marketing.
- **Metric:** total courier time = inter-stop distances + depot legs + service times.
  **Exceeding the compute limit = disqualification.**
- **Prototype ≤ 1 hour.** Any language (Python/C++/Julia/OR-Tools). Docker 4 CPU / 32 GB RAM.

**Track 3 «Контроль качества: выявление поддельных товаров»** — counterfeit from description +
metadata + images. **This is the direct ancestor of 2026 Task 2.**
- **Metric, verbatim:** *«F1-score. Итоговая метрика рассчитывается как средневзвешенное значение
  двух F1-оценок: F1-score по товарам с плашкой "Оригинал" берётся с весом 2; F1-score по товарам
  без плашки — с весом 1. Финальное значение: (2 × F1 с плашкой + 1 × F1 без плашки) / 3.»*
  → **a weighted-F1 over two subpopulations, not a single global F1.** Optimising the pooled number
  would have been the wrong objective. Expect a similarly composed metric in 2026.
- **Container envelope, verbatim — the only fully specified one in E-CUP history:**
  *«Python ≥ 3.10; любые open-source библиотеки; поддержка CUDA-GPU (если доступна). Контейнерные
  лимиты: 4 CPU, 32 GB RAM, 1 GPU (T4) / или CPU-режим; инференс ≤ 60 минут на тестовом наборе.
  Запрещены внешние API-вызовы в финальном запуске.»*
- Business framing: final adjudication involved manual operator verification, with **Recall weighted
  slightly above Precision**.

**No 2025 winning team names and no scores were ever published** — no leaderboard snapshot survives.

---

## What this predicts for 2026 — and how confident

| Prediction | Basis | Confidence |
|---|---|---|
| Container = **4 CPU / 32 GB RAM / 1× T4 / inference ≤ 60 min** | the only published E-CUP envelope (2025 T3), same "код-контейнер" format | `[INF]` — **plan for it, verify 10.08** |
| **External API calls banned in the final run** | explicit in both 2025 container tracks; 2026 §2.8 licensing rule points the same way | `[INF]` high |
| Task 2 metric is a **composed/weighted F1**, not plain F1 | 2025 T3 precedent + the two-class structure (легковоспламеняющиеся / БАДы) | `[INF]` medium |
| Task 1 metric is **PR AUC** or similar ranking-of-pairs metric | 2024 matching used PR AUC | `[INF]` medium |
| Public/private split ≈ **40/60** | 2025 T1 | `[INF]` low — only one data point |
| Candidate pairs are **given**, not generated | 2024 matching gave pairs | `[INF]` low — 2026 wording says "identify identical products", not "classify given pairs" |

**A T4 changes everything for tasks 1–2 if it holds.** 16 GB, Turing — no bf16, no FlashAttention-2,
no FP8. A 7B VLM in fp16 is ~14 GB of weights alone: it fits only quantised, and 60 minutes of
inference over an unknown test set is a hard throughput budget. If the envelope is confirmed on
10.08, model size is decided by arithmetic, not preference — measure images/sec before committing.

---

## Verified 2026 facts that were not in the first report

1. **Task 2 targets are named.** ODS block description, verbatim: *«разработать классификатор,
   который на основе заданных правил будет определять, относится ли товар к категории
   легковоспламеняющихся товаров или биологически активных добавок. Классификатор также должен
   формировать итоговый вердикт и объяснять принятое решение, используя данные о названии и
   описании товара, а также его изображения.»* `[V]`
   → two target categories: **легковоспламеняющиеся товары** and **БАДы**; the model must emit a
   **verdict + an explanation**; the classification is **rule-based conditioning** («на основе
   заданных правил») — the rules will be given as text, which is why an LLM is mandatory.
   **How the explanation is scored is unknown** and is a first-day question.
2. **Task 1 in 2026 is text-only** — *«на основе текстовой информации из их карточек»*. No images,
   unlike 2024. `[V]`
3. **Task 3 framing, fuller:** *«Нам важно глубже понимать поведение пользователя на площадке, чтобы
   определять покупательский потенциал, рост/снижение активности и на этой основе принимать решения,
   влияющие на долгосрочную выручку.»* `[V]`
4. **941 approved participants** on ODS as of 09.08.2026; teams/solutions still 0. `[V]`
5. **All three task endpoints return 403 `competition_unavailable`**; blocks unlock
   `2026-08-10T15:00:00Z` = 18:00 MSK, against the rules' 12:00 MSK. `[V]` both.
6. **Rules §4.5.1.3 explicitly reserves the right to change any task's metric mid-contest.** `[V]`
7. **2026 expert panel by track** `[V]`: Матчинг — Анастасия Киргизова, Антон Рябцев, Алексей Мохов,
   Андрей Попов, Никита Божедомов · Контроль качества — Александр Проскуряков, Кирилл Бобылев,
   Артем Коньшин · Поиск — Альбина Рухадзе, Даниил Ануфриев, Эрик Багдасарян, Алексей Лотников.
8. Habr announcement 07.08.2026: https://habr.com/ru/companies/ozontech/news/1067838/ ; registration
   short link `https://s.ozon.ru/4KRE3rB`. `[V]`

---

## Open questions carried into 10.08

1. All three 2026 **metrics** — unpublished.
2. 2026 **container limits** — unpublished. The single highest-value unknown.
3. 2026 **dataset sizes / schemas / row counts** — unpublished.
4. 2026 **public/private split** — unpublished.
5. **Task 2: how is the explanation scored**, if at all?
6. **Task 1: are candidate pairs given, or is candidate generation ours?** Decides whether the work
   is a reranker or a full retrieval+rerank+cluster pipeline.
7. No E-CUP winning or baseline **scores** have ever been published, for any edition, any track.
   → **There is no external calibration. Our own CV is the only ground truth this campaign will
   have.** Treat the public leaderboard as the only external signal and price it against its own
   noise floor before reading anything into it.

## Sources
- 2026: https://e-cup-ozon.ru/ · /terms · https://ods.ai/tracks/e-cup-2026-competitions · /faq ·
  rules PDF 23.07.2026 (storage.yandexcloud.net/ds-ods/…) · habr.com/ru/companies/ozontech/news/1067838/
- 2025: web.archive.org/web/20250719182754/https://e-cup-ozon.ru/rules2025 ·
  web.archive.org/web/20250901031030/https://e-cup-ozon.ru/ ·
  codenrock.com/blog/kejs-ozon-e-cup-2025-… · codenrock.com/blog/kak-pobedit-na-e-cup-2025-…
- 2024: web.archive.org/web/20240815014053/https://e-cup-ozon.ru/rules ·
  web.archive.org/web/20240809234511/https://e-cup-ozon.ru/ ·
  codenrock.com/blog/kejs-e-cup-… · codenrock.com/blog/ml-speczialisty-ozon-tech-…

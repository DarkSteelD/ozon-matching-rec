# Ozon-native facts: the 2024 schema, the production stack, encoder choice — and three contradictions

The most valuable file here, because it is about **Ozon's own data and Ozon's own systems**, not
about analogous competitions. `[V]` verified · `[INF]` inference · `[!]` contradiction, unresolved.

---

## 0. `[!]` THREE CONTRADICTIONS — resolve on 10.08 before anything depends on them

**0.1 `[!]` SUBMISSION SCORING — the operationally critical one.**
One agent reports, citing the site and news coverage: *"≤5 submissions/day and **the LAST one is
scored, not the best**"*. The **rules PDF §2.10** says something different and specific: you select
**≤2 final solutions per task by 31.08 12:00**, and if you do not, **the platform auto-picks the 2
best by test metric**. E-CUP **2024** did have a "last one counted" rule (3/day), so this looks like
a 2024 rule leaking into a 2026 summary — but it is not proven either way.
→ **These imply opposite endgames.** If "last scored" is right, every late submission is dangerous.
If the PDF is right, submissions are free and only the *selection* matters. **Read the task page and
the platform UI on day 1 and settle it before spending a single submission.**

**0.2 `[!]` OPENING TIME — three values, now ranked.** Rules PDF: **12:00 MSK**. ODS platform config
`open_dt`: **18:00 MSK**. Third value: **10:00 MSK**. → Poll from 10:00, expect 12:00, do not be
surprised by 18:00.
**Provenance traced 2026-08-09:** the 10:00 figure comes from third-party event aggregators
(`ict2go.ru/events/70960` — «Начало 10.08.2026 10:00»), **not from Ozon or ODS**. Authority order is
therefore rules PDF (12:00) > platform config (18:00) > aggregators (10:00). Keep polling from 10:00
as cheap insurance, but 10:00 is not a competing claim. `[V]`

**0.3 `[!]` SUBMISSION DEADLINE.** Rules PDF §1.4.3: solutions accepted **until 30.08 23:59**. This
agent: *"solutions until 11.09"*. The PDF is the authoritative document and its internal timeline is
self-consistent (select finals 31.08 → finalists notified 03.09 → pitches 11.09 → results 12.09).
→ **Treat 30.08 as the deadline.** Anyone planning against 11.09 loses twelve days they don't have.

---

## 1. The 2024 matching dataset — exact schema `[V]`

Cross-checked against executed notebook outputs in two independent repos (`TimeNtWait/Hack_OZON_2024`,
`nickalymov/multimodal-product-matching-ozon`); the row counts reconcile: 1 168 516 + 49 620 = 1 218 136.

| File | Contents |
|---|---|
| `train.parquet` | **1 168 516 pairs, 48.1% positive** — near-balanced, unusual for matching |
| `test.parquet` | 49 620 pairs |
| `attributes.parquet` | (2 252 569, 3): `categories` JSON keyed "1".."4", `characteristic_attributes_mapping` |
| `resnet.parquet` | **128-dim** `main_pic_embeddings_resnet_v1` + nullable extra-image embeddings |
| `text_and_bert.parquet` | `name`, `description`, **`name_bert_64` (64-dim)** |

**20 categories advertised, 24 actually present** in `cat_level_2`.

**Two things this tells us about 2026:**
- Ozon ships **precomputed embeddings** and their dimensionality is small (128 image, 64 text). The
  64 is not arbitrary — Ozon's production realtime matching BERT emits **exactly 64 dims** (§3).
- Pairs were **given** in 2024 and the positive rate was ~48%. If 2026 repeats this, задача 1 is a
  pair classifier, not a retrieval problem. **First-day question.**

## 2. The 2024 target number, and the free gain in the shipped baseline `[V]`

**4th place of 110, team "Skripka": public macro PR-AUC ≈ 0.9268, local CV 0.966.**
Progression through their experiment log: 0.8652 → 0.9089 → 0.9163 → 0.9223 → 0.9245 → **0.9268**.
→ **CV→LB gap ≈ 0.039.** Same lesson as everywhere in this campaign: **CV level is not comparable to
LB level, only direction.** ~0.92–0.93 macro PR AUC was a strong solution.

**The metric, from the organizers' own scorer:**
```python
for category in categories:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_by_category.append(auc(recall, precision))
macro_prauc = np.mean(pr_auc_by_category)
```
Three consequences:
1. **A category with zero positives scores 0, not skipped.** Rare categories sink the mean →
   **per-category calibration is the metric, not a refinement.**
2. It is a **ranking** metric with **no threshold** —
   **yet the official baseline binarizes at 0.5 before writing `submission.csv`.**
   → **Submitting raw probabilities instead is a free gain over the shipped baseline.** `[V]`
   Check on day 1 whether the 2026 baseline makes the same mistake.
3. The 4th-place team trained **20 per-category models** because the metric's native shape is
   per-category.

**Execution contract 2024:** Docker with **`--network none`**, GitLab CI on `git.codenrock.com`.
One 4th-place log entry records **dropping TF-IDF vectorisation entirely because "из-за неё не
проходили сабмиты"** — submissions were failing resource limits, and the numeric limits were never
published. `[V]` → **Budget the container envelope as a first-class constraint, not an afterthought.**

## 3. Ozon's production matching stack — a strong prior on how they think `[V]`

- **Prod2Vec**: one unified product vector. **2× rubert-tiny** (name, attributes) + **ResNet34**,
  **ArcFace** loss, intermediate softmax on cat1 to stabilise convergence. ~5M products / 1 300 cat3
  classes → **85% Acc@1, 94% Acc@5** vs fastText 60%; >3× faster than separate pipelines.
- **Pairs → groups by graph community detection**: ANN candidates → pairwise classifier →
  **Label Spreading / Label Propagation** on Spark. Explicitly requires pairwise **precision ≈ 1**;
  naive transitive closure catastrophically merges iPhone 14 and 15.
  → **If задача 1 asks for groups rather than pairs, this is the shape they expect, and precision
  matters far more than recall in the pairwise stage.**
- **Realtime**: BERT with a **64-dim output embedding** (the same 64 as `name_bert_64`), Triton on
  8× A40 (350 emb/s), HNSW over Spark Structured Streaming, **CatBoost ~50 features offline** plus a
  distilled fast model at 900 pairs/s/pod, 9 000 pairs/s cluster-wide, batch matching precision >95%.
- **Ranking**: Ozon moved from item-level to **offer-level** (`offer_id = item × price_decile ×
  delivery_decile`) with a 100×100 proximity matrix transferring behavioural statistics across offer
  states. A/B: GMV/user +0.9%, search→order conversion +0.4%.
- **Candidate generation**: they **replaced HNSW/ANN with exact brute-force KNN on GPU** (custom CUDA
  top-K kernels, 2.5–3× faster than `torch.topk`), embedding business logic into the same kernel.
  Their conclusion: *«честный перебор часто оказывается и точнее, и в сумме дешевле в поддержке, чем
  приближённый индекс»*.
- **SKU-warehouse forecasting** — the only Ozon post with a metric ladder: LinReg 1.15 → RF 1.10 →
  XGBoost 1.03 → **LightGBM 1.01** (`regression_l1`, chosen for speed), against baselines
  history-mean 1.45 and **"last week = next week" 1.26**. **Their production model beats naive
  persistence by only ~20%**, with 170 features and last week's sales as the top feature.
  → Useful calibration for задача 3: **persistence is a strong baseline in Ozon's own hands.**

## 4. Extra labelled data and mirrors `[V]`

- ~~**`Late-Dev/ozon-product-matching`** — a 2023 precursor competition with the same schema.~~
  **DEAD, checked 2026-08-09.** The HF API returns **HTTP 401** for it as a dataset *and* as a model,
  as does the web page — gated/private or nonexistent, and HF does not distinguish. Either way it is
  not a usable source. **Stop planning around extra 2023 pairs.** `[V]`
- **`evgmaslov/ozon_ecup`** on HuggingFace — mirror of the 2024 dataset. **Priced exactly
  2026-08-09:** 50 files, **16.38 GB download / 21.94 GB expanded**; `train` = 2 252 569 rows (this is
  the **attributes** table — `variantid`, `main_pic_embeddings_resnet_v1`, `pic_embeddings_resnet_v1`,
  `name`, `description`, `name_bert_64`, `categories`, `characteristic_attributes_mapping`),
  `test` 563 143, `cleaned` 104 261, plus a **`triplets` config of 34 844 mined pos/neg triplets**.
  ⚠ **No licence is declared on the repo** — the same §2.8 trap as ruCLIP. Fine for an offline dry
  run; **do not put it in a container**, and check whether external data is permitted at all. `[V]`
- **`l1ghtsource/ozon-ecup-matching` is 404 and was never archived** (Wayback availability API
  returns empty). **Stop looking for it.**

## 5. Russian encoder choice — pick the column, not the average `[V]`

No Russian product-taxonomy or query-relevance leaderboard exists. ruMTEB is the proxy, **and the
ranking changes by column**:

| Model | Class. | Cluster. | PairClass. | Rerank | Retrieval | STS |
|---|---|---|---|---|---|---|
| multilingual-e5-large-instruct | 66.28 | 63.13 | **63.89** | 64.35 | 68.23 | **76.48** |
| e5-mistral-7b-instruct | 69.07 | 64.24 | 60.81 | 69.96 | 74.19 | 73.71 |
| **BGE-M3** | 60.44 | 52.38 | **60.6** | 69.71 | **74.79** | 73.68 |
| GigaEmbeddings (2.5–3B) | **72.7** | **65.36** | 57.85 | **73.42** | 74.28 | 72.11 |

⚠ **`ru-en-RoSBERTa` wins on the ruMTEB *average* (60.4) but is the weakest modern model on exactly
Retrieval (66.52) and Reranking.** For matching and dedup, **BGE-M3 is the better default**
(PairClass 60.6, Retrieval 74.79). Picking by the headline average would pick the wrong model for
our task.

**encodechka mean-S:** `deepvk/USER-bge-m3` **0.799** > `BAAI/bge-m3` 0.787 >
`multilingual-e5-large-instruct` 0.784 > `LaBSE` 0.739 > `rubert-tiny2` 0.704 >
`ai-forever/sbert_large_nlu_ru` 0.688 (near-bottom, **497.7 ms on CPU** — avoid).

**RusBEIR, 17 datasets, nDCG@10:** BM25 **52.16** → BGE-M3 **61.13** → **BGE-M3 +
`bge-reranker-v2-m3` 65.85**. → **A cross-encoder reranker is worth ~+4.7 nDCG@10 over dense
retrieval alone in Russian** — the same lesson as ESCI, now measured on Russian text. BM25 still wins
on 4 of 17 datasets, all long-document.

**Russian peer datapoint (Kuper.tech):** FastText+Faiss+CatBoost (hit@5 74%, coverage 55%) →
LaBSE-en-ru (74%, 65%) → **LaBSE-ru-turbo + Matryoshka 768→256 + cross-encoder reranker: +10% hit@5,
+15% coverage**. Breakdown: embedding upgrade +6% hit@5 / +15% coverage; reranker +4% hit@5;
weekly index rebuild +1%.

**Wildberries at scale:** E5-Small-Multilingual fine-tuned + ViT-H/LAION-2B over ~130M products —
**swapping the image model moved precision 80% → 90%**.

## 6. Задача 2 — the shape Ozon actually asks for `[V]`

**2024 moderation was PRECISION-CONSTRAINED, not F1-constrained: detect smoking/tobacco in photos at
Precision ≥95%, with a ~1 000 RPS target.** Ozon's engineers recommended **detection models and
Siamese networks**, and called *"detailed feature extraction of cigarettes and their location on
photos"* essential. 2025 counterfeit used Precision + Recall with *«Recall чуть более приоритетная»*.

→ **Design for a precision floor, not for balanced F1.** That makes **threshold calibration on a
held-out set the deliverable**, and per-category thresholds beat a global one — the same conclusion
the matching family reached independently.

**Categorisation lessons that transfer to the classifier head:**
- **Stay flat.** Flat beats hierarchical three independent ways: the organizers' own bi-level
  cascading fastText was *"better by at least one percentage point absolute"* when made flat; a flat
  linear SVM (0.8366) beat the same team's top-down ensemble (0.8173); and on hierarchical-text-
  classification benchmarks **a plain BCE baseline beats HBGL and Conditional Softmax**. Only
  consider seq2seq path generation above ~20K leaves.
- **TF-IDF (uni+bi-gram) + flat linear SVM on titles = 98.3% of the Rakuten winner.** The Day-1 bar
  in this family is embarrassingly high.
- **Fuse at the decision level, never by concatenation** — concat fusion measured **−0.06 pp, i.e.
  worse than text alone**; late fusion **+1.01 pp**.
- **Adding capacity to a fusion head that already works degrades it**: extra dropout 92.67 → 90.00,
  extra FC → 91.11. *"The simplest architecture showed better overall performance."*
- **QA-reformulation is worth +18.41 F1** if any part of the task is attribute extraction
  (79.73 → 98.14, encoding the attribute as a question). BERT-base → BERT-large adds only +0.17.
- **Zero-shot LLMs lose to a fine-tuned encoder by ~17 F1** on attribute extraction (62.4–62.7 vs
  79.7–79.8); few-shot with a JSON schema wins by ~5 but costs **24.8× more per 1k pairs**.
- ⚠ **fastText nondeterminism:** 40 runs at identical settings gave 74.09 ± 0.07 accuracy, and
  **thread count changes generalisation** (Hogwild async SGD). Fix and log `-thread` if fastText is
  used anywhere.

## 7. One useful counterweight on hardware `[V]`

**The M5 winner used only free Kaggle notebooks — CPU only, no GPU** — *"Intel Xeon @ 2.20GHz, max
16 GB RAM, 2 cores"* — to train **220 LightGBM models**. Memory was the binding constraint, handled
with float16 rolling features, `max_bin=100`, per-store sharding and pickled frame parts.
→ Against the OTTO winner's 256 GB, this is the other end of the distribution. **Hardware is a
constraint to measure, not a story to tell in advance.**

## 8. Two more verified negatives worth carrying

- **Classical CF is not a viable starting point on Ozon's data.** On the 2025 recsys track,
  **top-popularity beat ALS and BPR by three orders of magnitude** (0.00175 vs ~1e-6), and two
  participants independently abandoned WALS/ALS/LightFM, CLIP+FAISS and a CatBoost pairwise ranker.
- **On a pre-retrieved candidate list, BM25 can score below a dummy submission** (ESCI: BM25 0.563 vs
  dummy 0.7486) — lexical re-ranking destroyed the given ordering. `[V both numbers, interpretation
  INF]` If задача 1 hands us candidate pairs, do not assume a lexical baseline is a floor.

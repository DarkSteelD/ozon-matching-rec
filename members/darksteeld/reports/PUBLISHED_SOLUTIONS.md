# Published E-CUP solutions — what exists, what is clonable, what the recipes are

`[V]` verified on the source · `[UNC]` unconfirmed/secondhand · `[flag]` internally contradictory

**Headline: no top-3 solution from any edition is public.** The one 2nd-place repo that existed
(2024 matching, `l1ghtsource/ozon-ecup-matching`) is **deleted — 404 as of 2026-08-09**, no fork, no
Wayback snapshot. Non-disclosure is contractual (§2.12 bans publishing solutions), which explains
the total silence on Habr, YouTube and Telegram. The best still-clonable artifacts are two
**4th-place** solutions and one very well documented mid-table one.

**There is also no organizer baseline repo, for any edition.** Baselines were pushed into each
team's *private* GitLab repo on Codenrock. But **participant mirrors of the baseline code exist and
run** — those are the fastest legal starting points. `[V]`

---

## 1. Metrics of past editions, recovered from organizer scoring code `[V]`

- **2024 matching = macro PR AUC over categories.** The organizer's `compute_score.py` iterates
  categories → `precision_recall_curve` → `auc(recall, precision)` → mean.
  → **optimise per-category, not globally.** Source: `Timik232/OzonProductMatcher/compute_score.py`.
- **2024 moderation = precision + recall ranked by the SUM**, floors P ≥ 0.95, R ≥ 0.6.
- **2025 recsys = NDCG@100**, ground truth = `delivered` purchases in a 2-week window, 40/60 split.
- **2025 counterfeit = (2·F1_«Оригинал» + 1·F1_no-badge)/3.** Participant repos call this "F1 Macro";
  the weighted formula is the authoritative one.
- **2025 logistics =** total route time **+ 3 000 s penalty per unassigned order**, plus compute time.

**Known LB numbers, 2024 matching:** 4th place **0.9260** `[V]`; 1st **0.9374** / 2nd **0.9341**
`[UNC]`. → the top of that leaderboard was separated by **~0.003 PR AUC**. Whatever the 2026 metric
is, expect the same: a dense top where the noise floor decides places. Measure it before reading
any delta.

## 2. Container envelopes are PER-TRACK, not per-edition `[V]`

| 2025 track | envelope |
|---|---|
| RecSys | **16 CPU / 32 GB RAM, no GPU guaranteed** |
| Counterfeit (QC) | **4 CPU / 32 GB / 1× T4, inference ≤ 60 min** |
| Logistics | 4 CPU / 32 GB, ≤ 1 h |

→ Correction to the earlier single-envelope inference: **the envelope varies by track**. For 2026
tasks 1–2 (both "контейнер с кодом") the T4/60-min shape is the relevant precedent, but it must be
read off each task page on 10.08, per task. `[INF]`

---

## 3. The recipes worth stealing

### 3.1 Matching — 2nd place 2024 (repo deleted; recipe recovered from search index) `[UNC but specific]`
`l1ghtsource/ozon-ecup-matching`, MISIS Neychev Loss, PR AUC ≈ 0.9341.
- **CatBoost only.** One global model set + one **per-category** set (20 categories).
- **109 features per model = 106 hand-crafted deterministic pair features + 3 OOF columns** from
  finetuned transformers.
- The 3 OOF transformer columns: `cointegrated/rubert-tiny2` on **attributes + description**;
  `distilbert-base-multilingual-cased` on **attributes only**.
- **Bagging + blend:** 5 seeds global, 5 seeds per-category, final weight **0.4 on the per-category
  branch**.
- → The lesson: cheap deterministic pair features carry most of the signal; transformers enter as a
  *few OOF columns*, not as the model.

### 3.2 Matching — 4th/110 2024, **clonable, with pitch deck** `[V]`
https://github.com/TimeNtWait/Hack_OZON_2024
- Three CatBoost bundles by category level: 20 / 3 / 20 models (seed+fold bagging).
- Shipped feature assets as pickles: category dictionaries for levels 2/3/4, **TF-IDF/IDF
  characteristic-importance maps**, colour dictionary, category encoder, **anti-words filter**,
  text vectoriser.
- `train_model_v9_3.ipynb` + `make_submission.py` + Dockerfile/entrypoint + defence presentation.

### 3.3 Matching — 7th/110 2024 `[V]`
https://github.com/nickalymov/multimodal-product-matching-ozon (mirror: `QurusX/Product-Matching-Engine`)
- AutoGluon `best_quality` multi-layer stack (LGBM+CatBoost+XGB+NN) → **0.9216**; HistGradientBoosting
  + Optuna TPE → 0.9100. *(README labels the metric ROC-AUC — almost certainly wrong given the
  official macro-PR-AUC and where 0.9216 sits.* `[flag]`*)*
- Features: Levenshtein, Jaccard, **regex extraction of dimensions/volumes/units** (author calls
  these the decisive ones), cosine+Euclidean on supplied ResNet embeddings, **embedding entropy**,
  JSON attribute parsing, **per-category Top-N most frequent attributes**, Jaccard on vital vs minor
  attribute keys.
- Author's own credited factor: *«quality of Feature Engineering (especially handling JSON
  attributes and Regex-based text parsing) was the decisive factor»*.

### 3.4 Quality control / counterfeit — the single most complete published recipe `[V]`
https://github.com/PaVeLlLlLX/Ozon-Tech-E-Cup-bebryata — 46 KB README, LB **F1 0.7492**, local
holdout 0.855. No place claimed.
- **Ensemble of three:** CatBoost on all 116 features (Optuna-tuned, **pseudo-labelled on test**) ·
  multimodal **TabNet + Cross-Attention** · multimodal **FT-Transformer** (all modalities as one
  token sequence).
- Encoders: text `sberbank-ai/ruBERT-base`; images **EfficientNet-V2-M** + Albumentations; text
  cleaning HTML-strip → lowercase → stopwords → **pymorphy3 lemmatisation**.
- **Blend weights `0.45 CatBoost / 0.10 TabNet / 0.45 FT-Transformer`, Optuna-tuned for the metric.**
  *(README's intro states 0.75/0.14/0.11 — §2.3 is authoritative.* `[flag]`*)*
- **The 8 feature groups — this is the reusable part:**
  1. *Text/style*: lengths, `caps_ratio`, digit ratio, repeated-word ratio; marker flags
     `has_копия / has_реплика / has_аналог`; **`brand_in_name_score` via Levenshtein**;
     `name_desc_similarity` (Jaccard); presence flags.
  2. *Price anomaly*: `price_zscore_cat`, **`robust_price_z_cat` (MAD-based)**,
     `log_price_minus_cat_median` — **and the same set per brand**.
  3. *Item ratios*: return ratios at 7/30/90d, **`item_fake_return_ratio_*`**, `sales_growth_7_30`,
     `item_age_ratio`.
  4. *Rating patterns*: `rating_polarization` (share of 1s and 5s), `negative_ratio`, variance.
  5. *Seller profile*: return/fake-return rates, **`seller_category_entropy` (Shannon over the
     seller's categories)**, `seller_newbie` (<30 days), sales/returns spikes.
  6. *Categorical interactions*: `brand_cat_rarity`, **`price_brand_cat_anomaly`** (cheap
     simultaneously vs category AND vs brand).
  7. *Temporal, leak-free*: **expanding-window** seller history, price deviation from seller mean,
     seller rating trend and price volatility.
  8. **Target encoding** with smoothing on CV folds over `brand_name`, `SellerID`, category.
- Author-credited wins: FE depth over model complexity; boosting + two deep multimodal nets with
  Optuna'd weights; **pseudo-labelling on test** for distribution shift.

### 3.5 The 2025 QC *winner's* one-line description — and why it matters `[V]`
Team **hype and chill**, 1st place: *«ensemble of text models, different BERT versions cascaded —
fast models for easy cases, hard cases escalated»*.
→ **A confidence-routed cascade is how you satisfy a 60-minute inference cap and still use a big
model.** For 2026 task 2 (LLM+VLM mandatory, container, likely T4), this is the most directly
transferable structural idea in the whole record: cheap text model first, escalate only the
uncertain tail to the VLM.

### 3.6 Exact 2025 QC dataset schema `[V]` — https://github.com/aleksey-karasev/Ozon-E-Cup-2025-solution
`description`, `name_rus`, `brand_name`, `rating_1..5_count`, `comments_published_count`,
`photos_published_count`, `videos_published_count`, `GmvTotal{7,30,90}`,
`ExemplarAcceptedCountTotal{7,30,90}`, `OrderAcceptedCountTotal{7,30,90}`,
`ExemplarReturnedCountTotal{7,30,90}`, `ExemplarReturnedValueTotal{7,30,90}`, `ItemAvailableCount`,
`CommercialTypeName4`, `PriceDiscounted`, `seller_time_alive`, `item_time_alive`,
`item_count_sales30`, `item_count_returns30`, **`item_count_fake_returns30`**, `ItemID`, `id`.
→ If 2026 task 2 reuses this table shape, the entire feature block in §3.4 is directly portable.

### 3.7 RecSys 2025 — including the negative results `[V]`
- Best public: `ilnitskii/ozon_recsys_2025` (18th/432). Two-stage, **400 candidates/user** from
  action histories + **co-occurrence (1-day window)** + CLIP-512 content neighbours + popularity →
  **CatBoostRanker**. Action weights `delivered 6.0 / processed 4.0 / to_cart 3.0 / favorite 2.0 /
  view_description 1.5 / review_view 1.5`.
- **`Sem-dmitry/ozon-cup-recsys`: matrix factorisation lost.** ALS / BPR / LightFM / CLIP+FAISS /
  CatBoostRanker were all beaten by **`LogisticRegression(class_weight='balanced')` on historical
  aggregates alone**. Independently corroborated by `GlebIsrailevich/ozon_recsys_25`, where
  top-popular (NDCG@10 0.00175) beat BPR (9.8e-07) and ALS (1.1e-06) by three orders of magnitude.
  → **Do not start from matrix factorisation on Ozon data.**

### 3.8 Logistics 2025 — 4th place, fully documented `[V]`
https://github.com/KaufmanDmitriy/ozon-ecup-2025-logistics — local 2 906 442 / public 2 908 351,
**16 min 30 s** total on 16 threads. Three stages: intra-polygon TSP (OR-Tools CHRISTOFIDES + GLS,
~2 min) → polygon→courier assignment (PATH_CHEAPEST_ARC + SIMULATED_ANNEALING, 480 s, ~11.5 min) →
per-courier TSP polish (GLS 20 s each, multiprocessed, ~3 min). Next tier up:
`KamilIskhakov/E-CUP-2025-ML-Challenge-VRP` — portals per micropolygon, ALNS, column generation with
ESPPRC pricing.

---

## 4. WHAT TO CLONE — ranked

**Baseline mirrors — start here, these are the organizer's own code**
1. https://github.com/Timik232/OzonProductMatcher (MIT) — **2024 matching baseline + the organizer's
   `compute_score.py` (macro PR AUC)**. Baseline itself: TF-IDF(3000) on concatenated pair text ‖
   supplied ResNet embeddings of both items → `LogisticRegression(max_iter=2000)`, 80/20 split.
   Ships `baseline.pkl`, `vectorizer.pkl`, inference under `docker run --network none --shm-size 2G`.
   **Note: in 2024 the organizers pre-computed ResNet image and BERT text embeddings — participants
   never ran the encoders.** Whether 2026 does the same is a first-day question.
2. https://github.com/recara/e-cup-2025-ml-challenge — 2025 recsys baseline + improved baseline,
   **prints validation NDCG@100 = 0.1799** — the only public baseline-level score in E-CUP history.
3. https://github.com/OlgaTora/E-CUP-Codenrock — 2024 moderation baseline (TF transfer learning) +
   `compute_score.py` printing P/R/F1.

**Highest-placed solutions still online**
4. https://github.com/TimeNtWait/Hack_OZON_2024 — 4th/110, 2024 matching, + pitch deck
5. https://github.com/KaufmanDmitriy/ozon-ecup-2025-logistics — 4th, 2025 logistics
6. https://github.com/nickalymov/multimodal-product-matching-ozon — 7th/110, 2024 matching
7. https://github.com/ilnitskii/ozon_recsys_2025 — 18th/432, 2025 recsys

**Deepest documentation**
8. https://github.com/PaVeLlLlLX/Ozon-Tech-E-Cup-bebryata — 2025 QC, 116-feature spec, full pipeline
9. https://github.com/KamilIskhakov/E-CUP-2025-ML-Challenge-VRP — 2025 logistics, ALNS + colgen
10. https://github.com/Sem-dmitry/ozon-cup-recsys — 2025 recsys, the LR-beats-everything result
11. https://github.com/Java-Boys-Hackathon-Team/e-cup-2025 — 2025 recsys, ClickHouse DDL + duckdb +
    **`ui-validator` submission checker** + full data dictionary

**Secondary:** `ValentinaFedorova/ozon_matching_project` · `galkin-v/item-matching-ozon` (`intfloat/e5-base-v2`
+ thefuzz + CatBoost) · `itsZENR/E-CUP` · `georgechaikin/moderatsiya-kartochek` · `AnutKu/ozon_ecup`
(top-25 QC 2025) · `aleksey-karasev/Ozon-E-Cup-2025-solution` · `ChemZhEg/ozon_ecup_2025` ·
`German229/Ozon-E-cup-2025` · `PE51K/ozon-e-cup-hack-2025` · `mihalko711/Ozon2025-Hackathon` ·
`kotoskar/Ozon_counterfeit` · **`maksim-cv/ozon_competition_2026`** · adjacent (LCT-2023 Ozon
matching): `dazzle-me/lct-2023`.

**Dead:** `l1ghtsource/ozon-ecup-matching` (404 — the 2nd-place 2024 solution) ·
`Pe-tro/ECUP2026Students_Task_3` (empty — someone already staked out 2026 task 3) ·
`GZakala/hack-ozon-recsys` (empty) · `Closing-AI/ozon-e-cup-2025` (scaffold only).

## 4b. The execution contract of past container tracks `[V]`

Both 2024 baselines ship the same Docker contract: **`docker run --network none --shm-size 2G`**,
`entrypoint.sh`, **model weights committed into the repo**.
→ **Inference ran fully offline, with no network.** 2025 restated it as "external API calls banned
in the final run". Assume the same for 2026 tasks 1–2: every weight must be *vendored into the
image*, nothing downloaded at runtime, and `--shm-size` is small enough that DataLoader worker
counts matter. `[INF for 2026, V for 2024/2025]`

Other details worth carrying:
- The 2024 matching baseline **deliberately ignores the supplied BERT embeddings** and uses only
  ResNet ‖ TF-IDF. Using them is the obvious first upgrade — and it means the organizers ship more
  signal than their own baseline consumes. `[V]`
- The organizer's `compute_score.py` **zero-scores categories with no positives** when averaging —
  a category with no positive pairs drags the macro mean down. Worth checking whether the 2026
  scorer does the same. `[V]`
- 2024 baseline prints **no score** — there is no published "baseline level" for that track. `[V]`
- Ozon's own pre-competition advice for the 2024 moderation track, verbatim: *«смотреть в сторону
  моделей детекции и сиамских сетей»*, split the target class into sub-classes, and handle
  **illustrated/anime depictions**, not just photographs. `[V]` → the host tells you what they want
  if you read their posts (ROGII rule 14).
- `galkin-v/item-matching-ozon` (team De Moivres) used **`intfloat/e5-base-v2`** rather than the
  supplied BERT, plus thefuzz `token_sort_ratio` and hand-written size/colour/ISBN comparison rules
  → CatBoost. `[V]`
- Adjacent, not E-CUP but the same problem on Ozon data: `dazzle-me/lct-2023` (ЛЦТ-2023 Ozon product
  matching), with PyTorch checkpoints named by score (`model_0.93023.pth`). `[V]`
- A second empty 2026 stub exists: `aaaaaaa0/E-CUP-2026-Students-3`. `[V]`
- The bebryata blend weights are contradictory **in both directions** (§1 says 0.75/0.14/0.11, §2.3
  says 0.45/0.10/0.45). Trust neither; rerun their `optimize_ensemble_weights.py`. `[flag]`

## 5. Confirmed absences
No participant Habr write-up for any edition. No YouTube/VK pitch recording. `@ozon_tech` has 3
E-CUP posts, all promotional, and never announced the 2025 winners. No archived leaderboard for
either edition. No vc.ru / ods.ai / dzen / tproger article.
→ **Nobody publishes here. Our own CV is the only ground truth, and no external score exists to
calibrate against.**

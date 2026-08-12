# RecSys family — no such track in 2026, but five findings transfer

**E-CUP 2026 has no recommendations track** (2025 did; 2026 replaced it with задача 3, a regression).
So this file is deliberately short: only what carries over to the tracks we actually have. Anchor
competitions: Kaggle OTTO (12.9M sessions, 216M events), RecSys Challenge 2022 (Dressipi), RecSys
Challenge 2023 (ShareChat). `[V]` verified · `[INF]` inference.

---

## 1. Two-stage beats end-to-end, and the gap is measured `[V]`

OTTO 1st place ran the only clean ablation in this literature (local weighted Recall@20):

| condition | weighted | Δ |
|---|---|---|
| full solution | **0.588** | — |
| without the NN source | 0.583 | −0.0053 |
| without co-visitation | 0.583 | −0.0052 |
| without aid-level features | 0.585 | −0.0035 |
| **only a single neural session model** | **0.515** | **−0.0732** |

**A good standalone neural sequence model reaches 87% of the winner. The candidate-generation +
GBDT-reranker stack on top is worth +0.073.** Nobody won OTTO with an end-to-end model.
→ Reinforces the task-1 architecture: retrieve, then rerank with a GBDT. Do not spend the campaign
trying to make one big model do both.

**But this is catalogue-size dependent, and there is a counterexample.** RecSys 2022 was won at 2nd
place by **a single Transformer with no reranker**, beating a 17-model stacked pipeline (Transformer
0.2121 vs XGBoost 0.1993 leaderboard MRR), and session-kNN beat GBDT there too. The likely
discriminator is catalogue size — 28k items vs OTTO's 1.86M. `[INF — item counts verified, the
causal claim is not stated in any source.]`

## 2. How you build the reranker's training data outranks feature engineering `[V]`

OTTO 3rd place (Benny), documented climb from the public rules-only baseline:

| stage | CV | public LB |
|---|---|---|
| rules only | 0.567 | 0.575 |
| + XGBoost reranker on truncated week 4 | 0.576 | 0.583 |
| + trained on 20% of week 4 truncated | 0.585 | **0.591** |
| + activity history | 0.587 | 0.594 |
| + 100% of week 4 truncated | 0.592 | **0.598** |
| + Word2Vec features | 0.593 | 0.599 |

**+0.008 from the reranker existing, then +0.015 purely from how its training data was constructed**
(how much of the last week is truncated, and generating train and inference features from the same
distribution) — then **+0.001** from adding Word2Vec.
→ **Data construction was worth 15× the feature.** When our reranker plateaus, the first place to
look is the construction of its training set, not the feature list.

## 3. The trivial baseline can be 95% of the winner — and that changes what "good" means `[V]`

| OTTO notebook | LB | % of winner |
|---|---|---|
| Word2Vec embeddings only | 0.521 | 86% |
| co-visitation matrix, 10 min | 0.542 | 90% |
| **3 co-visitation matrices, rules only, NO ML, 36 min GPU** | **0.575** | **95.1%** |
| **20 co-visitation matrices, still no ML** | **0.590** | **97.5%** — would have placed 49th alone |
| winner | 0.605 | 100% |

**All the work lived in the last 5%.** The ratio is a property of the *evaluation design*, not of the
task's difficulty — and it varies enormously across competitions in this family (the same
trivial-baseline-to-winner ratio was ~0.6% in a comparable Russian competition whose evaluation
anti-joins already-seen items).
→ **On day 1, measure this ratio for our own task**: submit the dumbest defensible baseline and read
where it lands. If it is at 95% of the visible top, the campaign is a fight over the last few
thousandths and the noise floor decides places (which is what E-CUP 2024's 0.003 top spread already
suggests). If it is at 10%, there is real headroom and the strategy is different. **This single
measurement should be made before planning the rest of the month.**

## 4. Adversarial validation for covariate shift — directly applicable to задача 3 `[V]`

RecSys 2023, 9th place: a **single LightGBM, no ensemble**, moved the leaderboard **0.408 NCE** on
feature engineering alone, using **adversarial validation to drop covariate-shifted features**. Their
CatBoost won locally (0.3599) and **lost on the leaderboard** (6.107 vs 6.059) — the one clean
local/LB divergence documented in this family.
→ **For задача 3** (predict the next 30 days from behavioural history, where train and test are
separated in time), adversarial validation between train and test is a cheap, high-yield first step:
train a classifier to distinguish the two, and drop or down-weight whatever it uses. Time-separated
tabular data almost always has drifting features.

Also from that competition: a **tabular Transformer beat a tuned GBDT by 0.0111** log loss with no
self-supervision, and **contrastive pre-training added 0.0174 more** — echoing the intermediate-
training result in `FAMILY_MATCHING.md` §1.1. Self-supervised pre-training on the competition's own
unlabelled data keeps showing up as the largest single lever across three unrelated families.

## 5. Speed and memory levers, with measured multipliers `[V]`

Relevant to every container track, where inference time is capped:
- **`pandas.merge` → `polars.join`: 40×** on large joins.
- **CatBoost GPU inference vs LightGBM CPU: 30×.** TreeLite for LightGBM inference: 2×.
- **CatBoost ranker beat LightGBM** on the same features: carts +0.002, clicks +0.0012, orders +0.0007.
- Downcast every column at creation; `cudf.set_option("default_integer_bitwidth", 32)`; process
  feature merges and inference in ~10 chunks; when downsampling negatives, **update the DMatrix group
  sizes to match** or the ranker silently trains on wrong groups.
- Co-visitation-style matrices on GPU: hundreds computed in minutes, 20 kept by CV.

**The compute-honesty note:** OTTO's winner used **256 GB RAM**; 3rd place used 20 CPU / 256 GB /
32 GB GPU and published **261 notebooks**. The best Kaggle-kernels-only competitor (59th, top 3%)
concluded outright: *"all of them used servers with way more RAM and GPU available… To win this
competition, one needed to have more computational resource."* → Budget the hardware question early
rather than discovering it in week three (`CAMPAIGN_RULES.md` #13, and the ROGII cluster lesson).

## 6. One more transferable trap `[V]`
**The reranker can only reorder what retrieval returned.** OTTO recall ceilings: at 20 candidates
clicks 52.7% / carts 40.7% / orders 64.8%; at 50, 60.4 / 45.2 / 67.9; at 75, barely better than 50 —
*"the model was able to select almost no candidates from those additional."* And a weak feature set
converted only 1–2 points of a ~4-point headroom, while the winner's ~200 features against ~1200
candidates did extract it.
→ **Candidate count is only worth raising in proportion to feature quality.** Compute the recall
ceiling first (`CAMPAIGN_RULES.md` #1 — oracle before building), then decide K.

## Repos, if задача 1's retrieval side needs a reference
- https://github.com/otto-de/recsys-dataset — official split/eval code
- https://github.com/cdeotte/Kaggle-OTTO-Comp — 3rd place, 261 notebooks, complete end-to-end
- https://github.com/Fedorov-Artem/kaggle_OTTO — silver medal built inside 13–30 GB RAM; effectively
  a textbook on doing this without a big machine

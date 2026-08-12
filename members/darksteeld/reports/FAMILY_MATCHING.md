# Task 1 — matching: what actually moves the score, with numbers

Evidence from Kaggle Shopee Price Match Guarantee, Amazon KDD Cup 2022 (ESCI), WDC Products (EDBT
2024), Ditto, R-SupCon, MWPD2020, Rakuten SIGIR-eCom. `[V]` verified on source · `[INF]` inference.

**Reminder of what we are actually facing:** E-CUP task 1 is **text-only** dedup of product cards,
submitted as a **code container**, with **LLM relabeling of the training data explicitly permitted**
(open-license models only). So the image half of the Shopee literature does not apply, and the
relabeling permission maps directly onto the self-distillation / confident-learning lane below.

---

## 1. The three findings that should drive the design

### 1.1 Intermediate training is the single largest verified delta in this whole family `[V]`
In-domain product-pair objective **+ MLM** before task fine-tuning, WDC Computers F1
(`ceur-ws.org/Vol-2726/paper1.pdf`, Tables 4–5):

| Fine-tune size | fine-tune only | + intermediate (PM) | **+ intermediate (PM+MLM)** | Δ |
|---|---|---|---|---|
| small (2 834) | 81.89 | 93.73 | **96.53** | **+14.64** |
| medium (8 094) | 89.31 | 94.88 | **96.58** | **+7.27** |
| large (33 359) | 93.29 | 94.09 | **95.82** | **+2.53** |
| xlarge (68 461) | 94.47 | 94.61 | **97.37** | **+2.90** |

Zero-shot *after intermediate training alone* reaches **94.16 F1** — beating fully fine-tuned BERT at
every training size below xlarge. Nothing else in this report comes close to +14 F1.
→ **Lane 1: MLM + pair-objective pre-training on the competition's own product text.** Cheap, uses
unlabelled data we will be given anyway, and pays most when labels are scarce.

### 1.2 How you mine negatives dominates every other choice in contrastive setups `[V]`
R-SupCon ablation (`arXiv:2202.02098`, Table 3): removing **source-aware negative sampling** costs
**55 F1 on Abt-Buy** (93.70 → 38.24) and **37 F1 on Amazon-Google**. Freezing the encoder during
fine-tuning beat leaving it unfrozen by ~14 F1 on Abt-Buy.
→ **Design the negative sampler before the model.** Hard negatives from the same category / same
brand / near-duplicate titles is the whole game.

### 1.3 Metric learning generalises WORSE to unseen entities — and our private test is unseen `[V]`
WDC Products, pair-wise F1 as Seen / Half-seen / **Unseen**:

| Train | RoBERTa | Ditto | R-SupCon |
|---|---|---|---|
| Large 80% | 78.15 / 75.52 / **69.75** | 79.46 / 68.81 / 67.94 | **82.15** / 67.27 / **53.31** |
| Large 20% | 87.80 / 82.17 / **78.64** | 87.52 / 82.81 / 77.92 | **89.04** / 74.59 / **62.45** |

**R-SupCon is best on seen entities and worst on unseen — it loses ~25 F1 seen→unseen while plain
RoBERTa loses ~9.** Contrastive pre-training buys in-distribution accuracy at the cost of
generalisation to entities never seen in training.
→ **Directly relevant:** E-CUP's private leaderboard is a closed test set. If it contains products
absent from train, a pure metric-learning solution will look great on CV and collapse on private.
**Build the validation split so it contains unseen products, and compare a cross-encoder against the
embedding model on that split specifically.** This is exactly the ROGII "design against the hidden
correlation, not the local one" rule (`CAMPAIGN_RULES.md` #6).

---

## 2. Score-movers, ranked by reported delta (Shopee public LB) `[V]`

| Δ | Move |
|---|---|
| **+0.024** | ArcFace metric learning replacing raw-embedding cosine (0.700→0.730) |
| **+0.020** | Piecewise-linear decision boundary over (image, text) cosine instead of independent thresholds |
| **+0.020** | A bundle of five post-processing steps: zero-match relaxation, graph neighbour-of-neighbour FP removal, cluster-level include/exclude by % overlap, per-product-type threshold stacking, **unit-mismatch filter** |
| **+0.019** | **min2** — force ≥1 non-self match (metric artifact, see §4) |
| **+0.019** | **Union of per-modality match sets** + threshold tuning (0.757→0.776) |
| **+0.017** | **Iterative Neighborhood Blending** (0.776→0.793) |
| +0.011 | Adding a second text encoder + multimodal similarity features to the GBDT stage |
| +0.010 | Emitting 3 embeddings (concat / image / text) from one model instead of 1 |
| +0.010 | L2-normalise **before** concat (0.743→0.753) |
| +0.005 | Unit-mismatch filter alone (reject a match at 50 ml when the query is 100 ml) |
| +0.004 | Training on **full data**, no held-out fold, for the final submission |
| +0.002 / +0.001 | GAT on top of LightGBM / query expansion / PageRank features |

**Read:** metric learning ≈ threshold shaping ≈ the post-processing bundle ≫ the fancy graph tricks.
The graph methods everyone talks about were the *cheapest* moves once a stage-2 model existed.

**Stage-2 (reranker) feature list worth copying verbatim** `[V]` (Shopee 2nd place): cosine sims per
embedding space; **mean and std of top-K sims for K = 5, 10, 15, 30**; PageRank; text length; word
count; Levenshtein distance.

## 3. ESCI-specific movers (text-only classification, closest analogue to our task) `[V]`

| Δ | Move |
|---|---|
| **>+0.010** | **Query-based TF-IDF augmentation** — pool all candidates' text per query, TF-IDF-extract keywords, append them + all candidates' brand/colour to the query text |
| +0.003 | **Self-distillation** from 10-fold OOF soft labels (soft label = mean(pred prob, one-hot true), retrain) |
| +0.0035 | EMA (+0.0011) → FGM adversarial ε=1.0 (+0.0011) → R-Drop (+0.0013). **Embedding Mixup added 0.000** |
| +0.0013 | **Poly1 loss** beat CE. **Focal (74.73) and GHM (74.85) were WORSE than CE (75.08)** |
| +0.0011 | Product2Query pre-training on top of MLM. **Contrastive learning added 0.000** |
| +0.005 nDCG / **−0.003 F1** | **Confident learning** removing ~4% noisy labels — **helps ranking, HURTS classification** |

→ Two directly actionable: **the augmentation trick is free and worth >1 point**; and if the metric
turns out to be classification-shaped, **do not reach for focal loss** — it lost to plain CE here.
→ **On LLM relabeling (which E-CUP explicitly invites):** the evidence says self-distillation is
worth ~+0.003 and confident-learning-style label cleaning is worth +0.005 on a *ranking* metric but
**−0.003 on a classification metric**. So relabeling is a real lane but a small one, and its sign
depends on the metric. Do not build it before the metric is known.

## 4. Metric mechanics and traps

- **Shopee's min2 (+0.019) is a pure artifact** of a per-row mean-F1 metric where every group has
  ≥2 members. It does not exist under a pairwise or ranking metric. **Do not port it blindly** —
  first read what E-CUP's metric actually averages over. The 2024 E-CUP matching metric was **macro
  PR AUC over categories** (`reports/PUBLISHED_SOLUTIONS.md`), which has no such artifact but has
  its own: **a category with no positive pairs is zero-scored and drags the mean down.**
- **Label-noise ceiling is real.** ESCI human agreement is **91%** on the 4-way label (>96% collapsed
  to binary), and the winner scored 0.8326 — close to the annotation ceiling. `[V]` Expect the same
  shape here: there is a level above which more modelling returns nothing.
- **The top is dense.** ESCI task 2 top-3 spread = **0.0053**. Shopee 1st→2nd = 0.003. E-CUP 2024
  matching 1st→4th ≈ 0.011. In all three, ensembling and post-processing decided places, not
  architecture.
- **Public→private haircut at the Shopee top was −0.013 to −0.015** `[INF from two verified numbers]`
  — normal, and a reason not to chase the last +0.005 of public-LB threshold tuning.

## 5. Validation

- **The scheme that worked: `GroupKFold(5)` grouped by the product-identity label** — stated by
  Shopee 6th place, endorsed in the top-solutions summary. Never a plain random or stratified split:
  the same product on both sides leaks the answer.
- **CV level ≫ LB level by construction.** Shopee 6th place: CV 0.873–0.877, private LB ~0.764. CV is
  comparable **in direction only, never in level.** Anyone quoting "CV 0.87 so LB 0.87" is wrong.
- **Threshold tuning is where the overfit lives.** The winner tuned only the two late-stage
  thresholds and pinned the first to 0; 14th place replaced the global threshold with an OOF-trained
  per-category adjustment. Treat thresholds as parameters that need regularisation.
- **Train-on-full-data for the final submission was verified safe and worth +0.004** — the CV split
  picks hyperparameters and thresholds, then is discarded.

## 6. Compute reality `[V]`

Shopee was a code competition with a **2-hour GPU inference cap**; the 2nd-place final submission ran
**~1h40m**, and they moved stage-2 inference from **40 min CPU → 2 min** with cuDF/cupy/cugraph/
ForestInference. 6th place spent ~1 month and **300–400 model runs** across a team. Public reference
runtimes on a P100: TF-IDF+pHash baseline **7m09s**; EffNetB1+TF-IDF inference **25m29s**; an ArcFace
*training* notebook **7h06m**.
→ If E-CUP task 1 lands with a 60-minute container cap, the stage-2 reranker must be GBDT-on-CPU or
GPU-accelerated, and candidate generation must be ANN, not a dense all-pairs matmul.
**No top-3 Shopee or ESCI write-up states its training GPU or total training hours** — that number
does not exist publicly.

## 7. Baseline ladder, with real numbers `[V]`

| Stage | Recipe | Score | vs winner |
|---|---|---|---|
| Day-1 | TF-IDF on titles, cosine, single threshold | F1 ≈ 0.646 (own split) | ~83% |
| Day-1 | TF-IDF + image emb + pHash union (7-min GPU notebook) | CV 0.700 / **private 0.690** | **88%** |
| Week-1 | EfficientNetB1 + TF-IDF inference notebook | **private 0.720** | 92% |
| Week-1 | ArcFace embeddings replacing raw cosine | LB 0.700 → **0.730** | 94% |
| Week-1 | ArcFace text-only, `paraphrase-xlm-r-multilingual-v1`, m=0.8 | F1 0.8211 → min2 0.8285 → INB **0.8345** (own split) | — |
| Winner | full ensemble + INB | **private 0.780** | 100% |

**A one-day TF-IDF + KNN + threshold baseline reaches ~88% of the winner's score.** Everything above
~0.76 came from neighbourhood post-processing, not better backbones. `[INF from the verified ladder]`

## 8. DAY-1 for E-CUP task 1 (text-only)

1. **Read the metric first**, then write the scorer as a standalone function and unit-test it against
   any example the organizers give. If it is macro-over-categories, build the per-category report
   from the first submission onward.
2. **TF-IDF (word + char n-gram) on the card text → ANN top-k → single cosine threshold.** This is
   the 88%-of-winner move and it costs hours. Submit it. It also gives the first LB reading.
3. **Immediately after: submit the identical file again** to measure the leaderboard's own noise
   floor (`CAMPAIGN_RULES.md` #2). Two of the five daily submissions, and every later delta becomes
   readable.
4. **Build the validation split with held-out products, not held-out pairs** — group by product
   identity, and carve a slice of *entirely unseen* products to mirror the private set (§1.3).
5. Compute the **oracle ceiling** of candidate generation: with the top-k retrieved set, what is the
   maximum achievable score if the reranker were perfect? That prices the whole reranking lane before
   it is built (`CAMPAIGN_RULES.md` #1).

## 9. WEEK-1 target

Fine-tuned Russian text encoder with ArcFace → L2-normalised embeddings → ANN candidates →
**GBDT reranker** on the stage-2 feature list from §2 (cosine sims, mean/std of top-K for
K=5/10/15/30, Levenshtein, length/word-count, unit- and number-mismatch flags from regex) →
threshold shaping, per-category if the metric is macro-by-category. In parallel, and started early
because it is the biggest single lever: **MLM + pair-objective intermediate training on the
competition's own text** (§1.1). Second decorrelated member built in week one, not week three
(`CAMPAIGN_RULES.md` #5).

## 10. Repos worth cloning
- https://github.com/kiccho1101/kaggle-shopee-6th-place-solution — full training code, 6th place
- https://github.com/jingxuanyang/Shopee-Product-Matching — text-only ArcFace + min2 + INB
- https://github.com/msdw/shopee-product-matching — image+text+PP, public 13th
- https://github.com/megagonlabs/ditto — reference cross-encoder for entity matching
- https://github.com/wbsg-uni-mannheim/contrastive-product-matching — R-SupCon (needs 64 GB+ RAM)
- https://github.com/wbsg-uni-mannheim/sc-block — blocking / candidate generation
- https://github.com/wbsg-uni-mannheim/wdcproducts — the seen/unseen benchmark of §1.3
- Write-ups: Shopee 1st `kaggle.com/…/discussion/238136`, 2nd `…/238022`, 14th (best PP doc)
  `…/238033`; ESCI winner `amazonkddcup.github.io/papers/3782.pdf`; ablations `arXiv:2301.13455`

---

## 12. CORRECTIONS after re-reading the primary write-ups verbatim

The §2 table was built from one pass over the Shopee threads. A second pass, reading them directly
rather than second-hand, changed four things. **These override §2 and §9 where they conflict.**

### 12.1 Query expansion / INB is CONDITIONAL, not a flat gain `[V]`

| Setting | QE / DBA / INB contribution |
|---|---|
| You threshold embedding similarities **directly** | 1st **+0.017**, 7th **+0.024 CV**, 8th **+0.011–0.016** |
| You already have a **learned reranker** consuming rank / density / graph features | 2nd place: **+0.001** |

**Why:** QE injects neighbourhood information into the embedding. A GBDT that already sees
neighbour-similarity mean/std at K∈{5,10,15,30}, rank-within-neighbour-list and local density
*already has that information*.
→ **Since our plan builds the GBDT reranker on day 1, QE drops from a priority to a late-week
experiment, gated on measuring a gain on top of the reranker.**

### 12.2 Feed the cross-encoder LESS text, not more `[V]`
**Ditto used only the `title` attribute on WDC — and that beat title + description + brand +
specTable.** Corroborated by their TF-IDF-summarisation result: on the Company dataset, F1 went
**41% → 93% by truncating** the input to high-TF-IDF tokens.
→ For product matching, **`name` carries nearly all the signal and `description` is mostly
dilution**. The first cross-encoder variant should be **name-only**; adding a TF-IDF-summarised
description is an ablation that must prove itself. This inverts the natural instinct, especially
against Ozon cards whose descriptions run to tens of thousands of characters.

### 12.3 Normalise each view BEFORE concatenating `[V, but the size is contested]`
Reordering two lines of code — concat-then-normalise → normalise-then-concat — was the 1st-place
solution's largest non-structural jump. Algebraically it converts *"whichever view has the larger
norm dominates"* into *"average the per-view cosines"*.
> **Contradiction in the record, unresolved.** One reading of thread 238136 has the ladder as
> 0.724 → min2 → 0.743 → normalise-before-concat → 0.753, i.e. **+0.010**. A second reading has
> normalise-before-concat carrying the whole 0.724 → 0.753, i.e. **+0.029**. The first is internally
> consistent with the separately-reported min2 gain of +0.019, so **+0.010 is the conservative
> number and +0.029 is the ceiling.** Either way the move is nearly free — do it — but do not quote
> +0.029 as established. *(ROGII rule: recompute headlines from artifacts; label contradictions.)*

### 12.4 Hand-written unit/quantity rules beat learned NER `[V]`
Ditto's *learned* domain-knowledge injection was worth **+0.22 F1** on WDC (its NER produced matching
span types for only 66.2% of pairs). Shopee 6th's **hand-written "reject unmatched units"** rule —
killing "Anmum Lacta 400gr" vs "Anmum Materna 200gr" — took affected rows from **F1 0.5 → 1.0**.
→ Hand-roll normalisation and mismatch rules for **объём, вес, размер, количество в упаковке, цвет,
модель/артикул**. Cheaper and more reliable than an NER stage — and it is exactly the "thoughtful
feature engineering" this jury says it scores on.

### 12.5 Two traps that change how we read our own numbers `[V]`
- **Corner-case ratio makes any borrowed F1 target meaningless.** Same model, same training size,
  WDC Large/Seen: RoBERTa scores **87.80 at 20% corner-cases and 65.45 at 80%** — a **22-point swing
  from negative-mining difficulty alone**. Never port an absolute target from another competition,
  and **expect our own score to move sharply whenever we change hard-negative mining** — which means
  a score change after a mining change is not evidence the model improved.
- **The text-only ceiling, quantified.** Shopee: text-only **0.640** LB vs image-only 0.700 vs
  combined 0.793 (CV: text 0.808, image 0.830, concat 0.888). **Roughly 0.09–0.15 of the winning
  score came from having images at all.** E-CUP task 1 is text-only, so that headroom does not
  exist — it has to be recovered from: multiple text views unioned at the candidate stage, the
  reranker, clustering, and intermediate/contrastive pre-training (§1.1).

### 12.6 Supporting numbers `[V]`
- **DeepBlocker — the union effect on candidate recall:** Abt-Buy DL 87.2 / RBB 82.9 / token-blocking
  81.9 → **union(DL, RBB) 95.7**; Hospital2 89.0 → **98.5**; Walmart-Amazon2 68.7 → **83.0**. Optimal
  K ranged **5 to 150** across datasets — so K is a per-dataset measurement, not a default.
- **Ditto label efficiency:** Ditto on **half** the data beats DeepMatcher on **all** of it by 2.89 F1
  on WDC-All; at 1/8 the data it is within 1%.
- **Ditto's one failure mode:** the Shoes category *regressed* (−2.50, −2.32 F1), attributed to a
  positive-rate gap between train (9.76%) and test (27.27%) — **a per-category prior-shift warning**,
  directly relevant if our metric is macro-over-categories.
- **Intermediate training zero-shot:** 92% F1 on WDC computers with **no task-specific fine-tuning at
  all**.
- **CIKM Cup 2016:** unsupervised **transitive inference** lifted final F1 0.4155 → 0.4204 and was
  called "critical to win"; k=18 chosen off an explicit recall-vs-k curve.
- **ESCI saturation:** 1st→10th on Task 2 spans **1.4 micro-F1 points**. Ensembling buys rank, not a
  category change.
- **ArcFace is genuinely hard to train.** Shopee 1st needed four simultaneous fixes (margin ramp
  0.2→1.0, large warmup, higher LR on the cosine head, gradient clipping); Shopee **3rd never beat
  triplet loss with it**; Shopee 2nd found **CurricularFace better**. → Budget time for it and keep a
  triplet / multi-similarity fallback rather than assuming ArcFace works.

### 12.7 Revised priority order
- **Moved up:** normalise-per-view-before-concat (~free, §12.3); unit/quantity mismatch rules (§12.4),
  alongside the attribute features.
- **Moved down:** QE / DBA / INB — from a week-1 item to a late experiment gated on a measured gain
  on top of the reranker (§12.1).
- **Changed default:** cross-encoder input starts **name-only**; description is an ablation (§12.2).

### 12.8 Tooling note
Kaggle discussion threads are fully readable through the reader proxy —
`curl -sSL "https://r.jina.ai/https://www.kaggle.com/competitions/<comp>/discussion/<id>"` — where a
plain fetch returns only the page title. **Thread 240667 is a curated index of every Shopee
place-solution.** Worth having during the competition.

---

## 11. Gaps in this evidence
No top-3 Shopee/ESCI write-up states its training hardware or hours. Shopee's private rank↔score
mapping below 2nd is reconstructed, not read. CurricularFace hyperparameters are named but not
published. The WDC EDBT camera-ready could not be located, so a small Ditto-column discrepancy
between the arXiv and website versions is unadjudicated. **DOLG appears in no Shopee top solution** —
if anyone recommends it citing Shopee, that citation is wrong.

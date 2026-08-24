# Architecture scout: cheap feasibility screen

## Inventory decision

The only complete classifier-ready checkpoint not already tested on matching is
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, recovered locally at
`scratch-q2/newtasks/cascade-bert-reranker/models/mmarco-mMiniLMv2-L12-H384-v1`.
It is a 12x384 distilled XLM-R encoder with a multilingual mMARCO cross-encoder
head. It is smaller and differently distilled/pretrained than current models,
although its SentencePiece vocabulary is the same XLM-R vocabulary used by e5;
therefore diversity is architectural/objective, not tokenizer-vocabulary diversity.

Snowflake Arctic and BGE-M3 are excluded because they repeat the already-tested
XLM-R-base/large embedding family. Cached 2-4B decoder/VL models lack a compatible
sequence-classification head and are not credible under the 20-minute container
budget. ruRoBERTa, rubase, e5, and mDeBERTa are excluded by prior evidence.

## Screen

- Held-out fold: fold_01, full 91,157-row evaluation.
- Training subset: deterministic 40,000 rows from folds 02-04, sampled equally
  across 20 categories and, inside category, up to the observed class mix.
- Every arm: exact original `name | category | attrs`, max length 160, batch 128,
  500 optimizer updates, AdamW LR 2e-5, seed 20260814, identical sampled rows/order.
- Arms: current `rubase_llmfull_e2` matched baseline; mMARCO MiniLM candidate;
  same MiniLM configuration with random weights as pipeline/pretraining control.
- Primary: macro-category PR-AUC. Secondary: pooled PR-AUC, fixed within-category
  rank blends (10%, 25%, 50% candidate), throughput, eval speed, peak GPU memory.

Promote MiniLM to matched full folds 01-02 only if it is within 0.03 macro AP of
the baseline and a preregistered blend improves macro AP by >0.001, while measured
inference speed is no slower than baseline. Otherwise stop after the screen.

In parallel, a CPU audit measures fold predictability from text-shape features and
tokenizer fertility. No validation, submission, push, or commit is allowed.

## Separate claim: fixed 10% blend-diversity confirmation

The standalone promotion gate failed, so this does not override or promote the
MiniLM model itself. Fold 01 nevertheless showed +0.0023618 macro-category AP for
a 10% within-category rank blend. Freeze that weight and run the exact same cheap
protocol on fold 02, with no 5%/25% tuning. Compare against two fixed controls:

- 10% blend with random-initialized MiniLM ranks;
- 10% blend with pretrained MiniLM ranks shuffled deterministically inside each
  category (same coverage and score distribution, destroyed row alignment).

Only if pretrained 10% is >+0.001 on both folds and exceeds each control by at
least 0.001 macro AP do we preregister a full-training folds 01-02 follow-up.

## Full-training folds 01-02 follow-up (preregistered after confirmation pass)

The fixed 10% diagnostic passed: `+0.0023618` on fold 01 and `+0.0015620`
on fold 02; random-init and shuffled controls were negative on both. The full
follow-up is therefore:

- candidate: pretrained mMARCO MiniLM, trained on
  `hand_pairs_pd_v3cal.parquet` with the exact standard hand budget: category +
  attrs, max length 224, pair-order swap augmentation and two-direction eval,
  2 epochs, batch 256, LR 2e-5, seed 20260814;
- exact strong baseline: the saved fixed-seed `rubase_llmfull_e2` v3cal rerun
  from `category_blend_distill`, produced with the same data and hand recipe;
- primary: fixed 90% baseline / 10% MiniLM within-fold/category percentile-rank
  blend, macro-category hard-label AP; no weight or category selection;
- full-data control: deterministically shuffle the trained MiniLM ranks inside
  each fold/category before the same 10% blend. The already-saved full cheap
  random-init control remains evidence about the discovery screen; a costly
  full random-init rerun is not useful for deployability and is not promoted;
- gate for folds 03-04: blend delta strictly greater than +0.001 on each of
  folds 01 and 02. Standalone MiniLM is reported but cannot be promoted by this
  claim.

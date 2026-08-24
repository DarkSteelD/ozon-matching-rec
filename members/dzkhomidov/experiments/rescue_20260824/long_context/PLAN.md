# Long-context teacher hypothesis

Claim: increasing RuBERT cross-encoder hand-FT context beyond 384 tokens improves
pooled fold PR-AUC by more than 0.001 on both fold_01 and fold_02, or gives a
clear macro-category / worst-category improvement, with all other settings fixed.

## Fixed setup

- init/tokenizer: `rubase_llmfull_e2`
- data: `hand_pairs.parquet` (`name | category | attrs`)
- folds: frozen `fold_01`..`fold_04`
- seed: 20260814
- epochs: 2
- effective batch: 256 (micro-batch 128, accumulation 2)
- learning rate: 2e-5; AdamW / OneCycleLR unchanged
- GPU: `avi-gn-fsk35`, physical GPU 3
- scratch only: `/home/dzkhomidov/matching-work/rescue_20260824/long_context`

## Matrix

| variant | folds 1-2 | folds 3-4 | role |
|---|---|---|---|
| len224 | checked | not planned | negative control |
| len384 | checked | not run: gate failed | baseline |
| len448 | checked | not run: gate failed | candidate |
| len512 | checked | not run: gate failed | candidate |

Gate: continue the winning candidate to folds 3-4 only if its delta vs len384
is >0.001 on both initial folds, or it has a clear macro-category / worst-category
gain. No container, submission, validation-directory write, push, or commit.

Gate decision: failed. Neither candidate exceeded +0.001 pooled on both folds,
and neither produced a clear macro or worst-category gain.

Historical reference (not a substitute for rerun): len384 fold_01=0.84155524,
fold_02=0.85106406, 4-fold mean=0.84531865.

Operational note: physical GPU 2 was initially assigned, but the staged runner
retained `CUDA_VISIBLE_DEVICES=3` and started exact PID 1686747 on idle physical
GPU 3. GPU 3 had no co-tenant; the coordinator explicitly authorized continuing
that exact process without migration or interruption.

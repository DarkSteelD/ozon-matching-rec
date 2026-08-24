# Preserving pair-pretraining by partial encoder freezing

## Outcome

**Negative for freezing the bottom six transformer blocks.** Against the exact
fully-unfrozen epoch-3 baseline, bottom-six freezing loses `-0.008859` macro
category AP on fold 01 and `-0.007930` on fold 02. It therefore fails the
predeclared `>+0.001` gate by a wide margin and with the same negative sign on
both folds.

| variant | fold | macro AP | delta vs full | pooled AP | pooled delta | status |
|---|---:|---:|---:|---:|---:|---|
| full | 01 | 0.801528 | - | 0.851552 | - | checked/reused exact |
| bottom6 | 01 | 0.792668 | -0.008859 | 0.844686 | -0.006867 | checked |
| full | 02 | 0.805875 | - | 0.859680 | - | checked/reused exact |
| bottom6 | 02 | 0.797945 | -0.007930 | 0.853878 | -0.005802 | checked |
| full mean | 01-02 | 0.803701 | - | 0.855616 | - | checked |
| bottom6 mean | 01-02 | 0.795306 | **-0.008395** | 0.849282 | **-0.006334** | gate failed |

The loss is broad, not a single-category accident: bottom6 is worse in all
40/40 category-fold cells. The smallest mean category losses are Hobby
`-0.003226`, Food `-0.003243`, and Pet supplies `-0.003337`; the largest are
Clothing `-0.019393`, Shoes `-0.018622`, Jewelry `-0.016847`, and Accessories
`-0.015681`.

## Exact protocol

- Initialization: epoch-3 pair-pretrained RuBERT checkpoint, model SHA256
  `44825c298e61dafa3b0a4f43a8eac93feff9df1ec0a16f3d09c36ac639d953dd`.
- Hand targets: `hand_pairs_pd_v3cal.parquet`, SHA256
  `b9ebd015f1881c1ac58b5966233b74390a25f13bf751af9a72dafc803c106af9`.
- Scoring labels: row-aligned `hand_pairs.parquet`, SHA256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`.
- Both variants use max length 224, symmetric pair-order augmentation and
  two-direction evaluation, two epochs, effective batch 256/microbatch 128,
  LR `2e-5`, OneCycle linear schedule, seed `20260814`, and exactly 2144/2142
  optimizer steps on folds 01/02.
- Bottom6 freezes only BERT encoder blocks 0-5. Embeddings, pooler, classifier,
  and blocks 6-11 remain trainable. This leaves 135,326,977 of 177,854,209
  parameters trainable (76.09%); 42,527,232 parameters are frozen.
- The full baseline predictions are the exact archived `e3_len224` factorial
  arm. Its trainer SHA256 was
  `46b10630fc1bb11140eb5326fded09f99c2e0c6cd9f7154c898c7e27b355970d`.
  The candidate trainer is that file plus only freeze selection, trainable
  parameter filtering/accounting, and omission of unused checkpoint writes.
  Baseline row hashes and scores reproduce the earlier factorial exactly.

The reused baseline is also a positive pipeline control: it reproduces the
previously recorded e3@224 macro values `0.801528/0.805875` exactly, so the
large negative candidate result is not a scoring or row-alignment failure.

## Speed and resource cost

Bottom6 took 502.5s and 502.6s per fold after shared tokenization, with about
4.70 optimizer steps/s, 12.7-13.1 GiB GPU memory, and 99% utilization on
`avi-ix-devbox02` H100 GPU3. The archived full baseline took 401.7s/399.1s and
about 5.82 steps/s on a different H100 session. Thus the observed frozen run
was about 25% slower per fold despite 23.9% fewer trainable parameters. This is
an observed cross-session cost, not a controlled speed claim: embeddings remain
trainable, so backward propagation still crosses the frozen lower blocks, and
host/session load differs.

Physical GPU3 was double-checked before launch, protected by the registry lock,
and released at the priority change. Final state was 1 MiB, 0% utilization, no
compute application. No process was launched on fsk35.

## Interpretation and limits

The tested preservation mechanism is not supported. Preventing hand-label
updates in lower blocks does not retain useful unseen-entity structure here; it
removes adaptation needed across every category, especially the four difficult
fashion categories.

The matched `top6` parameter-count control was not run because submission work
became the priority while bottom6 fold 02 was active. The queue runner was
stopped while its child safely finished writing predictions, then terminated
before top6 could start. Therefore this experiment cannot separate bottom-layer
specific harm from the harm of freezing any six blocks. That missing control
does not change the primary gate failure for bottom6.

Not run by design after the failed gate/priority change: top6 folds 01-02 and
all variants on folds 03-04. Unchecked: freezing embeddings with bottom blocks,
fewer frozen blocks, gradual unfreezing, other seeds, hidden-test transfer.
No direction-wide claim about all freezing schedules is made.

TP/FP/FN are not defined because this is rank-only PR-AUC without a binary
decision threshold.

## Artifacts

- `PLAN.md`, `train.py`, `score.py`, `run_locked.sh`.
- `preds/full/` and `preds/bottom6/`: complete gate-fold predictions.
- `metrics.json`: pooled, macro, and per-category AP.
- `runtime_{full,bottom6}_fold_0{1,2}.json`: timing and parameter counts.
- `logs/gate.log`, `logs/score_partial.log`: complete run/score output.
- `HASHES.sha256`, `COMMANDS.md`: artifact and process ledger.

No validation output, container, submission, repository source, commit, or push
was changed.

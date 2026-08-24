# 2x2 composition: epoch-3 pretrain × max_len384

## Compatibility decision

Composition is valid: epoch-2 and epoch-3 are both the same 12-layer RuBERT
`BertForSequenceClassification` (hidden 768, 12 heads, vocab 119547, type vocab
2, max positions 512) and have identical tokenizer SHA256
`aa68ce2535cea6f63667df615622150c10a72aae14e5ec44285edc4d1c88708a`.
Epoch-3 differs only by one continued pass over the same LLM-pretrain cache.

## Exact factorial

All four cells use the same v3cal soft-target hand data, symmetric pair-order
training/evaluation, seed 20260814, folds 01-02, two epochs, effective batch 256,
microbatch 128, LR 2e-5 and identical update counts:

| pretrain | len224 | len384 |
|---|---|---|
| epoch2 | fresh matched `e2_len224` | reuse exact `student_long_context/len384` |
| epoch3 | run `e3_len224` | run `e3_len384` |

The earlier len224 process loaded a pre-save copy of the trainer, so it is not
reused under the exact-hash rule; `e2_len224` is rerun fresh. Reuse of e2@384 is
allowed only after prediction row coverage, data hash, trainer/config and
checkpoint identities are verified. Composition delta is `e3_len384-e2_len224`.
Interaction is `(e3_len384-e2_len384)-(e3_len224-e2_len224)` per fold.

Gate: composition must improve macro category PR-AUC by >0.001 on each fold.
Single-factor cells are the mechanism controls. No folds 03-04 unless the gate
passes. No validation/submission/repository/commit/push writes.

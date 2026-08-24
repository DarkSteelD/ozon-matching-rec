# Deployable v3cal student: max_len 384 vs 224

Claim: keeping 384 tokens in the single symmetric RuBERT v3cal student improves
hard-label macro category PR-AUC over an otherwise exact max_len 224 rerun.

- Host/GPU: `avi-gn-fsk35`, physical GPU7 after live check.
- Same init `rubase_llmfull_e2`, soft targets `hand_pairs_pd_v3cal.parquet`, seed
  20260814, folds 01-02, two epochs, effective batch 256, lr 2e-5, identical
  update count, pair swap augmentation and two-direction evaluation.
- Baseline/negative control: unchanged len224 training at equal updates.
- Candidate: len384; only token budget differs.
- Gate: candidate delta >+0.001 mean macro AP and positive on both folds.
- Outputs remain here; no validation/submission/repository writes.

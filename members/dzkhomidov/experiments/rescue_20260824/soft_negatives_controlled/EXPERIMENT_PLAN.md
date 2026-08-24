# Controlled paired soft-negative rerun

Status at creation: all cases unchecked.

Claim: with all RNGs reset to the same fold-specific seed before each model is
initialized, `hard050` improves fresh macro-category PR-AUC by more than 0.001
against a fresh baseline on both folds 01 and 02. Only if that gate passes are
folds 03 and 04 run.

Fixed design:

- host `avi-ix-devbox03`, physical GPU 3 only after two live compute checks;
- data `/home/dzkhomidov/matching-work/data/hand_pairs.parquet`;
- OOF hardness `/home/dzkhomidov/matching-work/preds_disk/ce_rubase_e2_len224`;
- init `/home/dzkhomidov/matching-work/ckpt_disk/rubase_llmfull_e2`;
- seed `20260814 + fold_number`, reset via Python, NumPy, torch CPU and all CUDA RNGs;
- deterministic CUDA algorithms on, cuDNN benchmark off, TF32 off,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- two epochs, batch 192, max length 224, LR 2e-5;
- baseline and hard050 use identical initialization and minibatch order;
- hard050 downweights the per-category top 10% OOF-scored training negatives to 0.50;
- a duplicate baseline on folds 01/02 audits residual nondeterminism; no random
  treatment arm is run in this controlled experiment.

Primary metric: fresh hard050 minus fresh baseline macro-category PR-AUC.
Stage-1 gate: delta strictly above +0.001 on each of folds 01 and 02.

Artifacts remain isolated under this directory. Existing soft-negative outputs
are read-only evidence and are not overwritten. No validation, submission,
push, or commit is authorized.

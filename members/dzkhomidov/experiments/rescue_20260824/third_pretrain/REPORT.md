# Third rubase pretraining epoch — controlled result

## Outcome

**Positive.** One additional pass over the 11,187,780 soft-label pairs improves
the identical hand fine-tune on all four component-disjoint folds and repeats
on an independent Torch/NumPy seed on different hardware.

Primary four-fold mean pooled PR-AUC improves from `0.83414024` to `0.83555855`:
`+0.00141831`. Fold-delta standard deviation is `0.00048446`; the descriptive
four-fold t interval is `[+0.00064742, +0.00218920]`. This interval is not a
substitute for a new test set.

Competition-aligned macro-category PR-AUC improves from `0.77279458` to
`0.77400699`: `+0.00121241`. Nineteen of twenty categories have a positive
mean delta; jewelry is the exception (`-0.00315438`).

## Controlled results

| init | seed / hardware | fold | pooled PR-AUC | delta vs epoch2 | status | artifact |
|---|---|---:|---:|---:|---|---|
| epoch2 | 20260814 / H100 | 01 | 0.83116687 | — | checked | `preds/hand_e2_ctrl_gate/fold_01.csv` |
| epoch3 | 20260814 / H100 | 01 | 0.83299934 | +0.00183247 | positive | `preds/hand_e3_ctrl_gate/fold_01.csv` |
| epoch2 | 20260814 / H100 | 02 | 0.83972471 | — | checked | `preds/hand_e2_ctrl_gate/fold_02.csv` |
| epoch3 | 20260814 / H100 | 02 | 0.84151599 | +0.00179128 | positive | `preds/hand_e3_ctrl_gate/fold_02.csv` |
| epoch2 | 20260814 / H100 | 03 | 0.83138367 | — | checked | `preds/hand_e2_ctrl_rest/fold_03.csv` |
| epoch3 | 20260814 / H100 | 03 | 0.83261299 | +0.00122932 | positive | `preds/hand_e3_ctrl_rest/fold_03.csv` |
| epoch2 | 20260814 / H100 | 04 | 0.83428569 | — | checked | `preds/hand_e2_ctrl_rest/fold_04.csv` |
| epoch3 | 20260814 / H100 | 04 | 0.83510587 | +0.00082017 | positive, below 0.001 | `preds/hand_e3_ctrl_rest/fold_04.csv` |
| epoch2 | 20260825 / A100 | 01 | 0.83080600 | — | checked | `seed_replication_20260824/preds/hand_e2_seed20260825/fold_01.csv` |
| epoch3 | 20260825 / A100 | 01 | 0.83270610 | +0.00190010 | positive | `seed_replication_20260824/preds/hand_e3_seed20260825/fold_01.csv` |
| epoch2 | 20260825 / A100 | 02 | 0.83999806 | — | checked | `seed_replication_20260824/preds/hand_e2_seed20260825/fold_02.csv` |
| epoch3 | 20260825 / A100 | 02 | 0.84180572 | +0.00180766 | positive | `seed_replication_20260824/preds/hand_e3_seed20260825/fold_02.csv` |

The independent-seed mean delta on folds 01–02 is `+0.00185388`. The original
seed mean on the same two folds is `+0.00181187`; the difference is only
`+0.00004201`. Across all six checked fold/seed comparisons, the mean delta is
`+0.00156350` with standard deviation `0.00043849`.

## Controls and mechanism

- Baseline and candidate use the same rows, component folds, tokenization,
  max length 160, two hand epochs, batch 256, LR 2e-5, category/attributes,
  and explicit Torch/CUDA/NumPy seed. Only the initialization checkpoint
  differs.
- Archived epoch2 OOF row alignment and scoring were independently reproduced.
- Source and destination SHA-256 hashes match; see `SOURCE_HASHES.sha256`.
- The second seed ran on local physical A100 GPU2 after two 0%-utilization
  checks. The resident Streamlit and kernel processes remained untouched.
- The broad 19/20-category direction supports a general representation gain
  from continued pretraining rather than domination by one category. This is
  mechanism-consistent evidence, not proof of which learned features changed.

## Runtime and resource cost

- Original controlled H100 run: roughly 11–12 minutes per two-fold variant,
  based on saved logs at about 8.9–9.0 steps/s.
- Independent A100 run: about 20 minutes for epoch2 folds 01–02 and 19 minutes
  for epoch3 folds 01–02, at 4.3–4.4 steps/s, plus tokenization included in
  each log.
- The fsk35 checkpoint and predictions were copied read-only and verified;
  no new Python/CUDA process was launched there. Devbox03 GPU1 and its lock
  were released at 0 MiB / 0% utilization. Local GPU2 lock was released after
  completion; it returned to 0% utilization with only its pre-existing
  resident processes.

## Reproduction

The exact base command is recorded in `COMMANDS.md`. The independent replication
uses `seed_replication_20260824/run_hand_seed.py` with `--seed 20260825` and
folds `fold_01,fold_02`, once with epoch2 init and once with epoch3 init. Raw
logs, predictions, metrics JSON, four-fold category analysis, and hashes are
all stored beside this report.

Checked: seed 20260814 folds 01–04; seed 20260825 folds 01–02; pooled and
macro-category metrics; source/destination hashes. Unchecked by design: seed
20260825 folds 03–04, hidden test/leaderboard transfer, and deployment latency.
No validation write, submission, push, or commit was performed.

Repository provenance in the original plan is SHA
`2da459984a1207677ff9eb863ca28589027a4bc3`; the current local `.git` directory
is empty, so that SHA could not be re-verified in this recovery session.

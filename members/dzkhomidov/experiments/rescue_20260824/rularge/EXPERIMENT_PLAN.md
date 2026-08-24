# ruRoBERTa-large matching scale screen

## Measurable claim

With the frozen matching hand data, split, text format, seed, two epochs,
effective batch 256 and length 128, generic `ruRoberta-large` improves PR-AUC
over the comparable generic `rubert-base` hand-only control on both folds 01
and 02. A second claim is diversity: adding 10% rank weight from the large
model improves the frozen `final_combo` on both folds.

## Variants and controls (initially unchecked)

| variant | folds | status |
|---|---|---|
| `rubase_hand_control` (existing immutable OOF) | 01, 02 | available, not rescored |
| `rularge_hand` | 01, 02 | deferred: GPU ownership block |
| `final_combo_control` (existing immutable OOF) | 01, 02 | available, not rescored |
| `0.9*rank(final_combo)+0.1*rank(rularge)` | 01, 02 | deferred with candidate |
| group-unaware deterministic permuted rularge score | 01, 02 | deferred with candidate |

The permutation is only a pipeline sanity check, not evidence about model
quality. Acceptance for further LLM pretraining requires candidate improvement
on both folds; a mixed-sign result is inconclusive. Fold spread is reported as
the current noise proxy. No seed replication is claimed by this screen.

## Frozen inputs and configuration

- Data: `/home/dzkhomidov/matching-work/data/hand_pairs.parquet`, SHA-256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`.
- Candidate backbone: local `ai-forever/ruRoberta-large`, 24 layers, hidden
  1024, 355M parameters; generic MLM weights, no matching/quality checkpoint.
- Input: `name | category | attrs`, pair cross-encoding, max length 128.
- Seed 20260814; folds 01 and 02; two epochs; AdamW; LR 1e-5; effective batch
  256 (micro-batch 64, accumulation 4); bf16 autocast and TF32.
- Comparable control: existing `ce_rubase_hand`, max length 128, two epochs,
  batch 256, LR 2e-5, same seed/data/text fields. The LR differs by the usual
  stable large-model setting and is explicitly not a pure parameter-count
  ablation.
- Strong control: existing immutable `final_combo` OOF.
- Source repository identity: recovery branch SHA
  `2da459984a1207677ff9eb863ca28589027a4bc3`.

## Decision ladder

1. Run hand-only folds 01-02 and score candidate/control identically.
2. If scale signal is positive on both folds, run full 11.19M LLM pretrain.
3. If mixed-sign or negative, do not spend the 11M run; report the direction as
   inconclusive/negative for this exact configuration only.
4. If pretraining proceeds, fine-tune folds 01-02 first and require improvement
   over both `ce_rubase_e2_hand` and hand-only large before folds 03-04.

No files under any `validation/` directory are written. No submit, commit,
push, or external message is authorized.

## Live scheduling status

At 2026-08-24 01:26 MSK, devbox01 GPU 1 showed 1 MiB used, 0% utilization and
no compute-app rows. The root coordinator subsequently stated that devbox01
GPU 0/1 are occupied by services and must not be used. At 01:35 MSK the
coordinator explicitly assigned `avi-gn-fsk35` GPU 4 for the cheap controls.
Its fresh check showed 4 MiB used, 0% utilization and no compute-app rows.
Only hand-only folds 01-02 and the 200-update benchmark gate are authorized;
full pretraining still requires measured ETA below 12 hours.

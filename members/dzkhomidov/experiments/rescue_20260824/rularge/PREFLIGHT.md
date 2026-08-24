# Preflight — 2026-08-24

## Inventory

- Generic model is complete locally at
  `/home/dzkhomidov/ozon-hack/shared_models/ai-forever/ruRoberta-large`.
  `model.safetensors` is 1,627,787,592 bytes, SHA-256
  `d41f460530005a52224acf68dcec86fd8c7ce52da0530ab32663b4d2376321d8`.
  Architecture: RoBERTa, 24 layers, hidden size 1024, 16 heads, vocabulary
  50,265, max positions 514; README reports 355M parameters.
- Both `model.safetensors` and a redundant 1,421,754,926-byte
  `pytorch_model.bin` are present. Only safetensors plus config/tokenizer files
  need staging.
- Tokenizer loads as `RobertaTokenizerFast`; an exact pair produces 128 ids and
  no token-type ids. Tokenizer `pad_token_id=0`, while the upstream model config
  says `pad_token_id=1`. This is a real upstream compatibility risk and is one
  reason the hand-only positive control must precede 11M pretraining. Existing
  quality experiments used the same local model and did train successfully,
  but they do not prove matching quality.
- Padding decision for the cheap controls: preserve the upstream model
  `pad_token_id=1`/position-id convention and build the attention mask from the
  tokenizer's actual `<pad>` id 0. Do **not** patch the pretrained config to 0:
  that would shift positional behavior relative to the released weights and
  create a new unvalidated model. The inconsistency remains a documented
  upstream risk; practical acceptance depends on the hand-only control.
- Hand data: 121,525,891 bytes, SHA-256
  `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`.
- LLM data: 11,187,780 rows, 91 parquet row groups, 2,368,157,697 bytes,
  SHA-256 `47aa9ab675078a6b2074138d8a106ece48460fcdfbf4619efe60d9c21660028f`.
- No ruRoBERTa-large matching checkpoint or surviving token cache exists in
  `/home/dzkhomidov`, `/dev/shm`, or matching checkpoint directories.

## What `logs_tok_rularge.log` means

The log is a successful historical tokenization, not evidence of a surviving
cache or trained model. The exact command used `ruRoberta-large`, the full
11.19M LLM parquet, length 128, attrs+category, 12 workers, and cache prefix
`tok_rularge_len128`. All 46 two-row-group slices completed. File timestamps
span 22:16:54–22:29:17 MSK (about 12m23s); the last measured aggregate rate is
15,110 rows/s. Claude history then records explicit deletion of the ids,
targets and done-marker directory at about 22:31 MSK. Therefore the log's
`cache complete` is historically true and presently stale.

## Disk and time estimate

- Token ids at `[11,187,780, 128] int32`: 5,728,143,488 bytes (5.335 GiB).
- Targets: 44,751,248 bytes (0.042 GiB).
- Complete token cache: 5,772,894,736 bytes (5.376 GiB), plus tiny markers.
- Remote staging: about 4.0 GB for source parquet plus minimal model.
- Exact-resume state contains model+AdamW moments+scheduler. Atomic replacement
  temporarily holds both old and new states. With final model and cache, reserve
  **22 GB peak** (conservative); do not use the local 13 GB-free `/home`.
- Historical tokenization ETA: 13 minutes; budget 20 minutes.
- Measured hand-only runtime on fsk35 H100: 629.6s / 632.5s per fold, plus 78.7s
  one-time tokenization; about 22.3 minutes wall for both folds.
- Measured exact 200-update benchmark: 2.9186 updates/s, projecting 4.16 hours
  for the 43,702-update full epoch. The ETA gate passed, but the independent
  hand signal gate failed, so full pretraining was not started.

## Resources and ownership

- Local host `avi-ling-gpu03`: `/home` has only 13 GB available (100% rounded),
  so no cache regeneration is safe there. `/dev/shm` has 241 GB available but
  is shared and was not allocated by this task.
- Queried host `avi-ix-devbox01`, GPU 1: NVIDIA H100 PCIe 81,559 MiB; snapshot at
  01:26:04 MSK showed 1 MiB used, 0% utilization, and no compute-app rows.
  Remote `/home` had 3.8 TB free and `/dev/shm` 450 GB free.
- Root coordinator subsequently declared devbox01 GPU 0/1 occupied by services
  and forbidden. That ownership information overrides the empty compute-app
  list. No model/data were staged remotely and no GPU process was launched.
- The coordinator then assigned `avi-gn-fsk35` GPU 4 for cheap controls only.
  A fresh 01:35:01 MSK snapshot showed an NVIDIA H100 80GB HBM3, 4 MiB used,
  0% utilization and no compute-app rows. `/home` had 122 TB free and
  `/dev/shm` about 1 TB free. The run still performs a second live check around
  an atomic per-card lock.

## Controls

- Comparable negative/baseline control: existing `ce_rubase_hand`, folds
  01/02 PR-AUC 0.78703354 / 0.79647773, same hand data, length 128, two epochs,
  batch 256 and seed; backbone LR is 2e-5 versus registered 1e-5 for large.
- Candidate positive-control requirement: large hand-only must improve both
  folds before any LLM pretrain. A mixed-sign result is inconclusive.
- Pipeline negative control: deterministically permuted large OOF scores must
  collapse; this is prepared in `score_screen.py`.
- Strong marginal control: 10% rank blend into immutable `final_combo` must
  improve both folds, not only candidate standalone PR-AUC.
- Pretrain positive control: after LLM pretrain, large must beat its own
  hand-only result and the existing `ce_rubase_e2_hand` on both folds 01/02.
  Existing base-scale LLM gain is context, not proof for large.

All candidate/control results remain unchecked because GPU work was deferred.

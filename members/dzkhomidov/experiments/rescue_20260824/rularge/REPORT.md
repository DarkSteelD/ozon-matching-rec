# ruRoBERTa-large feasibility report

## Outcome

Status: **negative for hand-only scale; full pretrain not launched**. The model and data are
usable enough to prepare a controlled experiment, but there is no surviving
matching-pretrained large checkpoint or token cache. The only old large-model
artifact is a successful tokenization log whose 5.38 GiB cache was explicitly
deleted minutes later.

No work was launched on service-owned devbox01. The coordinator subsequently
assigned `avi-gn-fsk35` GPU 4 for cheap controls. Both allowed jobs completed;
GPU 4 is empty again. Hand lock PID was `1692407` (01:36:28–01:59:08 MSK), and
the 200-update benchmark lock PID was `1751360` (01:59:36–02:01:03 MSK).

## Result table

| variant | fold/seed | metric | delta vs baseline | status | artifact |
|---|---:|---:|---:|---|---|
| rubase hand control | fold_01 / 20260814 | 0.78703354 | — | existing immutable | `preds_disk/ce_rubase_hand/fold_01.csv` |
| rubase hand control | fold_02 / 20260814 | 0.79647773 | — | existing immutable | `preds_disk/ce_rubase_hand/fold_02.csv` |
| rularge hand | fold_01 / 20260814 | 0.78973171 | +0.00269818 | checked | `preds/rularge_hand/fold_01.csv` |
| rularge hand | fold_02 / 20260814 | 0.79467842 | -0.00179931 | checked | `preds/rularge_hand/fold_02.csv` |
| rularge hand | mean folds 01-02 | 0.79220507 | +0.00044944 | mixed-sign / inconclusive standalone | `metrics_screen.json` |
| final_combo control | fold_01 | 0.85195623 | — | existing immutable | `preds_disk/final_combo/fold_01.csv` |
| final_combo control | fold_02 | 0.86037441 | — | existing immutable | `preds_disk/final_combo/fold_02.csv` |
| final_combo + 10% rularge rank | fold_01 | 0.85143312 | -0.00052311 | checked negative | `metrics_screen.json` |
| final_combo + 10% rularge rank | fold_02 | 0.85953029 | -0.00084412 | checked negative | `metrics_screen.json` |
| final_combo + 10% rularge rank | mean folds 01-02 | 0.85548170 | -0.00068362 | checked negative | `metrics_screen.json` |
| permuted rularge OOF | mean folds 01-02 | 0.25660917 | — | checked sanity pass | `metrics_screen.json` |
| 11M pretrain throughput | updates 1-200 | 2.9186 updates/s | projected 4.16h/epoch | checked ETA pass, hard stop | remote `pretrain/training_state.pt` |

The two-fold standard deviation of the hand control is 0.00668; this is fold
heterogeneity, not a seed-noise estimate. Prior same-architecture work suggests
sub-0.001 changes require replication. This screen therefore requires the same
sign on both folds and does not call the +0.00045 mean delta positive. Pooled
over the two folds, large is actually -0.000280 versus base. Rank correlation
is 0.923 with hand-only base and 0.893 with `final_combo`, consistent with the
negative marginal blend result.

The category mechanism is also unfavorable for the actual rescue target.
Pooled folds 01-02, large loses PR-AUC versus hand-only base on all four fashion
categories: Обувь -0.02580, Ювелирные изделия -0.02239, Галантерея -0.02151,
Одежда -0.00857. Macro mean of the 20 category deltas is -0.00144. The gains
instead concentrate in Электроника (+0.01248), Музыкальные инструменты
(+0.01181), Строительство (+0.00765), Автотовары (+0.00655), and food
(+0.00537). Generic scale therefore does not address the hidden fashion
failure even locally.

Runtime/resource cost: fold 01 trained in 629.6s and fold 02 in 632.5s after
78.7s shared tokenization; GPU memory was about 15.2 GiB. The rebuilt full
cache completed all 46 slices in about 8.6 minutes. The 200-update benchmark
used the exact registered effective batch 256 and stopped at 2.9186 updates/s,
projecting 4.16 hours for 43,702 updates. Its exact resume state is 4,264,835,454
bytes and was verified to contain 393 model tensors, optimizer state, scheduler
`last_epoch=200`, epoch 0 and update 200.

## Prepared artifacts

- `EXPERIMENT_PLAN.md`: claim, controls, frozen settings and acceptance ladder.
- `PREFLIGHT.md`: model/data/cache audit, hashes, disk and ETA evidence.
- `train_hand_screen.py`: resumable-by-fold hand-only folds 01-02 run.
- `score_screen.py`: candidate, hand baseline, strong blend and permutation
  scoring outside all validation directories.
- `tokenize_llm.py`: resumable 46-slice full-cache build.
- `train_pretrain_resumable.py`: exact atomic resume including model, optimizer,
  scheduler, epoch and update.
- `FULL_PRETRAIN_COMMAND.md`: exact deferred commands and 200-update ETA gate.
- `run_locked.sh`: compute-app + atomic lock checks for the completed hand run.
- `run_benchmark_locked.sh` and `supervise_benchmark.sh`: exact sequential
  benchmark gate with fresh ownership checks and hard stop at update 200.

## Recommendation

Do not continue the exact resume state into a full epoch. The runtime gate
passed comfortably (4.16h < 12h), but the more important signal gate failed:
standalone scale is mixed-sign, the strong-stack marginal is negative on both
folds, and fashion macro behavior is materially worse. This does not prove
that LLM-pretrained ruRoBERTa-large can never help; it says the evidence does
not justify four more GPU-hours before cheaper, better-targeted arms finish.

After staging the task directory and immutable inputs on an explicitly cleared
host, the hand screen launcher is:

```bash
cd /home/dzkhomidov/matching-work/rescue_20260824/rularge
nohup bash run_locked.sh > hand_screen.log 2>&1 &
echo $! > hand_screen.pid
```

Its shell PID is the lock owner and remains recorded in `hand_screen.pid`, the
resource registry and lock `OWNER`. The full deferred command is recorded in
`FULL_PRETRAIN_COMMAND.md`.

Padding decision: retain upstream config `pad_token_id=1` for its positional
convention, while attention masks use tokenizer `<pad>` id 0; patching config
would create a different unvalidated pretrained model. The discrepancy remains
a caveat, but the tested path matches released config behavior and the prior
quality runs.

Checked: inventory, provenance, hashes, cache rebuild, syntax, hand metrics,
category deltas, negative control, strong-stack marginal, actual runtime/memory,
200-update H100 throughput, atomic locking, and exact resume contents.
Unchecked: seed replication, folds 03-04, corrected-pad ablation, and actual
post-LLM-pretrain gain. Accordingly the standalone scale conclusion is
`inconclusive`, while the marginal-to-current-stack result is `negative`.

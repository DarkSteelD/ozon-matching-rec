# mMARCO MiniLM architecture scout — recovered and finalized

## Conclusion

**NEGATIVE for the deployable diversity blend; standalone MiniLM remains NO-GO.**

The cheap 500-update screen produced a fixed 10% within-category rank-blend gain
on both discovery folds (`+0.002362`, `+0.001562`) and beat the random-init and
shuffled controls. That signal did not survive the preregistered standard full
hand fine-tune. Against the exact strong v3cal/sym rubase baseline, the fixed 10%
blend changed macro-category PR-AUC by `-0.000056` on fold 01 and `+0.000196` on
fold 02. Mean delta is `+0.000070`, below both the `+0.001` per-fold gate and the
observed two-fold delta spread (`0.000178` sample standard deviation).

Folds 03-04 were therefore not launched. No process was launched on fsk35 during
recovery/finalization.

## Claim and protocol

- Claim: a differently pretrained 12x384 mMARCO MiniLM contributes deployable
  ranking information to the strong rubase student after a standard full hand
  fine-tune.
- Primary metric: unweighted mean PR-AUC across 20 categories.
- Candidate: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, max length 224,
  category + attrs, pair-order swap augmentation, two-direction evaluation,
  2 epochs, batch 256, LR `2e-5`, seed `20260814`.
- Baseline: saved fixed-seed v3cal/sym `rubase_llmfull_e2` rerun predictions from
  `category_blend_distill`, same folds and hand recipe.
- Blend: fixed before the full run, `0.9 * category_rank(baseline) +
  0.1 * category_rank(MiniLM)`; no category or weight tuning.
- Dataset: `hand_pairs_pd_v3cal.parquet` for training and original hard targets
  in `hand_pairs.parquet` for evaluation; component-separated folds.
- Repository reference: `/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec`,
  git SHA `5099db5df398e6aa4fec9eccdaf6959f50cfbf29`, clean status at finalization.

## Full-training results

| variant | fold | macro-category PR-AUC | delta vs baseline | status | artifact |
|---|---:|---:|---:|---|---|
| exact strong baseline | 01 | 0.799520 | 0 | checked | `preds/full_baseline/fold_01.csv` |
| MiniLM standalone | 01 | 0.733276 | -0.066244 | checked, negative | `preds/full_minilm/fold_01.csv` |
| fixed pretrained 10% blend | 01 | 0.799464 | -0.000056 | checked, fails gate | `metrics_full.json` |
| shuffled MiniLM-rank 10% | 01 | 0.780875 | -0.018645 | checked negative control | `metrics_full_controls.json` |
| uniform-random-rank 10% | 01 | 0.780490 | -0.019031 | checked negative control | `metrics_full_controls.json` |
| exact strong baseline | 02 | 0.804345 | 0 | checked | `preds/full_baseline/fold_02.csv` |
| MiniLM standalone | 02 | 0.741169 | -0.063176 | checked, negative | `preds/full_minilm/fold_02.csv` |
| fixed pretrained 10% blend | 02 | 0.804541 | +0.000196 | checked, fails gate | `metrics_full.json` |
| shuffled MiniLM-rank 10% | 02 | 0.785584 | -0.018761 | checked negative control | `metrics_full_controls.json` |
| uniform-random-rank 10% | 02 | 0.786416 | -0.017928 | checked negative control | `metrics_full_controls.json` |
| baseline mean | 01-02 | 0.801932 | 0 | checked | `metrics_full_controls.json` |
| MiniLM standalone mean | 01-02 | 0.737222 | -0.064710 | checked, negative | `metrics_full.json` |
| fixed pretrained 10% mean | 01-02 | 0.802002 | +0.000070 | checked, negative gate | `metrics_full_controls.json` |

The full shuffled and uniform-random controls verify that merely injecting a
second category-rank vector is strongly harmful; row-aligned MiniLM information
recovers almost all of that damage, but does not improve the exact baseline by a
material amount. The already-saved cheap random-initialized MiniLM control was
also negative on both folds (`-0.007999`, `-0.009468`).

TP/FP/FN changes are not defined for this experiment because the competition
metric and preregistered decision are threshold-free PR-AUC/rank blending; no
classification threshold was selected after seeing results.

## Mechanism

Full task fine-tuning appears to erase the useful mMARCO diversity seen in the
short screen. Mean within-category Spearman correlation between MiniLM and the
matched rubase prediction rises from `0.6719 -> 0.8580` on fold 01 and
`0.6858 -> 0.8561` on fold 02 after the standard full fine-tune. This supports
the proposed mechanism: MiniLM retains complementary retrieval ordering early,
then converges toward the same hand-task ranking while remaining substantially
weaker standalone. It does not support adding the fully trained model to the
container.

## Runtime and deployability

| measurement | value |
|---|---:|
| parameters | 117,641,089 |
| checkpoint size | 470,592,698 bytes (448.8 MiB) |
| full folds 01-02 wall time on fsk35 H100 | 10m20s, including 135s tokenization |
| full-run observed device memory high-water | 16,579 MiB |
| H100 PCIe deploy benchmark, 100k pairs, two directions, len224 | 49.96s |
| same benchmark throughput | 2,001.7 input pairs/s |
| benchmark PyTorch peak allocated memory | 1,900,574,720 bytes (1.77 GiB) |
| estimated 365,654-pair two-direction H100 PCIe runtime | 182.67s |

The deploy benchmark ran on `avi-ix-devbox02` physical GPU0 only after two live
checks, under an experiment lock, and the lock was released. This confirms the
architecture is cheap on H100, but T4 latency remains unchecked; no T4 claim is
made.

## Recovery and reproducibility

Recovered read-only from fsk35 storage to the local artifact directory and then
to `/home/dzkhomidov/matching-work/rescue_20260824/architecture_scout` on
`avi-ix-devbox02`. `SHA256SUMS.recovered` verifies the original metrics,
predictions, and checkpoint on the destination. Re-running `score_full.py` on
devbox02 produced a byte-identical `metrics_full.json` SHA-256:
`cb9c0fdcf1ea3dadc794f32104bbe89c011b672aa54026fabb2fefa34f36fb16`.

Original full run command (historical; **must not be relaunched on fsk35**):

```bash
bash /home/dzkhomidov/matching-work/rescue_20260824/architecture_scout/launch_full.sh
```

Safe scoring reproduction on a non-fsk35 host:

```bash
cd /home/dzkhomidov/matching-work/rescue_20260824/architecture_scout
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score_full.py \
  --hard /home/dzkhomidov/matching-work/data/hand_pairs.parquet \
  --baseline preds/full_baseline --candidate preds/full_minilm \
  --output metrics_full.verify.json
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score_full_controls.py
```

## Checked and unchecked

Checked: cheap folds 01-02; cheap random-init and shuffled controls; standard
full fine-tune folds 01-02; exact baseline; fixed 10% full blend; full shuffled
and uniform-random controls; independent byte-identical rescoring; H100 PCIe
inference speed and peak allocation; artifact hashes.

Unchecked by design: folds 03-04 (failed gate); repeated full-training seeds;
costly full training from random initialization; T4 runtime and container
wall-clock. These unchecked cases cannot rescue the preregistered fixed blend
because its required folds 01-02 gate failed.

No validation directory, submission, push, or commit was touched.

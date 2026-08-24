# MiniLM early-diversity trajectory — fold-01 checkpoint

## Status

**NEGATIVE on the selection fold and paused before fold 02 by submission
priority.** devbox02 physical GPU0 is free and its experiment lock has been
released. Nothing was launched on fsk35.

The preregistered ladder was completed on fold 01. Every early checkpoint made
the fixed 10% blend worse than the exact strong baseline. The deterministic
fold-01 selection therefore chose the full endpoint, which is still slightly
negative and fails the required `+0.001` gate by a wide margin. Fold 02 and the
head-only mechanism control were not started after the resource-priority order.

## Fold-01 curve

| MiniLM update | fixed 10% blend macro AP | delta vs exact baseline | mean within-category Spearman vs baseline | status |
|---:|---:|---:|---:|---|
| 250 | 0.797594 | -0.001926 | 0.7171 | checked, negative |
| 500 | 0.798415 | -0.001105 | 0.7841 | checked, negative |
| 1000 | 0.799136 | -0.000385 | 0.8356 | checked, negative |
| full (2144) | 0.799452 | -0.000068 | 0.8571 | checked, selected, fails gate |

Exact baseline macro-category PR-AUC was `0.799520`. Selection used only fold 01,
fixed weight 10%, and no category tuning.

## Interpretation

The hypothesized useful early window does not exist along the exact standard
full-fine-tuning trajectory. Diversity is highest early, but that diversity is
harmful; as MiniLM converges toward rubase, correlation rises monotonically and
the blend loss shrinks toward zero. Thus the earlier positive cheap screen was
not explained by stopping the standard run at 500 updates. That screen also used
a 40k category-balanced subset, max length 160, batch 128, and its own 500-step
OneCycle schedule, so one or more of those protocol differences created the
screen-only effect.

The conclusion concerns only the fixed deployable blend, not MiniLM standalone.
TP/FP/FN are undefined because this was a threshold-free rank/PR-AUC experiment
and no threshold was selected.

## Resource evidence

- Host/GPU: `avi-ix-devbox02`, physical GPU0, H100 PCIe.
- Bidirectional tokenization: 121.36 seconds.
- Training plus four checkpoint evaluations: 373.07 seconds.
- Each 91,157-row two-direction evaluation: 23.16–23.42 seconds.
- Peak PyTorch allocation: 13,529,131,008 bytes (12.60 GiB).
- Full fold-01 predictions and the trajectory manifest are persisted in `preds/`.
- Exact output and selection are in `selection_fold01.json`.

## Checked and unchecked

Checked: fold-01 updates 250/500/1000/full, exact baseline, fixed 10% category
rank blend, correlation/gain curve, runtime and peak allocation.

Unchecked due submission priority: selected full endpoint on fold 02; fold-01
head-only control; folds 03-04. Because the selection-fold maximum is already
below zero, none of these unchecked runs can satisfy the preregistered requirement
of more than `+0.001` on both folds.

No validation directory, submission, push, or commit was touched.

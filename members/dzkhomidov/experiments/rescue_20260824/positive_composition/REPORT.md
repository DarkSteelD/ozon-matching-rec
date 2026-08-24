# Epoch3 pretrain × len384 composition — 4-fold result

Status: **positive**. All cells use the same v3cal soft targets, symmetric pair
order training/evaluation, seed 20260814, two hand-finetune epochs, effective
batch 256, LR 2e-5 and identical component folds. Cross-host runs were isolated
and merged only after row-count and pair-key validation; prediction hashes are
in `PRED_HASHES_4FOLD.sha256`.

| fold | e2@224 | e3@224 | e2@384 | e3@384 | composition delta |
|---|---:|---:|---:|---:|---:|
| 01 | .799450 | .801528 | .801746 | .804121 | +.004671 |
| 02 | .804079 | .805875 | .807445 | .808723 | +.004644 |
| 03 | .799372 | .801293 | .802832 | .804864 | +.005493 |
| 04 | .802055 | .802361 | .803868 | .804574 | +.002518 |
| mean | .801239 | .802764 | .803973 | .805570 | **+.004331** |

Mean fold effects (sample standard deviation):

- epoch3 at len224: `+.001525 ± .000821`
- len384 at epoch2: `+.002734 ± .000810`
- epoch3 at len384: `+.001598 ± .000751`
- composition e3@384 vs e2@224: `+.004331 ± .001271`, positive 4/4
- interaction: `+.000073 ± .000412`

The mechanism is supported: the gains are nearly additive, with interaction
close to zero. The >+.001 same-sign gate passes on every fold. The all-hand
epoch3@384 refit is therefore eligible for deployability checks.

Exact canonical `id1→id2` single-direction CV is also complete. Its mean macro
AP is `.804127` for e3@384 versus `.799980` for e2@224: `+.004147`, positive
on all four folds (`+.004851`, `+.004683`, `+.005147`, `+.001905`). Dropping
the reverse direction costs `.001444` macro AP for e3@384 versus its
two-direction score, but preserves the full composition gain. Full evidence is
in `single_direction_metrics_4fold.json` and
`single_direction_composition.json`.

No validation-directory write, submission, repository commit, push, or fsk35
compute was performed.

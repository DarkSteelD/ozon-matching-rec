# BCE + within-category RankNet

Status: gate complete. Result: **negative** for both tested within-category
RankNet weights; folds 03–04 correctly did not run.

## Hypothesis and gate

Against a fresh BCE baseline, add
`lambda * softplus(-(positive_logit - negative_logit))`, pairing positives and
negatives only inside the same category and minibatch. Test lambda 0.1 and 0.3;
use cross-category random positive/negative pairing at lambda 0.3 as the negative
control. Advance to folds 03–04 only if a candidate improves macro category AP
on both folds 01–02 and pooled macro category AP by more than 0.001.

## Checked controls

- Dataset SHA256: `d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`.
- Init config SHA256: `7ac471b7daa2628be40469ffc90000c903556b6412e8e9a0d26ebc0a38baa126`.
- Repository reference SHA: `2da459984a1207677ff9eb863ca28589027a4bc3`;
  repository was clean at preflight.
- Seed 20260814, batch 256, two epochs, max length 224, identical fold and
  permutation logic across arms. Pair sampling uses a separate RNG, so it cannot
  change batch order.
- Fold 01: 2,144/2,144 batches have valid within-category pairs, 60.15 pairs per
  batch on average (min 31, max 80), train/eval intersection 0.
- Fold 02: 2,142/2,142 batches have valid within-category pairs, 60.20 pairs per
  batch on average (min 39, max 80), train/eval intersection 0.
- Unit self-check and Python compilation pass locally and on the assigned host.
- Historical BCE reference (not used for the gate): fold macro category AP
  0.78120 / 0.78842; pooled two-fold macro category AP 0.78402.

## Gate results

Primary metric is macro category AP. Delta is paired against the fresh BCE arm.

| variant | fold | macro category AP | delta vs BCE | status |
|---|---|---:|---:|---|
| BCE | 01 | 0.778769 | — | checked |
| BCE | 02 | 0.784610 | — | checked |
| BCE | pooled categories | 0.781064 | — | baseline |
| within λ=0.1 | 01 | 0.778416 | -0.000353 | negative |
| within λ=0.1 | 02 | 0.784205 | -0.000405 | negative |
| within λ=0.1 | pooled categories | 0.780747 | -0.000317 | NO-GO |
| within λ=0.3 | 01 | 0.777323 | -0.001446 | negative |
| within λ=0.3 | 02 | 0.783732 | -0.000878 | negative |
| within λ=0.3 | pooled categories | 0.779921 | -0.001143 | NO-GO |
| random-pair λ=0.3 | 01 | 0.778302 | -0.000467 | negative |
| random-pair λ=0.3 | 02 | 0.783621 | -0.000988 | negative |
| random-pair λ=0.3 | pooled categories | 0.780443 | -0.000621 | NO-GO |

The conclusion is unchanged under pooled non-category AP: fresh BCE pooled AP
0.839823, within λ=0.1 0.839641, within λ=0.3 0.839094, random λ=0.3
0.839355. Thus macro aggregation alone did not manufacture the negative result.

Category detail is saved in `metrics/gate.json` for all 20 categories. Direction
was broad rather than one-category noise: λ=0.1 improved 4 and worsened 16
categories; λ=0.3 improved 3 and worsened 17; random improved 5 and worsened 15.
For λ=0.3 the largest drops were Одежда -0.00467, Галантерея -0.00389,
Обувь -0.00377 and Ювелирка -0.00161. These are precisely the already fragile
fashion groups, so this loss does not rescue the hidden-test weakness.

The random-pair control also lost, but less than within-category λ=0.3 overall.
Evidence therefore supports “the extra sampled rank gradient is harmful here”;
the extra damage on fashion is consistent with within-category pairing
overweighting noisy/hard local comparisons. This is a mechanism-compatible
interpretation, not proof of label-noise causality.

Observed fold-to-fold standard deviation of macro category AP is about
0.0027–0.0032, but the paired deltas have the same negative sign on both folds.
No positive effect approached the predeclared +0.001 gate.

## Runtime and resources

Training+eval time per two-fold arm: BCE 706.7 s, λ=0.1 712.2 s, λ=0.3
711.0 s, random 711.0 s. Full wall interval including four tokenizations and
scoring was 53.5 H100-minutes. Peak model allocation was about 26 GiB.

Final host/resource: `avi-gn-fsk35`, physical GPU 1, H100 80GB HBM3, UUID
`GPU-89e8e203-3f6c-a83b-c317-82264e0653b2`; wrapper PID 1687663. Child PIDs:
BCE 1687873, λ=0.1 1721498, λ=0.3 1755154, random 1785770. At completion the
compute-app list was empty and the atomic lock was released.

## Resource status

Initial assigned resource `avi-ix-devbox01` GPU 0 was rejected because a live
root `tritonserver` compute-app (PID 429190) occupied it. The queue was moved to
`avi-ix-devbox02` physical GPU 3 after `macro_balance` and `hard_negative`.
`avi-ix-devbox02` GPU 3 was then left to the queued macro/hard-negative tasks.
Root reassigned this experiment to the clean H100 above. No foreign process was
stopped.

## Checked / unchecked

Checked: fresh BCE, within-category λ=0.1 and λ=0.3, random-pair λ=0.3, folds
01–02, loss implementation, deterministic batch separation, fold isolation,
pair coverage, scorer, gate logic, checksums, runtime and launch/release guard.

Unchecked by design: folds 03–04 (no arm passed the gate), other lambdas or pair
samplers, and multi-seed noise. The tested claim is **negative**; the broader
family of every possible ranking loss is not declared closed.

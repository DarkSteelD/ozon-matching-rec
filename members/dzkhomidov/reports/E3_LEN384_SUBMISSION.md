# E3 len384 matching submission

Submitted to ODS `e-cup-2026-matching` on 2026-08-24 for team `Roma Bazuka`.

## Result

- Submission ID: `5e87fa28-21b0-43f6-af96-8709789f65e9`
- Status: `success`
- Mean PR-AUC: `0.4981669844`
- Previous team best: `0.49692917425196076`
- Public-LB gain: `+0.00123781014803922`
- File: `ecup_matching_e3_len384_student_runv5.zip`
- ODS created at: `2026-08-24T08:32:38.999Z`

The ODS leaderboard page still showed the previous best immediately after the
successful result was returned. The score above is the metric attached to the
submission receipt itself.

## Candidate

Single final student trained on all 365,654 labeled pairs from the epoch-3
pair-pretraining checkpoint, using the v3cal/symmetric recipe. Inference uses
prioritized attributes, a 384-token limit, one direction, adaptive batch size
128 for the long-context model, and the inherited fashion-size penalty.

The exact runtime source is `container/run_v5.py`; its model manifest is
`container/models_e3_len384.json`. Model weights are intentionally not stored
in Git.

## Validation evidence

- Exact submit-mode four-fold macro PR-AUC: `0.804126672`
- Epoch-2 / 224-token control: `0.799980018`
- Delta: `+0.004146654`; all four folds positive
- Two-direction composition delta versus the same control: mean
  `+0.00433137`, with fold deltas `+0.0046706`, `+0.0046437`, `+0.0054928`,
  `+0.0025184`; single direction was submitted for runtime safety
- Separate epoch-3 control: `+0.00121241` macro across categories, 19/20
  categories positive

## Full-run and artifact checks

- Full input: 365,654 pairs; 711,304 required item IDs retained
- A100 wall time: 400 s (24 s filtering, 97 s texts ready, 302 s inference)
- Process GPU-memory increment: 10,648 MiB
- Output: 365,654 rows, no nulls or duplicate pairs, exact input pair order
- Prediction CSV SHA256:
  `5a7d517b8b73446b0813553a1318524dd00f758e8f7bf89132f6ee4d0b05eb61`
- Pair stream SHA256:
  `b089a456d45d1547aeaa6d68f3f9783e76543bde93345c64b2e4e9afe6b9644f`
- ZIP bytes: 327,732,321
- ZIP SHA256:
  `fefd9f7ed5c05a846df0f7311bb5cbf4941e4b1693a1014630eb9561c70aec1b`
- ZIP MD5: `8c457fa747f6bc302cea7280339c5bfd`
- FP16 model SHA256:
  `77c0617a0b966d870d6a92877ad803e15788ebda8b2708466e617bcb14668c50`
- `run_v5.py` SHA256:
  `79261ab3531c416fb29f07181179a358cc8268b613f5e011a8ba381fb983b49b`

The packaged files were checked byte-for-byte against the source directory,
the extracted tokenizer and FP16 model loaded successfully, and all 201 model
tensors matched the FP32-to-FP16 conversion. The exact full A100 run did not
trigger length degradation. Actual ODS hardware timing is not exposed by the
receipt, but the container completed successfully within the grader limit.

The archived binary remains outside Git at:
`/home/dzkhomidov/matching-work/rescue_20260824/container_candidate_e3_len384/ecup_matching_e3_len384_student_runv5.zip`.

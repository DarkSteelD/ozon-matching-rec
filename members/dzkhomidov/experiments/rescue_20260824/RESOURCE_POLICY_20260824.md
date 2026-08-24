# GPU resource policy — 2026-08-24

Effective immediately by user request:

- **Do not launch any new process on `avi-gn-fsk35`.** At the policy snapshot,
  all eight GPUs were free (4 MiB, 0%, no compute applications); the host must
  remain released.
- Preferred remote pool after a fresh live check:
  - `avi-ix-devbox02`: physical GPUs 0, 2, 3 were free. GPU1 had a resident
    VLLM process and is forbidden.
  - `avi-ix-devbox03`: physical GPUs 0–3 were free.
- `avi-ix-devbox01` remains forbidden because Triton/VLLM services and service
  reservations occupy the host, including apparently idle cards.
- Local host `avi-ling-gpu03`: physical A100 GPUs 1 and 2 are the two allowed
  local cards. They had idle resident processes using about 5.2 and 7.8 GiB;
  do not kill them and launch only after a live memory/utilization check with a
  conservative memory budget. Local GPUs 0 and 3 were actively training and
  are forbidden.
- Every launch still requires two live compute-app checks and an experiment
  lock. Never co-tenant an active training/inference process.

This policy supersedes earlier `fsk35` assignments in experiment plans.

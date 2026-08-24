# Commands and process ledger

Host: `avi-ix-devbox02`; physical GPU3; UUID
`GPU-1e52e8ea-a6f8-3988-bc1d-602771414b10`.

Live checks at `2026-08-24T10:18:06+03:00` and
`2026-08-24T10:18:18+03:00` both reported 1 MiB, 0% utilization, no GPU3
compute process, and no active registry lock. GPU1's VLLM PID 4140935 was left
untouched.

```bash
cd /home/dzkhomidov/matching-work/rescue_20260824/freeze_pretrain
nohup ./run_locked.sh > logs/gate.log 2>&1 &
# runner PID 858147; bottom6 child PID 858152
```

The wrapper claimed
`/home/dzkhomidov/ozon-hack/scratch-q2/gpu_registry/avi-ix-devbox02_gpu3`
and began at `2026-08-24T10:18:19+03:00`.

When submission work became the priority, runner PID 858147 was sent SIGSTOP
while child PID 858152 continued the already-active bottom6 fold 02. After the
child wrote both fold predictions and became waitable, SIGTERM was queued for
the stopped runner and it was continued only to deliver that termination. The
EXIT trap removed the exact lock. No top6 process started. Final GPU3 state:
1 MiB, 0%, no compute process.

The scorer was then run directly:

```bash
/home/dzkhomidov/ozon-hack/.venv-ml/bin/python score.py
```

No fsk35 process was launched or modified.

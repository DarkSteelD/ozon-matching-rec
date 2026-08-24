# Matching rescue snapshot — 2026-08-24

This is the reproducible source-and-evidence subset of the local
`~/matching-work/rescue_20260824` campaign. It includes experiment code,
launch scripts, reports, configs and aggregate metrics.

Excluded on purpose: model/checkpoint directories, tokenizers, raw prediction
rows, logs, locks, host-migration copies, virtual environments and submission
ZIPs. Those are either generated artifacts, machine state or duplicates. The
winning deploy source and exact ODS receipt live separately in
`members/dzkhomidov/container/run_v5.py` and
`members/dzkhomidov/reports/E3_LEN384_SUBMISSION.md`.

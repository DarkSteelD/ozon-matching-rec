# e3 len384 submission candidate staging

Base source is `ecup_matching_consolidated_v2.zip`, SHA256
`7c115e7dd77653b4d19ecef80cf6772202e2e79fc690005a3970d0a68bc48bad`.
The inherited `run.py` has a 15.5-minute hard budget and progressively lowers
the expensive tail's token limit if projected runtime does not fit. The solo
candidate changes `models.json` from student224 + mdeb160 to only the fresh
all-hand epoch3 student at max_len384. The model is inserted only after the
four-fold gate passes and the all-hand refit finishes.

The old mDeBERTa branch is intentionally omitted unless a separate full-run
benchmark proves it fits the envelope. No fold checkpoint is deployable.

The exact full primary-package run passed on the complete 365,654-pair input:
400 seconds total on local A100, with 10,648 MiB incremental board memory.
Output has the required three-column schema, no nulls or duplicate pairs, and
its pair IDs match the input byte-for-byte in the original order. See
`PRIMARY_FULL_RUN.json` for hashes and measurements. Actual ODS/T4 runtime is
not directly measured; the 15.5-minute guard remains active.

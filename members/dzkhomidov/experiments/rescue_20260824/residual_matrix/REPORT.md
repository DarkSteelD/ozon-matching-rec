# Full OOF residual-matrix screen

Status: **inconclusive / no deployment gate**.

The existing positive five-channel category-shrunk blend was treated as the
strong baseline. Every complete additional OOF channel was converted to a
fold/category percentile rank and screened as a residual member. Candidate
weights were chosen on the other three folds only; category weights were
shrunk 75% toward the nested global choice. Row-shuffled channels were the
negative control.

Best single residual: `final_stack_all`, aggregate macro AP `+0.000924`.
Held-out fold deltas were `+0.000794`, `+0.000838`, `+0.000943`, and
`+0.001035`. Its shuffled control was only `+0.000127`. The broad same-sign
effect looks real, but it misses the predeclared `+0.001` per-fold gate.

The only allowed follow-up was a two-dimensional grid over `final_stack_all`
and `ce_final_combo`. It was weaker: aggregate `+0.000820`, fold deltas
`+0.000687`, `+0.000823`, `+0.000898`, `+0.000924`; its shuffled-pair control
was `-0.000238`.

No further matrix search is justified without new OOF channels. No validation
files, submission, GPU model, commit, or push were touched.

Artifacts: `rank_correlation.csv`, `metrics.csv`, `selected_weights.csv`,
`pair_metrics.csv`, `pair_weights.csv`, `run.py`, and `pair_grid.py`.

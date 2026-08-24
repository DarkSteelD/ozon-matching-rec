# Pair-graph / sparse-matrix audit

Status: **NO-GO**. This experiment is label-free: graph topology and OOF model
scores are used, but graph labels are never propagated.

## Coverage

- 365,654 pair edges.
- 9.51% of rows belong to a component with more than one edge.
- No candidate edge has a common neighbour (zero triangle rows).
- The frozen validation folds already separate connected components, so no item
  or component crosses an outer fold.

## Tests

The feature matrix includes endpoint degrees, component nodes/edges and
cyclomatic number, common-neighbour/Jaccard/Adamic-Adar features, incident-edge
score means/maxima excluding the current edge, and component score means.

1. A category-specific OOF histogram-gradient meta-model reduced aggregate
   macro AP by `-0.008482`. The matched row-shuffled graph control reduced it by
   `-0.008530`; real graph minus shuffled was only `+0.000049`.
2. Restricting all changes to the 9.51% multi-edge rows did not recover a
   mechanism. The best fixed one-step diffusion arm (`other_max_max`, weight
   0.01) gave only `+0.000041`; fold 4 was negative. Its matched shuffled
   control was larger at `+0.000078`.

The positive-looking effect is therefore generic rank perturbation, not graph
topology. No GPU, shared validation path, submission, commit, or push was used.

Artifacts: `run_graph_audit.py`, `score_diffusion.py`,
`graph_features.parquet`, `oof_predictions.parquet`, `metrics.json`, and
`diffusion_metrics.json` in this directory.

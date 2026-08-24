#!/usr/bin/env python3
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


MODELS = {
    "rubase": "ce_rubase_len384",
    "e5": "ce_e5_len288",
    "mdeb": "ce_mdeb_len224",
    "zs": "zs_llm_blend",
    "distill": "ce_priodistill",
}
BASE_W = np.full(5, 0.2)


def ap(y, p):
    return float(average_precision_score(y, p))


def macro(df, pred):
    return float(np.mean([ap(g.target, g[pred]) for _, g in df.groupby("category", sort=True)]))


def weights_grid():
    return np.array([x for x in itertools.product(range(5), repeat=5) if sum(x) == 4], dtype=float) / 4


def rank_inputs(df, cols):
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    groups = df.groupby(["fold", "category"], sort=False).indices
    for idx in groups.values():
        idx = np.asarray(idx)
        for j, col in enumerate(cols):
            out[idx, j] = df.iloc[idx][col].rank(method="average", pct=True).to_numpy(np.float32)
    return out


def ap_table(df, x, grid, y):
    table = {}
    for (fold, cat), idx in df.groupby(["fold", "category"], sort=True).indices.items():
        idx = np.asarray(idx)
        candidate = x[idx] @ grid.T
        table[(fold, cat)] = np.array([ap(y[idx], candidate[:, j]) for j in range(len(grid))])
    return table


def choose(table, folds, categories, held, grid):
    chosen = {}
    for cat in categories:
        scores = np.mean([table[(fold, cat)] for fold in folds if fold != held], axis=0)
        best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
        # Stable conservative tie-break: closest to the equal global blend.
        k = best[np.argmin(np.sum((grid[best] - BASE_W) ** 2, axis=1))]
        chosen[cat] = grid[k]
    return chosen


def choose_global(table, folds, categories, held, grid):
    scores = np.mean([table[(fold, cat)] for fold in folds if fold != held for cat in categories], axis=0)
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
    return grid[best[np.argmin(np.sum((grid[best] - BASE_W) ** 2, axis=1))]]


def assign(df, chosen, alpha=1.0):
    w = np.vstack([chosen[c] for c in df.category])
    return alpha * w + (1 - alpha) * BASE_W


def paired_category_bootstrap(df, candidate, baseline, rng, n=10000):
    deltas = []
    for _, g in df.groupby("category", sort=True):
        deltas.append(ap(g.target, g[candidate]) - ap(g.target, g[baseline]))
    deltas = np.asarray(deltas)
    draws = rng.choice(deltas, size=(n, len(deltas)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(MODELS.values())
    df = pd.read_parquet(args.input, columns=["fold", "target", "category", *cols])
    assert df[cols].notna().all().all() and df.fold.nunique() == 4
    x = rank_inputs(df, cols)
    grid = weights_grid()
    singles = np.eye(5)
    folds = sorted(df.fold.unique())
    categories = sorted(df.category.unique())
    actual_grid_ap = ap_table(df, x, grid, df.target.to_numpy())
    actual_single_ap = ap_table(df, x, singles, df.target.to_numpy())
    pred = {name: np.empty(len(df), np.float32) for name in [
        "global_equal", "global_grid_nested", "category_best_single", "category_grid_raw",
        "category_grid_shrink25", "category_grid_shrink50", "category_grid_shrink75",
    ]}
    selected = []
    for held in folds:
        test = df.fold.eq(held).to_numpy()
        single = choose(actual_single_ap, folds, categories, held, singles)
        coarse = choose(actual_grid_ap, folds, categories, held, grid)
        global_w = choose_global(actual_grid_ap, folds, categories, held, grid)
        pred["global_equal"][test] = x[test] @ BASE_W
        pred["global_grid_nested"][test] = x[test] @ global_w
        pred["category_best_single"][test] = np.sum(x[test] * assign(df.loc[test], single), axis=1)
        for name, alpha in [("category_grid_raw", 1), ("category_grid_shrink25", .25),
                            ("category_grid_shrink50", .5), ("category_grid_shrink75", .75)]:
            pred[name][test] = np.sum(x[test] * assign(df.loc[test], coarse, alpha), axis=1)
        for cat in categories:
            selected.append({"held_fold": held, "category": cat,
                             **{f"single_{m}": float(v) for m, v in zip(MODELS, single[cat])},
                             **{f"grid_{m}": float(v) for m, v in zip(MODELS, coarse[cat])}})
        selected.append({"held_fold": held, "category": "__GLOBAL__",
                         **{f"grid_{m}": float(v) for m, v in zip(MODELS, global_w)}})

    # Controls are repeated and summarized, not candidates for deployment.
    controls = []
    for kind, repeats in [("permuted_selection", 10), ("random_weights", 25)]:
        for seed_i in range(repeats):
            rng = np.random.default_rng(args.seed + seed_i + (0 if kind.startswith("perm") else 1000))
            p = np.empty(len(df), np.float32)
            if kind.startswith("perm"):
                shuffled = df.target.to_numpy().copy()
                for idx in df.groupby(["fold", "category"], sort=False).indices.values():
                    idx = np.asarray(idx)
                    shuffled[idx] = rng.permutation(shuffled[idx])
                shuffled_ap = ap_table(df, x, grid, shuffled)
            for held in folds:
                test = df.fold.eq(held).to_numpy()
                if kind.startswith("perm"):
                    chosen = choose(shuffled_ap, folds, categories, held, grid)
                else:
                    chosen = {c: rng.dirichlet(np.ones(5)) for c in categories}
                p[test] = np.sum(x[test] * assign(df.loc[test], chosen), axis=1)
            controls.append({"variant": kind, "seed": int(seed_i), "primary": macro(df.assign(p=p), "p"),
                             "secondary": float(np.mean([macro(df.loc[df.fold.eq(f)].assign(p=p[df.fold.eq(f)]), "p") for f in folds]))})

    scored = df[["fold", "target", "category"]].copy()
    for name, p in pred.items():
        scored[name] = p
    rows = []
    baseline_folds = {}
    for fold in folds:
        g = scored.loc[scored.fold.eq(fold)]
        baseline_folds[fold] = macro(g, "global_equal")
    rng = np.random.default_rng(args.seed)
    strong_primary = macro(scored, "global_grid_nested")
    strong_fold = {fold: macro(scored.loc[scored.fold.eq(fold)], "global_grid_nested") for fold in folds}
    for name in pred:
        fold_scores = []
        for fold in folds:
            g = scored.loc[scored.fold.eq(fold)]
            value = macro(g, name)
            fold_scores.append(value)
            rows.append({"variant": name, "fold": fold, "metric": value,
                         "delta_vs_baseline": value - baseline_folds[fold],
                         "delta_vs_global_grid": value - strong_fold[fold], "status": "checked"})
        primary = macro(scored, name)
        base_primary = macro(scored, "global_equal")
        ci = [0.0, 0.0] if name == "global_equal" else paired_category_bootstrap(scored, name, "global_equal", rng)
        strong_ci = [0.0, 0.0] if name == "global_grid_nested" else paired_category_bootstrap(scored, name, "global_grid_nested", rng)
        rows.append({"variant": name, "fold": "aggregate", "metric": primary,
                     "delta_vs_baseline": primary - base_primary, "status": "checked",
                     "delta_vs_global_grid": primary - strong_primary,
                     "mean_fold_metric": float(np.mean(fold_scores)), "std_fold_metric": float(np.std(fold_scores, ddof=1)),
                     "min_fold_delta": float(np.min(np.asarray(fold_scores) - np.asarray(list(baseline_folds.values())))),
                     "bootstrap95_low": ci[0], "bootstrap95_high": ci[1],
                     "strong_bootstrap95_low": strong_ci[0], "strong_bootstrap95_high": strong_ci[1]})

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "selected_weights.csv", index=False)
    pd.DataFrame(controls).to_csv(out / "controls.csv", index=False)
    scored.to_parquet(out / "heldout_predictions.parquet", index=False)
    metadata = {"input": str(Path(args.input).resolve()),
                "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                "rows": len(df), "folds": folds, "categories": int(df.category.nunique()),
                "models": MODELS, "grid_size": len(grid), "seed": args.seed,
                "runtime_seconds": time.time() - started}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(pd.DataFrame(rows).query("fold == 'aggregate'").to_string(index=False))
    print(pd.DataFrame(controls).groupby("variant")[["primary", "secondary"]].agg(["mean", "std", "min", "max"]))


if __name__ == "__main__":
    main()

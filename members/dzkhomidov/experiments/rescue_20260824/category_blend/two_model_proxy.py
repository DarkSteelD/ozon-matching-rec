#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


COLS = ["ce_mdeb_len224", "ce_priodistill"]
FIXED = np.array([1 / 3, 2 / 3])  # proxy for student + 0.5*mdeb
GRID = np.array([[x / 4, 1 - x / 4] for x in range(5)])


def ap(y, p):
    return float(average_precision_score(y, p))


def rank_inputs(df):
    x = np.empty((len(df), 2), np.float32)
    for idx in df.groupby(["fold", "category"], sort=False).indices.values():
        idx = np.asarray(idx)
        for j, col in enumerate(COLS):
            x[idx, j] = df.iloc[idx][col].rank(method="average", pct=True).to_numpy(np.float32)
    return x


def ap_table(df, x):
    table = {}
    for key, idx in df.groupby(["fold", "category"], sort=True).indices.items():
        idx = np.asarray(idx)
        candidate = x[idx] @ GRID.T
        table[key] = np.array([ap(df.target.to_numpy()[idx], candidate[:, j]) for j in range(len(GRID))])
    return table


def pick(scores):
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
    return GRID[best[np.argmin(np.sum((GRID[best] - FIXED) ** 2, axis=1))]]


def macro(df, col):
    return float(np.mean([ap(g.target, g[col]) for _, g in df.groupby("category", sort=True)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input, columns=["fold", "target", "category", *COLS])
    x = rank_inputs(df)
    table = ap_table(df, x)
    folds, cats = sorted(df.fold.unique()), sorted(df.category.unique())
    names = ["two_global_fixed_2to1", "two_global_nested", "two_category_shrink75"]
    pred = {name: np.empty(len(df), np.float32) for name in names}
    weights = []
    for held in folds:
        test = df.fold.eq(held).to_numpy()
        global_w = pick(np.mean([table[(f, c)] for f in folds if f != held for c in cats], axis=0))
        pred["two_global_fixed_2to1"][test] = x[test] @ FIXED
        pred["two_global_nested"][test] = x[test] @ global_w
        for cat in cats:
            cat_w = pick(np.mean([table[(f, cat)] for f in folds if f != held], axis=0))
            shrunk = .75 * cat_w + .25 * global_w
            rows = test & df.category.eq(cat).to_numpy()
            pred["two_category_shrink75"][rows] = x[rows] @ shrunk
            weights.append({"held_fold": held, "category": cat,
                            "global_mdeb": global_w[0], "global_distill": global_w[1],
                            "category_mdeb": cat_w[0], "category_distill": cat_w[1],
                            "shrunk_mdeb": shrunk[0], "shrunk_distill": shrunk[1],
                            "changes_global": bool(np.any(cat_w != global_w))})
    scored = df[["fold", "target", "category"]].copy()
    for name in names:
        scored[name] = pred[name]
    rows = []
    for fold in folds + ["aggregate"]:
        g = scored if fold == "aggregate" else scored.loc[scored.fold.eq(fold)]
        base = macro(g, "two_global_nested")
        for name in names:
            value = macro(g, name)
            rows.append({"variant": name, "fold": fold, "metric": value,
                         "delta_vs_global_nested": value - base})
    cat_rows = []
    for cat, g in scored.groupby("category", sort=True):
        base = ap(g.target, g.two_global_nested)
        value = ap(g.target, g.two_category_shrink75)
        cat_rows.append({"category": cat, "two_global_nested": base,
                         "two_category_shrink75": value, "delta": value - base})
    rng = np.random.default_rng(20260824)
    delta = np.array([r["delta"] for r in cat_rows])
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    summary = {"bootstrap95_low": np.quantile(boot, .025), "bootstrap95_high": np.quantile(boot, .975),
               "changed_category_fold_choices": int(sum(r["changes_global"] for r in weights)),
               "total_category_fold_choices": len(weights)}
    pd.DataFrame(rows).to_csv(out / "two_model_metrics.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(out / "two_model_category_metrics.csv", index=False)
    pd.DataFrame(weights).to_csv(out / "two_model_weights.csv", index=False)
    scored.to_parquet(out / "two_model_heldout_predictions.parquet", index=False)
    pd.Series(summary).to_json(out / "two_model_summary.json", indent=2)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(cat_rows).sort_values("delta", ascending=False).to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def manual_ap(y, p):
    order = np.argsort(-p, kind="mergesort")
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    tp = np.cumsum(y)[ends]
    added = np.diff(np.r_[0, tp])
    return float(np.sum(added * tp / (ends + 1)) / np.sum(y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    args = parser.parse_args()
    root = Path(args.artifacts)
    df = pd.read_parquet(root / "heldout_predictions.parquet")
    rows = []
    max_check_error = 0.0
    for cat, g in df.groupby("category", sort=True):
        base = average_precision_score(g.target, g.global_grid_nested)
        cand = average_precision_score(g.target, g.category_grid_shrink75)
        max_check_error = max(max_check_error,
                              abs(base - manual_ap(g.target, g.global_grid_nested)),
                              abs(cand - manual_ap(g.target, g.category_grid_shrink75)))
        rows.append({"category": cat, "global_grid_nested": base,
                     "category_grid_shrink75": cand, "delta": cand - base,
                     "rows": len(g), "positives": int(g.target.sum())})
    result = pd.DataFrame(rows).sort_values("delta", ascending=False)
    result.to_csv(root / "category_metrics.csv", index=False)
    fold_rows = []
    for fold, g in df.groupby("fold", sort=True):
        base = np.mean([average_precision_score(x.target, x.global_grid_nested)
                        for _, x in g.groupby("category")])
        cand = np.mean([average_precision_score(x.target, x.category_grid_shrink75)
                        for _, x in g.groupby("category")])
        fold_rows.append({"fold": fold, "global_grid_nested": base,
                          "category_grid_shrink75": cand, "delta": cand - base})
    pd.DataFrame(fold_rows).to_csv(root / "fold_comparison.csv", index=False)
    assert max_check_error < 1e-12, max_check_error
    positive = result.delta[result.delta > 0]
    print(result.to_string(index=False))
    print("manual_ap_max_abs_error", max_check_error)
    print("positive_categories", len(positive), "of", len(result))
    print("top2_positive_share", float(positive.head(2).sum() / positive.sum()))


if __name__ == "__main__":
    main()

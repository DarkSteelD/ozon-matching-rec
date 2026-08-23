"""Score a zero-shot sweep CSV (fold,id1,id2,target,category,predict).

mean_prauc = mean over folds of pooled AP — same aggregation as
validation.evaluate; AP implementation copied verbatim from there.
"""
import argparse

import numpy as np
import polars as pl


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = float(y_true.sum())
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    boundaries = np.flatnonzero(np.diff(scores_sorted)) if y_sorted.size > 1 else np.array([], int)
    block_ends = np.concatenate([boundaries, [y_sorted.size - 1]])
    tp_cum = np.cumsum(y_sorted)[block_ends]
    counts = block_ends + 1.0
    precision = tp_cum / counts
    recall = tp_cum / positives
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args()
    for path in args.csv:
        df = pl.read_csv(path)
        fold_aps = []
        for fold in sorted(df["fold"].unique()):
            fd = df.filter(pl.col("fold") == fold)
            fold_aps.append(average_precision(
                fd["target"].to_numpy().astype(float), fd["predict"].to_numpy()))
        base = df["target"].mean()
        print(f"{path}: mean_prauc={np.mean(fold_aps):.5f} "
              f"folds={[round(a, 4) for a in fold_aps]} pos_rate={base:.3f}")


if __name__ == "__main__":
    main()

"""CPU batch-pair audit and AP scorer for the rank-loss gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def batch_audit(data, folds, batch_size, epochs, seed):
    fold_col = data["fold"].to_numpy()
    labels = data["target"].to_numpy()
    names = sorted(data["category"].unique().to_list())
    codes = np.asarray([names.index(x) for x in data["category"].to_list()])
    rows = []
    for fold in folds:
        train = np.flatnonzero(fold_col != fold)
        valid = np.flatnonzero(fold_col == fold)
        assert not np.intersect1d(train, valid).size
        rng = np.random.default_rng(seed + FOLDS.index(fold))
        counts = []
        for _ in range(epochs):
            permutation = rng.permutation(len(train))
            for start in range(0, len(train) - batch_size + 1, batch_size):
                idx = train[permutation[start:start + batch_size]]
                count = 0
                for category in np.unique(codes[idx]):
                    mask = codes[idx] == category
                    count += min(np.count_nonzero(labels[idx][mask] > 0.5),
                                 np.count_nonzero(labels[idx][mask] <= 0.5))
                counts.append(count)
        rows.append({
            "fold": fold,
            "train_rows": len(train),
            "eval_rows": len(valid),
            "leak_rows": int(np.intersect1d(train, valid).size),
            "batches": len(counts),
            "valid_pair_batches": int(np.count_nonzero(counts)),
            "pairs_total": int(np.sum(counts)),
            "pairs_per_batch_mean": float(np.mean(counts)),
            "pairs_per_batch_min": int(np.min(counts)),
            "pairs_per_batch_max": int(np.max(counts)),
        })
    return rows


def score(data, pred_root, variant, folds):
    parts = []
    fold_rows = []
    for fold in folds:
        truth = data.filter(pl.col("fold") == fold).select("id1", "id2", "target", "category")
        pred = pl.read_csv(pred_root / variant / f"{fold}.csv")
        joined = truth.join(pred, on=["id1", "id2"], how="inner", validate="1:1")
        if joined.height != truth.height:
            raise ValueError(f"{variant}/{fold}: {joined.height} predictions for {truth.height} rows")
        category_ap = []
        for category in sorted(joined["category"].unique().to_list()):
            group = joined.filter(pl.col("category") == category)
            category_ap.append(float(average_precision_score(group["target"], group["predict"])))
        fold_rows.append({"variant": variant, "fold": fold,
                          "macro_category_ap": float(np.mean(category_ap))})
        parts.append(joined)
    pooled = pl.concat(parts)
    categories = []
    for category in sorted(pooled["category"].unique().to_list()):
        group = pooled.filter(pl.col("category") == category)
        categories.append({"variant": variant, "category": category, "rows": group.height,
                           "positives": int(group["target"].sum()),
                           "ap": float(average_precision_score(group["target"], group["predict"]))})
    return fold_rows, categories, float(np.mean([x["ap"] for x in categories]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pred-root")
    parser.add_argument("--variants", default="")
    parser.add_argument("--folds", default="fold_01,fold_02")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    data = pl.read_parquet(args.data)
    folds = args.folds.split(",")
    result = {"batch_audit": batch_audit(data, folds, args.bs, args.epochs, args.seed)}
    if args.variants:
        result["scores"] = []
        for variant in args.variants.split(","):
            fold_rows, category_rows, macro = score(data, Path(args.pred_root), variant, folds)
            result["scores"].append({"variant": variant, "folds": fold_rows,
                                     "per_category": category_rows,
                                     "macro_category_ap": macro})
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

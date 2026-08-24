from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score

FASHION = {"Обувь", "Одежда", "Галантерея и аксессуары", "Ювелирные изделия"}


def load_predictions(path, targets):
    predictions = pl.read_csv(path)
    joined = targets.join(predictions, on=["id1", "id2"], how="left", validate="1:1")
    if joined["predict"].null_count() or len(joined) != len(targets):
        raise ValueError(f"incomplete predictions: {path}")
    return joined


def score_frame(frame):
    y = frame["target"].to_numpy()
    p = frame["predict"].to_numpy()
    by_category = []
    for category in sorted(frame["category"].unique().to_list()):
        part = frame.filter(pl.col("category") == category)
        by_category.append({
            "category": category,
            "rows": len(part),
            "prauc": average_precision_score(part["target"].to_numpy(),
                                               part["predict"].to_numpy())
        })
    fashion = [row["prauc"] for row in by_category if row["category"] in FASHION]
    return {
        "prauc": average_precision_score(y, p),
        "macro_category_prauc": float(np.mean([row["prauc"] for row in by_category])),
        "worst_fashion_prauc": float(min(fashion)),
        "mean_fashion_prauc": float(np.mean(fashion)),
        "per_category": by_category,
    }


def ranking_change(baseline, candidate):
    a = baseline["predict"].to_numpy()
    b = candidate["predict"].to_numpy()
    rank_a = rankdata(a, method="average")
    rank_b = rankdata(b, method="average")
    movement = np.abs(rank_a - rank_b)
    n = len(a)
    top_a = rank_a > 0.9 * n
    top_b = rank_b > 0.9 * n
    return {
        "spearman": float(spearmanr(a, b).statistic),
        "mean_absolute_rank_change": float(movement.mean()),
        "median_absolute_rank_change": float(np.median(movement)),
        "rows_moved_at_least_1pct": int((movement >= 0.01 * n).sum()),
        "rows_moved_at_least_1pct_fraction": float((movement >= 0.01 * n).mean()),
        "top_decile_jaccard": float((top_a & top_b).sum() / (top_a | top_b).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", default="fold_01,fold_02")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_targets = pl.read_parquet(args.targets).select(
        "fold", "id1", "id2", "target", "category")
    results = []
    per_category_rows = []
    frames = {"baseline": [], "category_balanced": []}
    for fold in args.folds.split(","):
        targets = all_targets.filter(pl.col("fold") == fold)
        fold_frames = {}
        for variant, root in (("baseline", args.baseline),
                              ("category_balanced", args.candidate)):
            frame = load_predictions(Path(root) / f"{fold}.csv", targets)
            fold_frames[variant] = frame
            frames[variant].append(frame)
            scores = score_frame(frame)
            results.append({"variant": variant, "fold": fold,
                            **{k: v for k, v in scores.items() if k != "per_category"}})
            for row in scores["per_category"]:
                per_category_rows.append({"variant": variant, "fold": fold, **row})
        change = ranking_change(fold_frames["baseline"], fold_frames["category_balanced"])
        (output / f"ranking_{fold}.json").write_text(
            json.dumps(change, ensure_ascii=False, indent=2) + "\n")

    for variant in ("baseline", "category_balanced"):
        pooled = pl.concat(frames[variant])
        scores = score_frame(pooled)
        fold_rows = [row for row in results if row["variant"] == variant]
        results.append({
            "variant": variant, "fold": "mean_2fold",
            "prauc": float(np.mean([row["prauc"] for row in fold_rows])),
            "macro_category_prauc": scores["macro_category_prauc"],
            "worst_fashion_prauc": scores["worst_fashion_prauc"],
            "mean_fashion_prauc": scores["mean_fashion_prauc"],
        })
        for row in scores["per_category"]:
            per_category_rows.append({"variant": variant, "fold": "pooled_2fold", **row})
    ranking = ranking_change(pl.concat(frames["baseline"]),
                             pl.concat(frames["category_balanced"]))
    (output / "ranking_pooled_2fold.json").write_text(
        json.dumps(ranking, ensure_ascii=False, indent=2) + "\n")
    pl.DataFrame(results).write_csv(output / "metrics.csv")
    pl.DataFrame(per_category_rows).write_csv(output / "per_category_metrics.csv")
    payload = {"metrics": results, "ranking_pooled_2fold": ranking}
    (output / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

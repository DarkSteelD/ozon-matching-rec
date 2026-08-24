from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def ranks(scores, categories):
    result = np.empty(len(scores), np.float64)
    for category in np.unique(categories):
        index = np.flatnonzero(categories == category)
        order = np.argsort(scores[index], kind="stable")
        category_ranks = np.empty(len(index), np.float64)
        category_ranks[order] = (np.arange(len(index)) + 0.5) / len(index)
        result[index] = category_ranks
    return result


def macro_ap(target, scores, categories):
    return float(np.mean([average_precision_score(target[categories == category], scores[categories == category])
                          for category in np.unique(categories)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()

    hard = pl.read_parquet(args.hard).filter(pl.col("fold") == args.fold).select(
        "id1", "id2", "target", "category"
    )
    base = pl.read_csv(Path(args.baseline) / f"{args.fold}.csv").rename({"predict": "baseline"})
    frame = hard.join(base, on=["id1", "id2"], how="left", validate="1:1")
    target = frame["target"].to_numpy()
    categories = frame["category"].to_numpy()
    baseline = frame["baseline"].to_numpy()
    baseline_rank = ranks(baseline, categories)
    baseline_metric = macro_ap(target, baseline, categories)
    rows = []
    for label in args.labels.split(","):
        prediction = pl.read_csv(Path(args.predictions) / f"{args.fold}_step_{label}.csv").rename(
            {"predict": "candidate"}
        )
        joined = frame.join(prediction, on=["id1", "id2"], how="left", validate="1:1")
        candidate = joined["candidate"].to_numpy()
        candidate_rank = ranks(candidate, categories)
        blend = 0.9 * baseline_rank + 0.1 * candidate_rank
        correlations = [spearmanr(baseline[categories == category], candidate[categories == category]).statistic
                        for category in np.unique(categories)]
        blend_metric = macro_ap(target, blend, categories)
        rows.append({"label": label, "baseline_macro_ap": baseline_metric,
                     "blend_macro_ap": blend_metric, "delta": blend_metric - baseline_metric,
                     "mean_category_spearman": float(np.mean(correlations))})
    result = {"fold": args.fold, "weight": 0.1, "rows": rows}
    if args.select:
        selected = max(enumerate(rows), key=lambda pair: (pair[1]["delta"], -pair[0]))[1]
        result["selection"] = selected
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

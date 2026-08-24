from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss


def ece(y, p, bins=20):
    edges = np.linspace(0, 1, bins + 1); groups = np.minimum(np.digitize(p, edges) - 1, bins - 1)
    return float(sum(abs(y[groups == b].mean() - p[groups == b].mean()) * (groups == b).mean()
                     for b in range(bins) if np.any(groups == b)))


def metrics(frame):
    y, p, cats = frame["target"].to_numpy(), frame["predict"].to_numpy(), frame["category"].to_numpy()
    by_category = {}
    for category in sorted(set(cats)):
        mask = cats == category
        by_category[category] = {"ap": float(average_precision_score(y[mask], p[mask])),
            "prevalence": float(y[mask].mean()), "prediction_mean": float(p[mask].mean()),
            "brier": float(brier_score_loss(y[mask], p[mask])), "ece20": ece(y[mask], p[mask]), "rows": int(mask.sum())}
    return {"pooled_ap": float(average_precision_score(y, p)),
        "macro_category_ap": float(np.mean([x["ap"] for x in by_category.values()])),
        "prevalence": float(y.mean()), "prediction_mean": float(p.mean()),
        "brier": float(brier_score_loss(y, p)), "ece20": ece(y, p), "per_category": by_category}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--root", required=True)
    ap.add_argument("--archived-baseline", required=True); ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--variants", default="baseline_rerun,baseline_archived,hard075,random075,hard050,random050")
    ap.add_argument("--output", required=True); args = ap.parse_args(); df = pl.read_parquet(args.data)
    variants = args.variants.split(",")
    rows = []
    for fold in args.folds.split(","):
        truth = df.filter(pl.col("fold") == fold).select("id1", "id2", "target", "category")
        fold_metrics = {}
        for variant in variants:
            path = Path(args.archived_baseline if variant == "baseline_archived" else args.root,
                        f"{fold}.csv" if variant == "baseline_archived" else variant.removesuffix("_rerun") + f"/{fold}.csv")
            pred = pl.read_csv(path); got = truth.join(pred, on=["id1", "id2"], validate="1:1")
            assert got.height == truth.height
            fold_metrics[variant] = metrics(got)
        base = fold_metrics["baseline_rerun"]
        for variant, value in fold_metrics.items():
            rows.append({"variant": variant, "fold": fold, **value,
                "delta_macro": value["macro_category_ap"] - base["macro_category_ap"],
                "delta_pooled": value["pooled_ap"] - base["pooled_ap"],
                "per_category_ap_delta": {k: v["ap"] - base["per_category"][k]["ap"] for k, v in value["per_category"].items()}})
    aggregates = []
    for variant in variants:
        selected = [x for x in rows if x["variant"] == variant]; deltas = [x["delta_macro"] for x in selected]
        aggregates.append({"variant": variant, "folds": len(selected),
            "macro_mean": float(np.mean([x["macro_category_ap"] for x in selected])),
            "macro_std": float(np.std([x["macro_category_ap"] for x in selected], ddof=1)),
            "delta_macro_mean": float(np.mean(deltas)), "same_positive_sign": all(x > 0 for x in deltas),
            "pooled_mean": float(np.mean([x["pooled_ap"] for x in selected])),
            "prediction_mean": float(np.mean([x["prediction_mean"] for x in selected])),
            "brier_mean": float(np.mean([x["brier"] for x in selected])),
            "ece20_mean": float(np.mean([x["ece20"] for x in selected]))})
    hard = [x for x in aggregates if x["variant"].startswith("hard")]
    best = max(hard, key=lambda x: x["delta_macro_mean"])
    result = {"rows": rows, "aggregates": aggregates, "best_hard_variant": best["variant"],
              "gate_pass": best["delta_macro_mean"] > 0.001 and best["same_positive_sign"]}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(aggregates, ensure_ascii=False, indent=2)); print("gate_pass", result["gate_pass"])


if __name__ == "__main__": main()

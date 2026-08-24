from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def metrics(frame):
    y, p, cats = frame["target"].to_numpy(), frame["predict"].to_numpy(), frame["category"].to_numpy()
    by_cat = {cat: float(average_precision_score(y[cats == cat], p[cats == cat])) for cat in sorted(set(cats))}
    return {"pooled_prauc": float(average_precision_score(y, p)),
            "macro_category_prauc": float(np.mean(list(by_cat.values()))), "per_category": by_cat}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--truth", default=None); ap.add_argument("--root", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02"); ap.add_argument("--output", required=True)
    args = ap.parse_args(); data = pl.read_parquet(args.truth or args.data)
    rows = []
    for fold in args.folds.split(","):
        truth = data.filter(pl.col("fold") == fold).select("id1", "id2", "target", "category")
        fold_metrics = {}
        for variant in ("baseline", "structured", "shuffled"):
            pred = pl.read_csv(Path(args.root, variant, f"{fold}.csv"))
            got = truth.join(pred, on=["id1", "id2"], validate="1:1")
            if got.height != truth.height: raise RuntimeError(f"missing rows: {variant} {fold}")
            fold_metrics[variant] = metrics(got)
        base = fold_metrics["baseline"]
        for variant, m in fold_metrics.items():
            rows.append({"variant": variant, "fold": fold, **m,
                         "delta_macro": m["macro_category_prauc"] - base["macro_category_prauc"],
                         "delta_pooled": m["pooled_prauc"] - base["pooled_prauc"],
                         "per_category_delta": {k: m["per_category"][k] - base["per_category"][k] for k in m["per_category"]}})
    aggregates = []
    for variant in ("baseline", "structured", "shuffled"):
        selected = [r for r in rows if r["variant"] == variant]
        deltas = [r["delta_macro"] for r in selected]
        aggregates.append({"variant": variant, "folds": len(selected),
            "macro_mean": float(np.mean([r["macro_category_prauc"] for r in selected])),
            "macro_std": float(np.std([r["macro_category_prauc"] for r in selected], ddof=1)) if len(selected)>1 else None,
            "delta_macro_mean": float(np.mean(deltas)), "same_positive_sign": all(x > 0 for x in deltas),
            "pooled_mean": float(np.mean([r["pooled_prauc"] for r in selected]))})
    structured = next(x for x in aggregates if x["variant"] == "structured")
    result = {"rows": rows, "aggregates": aggregates,
              "gate_pass": structured["same_positive_sign"] and structured["delta_macro_mean"] > 0.001}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["aggregates"], ensure_ascii=False, indent=2)); print("gate_pass", result["gate_pass"])


if __name__ == "__main__": main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def score(y, p, cats):
    pooled = float(average_precision_score(y, p))
    by_cat = {}
    for cat in sorted(set(cats)):
        m = cats == cat
        by_cat[cat] = float(average_precision_score(y[m], p[m]))
    return pooled, float(np.mean(list(by_cat.values()))), by_cat


def load_preds(path):
    return pl.read_csv(path).select("id1", "id2", "predict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--baseline", required=True)
    ap.add_argument("--pred-root", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02")
    args = ap.parse_args()
    data = pl.read_parquet(args.data)
    rows = []
    for fold in args.folds.split(","):
        truth = data.filter(pl.col("fold") == fold).select("id1", "id2", "target", "category")
        base = truth.join(load_preds(Path(args.baseline) / f"{fold}.csv"), on=["id1", "id2"], validate="1:1")
        y, cats = base["target"].to_numpy(), base["category"].to_numpy()
        bp, bm, bc = score(y, base["predict"].to_numpy(), cats)
        rows.append({"variant": "baseline", "fold": fold, "prauc": bp, "macro_category_prauc": bm,
                     "delta_prauc": 0.0, "delta_macro": 0.0, "per_category": bc})
        for expdir in sorted(Path(args.pred_root).iterdir()):
            path = expdir / f"{fold}.csv"
            if not path.exists(): continue
            got = truth.join(load_preds(path), on=["id1", "id2"], validate="1:1")
            p, m, c = score(got["target"].to_numpy(), got["predict"].to_numpy(), got["category"].to_numpy())
            rows.append({"variant": expdir.name, "fold": fold, "prauc": p, "macro_category_prauc": m,
                         "delta_prauc": p-bp, "delta_macro": m-bm,
                         "worst_category_delta": min(c[k]-bc[k] for k in c), "per_category": c,
                         "per_category_delta": {k: c[k]-bc[k] for k in c}})
    aggregates = []
    for variant in sorted(set(r["variant"] for r in rows)):
        vr = [r for r in rows if r["variant"] == variant]
        aggregates.append({"variant": variant, "folds": len(vr),
                           "prauc_mean": float(np.mean([r["prauc"] for r in vr])),
                           "prauc_std": float(np.std([r["prauc"] for r in vr], ddof=1)) if len(vr)>1 else None,
                           "delta_prauc_mean": float(np.mean([r["delta_prauc"] for r in vr])),
                           "macro_mean": float(np.mean([r["macro_category_prauc"] for r in vr])),
                           "delta_macro_mean": float(np.mean([r["delta_macro"] for r in vr])),
                           "same_delta_sign": all(r["delta_prauc"] > 0 for r in vr) or all(r["delta_prauc"] < 0 for r in vr),
                           "worst_category_delta": min((r.get("worst_category_delta", 0) for r in vr))})
    result = {"rows": rows, "aggregates": aggregates}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for a in aggregates: print(a)


if __name__ == "__main__": main()

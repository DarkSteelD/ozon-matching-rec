import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    df = pl.read_parquet(args.data)
    results = {}
    for variant in args.variants:
        rows = []
        for path in sorted((Path(args.pred_root) / variant).glob("fold_*.csv")):
            fold = path.stem
            truth = df.filter(pl.col("fold") == fold)
            pred = pl.read_csv(path)
            assert pred.select("id1", "id2").equals(truth.select("id1", "id2"))
            y, p = truth["target"].to_numpy(), pred["predict"].to_numpy()
            cats = truth["category"].to_numpy()
            per_cat = {str(c): float(average_precision_score(y[cats == c], p[cats == c]))
                       for c in sorted(set(cats))}
            rows.append({"fold": fold, "prauc": float(average_precision_score(y, p)),
                         "macro_category_prauc": float(np.mean(list(per_cat.values()))),
                         "worst_category": min(per_cat, key=per_cat.get),
                         "worst_category_prauc": min(per_cat.values()),
                         "per_category": per_cat})
        results[variant] = {"folds": rows,
                            "mean_prauc": float(np.mean([x["prauc"] for x in rows])),
                            "mean_macro_category_prauc": float(np.mean([x["macro_category_prauc"] for x in rows]))}
    base = {x["fold"]: x for x in results.get("len384", {}).get("folds", [])}
    for variant, result in results.items():
        for row in result["folds"]:
            if row["fold"] in base:
                row["delta_vs_len384"] = row["prauc"] - base[row["fold"]]["prauc"]
                row["macro_delta_vs_len384"] = row["macro_category_prauc"] - base[row["fold"]]["macro_category_prauc"]
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

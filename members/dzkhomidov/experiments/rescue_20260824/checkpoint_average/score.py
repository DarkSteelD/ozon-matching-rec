from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss


VARIANTS = ("final", "late_avg", "ema", "early_avg", "late_pred_avg")
FOLDS = ("fold_01", "fold_02")


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.minimum(np.digitize(p, edges) - 1, bins - 1)
    return sum((ids == i).mean() * abs(y[ids == i].mean() - p[ids == i].mean())
               for i in range(bins) if np.any(ids == i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-data", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    hard = pl.read_parquet(args.hard_data).select("id1", "id2", "fold", "category", "target")
    rows = []
    per_category = []
    merged_preds = {}
    for fold in FOLDS:
        fold_hard = hard.filter(pl.col("fold") == fold)
        for variant in VARIANTS:
            pred = pl.read_csv(args.root / "preds" / variant / f"{fold}.csv")
            joined = fold_hard.join(pred, on=["id1", "id2"], how="inner", validate="1:1")
            assert joined.height == fold_hard.height
            y = joined["target"].to_numpy()
            p = joined["predict"].to_numpy()
            cat_scores = []
            for cat, group in joined.group_by("category"):
                cy = group["target"].to_numpy()
                cp = group["predict"].to_numpy()
                score = average_precision_score(cy, cp)
                cat_name = cat[0] if isinstance(cat, tuple) else cat
                cat_scores.append(score)
                per_category.append({"fold": fold, "variant": variant,
                                     "category": cat_name, "ap": score})
            record = {"fold": fold, "variant": variant,
                      "macro_ap": float(np.mean(cat_scores)),
                      "pooled_ap": float(average_precision_score(y, p)),
                      "brier": float(brier_score_loss(y, p)),
                      "logloss": float(log_loss(y, np.clip(p, 1e-7, 1-1e-7))),
                      "ece15": float(ece(y, p)), "n": len(y)}
            rows.append(record)
            merged_preds[(fold, variant)] = p
    base = {(r["fold"]): r for r in rows if r["variant"] == "final"}
    for r in rows:
        r["delta_macro"] = r["macro_ap"] - base[r["fold"]]["macro_ap"]
        r["delta_pooled"] = r["pooled_ap"] - base[r["fold"]]["pooled_ap"]
    aggregates = []
    for variant in VARIANTS:
        vr = [r for r in rows if r["variant"] == variant]
        aggregates.append({"variant": variant,
                           **{f"mean_{key}": float(np.mean([r[key] for r in vr]))
                              for key in ("macro_ap", "pooled_ap", "brier", "logloss", "ece15",
                                          "delta_macro", "delta_pooled")},
                           "delta_macro_std": float(np.std([r["delta_macro"] for r in vr], ddof=1))})
    variability = {}
    for fold in FOLDS:
        arr = np.stack([merged_preds[(fold, v)] for v in ("late_avg", "ema", "final")])
        raw = np.load(args.root / "diagnostics" / f"{fold}_late_preds.npz")
        checkpoints = np.stack([raw[k] for k in ("q75", "q875", "q100")])
        variability[fold] = {
            "candidate_row_std_mean": float(arr.std(0).mean()),
            "late_checkpoint_row_std_mean": float(checkpoints.std(0).mean()),
            "late_checkpoint_correlations": np.corrcoef(checkpoints).tolist(),
        }
    result = {"rows": rows, "aggregates": aggregates, "prediction_variability": variability}
    (args.root / "metrics_folds12.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    pl.DataFrame(per_category).write_csv(args.root / "category_metrics_folds12.csv")
    for r in rows:
        print(r)
    print("AGGREGATES")
    for r in aggregates:
        print(r)


if __name__ == "__main__":
    main()

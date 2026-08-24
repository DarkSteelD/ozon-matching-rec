"""Summarize controlled epoch-2 versus epoch-3 predictions."""
from pathlib import Path
import json

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
HAND = Path("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")


def ap(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.concatenate([np.flatnonzero(np.diff(p)), [len(y) - 1]])
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    return float(np.sum((recall - np.concatenate([[0], recall[:-1]])) * tp / (ends + 1)))


def pred_path(epoch: int, fold: str) -> Path:
    stage = "gate" if fold in {"fold_01", "fold_02"} else "rest"
    return ROOT / "preds" / f"hand_e{epoch}_ctrl_{stage}" / f"{fold}.csv"


hand = pl.read_parquet(HAND)
fold_rows = []
category_scores: dict[str, list[dict]] = {}
for fold in ["fold_01", "fold_02", "fold_03", "fold_04"]:
    truth = hand.filter(pl.col("fold") == fold)
    joined = truth
    for epoch in [2, 3]:
        pred = pl.read_csv(pred_path(epoch, fold)).rename({"predict": f"p{epoch}"})
        joined = joined.join(pred, on=["id1", "id2"], how="left", validate="1:1")
    assert joined["p2"].null_count() == joined["p3"].null_count() == 0
    y = joined["target"].to_numpy().astype(np.int8)
    s2, s3 = ap(y, joined["p2"].to_numpy()), ap(y, joined["p3"].to_numpy())
    fold_rows.append({"fold": fold, "n": len(y), "epoch2": s2, "epoch3": s3, "delta": s3 - s2})
    for cat in joined["category"].unique().sort().to_list():
        part = joined.filter(pl.col("category") == cat)
        yc = part["target"].to_numpy().astype(np.int8)
        if yc.sum() == 0:
            continue
        c2, c3 = ap(yc, part["p2"].to_numpy()), ap(yc, part["p3"].to_numpy())
        category_scores.setdefault(cat, []).append(
            {"fold": fold, "n": len(yc), "epoch2": c2, "epoch3": c3, "delta": c3 - c2}
        )

deltas = np.array([row["delta"] for row in fold_rows])
summary = {
    "folds": fold_rows,
    "fold_mean": {
        "epoch2": float(np.mean([row["epoch2"] for row in fold_rows])),
        "epoch3": float(np.mean([row["epoch3"] for row in fold_rows])),
        "delta": float(deltas.mean()),
        "delta_std": float(deltas.std(ddof=1)),
        "positive_folds": int((deltas > 0).sum()),
        "above_0.001_folds": int((deltas > 0.001).sum()),
    },
    "categories": {
        cat: {
            "mean_epoch2": float(np.mean([x["epoch2"] for x in rows])),
            "mean_epoch3": float(np.mean([x["epoch3"] for x in rows])),
            "mean_delta": float(np.mean([x["delta"] for x in rows])),
            "positive_folds": int(sum(x["delta"] > 0 for x in rows)),
            "folds": rows,
        }
        for cat, rows in category_scores.items()
    },
}
cat_values = list(summary["categories"].values())
summary["macro_category_mean"] = {
    "epoch2": float(np.mean([x["mean_epoch2"] for x in cat_values])),
    "epoch3": float(np.mean([x["mean_epoch3"] for x in cat_values])),
    "delta": float(np.mean([x["mean_delta"] for x in cat_values])),
    "positive_categories": int(sum(x["mean_delta"] > 0 for x in cat_values)),
    "category_count": len(cat_values),
}
(ROOT / "controlled_4fold_analysis.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary["fold_mean"], ensure_ascii=False, indent=2))
print(json.dumps(summary["macro_category_mean"], ensure_ascii=False, indent=2))

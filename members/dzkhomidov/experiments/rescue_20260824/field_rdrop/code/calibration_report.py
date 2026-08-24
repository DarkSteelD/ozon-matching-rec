"""Paired four-fold calibration and fixed-threshold diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import log_loss


ROOT = Path("/home/dzkhomidov/matching-work/rescue_20260824/field_rdrop")
DATA = Path("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]
VARIANTS = ["bce2view", "field05"]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    result = 0.0
    for bin_index in range(bins):
        mask = index == bin_index
        if mask.any():
            result += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    positive = y == 1
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    return {
        "rows": len(y),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece20_equal_width": ece(y, p),
        "threshold_0.5": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": float(2 * tp / (2 * tp + fp + fn)),
        },
    }


data = pl.read_parquet(DATA).select("fold", "id1", "id2", "target", "category")
result = {"definitions": {"ece": "20 equal-width probability bins", "threshold": 0.5}, "variants": {}}
for variant in VARIANTS:
    joined_parts = []
    fold_metrics = {}
    for fold in FOLDS:
        truth = data.filter(pl.col("fold") == fold)
        pred = pl.read_csv(ROOT / "preds" / variant / f"{fold}.csv")
        joined = truth.join(pred, on=["id1", "id2"], how="inner", validate="1:1")
        if joined.height != truth.height:
            raise RuntimeError(f"incomplete predictions: {variant}/{fold}")
        fold_metrics[fold] = metrics(joined["target"].to_numpy(), joined["predict"].to_numpy())
        joined_parts.append(joined)
    all_rows = pl.concat(joined_parts)
    per_category = {}
    for category in sorted(all_rows["category"].unique().to_list()):
        group = all_rows.filter(pl.col("category") == category)
        per_category[category] = metrics(group["target"].to_numpy(), group["predict"].to_numpy())
    category_macro = {
        key: float(np.mean([row[key] for row in per_category.values()]))
        for key in ("brier", "log_loss", "ece20_equal_width")
    }
    result["variants"][variant] = {
        "pooled": metrics(all_rows["target"].to_numpy(), all_rows["predict"].to_numpy()),
        "folds": fold_metrics,
        "category_macro": category_macro,
        "per_category": per_category,
    }

base = result["variants"]["bce2view"]
candidate = result["variants"]["field05"]
result["delta_field05_minus_bce2view"] = {
    "pooled": {
        key: candidate["pooled"][key] - base["pooled"][key]
        for key in ("brier", "log_loss", "ece20_equal_width")
    },
    "category_macro": {
        key: candidate["category_macro"][key] - base["category_macro"][key]
        for key in ("brier", "log_loss", "ece20_equal_width")
    },
    "threshold_0.5": {
        key: candidate["pooled"]["threshold_0.5"][key] - base["pooled"]["threshold_0.5"][key]
        for key in ("tp", "fp", "fn", "f1")
    },
}
(ROOT / "metrics" / "calibration_full4.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(result["delta_field05_minus_bce2view"], ensure_ascii=False, indent=2))

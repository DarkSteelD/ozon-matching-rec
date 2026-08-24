from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).parent
FOLDS = [f"fold_{i:02d}" for i in range(1, 5)]


def confusion(y: np.ndarray, p: np.ndarray, threshold: float = 0.5):
    z = p >= threshold
    tp = int(((y == 1) & z).sum())
    fp = int(((y == 0) & z).sum())
    fn = int(((y == 1) & ~z).sum())
    tn = int(((y == 0) & ~z).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / (2 * tp + fp + fn)}


metrics = json.loads((ROOT / "metrics_all.json").read_text())
by = {(x["fold"], x["variant"]): x for x in metrics["rows"]}
categories = sorted(by[(FOLDS[0], "hard050")]["per_category"])

category_effects = {}
for category in categories:
    values = []
    for fold in FOLDS:
        hard = by[(fold, "hard050")]["per_category"][category]["ap"]
        base = by[(fold, "baseline_rerun")]["per_category"][category]["ap"]
        values.append(hard - base)
    category_effects[category] = {"fold_deltas": values, "mean": float(np.mean(values))}

values = np.array([category_effects[x]["mean"] for x in categories])
rng = np.random.default_rng(20260824)
boot = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(axis=1)

data = pl.read_parquet("/home/dzkhomidov/matching-work/data/hand_pairs.parquet").select(
    "id1", "id2", "target", "fold"
)
confusions = {}
for variant in ["baseline", "hard050"]:
    parts = []
    for fold in FOLDS:
        pred = pl.read_csv(ROOT / "preds" / variant / f"{fold}.csv")
        truth = data.filter(pl.col("fold") == fold)
        got = truth.join(pred, on=["id1", "id2"], validate="1:1")
        assert got.height == truth.height
        parts.append(got)
    frame = pl.concat(parts)
    confusions[variant] = confusion(frame["target"].to_numpy(), frame["predict"].to_numpy())

manifests = [json.loads((ROOT / x).read_text()) for x in
             ["run_manifest_stage1.json", "run_manifest_stage2.json"]]
runs = [run for manifest in manifests for run in manifest["runs"]]

result = {
    "fold_deltas": {fold: by[(fold, "hard050")]["delta_macro"] for fold in FOLDS},
    "delta_mean": float(np.mean([by[(fold, "hard050")]["delta_macro"] for fold in FOLDS])),
    "category_positive": int((values > 0).sum()),
    "category_bootstrap_seed": 20260824,
    "category_bootstrap_samples": 20_000,
    "category_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
    "category_effects": category_effects,
    "confusion_at_0_5": confusions,
    "runtime_seconds_all_runs": float(sum(x["runtime_seconds"] for x in runs)),
    "runtime_seconds_primary_baseline_hard": float(sum(
        x["runtime_seconds"] for x in runs if x["variant"] in {"baseline", "hard050"}
    )),
}
(ROOT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))

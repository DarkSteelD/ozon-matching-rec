from pathlib import Path
import json
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

root = Path(__file__).parent
hard = pl.read_parquet("/home/dzkhomidov/matching-work/data/hand_pairs.parquet").select(
    "id1", "id2", "fold", "category", "target")
variants = ("e2_len224", "e3_len224", "e2_len384", "e3_len384")
rows, cats = [], []
for fold in ("fold_01", "fold_02"):
    h = hard.filter(pl.col("fold") == fold)
    for variant in variants:
        pred = pl.read_csv(root / "preds" / variant / f"{fold}.csv")
        data = h.join(pred, on=["id1", "id2"], validate="1:1")
        assert data.height == h.height
        scores = []
        for cat, group in data.group_by("category"):
            name = cat[0] if isinstance(cat, tuple) else cat
            ap = average_precision_score(group["target"], group["predict"])
            scores.append(ap)
            cats.append({"fold": fold, "variant": variant, "category": name, "ap": ap})
        rows.append({"fold": fold, "variant": variant,
                     "macro_ap": float(np.mean(scores)),
                     "pooled_ap": float(average_precision_score(data["target"], data["predict"]))})

by = {(r["fold"], r["variant"]): r for r in rows}
effects = []
for fold in ("fold_01", "fold_02"):
    a, b = by[(fold, "e2_len224")], by[(fold, "e3_len224")]
    c, d = by[(fold, "e2_len384")], by[(fold, "e3_len384")]
    effects.append({"fold": fold,
                    "epoch3_at_224_macro": b["macro_ap"] - a["macro_ap"],
                    "context_at_e2_macro": c["macro_ap"] - a["macro_ap"],
                    "composition_macro": d["macro_ap"] - a["macro_ap"],
                    "epoch3_at_384_macro": d["macro_ap"] - c["macro_ap"],
                    "interaction_macro": (d["macro_ap"] - c["macro_ap"]) - (b["macro_ap"] - a["macro_ap"]),
                    "epoch3_at_224_pooled": b["pooled_ap"] - a["pooled_ap"],
                    "context_at_e2_pooled": c["pooled_ap"] - a["pooled_ap"],
                    "composition_pooled": d["pooled_ap"] - a["pooled_ap"],
                    "epoch3_at_384_pooled": d["pooled_ap"] - c["pooled_ap"],
                    "interaction_pooled": (d["pooled_ap"] - c["pooled_ap"]) - (b["pooled_ap"] - a["pooled_ap"])})
means = {variant: {metric: float(np.mean([r[metric] for r in rows if r["variant"] == variant]))
                   for metric in ("macro_ap", "pooled_ap")} for variant in variants}
effect_means = {key: float(np.mean([r[key] for r in effects]))
                for key in effects[0] if key != "fold"}
result = {"rows": rows, "effects": effects, "means": means, "effect_means": effect_means}
(root / "metrics_folds12.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
pl.DataFrame(cats).write_csv(root / "category_metrics_folds12.csv")
print(json.dumps(result, ensure_ascii=False, indent=2))

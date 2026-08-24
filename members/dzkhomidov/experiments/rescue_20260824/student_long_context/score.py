from pathlib import Path
import json
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

root = Path(__file__).parent
hard = pl.read_parquet("/home/dzkhomidov/matching-work/data/hand_pairs.parquet").select(
    "id1", "id2", "fold", "category", "target")
rows, cats = [], []
for fold in ("fold_01", "fold_02"):
    h = hard.filter(pl.col("fold") == fold)
    for variant in ("len224", "len384"):
        p = pl.read_csv(root / "preds" / variant / f"{fold}.csv")
        d = h.join(p, on=["id1", "id2"], validate="1:1")
        assert d.height == h.height
        values = []
        for cat, g in d.group_by("category"):
            ap = average_precision_score(g["target"], g["predict"])
            name = cat[0] if isinstance(cat, tuple) else cat
            values.append(ap); cats.append({"fold": fold, "variant": variant, "category": name, "ap": ap})
        rows.append({"fold": fold, "variant": variant, "macro_ap": float(np.mean(values)),
                     "pooled_ap": float(average_precision_score(d["target"], d["predict"]))})
base = {r["fold"]: r for r in rows if r["variant"] == "len224"}
for r in rows:
    r["delta_macro"] = r["macro_ap"] - base[r["fold"]]["macro_ap"]
    r["delta_pooled"] = r["pooled_ap"] - base[r["fold"]]["pooled_ap"]
result = {"rows": rows, "means": {v: {k: float(np.mean([r[k] for r in rows if r["variant"] == v]))
                                       for k in ("macro_ap", "pooled_ap", "delta_macro", "delta_pooled")}
                                   for v in ("len224", "len384")}}
(root / "metrics_folds12.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
pl.DataFrame(cats).write_csv(root / "category_metrics_folds12.csv")
print(json.dumps(result, ensure_ascii=False, indent=2))

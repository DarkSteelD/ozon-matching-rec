import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).parent
hard = pl.read_parquet(ROOT / "input" / "hand_pairs.parquet")
out = {}
for variant in ["shared", "random", "category"]:
    folds = []
    for fold in ["fold_01", "fold_02"]:
        d = hard.filter(pl.col("fold") == fold)
        p = pl.read_csv(ROOT / "preds" / variant / f"{fold}.csv")
        assert p.select("id1","id2").equals(d.select("id1","id2"))
        y, s, c = d["target"].to_numpy(), p["predict"].to_numpy(), d["category"].to_numpy()
        pcs = {x: float(average_precision_score(y[c==x],s[c==x])) for x in sorted(set(c))}
        folds.append({"fold":fold,"pooled":float(average_precision_score(y,s)),
                      "macro":float(np.mean(list(pcs.values()))),"per_category":pcs})
    out[variant]={"folds":folds,"mean_macro":float(np.mean([x["macro"] for x in folds]))}
base={x["fold"]:x for x in out["shared"]["folds"]}
for v in out:
    for x in out[v]["folds"]:
        x["macro_delta_shared"]=x["macro"]-base[x["fold"]]["macro"]
Path(ROOT / "metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")

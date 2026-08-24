import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    hard = pl.read_parquet(args.root / "input" / "hand_pairs.parquet")
    out = {"variants": {}, "gate_threshold": 0.001}
    for variant in ("full", "bottom6", "top6"):
        rows = []
        for fold in ("fold_01", "fold_02", "fold_03", "fold_04"):
            path = args.root / "preds" / variant / f"{fold}.csv"
            if not path.exists():
                continue
            d = hard.filter(pl.col("fold") == fold)
            p = pl.read_csv(path)
            assert p.select("id1", "id2").equals(d.select("id1", "id2"))
            y, s, c = d["target"].to_numpy(), p["predict"].to_numpy(), d["category"].to_numpy()
            per = {x: float(average_precision_score(y[c == x], s[c == x])) for x in sorted(set(c))}
            rows.append({"fold": fold, "macro": float(np.mean(list(per.values()))),
                         "pooled": float(average_precision_score(y, s)), "per_category": per})
        out["variants"][variant] = rows
    base = {x["fold"]: x for x in out["variants"]["full"]}
    control = {x["fold"]: x for x in out["variants"]["top6"]}
    for variant, rows in out["variants"].items():
        for row in rows:
            row["delta_full"] = row["macro"] - base[row["fold"]]["macro"]
            if row["fold"] in control:
                row["delta_top6"] = row["macro"] - control[row["fold"]]["macro"]
    cand = {x["fold"]: x for x in out["variants"]["bottom6"]}
    out["gate_pass"] = all(f in cand and cand[f]["delta_full"] > 0.001
                           for f in ("fold_01", "fold_02"))
    (args.root / "metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate_pass": out["gate_pass"],
                      "gate": {f: cand.get(f, {}) for f in ("fold_01", "fold_02")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

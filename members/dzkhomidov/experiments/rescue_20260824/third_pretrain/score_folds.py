"""Score saved fold predictions without touching the repository validator."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def ap(y, p):
    order = np.argsort(-p, kind="stable")
    y, p = y[order], p[order]
    ends = np.concatenate([np.flatnonzero(np.diff(p)), [len(y) - 1]])
    tp = np.cumsum(y)[ends]
    recall = tp / y.sum()
    return float(np.sum((recall - np.concatenate([[0], recall[:-1]])) * tp / (ends + 1)))


parser = argparse.ArgumentParser()
parser.add_argument("pred_dir")
parser.add_argument("--out", required=True)
parser.add_argument("--folds", default="fold_01,fold_02")
args = parser.parse_args()

hand = pl.read_parquet("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")
rows = []
for fold in args.folds.split(","):
    truth = hand.filter(pl.col("fold") == fold).select("id1", "id2", "target")
    pred = pl.read_csv(Path(args.pred_dir) / f"{fold}.csv")
    joined = truth.join(pred, on=["id1", "id2"], how="left", validate="1:1")
    assert joined.height == truth.height and joined["predict"].null_count() == 0
    y = joined["target"].to_numpy().astype(np.int8)
    p = joined["predict"].to_numpy()
    rows.append({"fold": fold, "n": len(y), "prauc": ap(y, p)})
result = {"folds": rows, "mean_prauc": float(np.mean([x["prauc"] for x in rows]))}
Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

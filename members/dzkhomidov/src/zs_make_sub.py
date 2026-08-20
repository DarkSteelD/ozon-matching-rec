"""Build fold_0K.csv submission files from full-run zero-shot CSVs.

Rank-average per fold over the given prediction CSVs (fold,id1,id2,...,predict),
row order taken from hand_pairs.parquet (canonical target order).

Usage: zs_make_sub.py --out <dir> full_a.csv full_b.csv ...
"""
import argparse
from pathlib import Path

import polars as pl

DATA = Path.home() / "matching-work/data/hand_pairs.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("csvs", nargs="+")
    args = ap.parse_args()

    base = pl.read_parquet(DATA).select("fold", "id1", "id2")
    acc = None
    for i, path in enumerate(args.csvs):
        p = pl.read_csv(path).select("fold", "id1", "id2", "predict")
        p = p.with_columns(
            (pl.col("predict").rank().over("fold") / pl.len().over("fold"))
            .alias(f"r{i}"))
        acc = p.drop("predict") if acc is None else acc.join(
            p.drop("predict"), on=["fold", "id1", "id2"], how="inner")
    rcols = [c for c in acc.columns if c.startswith("r")]
    acc = acc.with_columns(pl.mean_horizontal(rcols).alias("predict"))

    out = base.join(acc.select("fold", "id1", "id2", "predict"),
                    on=["fold", "id1", "id2"], how="left", maintain_order="left")
    assert out["predict"].null_count() == 0, "missing pairs in blend"
    assert out.height == base.height
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    for fold in sorted(out["fold"].unique()):
        fd = out.filter(pl.col("fold") == fold).select("id1", "id2", "predict")
        fd.write_csv(dest / f"{fold}.csv")
        print(fold, fd.height)
    print("wrote", dest)


if __name__ == "__main__":
    main()

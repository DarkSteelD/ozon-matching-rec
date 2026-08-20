"""Stratified subsample of hand_pairs for the zero-shot LLM sweep.

Per fold: N pairs, preserving the fold's positive rate. Fixed seed.
Output: ~/matching-work/data/zs_sample_<N>.parquet (same columns as hand_pairs).
"""
import argparse
from pathlib import Path

import polars as pl

DATA = Path.home() / "matching-work/data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    df = pl.read_parquet(DATA / "hand_pairs.parquet")
    parts = []
    for fold in sorted(df["fold"].unique()):
        fd = df.filter(pl.col("fold") == fold)
        for tgt in (0, 1):
            sub = fd.filter(pl.col("target") == tgt)
            n = round(args.per_fold * sub.height / fd.height)
            parts.append(sub.sample(n=min(n, sub.height), seed=args.seed))
    out = pl.concat(parts)
    dest = DATA / f"zs_sample_{args.per_fold}.parquet"
    out.write_parquet(dest)
    print(dest, out.shape, out.group_by("fold", "target").len().sort("fold", "target"))


if __name__ == "__main__":
    main()

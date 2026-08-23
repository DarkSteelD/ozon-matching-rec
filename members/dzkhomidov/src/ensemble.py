"""Rank-average (or mean) ensemble of prediction experiments, per fold."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

WORK = Path.home() / "matching-work"
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def read_preds(exp: str, fold: str):
    pairs, vals = [], []
    with (WORK / "preds" / exp / f"{fold}.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append((row["id1"], row["id2"]))
            vals.append(float(row["predict"]))
    return pairs, np.asarray(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="exp[:weight] entries")
    ap.add_argument("--mode", choices=["rank", "mean", "logit"], default="rank")
    args = ap.parse_args()

    outdir = WORK / "preds" / args.exp
    outdir.mkdir(parents=True, exist_ok=True)
    parsed = []
    for spec in args.inputs:
        name, _, w = spec.partition(":")
        parsed.append((name, float(w) if w else 1.0))

    for fold in FOLDS:
        ref_pairs = None
        acc = None
        wsum = 0.0
        for name, w in parsed:
            pairs, vals = read_preds(name, fold)
            if ref_pairs is None:
                ref_pairs = pairs
            else:
                assert pairs == ref_pairs, f"pair order mismatch {name} {fold}"
            if args.mode == "rank":
                v = np.argsort(np.argsort(vals)) / (len(vals) - 1)
            elif args.mode == "logit":
                c = np.clip(vals, 1e-7, 1 - 1e-7)
                v = np.log(c / (1 - c))
            else:
                v = vals
            acc = v * w if acc is None else acc + v * w
            wsum += w
        acc /= wsum
        with (outdir / f"{fold}.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(["id1", "id2", "predict"])
            for (a, b), s in zip(ref_pairs, acc.tolist(), strict=True):
                wr.writerow([a, b, f"{s:.8f}"])
        print(fold, "done", flush=True)


if __name__ == "__main__":
    main()

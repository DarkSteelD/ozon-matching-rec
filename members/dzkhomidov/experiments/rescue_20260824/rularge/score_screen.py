"""Score partial OOF predictions and comparable on-disk controls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def read_predictions(path: Path) -> dict[tuple[int, int], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["id1"]), int(row["id2"])): float(row["predict"])
            for row in csv.DictReader(handle)
        }


def rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values, kind="stable"), kind="stable") / (len(values) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hand-control", type=Path, required=True)
    parser.add_argument("--strong-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="fold_01,fold_02")
    args = parser.parse_args()
    data = pl.read_parquet(args.data, columns=["fold", "id1", "id2", "target"])
    rows = []
    for fold in args.folds.split(","):
        part = data.filter(pl.col("fold") == fold)
        keys = list(zip(part["id1"].to_list(), part["id2"].to_list(), strict=True))
        y = part["target"].to_numpy()
        candidate_map = read_predictions(args.candidate / f"{fold}.csv")
        hand_map = read_predictions(args.hand_control / f"{fold}.csv")
        strong_map = read_predictions(args.strong_control / f"{fold}.csv")
        candidate = np.array([candidate_map[key] for key in keys])
        hand = np.array([hand_map[key] for key in keys])
        strong = np.array([strong_map[key] for key in keys])
        variants = {
            "rularge_hand": candidate,
            "rubase_hand_control": hand,
            "final_combo_control": strong,
            "final_combo_plus_rularge_rank_10pct": 0.9 * rank(strong) + 0.1 * rank(candidate),
            "negative_permuted_rularge": np.random.default_rng(20260824).permutation(candidate),
        }
        for variant, predictions in variants.items():
            rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "prauc": float(average_precision_score(y, predictions)),
                    "rows": len(y),
                    "positives": int(y.sum()),
                }
            )
    for variant in sorted({row["variant"] for row in rows}):
        scores = [row["prauc"] for row in rows if row["variant"] == variant]
        rows.append(
            {
                "variant": variant,
                "fold": "mean_01_02",
                "prauc": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)),
            }
        )
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        if row["fold"] == "mean_01_02":
            print(f'{row["variant"]:42s} {row["prauc"]:.8f} std={row["std"]:.8f}')


if __name__ == "__main__":
    main()

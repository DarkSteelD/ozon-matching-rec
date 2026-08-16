"""Weighted average of two or more OOF prediction sets on the frozen folds.

A blend is not a model: it has no training and nothing to re-run, only the
component prediction files. This script writes the averaged predictions in the
same layout as any experiment, so the blend is scored and registered by the
ordinary ``make score-v2`` path instead of a one-off number pasted by hand.

The pair order of every component is checked against the fold targets rather
than assumed — averaging two files whose rows are permuted differently would
produce a plausible-looking result that is silently wrong.

    .venv/bin/python members/darksteeld/src/blend_predictions.py \\
        --out blend_lgbm_knrm_audit_50 \\
        --component lgbm_cheap_audit:0.5 --component knrm_llm_audit:0.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FOLDS = [f"fold_{k:02d}" for k in range(1, 5)]


def read_column(path: Path, column: str) -> tuple[list[tuple[str, str]], np.ndarray]:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    keys = [(r["id1"], r["id2"]) for r in rows]
    return keys, np.array([float(r[column]) for r in rows], dtype=np.float64)


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return float("nan")
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="имя эксперимента-бленда")
    parser.add_argument("--component", action="append", required=True, metavar="ИМЯ:ВЕС",
                        help="можно повторять; веса нормируются к сумме 1")
    parser.add_argument("--predictions-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2" / "darksteeld")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    args = parser.parse_args()

    components = []
    for item in args.component:
        name, _, weight = item.partition(":")
        components.append((name, float(weight) if weight else 1.0))
    total = sum(w for _, w in components)
    if total <= 0:
        raise SystemExit("сумма весов должна быть положительной")
    components = [(name, weight / total) for name, weight in components]
    print("компоненты: " + ", ".join(f"{n} × {w:.3f}" for n, w in components))

    out_dir = args.predictions_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for fold in FOLDS:
        keys, target = read_column(args.targets_dir / f"{fold}.csv", "target")
        blended = np.zeros(len(target), dtype=np.float64)
        for name, weight in components:
            component_keys, prediction = read_column(
                args.predictions_dir / name / f"{fold}.csv", "predict")
            if component_keys != keys:
                raise SystemExit(f"{name}/{fold}: порядок пар не совпадает с целями фолда")
            blended += weight * prediction
        with (out_dir / f"{fold}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), value in zip(keys, blended.tolist(), strict=True):
                writer.writerow([id1, id2, f"{value:.8f}"])
        fold_score = average_precision(target, blended)
        scores.append(fold_score)
        print(f"  {fold}: {len(target):,} пар, PR-AUC {fold_score:.6f}")

    print(f"\nmean PR-AUC {np.mean(scores):.6f} -> {out_dir}")
    for name, _ in components:
        per_fold = []
        for fold in FOLDS:
            _, target = read_column(args.targets_dir / f"{fold}.csv", "target")
            _, prediction = read_column(args.predictions_dir / name / f"{fold}.csv", "predict")
            per_fold.append(average_precision(target, prediction))
        print(f"  компонент {name:<24} mean {np.mean(per_fold):.6f}")


if __name__ == "__main__":
    main()

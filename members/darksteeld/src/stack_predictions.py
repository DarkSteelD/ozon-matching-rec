"""Stacking of two or more OOF prediction sets on the frozen folds.

A blend fixes its weights by hand; a stack learns them. The meta-model here is a
logistic regression over the component scores, and the one thing it must not do
is learn them on the fold it is scored on. So the weights are fitted
leave-one-fold-out: predictions for ``fold_03`` come from a meta-model fitted on
folds 01, 02, 04 only. That mirrors what the base models already do — every
component file is itself out-of-fold — and it is the honest counterpart of
sweeping a weight grid against the same four folds and reporting the best cell.

Scores are combined in **logit** space by default. The components are
probabilities from three differently-shaped models, and a linear rule over
logits is the one that composes with their training objective (BCE) instead of
fighting it; ``--transform prob`` and ``--transform rank`` are kept as controls.

The pair order of every component is checked against the fold targets rather
than assumed — stacking two files whose rows are permuted differently would
produce a plausible-looking result that is silently wrong.

The script also fits a final meta-model on **all four** folds and writes it to
``--coefficients``. That file is what a container ships: the per-fold models
exist to measure the stack, the all-folds model is the one that scores the test
set, exactly as each component ships a model trained on all hand pairs.

    .venv/bin/python members/darksteeld/src/stack_predictions.py \\
        --out stack_knrm_name_attrs_lgbm \\
        --component knrm_llm_pretrain --component knrm_attrs_llm \\
        --component lgbm_cheap_v1 \\
        --coefficients members/darksteeld/container/stack3/meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FOLDS = [f"fold_{k:02d}" for k in range(1, 5)]
EPSILON = 1e-6


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


def rank01(column: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1] — the calibration-free view of one fold."""
    order = np.argsort(column, kind="mergesort")
    ranks = np.empty(len(column), dtype=np.float64)
    ranks[order] = np.arange(len(column), dtype=np.float64)
    return ranks / max(len(column) - 1, 1)


def transform_features(matrix: np.ndarray, transform: str) -> np.ndarray:
    if transform == "prob":
        return matrix
    if transform == "logit":
        clipped = np.clip(matrix, EPSILON, 1.0 - EPSILON)
        return np.log(clipped / (1.0 - clipped))
    if transform == "rank":
        return np.column_stack([rank01(matrix[:, j]) for j in range(matrix.shape[1])])
    raise SystemExit(f"неизвестное преобразование: {transform}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="имя эксперимента-стака")
    parser.add_argument("--component", action="append", required=True, metavar="ИМЯ",
                        help="можно повторять; порядок задаёт порядок коэффициентов")
    parser.add_argument("--transform", choices=("logit", "prob", "rank"), default="logit",
                        help="пространство, в котором линеен мета-классификатор")
    parser.add_argument("--C", type=float, default=1.0, help="обратная регуляризация логрегрессии")
    parser.add_argument("--coefficients", type=Path,
                        help="куда записать мета-модель, обученную на всех фолдах")
    parser.add_argument("--predictions-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2" / "darksteeld")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    args = parser.parse_args()

    components = list(args.component)
    if len(components) < 2:
        raise SystemExit("стак имеет смысл от двух компонент")
    print("компоненты: " + ", ".join(components) + f" | пространство: {args.transform}")

    keys: dict[str, list[tuple[str, str]]] = {}
    targets: dict[str, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for fold in FOLDS:
        keys[fold], targets[fold] = read_column(args.targets_dir / f"{fold}.csv", "target")
        columns = []
        for name in components:
            component_keys, prediction = read_column(
                args.predictions_dir / name / f"{fold}.csv", "predict")
            if component_keys != keys[fold]:
                raise SystemExit(f"{name}/{fold}: порядок пар не совпадает с целями фолда")
            columns.append(prediction)
        features[fold] = np.column_stack(columns)

    out_dir = args.predictions_dir / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, weights = [], []
    for held in FOLDS:
        train = [fold for fold in FOLDS if fold != held]
        design = np.vstack([transform_features(features[fold], args.transform) for fold in train])
        labels = np.concatenate([targets[fold] for fold in train])
        model = LogisticRegression(C=args.C, max_iter=2000)
        model.fit(design, labels)
        stacked = model.predict_proba(
            transform_features(features[held], args.transform))[:, 1]

        with (out_dir / f"{held}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), value in zip(keys[held], stacked.tolist(), strict=True):
                writer.writerow([id1, id2, f"{value:.8f}"])

        fold_score = average_precision(targets[held], stacked)
        scores.append(fold_score)
        weights.append(np.r_[model.coef_[0], model.intercept_])
        coefficient_text = " ".join(
            f"{name}={value:+.4f}" for name, value in zip(components, model.coef_[0], strict=True))
        print(f"  {held}: {len(targets[held]):,} пар, PR-AUC {fold_score:.6f} | "
              f"обучен на {', '.join(train)} | {coefficient_text} "
              f"bias={model.intercept_[0]:+.4f}")

    weights = np.array(weights)
    print(f"\nmean PR-AUC {np.mean(scores):.6f} -> {out_dir}")
    print("устойчивость весов по фолдам (std): " + " ".join(
        f"{name}={value:.4f}" for name, value in zip(components, weights[:, :-1].std(axis=0),
                                                     strict=True)))
    share = weights[:, :-1].mean(axis=0)
    share = share / share.sum()
    print("средние веса, нормированные к сумме 1: " + " ".join(
        f"{name}={value:.3f}" for name, value in zip(components, share, strict=True)))

    for index, name in enumerate(components):
        per_fold = [average_precision(targets[fold], features[fold][:, index]) for fold in FOLDS]
        print(f"  компонент {name:<24} mean {np.mean(per_fold):.6f}")

    if args.coefficients:
        design = np.vstack([transform_features(features[fold], args.transform) for fold in FOLDS])
        labels = np.concatenate([targets[fold] for fold in FOLDS])
        final = LogisticRegression(C=args.C, max_iter=2000)
        final.fit(design, labels)
        args.coefficients.parent.mkdir(parents=True, exist_ok=True)
        args.coefficients.write_text(json.dumps({
            "components": components,
            "transform": args.transform,
            "epsilon": EPSILON,
            "C": args.C,
            "coefficients": final.coef_[0].tolist(),
            "intercept": float(final.intercept_[0]),
            "fitted_on": FOLDS,
            "leave_one_fold_out_mean_prauc": float(np.mean(scores)),
            "leave_one_fold_out_per_fold": {
                fold: score for fold, score in zip(FOLDS, scores, strict=True)},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        text = " ".join(f"{name}={value:+.4f}"
                        for name, value in zip(components, final.coef_[0], strict=True))
        print(f"\nмета-модель на всех фолдах -> {args.coefficients}\n  {text} "
              f"bias={final.intercept_[0]:+.4f}")


if __name__ == "__main__":
    main()

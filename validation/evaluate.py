"""Validate shared-fold predictions and update the team leaderboard.

Metric: PR-AUC as ``average_precision_score`` (step-wise interpolation, ties
handled as score blocks — numerically identical to scikit-learn). The official
``total_prauc`` is the macro-average over the 20 categories; per-fold and pooled
scores are retained as diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from validation.render_leaderboard import render_leaderboard
from validation.spec import FoldSpec, load_spec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPOSITORY_ROOT / "validation" / "folds.json"
DEFAULT_TARGETS = REPOSITORY_ROOT / "validation" / "targets"
DEFAULT_RESULTS = REPOSITORY_ROOT / "validation" / "results"
DEFAULT_LEADERBOARD = REPOSITORY_ROOT / "validation" / "leaderboard.csv"
SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC (average precision), identical to sklearn.average_precision_score."""
    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_score.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot score empty inputs")
    positives = float(y_true.sum())
    if positives == 0:
        raise ValueError("Cannot compute PR-AUC without positive pairs")

    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    scores_sorted = y_score[order]
    # threshold block boundaries: last index of every distinct score value
    boundaries = np.flatnonzero(np.diff(scores_sorted)) if y_sorted.size > 1 else np.array([], int)
    block_ends = np.concatenate([boundaries, [y_sorted.size - 1]])
    tp_cum = np.cumsum(y_sorted)[block_ends]
    counts = block_ends + 1.0
    precision = tp_cum / counts
    recall = tp_cum / positives
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def read_pairs(
    path: Path, value_column: str, *, with_category: bool = False
) -> tuple[list[tuple[int, int]], np.ndarray, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing file: {path}"
            + (
                " (build local targets first: make validation-targets)"
                if "targets" in str(path)
                else ""
            )
        )
    expected = ["id1", "id2", value_column] + (["category"] if with_category else [])
    pair_keys: list[tuple[int, int]] = []
    values: list[float] = []
    categories: list[str] = []
    seen: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected}, found {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            try:
                pair = (int(row["id1"]), int(row["id2"]))
                value = float(row[value_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from error
            if pair in seen:
                raise ValueError(f"{path}:{line_number}: duplicate pair {pair}")
            if not math.isfinite(value):
                raise ValueError(f"{path}:{line_number}: value must be finite")
            seen.add(pair)
            pair_keys.append(pair)
            values.append(value)
            if with_category:
                categories.append(row["category"])
    if not pair_keys:
        raise ValueError(f"{path}: file is empty")
    return pair_keys, np.asarray(values, dtype=np.float64), categories


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_targets(targets_dir: Path, spec: FoldSpec) -> None:
    for fold in spec.folds:
        if fold.sha256 is None:
            raise ValueError("folds.json has no pinned SHA256 hashes; pin them first")
        path = targets_dir / f"{fold.id}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing target file {path}; build it with: make validation-targets"
            )
        actual = sha256_of_file(path)
        if actual != fold.sha256:
            raise ValueError(
                f"{path}: SHA256 {actual} does not match the pinned spec {fold.sha256}; "
                "rebuild with make validation-targets and do not edit targets by hand"
            )


def macro_over_categories(
    y_true: np.ndarray, y_score: np.ndarray, categories: list[str]
) -> tuple[float, list[dict[str, object]], int]:
    per_category: list[dict[str, object]] = []
    skipped = 0
    for name in sorted(set(categories)):
        mask = np.asarray([category == name for category in categories])
        positives = int(y_true[mask].sum())
        if positives == 0 or positives == int(mask.sum()):
            skipped += 1
            continue
        per_category.append(
            {
                "category": name,
                "rows": int(mask.sum()),
                "positives": positives,
                "prauc": average_precision(y_true[mask], y_score[mask]),
            }
        )
    if not per_category:
        raise ValueError("No category has both positive and negative pairs")
    macro = float(np.mean([float(item["prauc"]) for item in per_category]))
    return macro, per_category, skipped


def score_predictions(
    predictions_dir: Path, targets_dir: Path, spec: FoldSpec
) -> dict[str, object]:
    fold_results: list[dict[str, object]] = []
    pooled_true: list[np.ndarray] = []
    pooled_score: list[np.ndarray] = []
    pooled_categories: list[str] = []
    for fold in spec.folds:
        target_pairs, targets, categories = read_pairs(
            targets_dir / f"{fold.id}.csv", "target", with_category=True
        )
        prediction_pairs, predictions, _ = read_pairs(
            predictions_dir / f"{fold.id}.csv", "predict"
        )
        if prediction_pairs != target_pairs:
            raise ValueError(
                f"{fold.id}: prediction (id1, id2) values or order differ from canonical target"
            )
        fold_macro, _, fold_skipped = macro_over_categories(targets, predictions, categories)
        fold_results.append(
            {
                "id": fold.id,
                "rows": len(targets),
                "positives": int(targets.sum()),
                "prauc": average_precision(targets, predictions),
                "macro_category_prauc": fold_macro,
                "categories_skipped": fold_skipped,
            }
        )
        pooled_true.append(targets)
        pooled_score.append(predictions)
        pooled_categories.extend(categories)

    y_true = np.concatenate(pooled_true)
    y_score = np.concatenate(pooled_score)
    macro, per_category, skipped = macro_over_categories(y_true, y_score, pooled_categories)
    per_category.sort(key=lambda item: float(item["prauc"]))
    fold_praucs = [float(item["prauc"]) for item in fold_results]
    return {
        "mean_prauc": float(np.mean(fold_praucs)),
        "pooled_prauc": average_precision(y_true, y_score),
        "macro_category_prauc": macro,
        "min_fold_prauc": min(fold_praucs),
        "folds": fold_results,
        "per_category": per_category,
        "worst_category": per_category[0]["category"],
        "pooled_categories_skipped": skipped,
    }


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_slug(value: str, label: str) -> None:
    if not SLUG.fullmatch(value):
        raise ValueError(
            f"{label} must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )


def write_result(results_dir: Path, result: dict[str, object]) -> Path:
    destination = results_dir / str(result["member"]) / f'{result["experiment"]}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def rebuild_leaderboard(results_dir: Path, leaderboard_path: Path, spec: FoldSpec) -> int:
    results: list[dict[str, object]] = []
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("fold_spec_version") != spec.version:
                continue
            results.append(payload)
    results.sort(
        key=lambda item: (
            -float(item["macro_category_prauc"]),
            str(item["member"]),
            str(item["experiment"]),
        )
    )

    public_rows: list[tuple[float, str, str]] = []
    for result in results:
        public_prauc = result.get("public_prauc")
        if public_prauc is None:
            continue
        public_score = float(public_prauc)
        if not math.isfinite(public_score) or not 0.0 <= public_score <= 1.0:
            raise ValueError(
                f'{result["member"]}/{result["experiment"]}: public_prauc must be in [0, 1]'
            )
        public_rows.append((-public_score, str(result["member"]), str(result["experiment"])))
    public_rows.sort()
    public_ranks = {
        (member, experiment): rank
        for rank, (_, member, experiment) in enumerate(public_rows, start=1)
    }

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "member",
        "experiment",
        "mean_prauc",
        "pooled_prauc",
        "macro_cat_prauc",
        "public_prauc",
        "public_rank",
        "public_delta",
        *[f"{fold.id}_prauc" for fold in spec.folds],
        "min_fold_prauc",
        "worst_category",
        "evaluated_at",
        "commit",
        "notes",
    ]
    with leaderboard_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            public_prauc = result.get("public_prauc")
            public_score = float(public_prauc) if public_prauc is not None else None
            fold_scores = {
                str(item["id"]): float(item["prauc"])
                for item in result["folds"]  # type: ignore[union-attr]
            }
            writer.writerow(
                {
                    "rank": rank,
                    "member": result["member"],
                    "experiment": result["experiment"],
                    "mean_prauc": f'{float(result["mean_prauc"]):.8f}',
                    "pooled_prauc": f'{float(result["pooled_prauc"]):.8f}',
                    "macro_cat_prauc": f'{float(result["macro_category_prauc"]):.8f}',
                    "public_prauc": (
                        f"{public_score:.10f}" if public_score is not None else ""
                    ),
                    "public_rank": (
                        public_ranks[(str(result["member"]), str(result["experiment"]))]
                        if public_score is not None
                        else ""
                    ),
                    "public_delta": (
                        f'{public_score - float(result["macro_category_prauc"]):.8f}'
                        if public_score is not None
                        else ""
                    ),
                    **{
                        f"{fold.id}_prauc": f"{fold_scores[fold.id]:.8f}"
                        for fold in spec.folds
                    },
                    "min_fold_prauc": f'{float(result["min_fold_prauc"]):.8f}',
                    "worst_category": result.get("worst_category", ""),
                    "evaluated_at": result["evaluated_at"],
                    "commit": result["commit"],
                    "notes": result.get("notes", ""),
                }
            )
    render_leaderboard(leaderboard_path)
    return len(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member")
    parser.add_argument("--experiment")
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--public-prauc",
        type=float,
        help="Optional observed ODS leaderboard total_prauc for the same model",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--rebuild-only", action="store_true")
    return parser


def main() -> None:
    from datetime import UTC, datetime

    args = build_parser().parse_args()
    spec = load_spec(args.spec)
    if args.rebuild_only:
        count = rebuild_leaderboard(args.results_dir, args.leaderboard, spec)
        print(f"Rebuilt leaderboard with {count} experiment(s): {args.leaderboard}")
        return

    if not args.member or not args.experiment or args.predictions_dir is None:
        raise SystemExit("--member, --experiment, and --predictions-dir are required")
    validate_slug(args.member, "member")
    validate_slug(args.experiment, "experiment")
    verify_targets(args.targets_dir, spec)
    scores = score_predictions(args.predictions_dir, args.targets_dir, spec)
    result: dict[str, object] = {
        "fold_spec_version": spec.version,
        "member": args.member,
        "experiment": args.experiment,
        **scores,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": current_commit(),
        "notes": args.notes,
    }
    if args.public_prauc is not None:
        if not math.isfinite(args.public_prauc) or not 0.0 <= args.public_prauc <= 1.0:
            raise ValueError("public_prauc must be within [0, 1]")
        result["public_prauc"] = args.public_prauc
    result_path = write_result(args.results_dir, result)
    rebuild_leaderboard(args.results_dir, args.leaderboard, spec)
    summary = {
        key: result[key]
        for key in (
            "member",
            "experiment",
            "mean_prauc",
            "pooled_prauc",
            "macro_category_prauc",
            "min_fold_prauc",
            "worst_category",
        )
    }
    summary["folds"] = {
        str(item["id"]): round(float(item["prauc"]), 6) for item in scores["folds"]  # type: ignore[index]
    }
    summary["result_path"] = str(result_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

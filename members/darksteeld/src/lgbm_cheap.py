"""First trained reference: LightGBM over cheap pair features, OOF by folds.

For every frozen fold K the model is trained on the labels of the other three
folds only and predicts fold K, so the out-of-fold predictions respect the
grouped split (no item of fold K appears in its training pairs by
construction of the folds). Fixed hyperparameters, fixed seed, no early
stopping on the evaluation fold: this is a reference point, not a tuned model.

Features per pair (items_human texts only, label-free):
  name char_wb 3-5gram TF-IDF cosine; normalized-name token Jaccard; exact
  normalized-name equality; common-prefix ratio; name lengths and diffs;
  number-token sets from names (Jaccard, equality, one-sided counts);
  attributes key=value Jaccard; key-set Jaccard; shared keys; agreeing keys;
  conflicting keys (same key, no common value); attribute set sizes; category.

Writes ``validation/predictions/darksteeld/lgbm_cheap_v1/fold_0K.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_features import CATEGORICAL_FEATURES, FEATURE_NAMES, build_features  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
TARGETS_DIR = REPOSITORY_ROOT / "validation" / "targets"
EXPERIMENT = "lgbm_cheap_v1"
PREDICTIONS_DIR = REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld" / EXPERIMENT
SEED = 20260813

AUDIT_FILE = REPOSITORY_ROOT / "members" / "darksteeld" / "data" / "label_audit.jsonl"


def load_audit() -> dict[tuple[int, int], int]:
    """Ручные исправления меток; последнее судейство по паре побеждает."""
    import json

    if not AUDIT_FILE.is_file():
        return {}
    latest: dict[tuple[int, int], dict] = {}
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["id1"], r["id2"])] = r
    return {k: v["audited_label"] for k, v in latest.items()
            if v["audited_label"] >= 0 and v["audited_label"] != v["original_target"]}


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


def load_folds(targets_dir: Path = TARGETS_DIR) -> dict[str, tuple[list[tuple[int, int]], np.ndarray, list[str]]]:
    folds: dict[str, tuple[list[tuple[int, int]], np.ndarray, list[str]]] = {}
    paths = sorted(targets_dir.glob("fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold targets in {targets_dir}; run make validation-targets")
    for path in paths:
        pairs: list[tuple[int, int]] = []
        targets: list[int] = []
        categories: list[str] = []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                pairs.append((int(row["id1"]), int(row["id2"])))
                targets.append(int(row["target"]))
                categories.append(row["category"])
        folds[path.stem] = (pairs, np.asarray(targets, dtype=np.float64), categories)
    return folds


def main() -> None:
    import argparse

    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", type=Path, default=TARGETS_DIR,
                        help="fold targets; use validation/targets_v2 for the stratified spec v2")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--audit", action="store_true",
                        help="обучать на метках, исправленных в label_audit.jsonl")
    args = parser.parse_args()
    predictions_dir = args.predictions_dir

    folds = load_folds(args.targets_dir)
    fold_ids = list(folds)
    all_pairs = [pair for fold_id in fold_ids for pair in folds[fold_id][0]]
    y = np.concatenate([folds[fold_id][1] for fold_id in fold_ids])
    fold_of_row = np.concatenate(
        [np.full(len(folds[fold_id][0]), index) for index, fold_id in enumerate(fold_ids)]
    )
    categories_all = [category for fold_id in fold_ids for category in folds[fold_id][2]]
    category_codes = {name: code for code, name in enumerate(sorted(set(categories_all)))}
    print(f"pairs={len(all_pairs)} folds={ {f: len(folds[f][0]) for f in fold_ids} }")

    y_original = y.copy()
    if args.audit:
        corrections = load_audit()
        applied = 0
        for position, pair in enumerate(all_pairs):
            if pair in corrections:
                y[position] = corrections[pair]; applied += 1
        print(f"доразметка: применено {applied} исправлений из {len(corrections)} в журнале")
    else:
        print("доразметка: не применяется (--audit чтобы включить)")

    items = pl.read_parquet(
        RAW_DIR / "items_human.parquet", columns=["id", "name", "attributes", "category"]
    )
    features, known = build_features(
        items["id"].to_list(),
        items["name"].to_list(),
        items["attributes"].to_list(),
        items["category"].to_list(),
        np.asarray([a for a, _ in all_pairs], dtype=np.int64),
        np.asarray([b for _, b in all_pairs], dtype=np.int64),
        category_codes,
    )
    if not known.all():
        raise AssertionError("items_human must cover every hand pair")
    feature_names = FEATURE_NAMES

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    predictions_dir.mkdir(parents=True, exist_ok=True)
    fold_scores = []
    for fold_index, fold_id in enumerate(fold_ids):
        train_mask = fold_of_row != fold_index
        dataset = lgb.Dataset(
            features[train_mask],
            label=y[train_mask],
            feature_name=feature_names,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=True,
        )
        booster = lgb.train(params, dataset, num_boost_round=400)
        scores = booster.predict(features[~train_mask])
        pairs = folds[fold_id][0]
        with (predictions_dir / f"{fold_id}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), score in zip(pairs, scores.tolist(), strict=True):
                writer.writerow([id1, id2, f"{score:.8f}"])
        on_original = average_precision(y_original[~train_mask], scores)
        on_corrected = average_precision(y[~train_mask], scores)
        fold_scores.append((on_original, on_corrected))
        print(f"{fold_id}: trained on {int(train_mask.sum())} pairs, predicted {len(pairs)}"
              f"  |  PR-AUC на исходных метках {on_original:.6f}, на исправленных {on_corrected:.6f}")
        if fold_index == 0:
            gains = booster.feature_importance("gain")
            order = np.argsort(-gains)[:10]
            print("  top gain:", [(feature_names[k], round(float(gains[k]), 1)) for k in order])


    import numpy as _np
    a = _np.mean([x for x, _ in fold_scores]); b = _np.mean([x for _, x in fold_scores])
    print(f"\nmean PR-AUC: на исходных метках {a:.6f}, на исправленных {b:.6f}")
    print("контроль lgbm_cheap_v1 (обучен на исходных, spec-v2): 0.638171")


if __name__ == "__main__":
    main()

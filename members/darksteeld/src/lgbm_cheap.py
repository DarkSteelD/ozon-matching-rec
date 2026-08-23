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

**Шумовой пол этого пайплайна — 0.0003.** Пять прогонов одного и того же кода на
одних и тех же данных, различающихся только сидом бустинга, дали 0.637553,
0.637751, 0.637908, 0.638185, 0.638309: размах 0.00076, σ 0.00031. Любая дельта
меньше примерно 0.0006 здесь неотличима от перестановки сида, и мерить её надо
парно по нескольким сидам, а не одним прогоном. Измеренные так:

    доразметка нашим журналом (106 испр.)   +0.000024 +- 0.000095   не значимо
    доразметка журналом команды (244 испр.) +0.000069 +- 0.000097   не значимо
    транзитивное замыкание, вес 1.0         -0.000112               в пределах шума
    транзитивное замыкание, вес 3.0         -0.001568               значимо, во вред

Зарегистрированный ``lgbm_cheap_v1`` = 0.638171 (среднее по фолдам) воспроизводится
побитово только кодом ревизии 2ff04b5. Начиная с b8ae5d6 тот же расчёт даёт
0.637751. Причина не в логике: ``fit_transform`` отдавал разреженную матрицу с
несортированными индексами, а пришедший ему на смену ``fit().transform()`` — с
сортированными, и порядок накопления суммы менял косинус в последнем знаке
float32. Теперь порядок канонизируется явно (см. pair_features.py), так что
расчёт больше не зависит от того, каким путём получена матрица; цена перехода
0.00042 — внутри шумового пола выше.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closure_pairs import build_closure  # noqa: E402
from pair_features import CATEGORICAL_FEATURES, FEATURE_NAMES, build_features  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
TARGETS_DIR = REPOSITORY_ROOT / "validation" / "targets"
EXPERIMENT = "lgbm_cheap_v1"
PREDICTIONS_DIR = REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld" / EXPERIMENT
SEED = 20260813

AUDIT_FILE = REPOSITORY_ROOT / "members" / "darksteeld" / "data" / "label_audit.jsonl"


def load_audit(path: Path = AUDIT_FILE) -> dict[tuple[int, int], int]:
    """Ручные исправления меток; последнее судейство по паре побеждает."""
    import json

    if not path.is_file():
        return {}
    latest: dict[tuple[int, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
    parser.add_argument("--seed", type=int, default=SEED,
                        help="сид бустинга; варьируя его, меряют шумовой пол пайплайна")
    parser.add_argument("--audit-file", type=Path, default=AUDIT_FILE,
                        help="другой журнал: например общий с разметкой всей команды")
    parser.add_argument("--closure", action="store_true",
                        help="добавить в ОБУЧЕНИЕ пары, выводимые транзитивностью")
    parser.add_argument("--closure-weight", type=float, default=1.0,
                        help="вес выведенной строки против размеченной")
    parser.add_argument("--closure-fraction", type=float, default=1.0,
                        help="доля выведенных пар, взятая случайно (seed фиксирован)")
    parser.add_argument("--closure-kind", choices=["both", "pos", "neg"], default="both",
                        help="какие выведенные пары брать")
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
        corrections = load_audit(args.audit_file)
        applied = 0
        for position, pair in enumerate(all_pairs):
            if pair in corrections:
                y[position] = corrections[pair]; applied += 1
        print(f"доразметка: применено {applied} исправлений из {len(corrections)} "
              f"в журнале {args.audit_file.name}")
    else:
        print("доразметка: не применяется (--audit чтобы включить)")

    extra_pairs: list[tuple[int, int]] = []
    extra_y = np.zeros(0)
    extra_fold = np.zeros(0, dtype=int)
    if args.closure:
        # выведенные пары, отсуженные вручную как «не дубль», рвут свою цепочку
        import json as _json
        rejected = set()
        if AUDIT_FILE.is_file():
            for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = _json.loads(line)
                    if r.get("mode") == "closure" and r["audited_label"] == 0:
                        rejected.add((r["id1"], r["id2"]))
        if rejected:
            print(f"замыкание: {len(rejected)} выведенных пар отсужены как «не дубль» — "
                  f"их компоненты исключаются")
        produced, produced_y, produced_fold, contradictions = build_closure(
            all_pairs, y.tolist(), fold_of_row.tolist(), rejected)
        if contradictions:
            print(f"замыкание: {len(contradictions)} противоречий после исправлений — "
                  f"их компоненты исключены целиком")
        if args.closure_kind != "both":
            want = 1.0 if args.closure_kind == "pos" else 0.0
            keep = [v == want for v in produced_y]
            produced = [p for p, k in zip(produced, keep) if k]
            produced_y = [v for v, k in zip(produced_y, keep) if k]
            produced_fold = [f for f, k in zip(produced_fold, keep) if k]
        if args.closure_fraction < 1.0:
            rng = np.random.default_rng(SEED)
            keep = rng.random(len(produced)) < args.closure_fraction
            produced = [p for p, k in zip(produced, keep) if k]
            produced_y = [v for v, k in zip(produced_y, keep) if k]
            produced_fold = [f for f, k in zip(produced_fold, keep) if k]
        extra_pairs = produced
        extra_y = np.asarray(produced_y, dtype=np.float64)
        extra_fold = np.asarray(produced_fold, dtype=int)
        positives = int(extra_y.sum())
        print(f"замыкание: {len(extra_pairs)} выведенных пар "
              f"(+{positives} положительных, +{len(extra_pairs) - positives} отрицательных), "
              f"вес {args.closure_weight}, доля {args.closure_fraction}")
        print(f"  по фолдам: { {k: int((extra_fold == k).sum()) for k in range(len(fold_ids))} }")
    else:
        print("замыкание: не применяется (--closure чтобы включить)")

    items = pl.read_parquet(
        RAW_DIR / "items_human.parquet", columns=["id", "name", "attributes", "category"]
    )
    features, known = build_features(
        items["id"].to_list(),
        items["name"].to_list(),
        items["attributes"].to_list(),
        items["category"].to_list(),
        np.asarray([a for a, _ in all_pairs + extra_pairs], dtype=np.int64),
        np.asarray([b for _, b in all_pairs + extra_pairs], dtype=np.int64),
        category_codes,
    )
    if not known.all():
        raise AssertionError("items_human must cover every hand pair")
    feature_names = FEATURE_NAMES

    # выведенные строки идут ТОЛЬКО в обучение: маска отделяет их от размеченных,
    # по которым считается метрика
    n_labelled = len(all_pairs)
    y_train = np.concatenate([y, extra_y])
    fold_train = np.concatenate([fold_of_row, extra_fold])
    weights = np.concatenate([np.ones(n_labelled), np.full(len(extra_pairs), args.closure_weight)])
    is_labelled = np.arange(len(fold_train)) < n_labelled

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": args.seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }
    predictions_dir.mkdir(parents=True, exist_ok=True)
    fold_scores = []
    for fold_index, fold_id in enumerate(fold_ids):
        train_mask = fold_train != fold_index
        evaluate_mask = (fold_train == fold_index) & is_labelled
        dataset = lgb.Dataset(
            features[train_mask],
            label=y_train[train_mask],
            # None, а не вектор единиц: LightGBM по-разному считает границы листа
            # для взвешенной и невзвешенной задачи, и вектор единиц сдвигает скор
            weight=(weights[train_mask] if extra_pairs else None),
            feature_name=feature_names,
            categorical_feature=CATEGORICAL_FEATURES,
            free_raw_data=True,
        )
        booster = lgb.train(params, dataset, num_boost_round=400)
        scores = booster.predict(features[evaluate_mask])
        pairs = folds[fold_id][0]
        with (predictions_dir / f"{fold_id}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), score in zip(pairs, scores.tolist(), strict=True):
                writer.writerow([id1, id2, f"{score:.8f}"])
        fold_rows = fold_of_row == fold_index
        on_original = average_precision(y_original[fold_rows], scores)
        on_corrected = average_precision(y[fold_rows], scores)
        fold_scores.append((on_original, on_corrected))
        print(f"{fold_id}: trained on {int(train_mask.sum())} rows "
              f"({int((train_mask & ~is_labelled).sum())} выведенных), predicted {len(pairs)}"
              f"  |  PR-AUC на исходных метках {on_original:.6f}, на исправленных {on_corrected:.6f}")
        if fold_index == 0:
            gains = booster.feature_importance("gain")
            order = np.argsort(-gains)[:10]
            print("  top gain:", [(feature_names[k], round(float(gains[k]), 1)) for k in order])


    import numpy as _np
    a = _np.mean([x for x, _ in fold_scores]); b = _np.mean([x for _, x in fold_scores])
    print(f"\nmean PR-AUC: на исходных метках {a:.6f}, на исправленных {b:.6f}")
    print("контроль lgbm_cheap_v1 (обучен на исходных, spec-v2): 0.638171 — "
          "воспроизводится только кодом ревизии 2ff04b5, см. docstring")


if __name__ == "__main__":
    main()

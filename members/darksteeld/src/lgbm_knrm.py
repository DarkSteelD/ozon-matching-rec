"""LightGBM над дешёвыми парными признаками ПЛЮС выходы двух KNRM как признаки.

Это стекинг на уровне признаков, а не предсказаний: вместо того чтобы усреднять
три готовых скора, выходы KNRM по названию и KNRM по атрибутам подаются в
LightGBM двумя дополнительными колонками рядом с 21 дешёвым признаком. Модель
второго уровня получает право резать по ним и комбинировать их с остальными
нелинейно — например, доверять KNRM по атрибутам только там, где атрибутов
достаточно много.

**Где здесь утечка и как она закрыта.** Для фолда K обе KNRM обучены на folds≠K,
поэтому предсказание на самом фолде K честное. Опасность в другом: на
ОБУЧАЮЩИХ строках та же KNRM даёт in-sample предсказания, а они систематически
увереннее, чем то, что LightGBM увидит на тесте. Обученная на таких признаках
модель переоценивает KNRM-колонки. Скрипт умеет обе схемы, и разница между ними
и есть цена этой ошибки:

* ``--train-feature-mode insample`` — буквальная схема: признак обучающей строки
  берётся у той же KNRM, что обучалась на её фолде. Честно по фолду K,
  оптимистично по признаку.
* ``--train-feature-mode nested`` — вложенный OOF: признак строки из фолда J
  берётся у KNRM, не видевшей ни J, ни K. Признак на обучении и на тесте
  становится одной и той же величиной.

Раскладка обоих источников одинакова, отличается только каталог:

    <train-dir>/trained_without_<K>/<J>.csv   — признак строки фолда J при внешнем фолде K
    <oof-dir>/<K>.csv                         — признак строк самого фолда K

Порядок пар в каждом файле сверяется с целями фолда, а не предполагается.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pair_features import CATEGORICAL_FEATURES, FEATURE_NAMES, build_features  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
SEED = 20260813
KNRM_FEATURES = ["knrm_name", "knrm_attrs"]


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


def load_folds(targets_dir: Path):
    folds = {}
    for path in sorted(targets_dir.glob("fold_*.csv")):
        pairs, targets, categories = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                pairs.append((int(row["id1"]), int(row["id2"])))
                targets.append(int(row["target"]))
                categories.append(row["category"])
        folds[path.stem] = (pairs, np.asarray(targets, dtype=np.float64), categories)
    if not folds:
        raise FileNotFoundError(f"нет файлов целей в {targets_dir}")
    return folds


def read_prediction(path: Path, expected_pairs: list[tuple[int, int]]) -> np.ndarray:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(expected_pairs):
        raise SystemExit(f"{path}: {len(rows):,} строк против {len(expected_pairs):,} пар фолда")
    values = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows):
        if (int(row["id1"]), int(row["id2"])) != expected_pairs[position]:
            raise SystemExit(f"{path}: порядок пар не совпадает с целями фолда (строка {position})")
        values[position] = float(row["predict"])
    return values


def main() -> None:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--knrm-name-oof", type=Path, required=True,
                        help="каталог с <fold>.csv: KNRM по имени, обучен на folds!=fold")
    parser.add_argument("--knrm-name-train", type=Path, required=True,
                        help="каталог с trained_without_<K>/<J>.csv для обучающих строк")
    parser.add_argument("--knrm-attrs-oof", type=Path, required=True)
    parser.add_argument("--knrm-attrs-train", type=Path, required=True)
    parser.add_argument("--train-feature-mode", choices=("insample", "nested"), required=True,
                        help="только для протокола: раскладка каталогов одна и та же, "
                             "флаг попадает в лог и в заметку эксперимента")
    parser.add_argument("--num-boost-round", type=int, default=400)
    args = parser.parse_args()

    folds = load_folds(args.targets_dir)
    fold_ids = list(folds)
    all_pairs = [pair for fold_id in fold_ids for pair in folds[fold_id][0]]
    y = np.concatenate([folds[fold_id][1] for fold_id in fold_ids])
    fold_of_row = np.concatenate(
        [np.full(len(folds[fold_id][0]), index) for index, fold_id in enumerate(fold_ids)])
    categories_all = [category for fold_id in fold_ids for category in folds[fold_id][2]]
    category_codes = {name: code for code, name in enumerate(sorted(set(categories_all)))}
    print(f"пар {len(all_pairs):,}, фолды { {f: len(folds[f][0]) for f in fold_ids} }")
    print(f"режим признаков на обучающих строках: {args.train_feature_mode}")

    items = pl.read_parquet(RAW_DIR / "items_human.parquet",
                            columns=["id", "name", "attributes", "category"])
    cheap, known = build_features(
        items["id"].to_list(), items["name"].to_list(), items["attributes"].to_list(),
        items["category"].to_list(),
        np.asarray([a for a, _ in all_pairs], dtype=np.int64),
        np.asarray([b for _, b in all_pairs], dtype=np.int64),
        category_codes)
    if not known.all():
        raise AssertionError("items_human обязан покрывать каждую ручную пару")
    print(f"дешёвые признаки {cheap.shape}")

    # Столбцы KNRM: для каждого внешнего фолда K своя колонка признаков, потому
    # что источник у обучающих и оценочных строк разный по построению.
    sources = [(args.knrm_name_oof, args.knrm_name_train), (args.knrm_attrs_oof,
                                                            args.knrm_attrs_train)]
    knrm = {}
    for held_out in fold_ids:
        columns = []
        for oof_dir, train_dir in sources:
            column = np.empty(len(all_pairs), dtype=np.float64)
            offset = 0
            for fold_id in fold_ids:
                pairs = folds[fold_id][0]
                if fold_id == held_out:
                    path = oof_dir / f"{fold_id}.csv"
                else:
                    path = train_dir / f"trained_without_{held_out}" / f"{fold_id}.csv"
                column[offset:offset + len(pairs)] = read_prediction(path, pairs)
                offset += len(pairs)
            columns.append(column)
        knrm[held_out] = np.column_stack(columns)

    feature_names = FEATURE_NAMES + KNRM_FEATURES
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
    args.predictions_dir.mkdir(parents=True, exist_ok=True)
    fold_scores = []
    for fold_index, fold_id in enumerate(fold_ids):
        features = np.column_stack([cheap, knrm[fold_id]])
        train_mask = fold_of_row != fold_index
        dataset = lgb.Dataset(features[train_mask], label=y[train_mask],
                              feature_name=feature_names,
                              categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
        booster = lgb.train(params, dataset, num_boost_round=args.num_boost_round)
        scores = booster.predict(features[~train_mask])

        pairs = folds[fold_id][0]
        with (args.predictions_dir / f"{fold_id}.csv").open("w", newline="",
                                                            encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), score in zip(pairs, scores.tolist(), strict=True):
                writer.writerow([id1, id2, f"{score:.8f}"])

        score = average_precision(y[~train_mask], scores)
        fold_scores.append(score)
        gains = booster.feature_importance("gain")
        total = float(gains.sum()) or 1.0
        share = {name: 100 * float(gains[feature_names.index(name)]) / total
                 for name in KNRM_FEATURES}
        order = np.argsort(-gains)[:6]
        print(f"{fold_id}: обучен на {int(train_mask.sum()):,} парах, PR-AUC {score:.6f} | "
              f"доля gain: knrm_name {share['knrm_name']:.1f}%, "
              f"knrm_attrs {share['knrm_attrs']:.1f}%")
        print("    топ по gain:", [feature_names[k] for k in order])

    print(f"\nmean PR-AUC {np.mean(fold_scores):.6f} -> {args.predictions_dir}")
    print("контроли: lgbm_cheap_v1 0.638171 | blend3_opt 0.681100 | stack3_logit_full 0.683971")


if __name__ == "__main__":
    main()

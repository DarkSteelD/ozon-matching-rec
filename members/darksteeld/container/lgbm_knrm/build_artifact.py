"""Обучить отгружаемый LightGBM второго уровня: 21 дешёвый признак + два KNRM.

**Чем обучающие признаки обязаны быть.** В контейнере обе KNRM обучены на всех
ручных парах и предсказывают тестовые пары, которых не видели, — то есть
out-of-sample. Значит и обучать LightGBM надо на out-of-sample колонках, иначе
он увидит на обучении величину, которой на тесте не будет: измерено на фолдах,
in-sample признак стоит 0.0388 PR-AUC (0.666097 против 0.704896). Поэтому сюда
подаются OOF-предсказания — те же, на которых считался эксперимент
``lgbm_knrm_nested``: для каждой пары скор от KNRM, обученной на трёх фолдах без
её собственного.

Сам LightGBM при этом обучается на **всех** 365,654 парах: фолды нужны были,
чтобы честно получить колонку признака, а не чтобы делить обучение.

Рецепты KNRM, чьи OOF сюда идут, обязаны совпадать с теми, что лежат в
контейнере: иначе распределение признака на обучении и на инференсе разъедется.
Проверка вынесена в ``--knrm-name-artifact``/``--knrm-attrs-artifact``: из них
читается имя эксперимента и записывается в артефакт.

    .venv/bin/python members/darksteeld/container/lgbm_knrm/build_artifact.py \\
        --knrm-name-oof <scratch>/kf/names_oof \\
        --knrm-attrs-oof <scratch>/kf/attrs_oof
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from pair_features import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_NAMES, build_category_codes, build_features)

SEED = 20260813
KNRM_FEATURES = ["knrm_name", "knrm_attrs"]


def read_prediction(path: Path, expected: list[tuple[int, int]]) -> np.ndarray:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(expected):
        raise SystemExit(f"{path}: {len(rows):,} строк против {len(expected):,} пар фолда")
    values = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows):
        if (int(row["id1"]), int(row["id2"])) != expected[position]:
            raise SystemExit(f"{path}: порядок пар разошёлся с целями фолда на строке {position}")
        values[position] = float(row["predict"])
    return values


def main() -> None:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--knrm-name-oof", type=Path, required=True)
    parser.add_argument("--knrm-attrs-oof", type=Path, required=True)
    parser.add_argument("--knrm-name-artifact", type=Path,
                        default=HERE / "knrm_name" / "artifact.json")
    parser.add_argument("--knrm-attrs-artifact", type=Path,
                        default=HERE / "knrm_attrs" / "artifact.json")
    parser.add_argument("--num-boost-round", type=int, default=400)
    args = parser.parse_args()

    pairs: list[tuple[int, int]] = []
    targets: list[int] = []
    categories: list[str] = []
    fold_slices: dict[str, tuple[int, int]] = {}
    for path in sorted(args.targets_dir.glob("fold_*.csv")):
        start = len(pairs)
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                pairs.append((int(row["id1"]), int(row["id2"])))
                targets.append(int(row["target"]))
                categories.append(row["category"])
        fold_slices[path.stem] = (start, len(pairs))
    if not pairs:
        raise SystemExit(f"нет файлов целей в {args.targets_dir}")
    y = np.asarray(targets, dtype=np.float64)
    print(f"ручных пар {len(pairs):,}, фолдов {len(fold_slices)}")

    knrm = np.empty((len(pairs), 2), dtype=np.float64)
    for column, oof_dir in enumerate((args.knrm_name_oof, args.knrm_attrs_oof)):
        for fold_id, (start, stop) in fold_slices.items():
            knrm[start:stop, column] = read_prediction(oof_dir / f"{fold_id}.csv",
                                                       pairs[start:stop])
        print(f"  {KNRM_FEATURES[column]}: OOF из {oof_dir}, "
              f"диапазон {knrm[:, column].min():.6f}..{knrm[:, column].max():.6f}")

    items = pl.read_parquet(args.data_dir / "items_human.parquet",
                            columns=["id", "name", "attributes", "category"])
    category_codes = build_category_codes(items["category"].to_list())
    cheap, known = build_features(
        items["id"].to_list(), items["name"].to_list(), items["attributes"].to_list(),
        items["category"].to_list(),
        np.asarray([a for a, _ in pairs], dtype=np.int64),
        np.asarray([b for _, b in pairs], dtype=np.int64),
        category_codes)
    if not known.all():
        raise AssertionError("items_human обязан покрывать каждую ручную пару")
    features = np.column_stack([cheap, knrm])
    feature_names = FEATURE_NAMES + KNRM_FEATURES
    print(f"признаки {features.shape}: {len(FEATURE_NAMES)} дешёвых + {len(KNRM_FEATURES)} KNRM")

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
    dataset = lgb.Dataset(features, label=y, feature_name=feature_names,
                          categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
    booster = lgb.train(params, dataset, num_boost_round=args.num_boost_round)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.out_dir / "model.txt"))
    gains = booster.feature_importance("gain")
    total = float(gains.sum()) or 1.0
    order = np.argsort(-gains)[:8]
    print("топ по gain:", [(feature_names[k], f"{100 * gains[k] / total:.1f}%") for k in order])

    def experiment_of(path: Path) -> str | None:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8")).get("experiment")
        return None

    artifact = {
        "experiment": "lgbm_knrm_nested",
        "trained_on_pairs": len(pairs),
        "feature_names": feature_names,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_codes": category_codes,
        "prior": float(y.mean()),
        "num_boost_round": args.num_boost_round,
        "params": params,
        "lightgbm_version": lgb.__version__,
        "knrm_name_experiment": experiment_of(args.knrm_name_artifact),
        "knrm_attrs_experiment": experiment_of(args.knrm_attrs_artifact),
        "knrm_features_are": "out-of-fold; in-sample стоит 0.0388 PR-AUC (см. STACK_FEATURES.md)",
        "local_cv": {
            "spec_v2_mean_prauc": 0.70489562,
            "control_stack3_logit_full": 0.68397125,
            "control_lgbm_cheap_v1": 0.63817140,
            "control_insample_features": 0.66609694,
        },
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nартефакт -> {args.out_dir}/model.txt, artifact.json "
          f"({booster.num_trees()} деревьев, prior {artifact['prior']:.4f})")


if __name__ == "__main__":
    main()

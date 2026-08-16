"""LightGBM, обученный на matches_llm вместе с тренировочными фолдами.

Схема: для фолда K модель учится на **всех отобранных LLM-парах плюс парах трёх
остальных фолдов**, предсказывает фолд K. Контроль — `lgbm_cheap_v1` (mean
0.63817 на spec-v2), обученный только на ручной разметке; набор признаков и
гиперпараметры те же, единственная изменённая переменная — добавленные данные.

Утечки нет: вселенные товаров не пересекаются (0 из 12 384 610 LLM-товаров
встречается в `items_human`), и это проверяется в рантайме. Поэтому LLM-пары
можно подмешивать в обучение любого фолда.

Три решения, которые пришлось принять, и почему:

* **Один векторайзер на обе вселенных.** TF-IDF учится на `items_human` и тем же
  объектом применяется к LLM-именам. Отдельный fit на каждую вселенную поставил
  бы косинус на две разные шкалы, и модель видела бы одну колонку в двух
  значениях.
* **LLM-таргет бинаризуется по 0.5.** Мягкие метки просятся в
  `objective="cross_entropy"`, но тогда менялись бы сразу две вещи — данные и
  функция потерь. Здесь меняется только первое; мягкие метки как отдельный шаг.
* **Подвыборка LLM-пар.** Атрибуты 12.4 млн товаров не разобрать в память:
  интернирование key=value множеств для такого числа item'ов стоит десятки
  гигабайт. `--llm-pairs` ограничивает выборку; для GBDT на 21 признаке
  насыщение наступает задолго до 11 млн строк.

Ручные метки берутся с учётом журнала доразметки
(`members/darksteeld/data/label_audit.jsonl`), если он есть.

    .venv/bin/python members/darksteeld/experiments/lgbm_llm/train.py \
        --targets-dir validation/targets_v2 \
        --out-dir validation/predictions_v2/darksteeld/lgbm_llm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from pair_features import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_NAMES, build_category_codes, build_features, fit_name_vectorizer,
)

SEED = 20260813
NUM_BOOST_ROUND = 400
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 100, "feature_fraction": 0.9, "bagging_fraction": 0.9,
    "bagging_freq": 1, "seed": SEED, "deterministic": True,
    "force_row_wise": True, "verbosity": -1,
}
AUDIT_FILE = REPOSITORY_ROOT / "members" / "darksteeld" / "data" / "label_audit.jsonl"


def log(message: str) -> None:
    print(message, flush=True)


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


def load_audit() -> dict[tuple[int, int], int]:
    """Исправленные вручную метки; последнее судейство по паре побеждает."""
    if not AUDIT_FILE.is_file():
        return {}
    latest: dict[tuple[int, int], dict] = {}
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["id1"], r["id2"])] = r
    return {k: v["audited_label"] for k, v in latest.items()
            if v["audited_label"] >= 0 and v["audited_label"] != v["original_target"]}


def load_llm_items(items_path: Path, wanted: np.ndarray):
    """id/name/attributes/category только для нужных товаров, потоково."""
    import pyarrow.parquet as pq

    ids, names, attributes, categories = [], [], [], []
    scanned, started = 0, time.time()
    for batch in pq.ParquetFile(items_path).iter_batches(
            batch_size=200_000, columns=["id", "name", "attributes", "category"]):
        column = np.asarray(batch.column("id"), dtype=np.int64)
        position = np.searchsorted(wanted, column)
        position[position >= len(wanted)] = 0
        hit = wanted[position] == column
        if hit.any():
            index = np.flatnonzero(hit)
            n = batch.column("name").to_pylist()
            a = batch.column("attributes").to_pylist()
            c = batch.column("category").to_pylist()
            for k in index.tolist():
                ids.append(int(column[k])); names.append(n[k])
                attributes.append(a[k]); categories.append(c[k])
        scanned += batch.num_rows
        if scanned % 4_000_000 == 0:
            log(f"    просмотрено {scanned:,}, собрано {len(ids):,}, {time.time()-started:.0f}s")
    log(f"  собрано {len(ids):,}/{len(wanted):,} товаров за {time.time()-started:.0f}s")
    return ids, names, attributes, categories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2" / "darksteeld" / "lgbm_llm")
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04")
    parser.add_argument("--llm-pairs", type=int, default=1_000_000)
    parser.add_argument("--llm-weight", default="1.0",
                        help="вес LLM-строки относительно ручной; можно список через запятую "
                             "для свипа — признаки считаются один раз на все веса")
    parser.add_argument("--no-audit", action="store_true", help="игнорировать доразметку")
    args = parser.parse_args()

    import lightgbm as lgb
    import polars as pl

    fold_ids = [f.strip() for f in args.folds.split(",") if f.strip()]
    started = time.time()

    # ---- ручные фолды ------------------------------------------------------
    folds = {}
    for fold_id in fold_ids:
        rows = list(csv.DictReader((args.targets_dir / f"{fold_id}.csv").open(encoding="utf-8")))
        folds[fold_id] = {
            "id1": np.array([int(r["id1"]) for r in rows], dtype=np.int64),
            "id2": np.array([int(r["id2"]) for r in rows], dtype=np.int64),
            "target": np.array([float(r["target"]) for r in rows], dtype=np.float64),
            "category": [r["category"] for r in rows],
        }
    hand_id1 = np.concatenate([folds[f]["id1"] for f in fold_ids])
    hand_id2 = np.concatenate([folds[f]["id2"] for f in fold_ids])
    hand_y = np.concatenate([folds[f]["target"] for f in fold_ids])
    fold_of_row = np.concatenate([np.full(len(folds[f]["target"]), k)
                                  for k, f in enumerate(fold_ids)])
    pair_categories = [c for f in fold_ids for c in folds[f]["category"]]
    category_codes = build_category_codes(pair_categories)

    corrections = {} if args.no_audit else load_audit()
    if corrections:
        applied = 0
        for position, key in enumerate(zip(hand_id1.tolist(), hand_id2.tolist())):
            if key in corrections:
                hand_y[position] = corrections[key]; applied += 1
        log(f"доразметка: применено {applied} исправлений из {len(corrections)} в журнале")
    else:
        log("доразметка: журнал пуст или отключён")

    # ---- признаки ручной вселенной ----------------------------------------
    items = pl.read_parquet(args.data_dir / "items_human.parquet",
                            columns=["id", "name", "attributes", "category"])
    log(f"items_human {items.height:,}; учу векторайзер на его именах")
    vectorizer = fit_name_vectorizer(items["name"].to_list())
    hand_features, known = build_features(
        items["id"].to_list(), items["name"].to_list(), items["attributes"].to_list(),
        items["category"].to_list(), hand_id1, hand_id2, category_codes,
        vectorizer=vectorizer, log=log)
    if not known.all():
        raise AssertionError("items_human не покрывает все ручные пары")
    hand_features = hand_features.astype(np.float32)
    del items
    log(f"ручные признаки {hand_features.shape}, {time.time()-started:.0f}s")

    # ---- признаки LLM-вселенной -------------------------------------------
    llm = pl.read_parquet(args.data_dir / "matches_llm.parquet")
    if args.llm_pairs and args.llm_pairs < llm.height:
        llm = llm.sample(n=args.llm_pairs, seed=SEED, shuffle=True)
    llm_id1 = llm["id1"].to_numpy(); llm_id2 = llm["id2"].to_numpy()
    llm_y = (llm["target"].to_numpy() >= 0.5).astype(np.float64)
    log(f"\nLLM-пар взято {llm.height:,} из 11,187,780; позитивов после бинаризации "
        f"{llm_y.mean():.4f}")
    del llm

    wanted = np.unique(np.concatenate([llm_id1, llm_id2]))
    log(f"нужно товаров: {len(wanted):,} — тяну из items.parquet")
    ids, names, attributes, categories = load_llm_items(args.data_dir / "items.parquet", wanted)

    hand_ids = set(int(x) for x in np.concatenate([hand_id1, hand_id2]))
    overlap = sum(1 for x in ids if x in hand_ids)
    if overlap:
        raise AssertionError(f"вселенные пересекаются в {overlap} товарах — обучение потечёт")
    log(f"пересечение с ручной вселенной: {overlap} (проверено)")

    llm_features, llm_known = build_features(
        ids, names, attributes, categories, llm_id1, llm_id2, category_codes,
        vectorizer=vectorizer, log=log)
    llm_features = llm_features.astype(np.float32)
    del ids, names, attributes, categories
    log(f"LLM-признаки {llm_features.shape}, покрыто пар {llm_known.mean():.4f}, "
        f"{time.time()-started:.0f}s")
    llm_features = llm_features[llm_known]
    llm_y = llm_y[llm_known]

    # ---- обучение по фолдам, для каждого веса -------------------------------
    weights = [float(w) for w in str(args.llm_weight).split(",") if w.strip()]
    balanced = len(hand_y) * 0.75 / max(len(llm_y), 1)
    log(f"\nвеса LLM-строк: {weights}   (равный суммарный вклад двух источников "
        f"был бы при {balanced:.3f})")
    summary = {}
    for weight_value in weights:
        out_dir = args.out_dir if len(weights) == 1 else Path(f"{args.out_dir}_w{weight_value:g}")
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_per_fold = {}
        for k, held_out in enumerate(fold_ids):
            train_mask = fold_of_row != k
            x = np.vstack([llm_features, hand_features[train_mask]])
            y = np.concatenate([llm_y, hand_y[train_mask]])
            weight = np.concatenate([
                np.full(len(llm_y), weight_value, dtype=np.float32),
                np.ones(int(train_mask.sum()), dtype=np.float32),
            ])
            dataset = lgb.Dataset(x, label=y, weight=weight, feature_name=FEATURE_NAMES,
                                  categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True)
            booster = lgb.train(PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)
            del dataset, x, y, weight

            predictions = booster.predict(hand_features[~train_mask])
            fold = folds[held_out]
            with (out_dir / f"{held_out}.csv").open("w", newline="", encoding="utf-8") as sink:
                writer = csv.writer(sink, lineterminator="\n")
                writer.writerow(["id1", "id2", "predict"])
                for a, b, s in zip(fold["id1"].tolist(), fold["id2"].tolist(),
                                   predictions.tolist(), strict=True):
                    writer.writerow([a, b, f"{s:.8f}"])
            scores_per_fold[held_out] = average_precision(hand_y[~train_mask], predictions)
            if k == 0 and weight_value == weights[0]:
                gains = booster.feature_importance("gain")
                order = np.argsort(-gains)[:8]
                log(f"  топ по gain: {[(FEATURE_NAMES[i], round(float(gains[i]))) for i in order]}")
        values = list(scores_per_fold.values())
        summary[weight_value] = (float(np.mean(values)), max(values) - min(values), out_dir)
        log(f"  llm_weight={weight_value:<6g} mean PR-AUC {np.mean(values):.6f}  "
            f"разброс {max(values)-min(values):.6f}  -> {out_dir.name}")

    log(f"\n{'llm_weight':>11}{'mean PR-AUC':>14}{'к контролю':>13}")
    log(f"{'0 (контроль)':>11}{0.638171:>14.6f}{0.0:>+13.5f}")
    for weight_value, (mean, spread, _) in sorted(summary.items()):
        log(f"{weight_value:>11g}{mean:>14.6f}{mean - 0.638171:>+13.5f}")
    log(f"\nвсего {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()

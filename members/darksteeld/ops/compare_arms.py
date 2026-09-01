"""Сравнение веток свипа по фолдовым предсказаниям ``train_distill.py --stage folds``.

Ветка — это один прогон по всем фолдам с фиксированными гиперпараметрами; на
диске она выглядит как каталог с ``fold_01.csv`` ... ``fold_04.csv``, где лежат
``id1, id2, predict``.

Считается три вещи, и все три нужны по разным причинам:

*   **PR-AUC по фолду и среднее по ним.** Это то же число, что печатает сам
    ``train_distill``, поэтому оно напрямую сравнимо с уже измеренными
    ориентирами: 0.855566 у лучшей конфигурации на rubert-base и 0.851713 у
    ``ce_priodistill``.
*   **Макро-среднее по категориям.** На лидерборде метрика макро по 20
    категориям, а не общая. Модель может выигрывать в среднем и проигрывать
    макро, если её выигрыш сосредоточен в крупных категориях.
*   **Парные разности по фолдам.** Среднее из четырёх значений само по себе
    почти ничего не доказывает: шумовой пол пайплайна около 0.0003, а разброс
    между ветками бывает того же порядка. Одна и та же разность, устойчиво
    воспроизводящаяся на 4 фолдах из 4, — куда более сильное свидетельство, чем
    та же разность в среднем. Поэтому знак разности по каждому фолду печатается
    отдельно.

    python compare_arms.py --runs runs/mb_adan_5e-5 runs/mb_adan_1e-4 \
        --data data/hand_pairs_distill_aug.parquet --items data/raw/items.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

# Ориентиры, померенные раньше на тех же фолдах и той же метрике.
BASELINES = {
    "A_base (rubert-base, AdamW, без замыкания)": 0.855566,
    "ce_priodistill у dzkhomidov": 0.851713,
}


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC, идентичный sklearn.average_precision_score и версии train_distill."""
    order = np.argsort(-score, kind="stable")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return float("nan")
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def macro_over_categories(target: np.ndarray, score: np.ndarray,
                          category: np.ndarray) -> tuple[float, int]:
    """Среднее PR-AUC по категориям. Категории без обоих классов пропускаются."""
    scores = []
    for name in np.unique(category):
        mask = category == name
        positives = int(target[mask].sum())
        # В категории из одних позитивов или одних негативов PR-AUC не определён.
        if positives == 0 or positives == int(mask.sum()):
            continue
        scores.append(average_precision(target[mask], score[mask]))
    if not scores:
        raise ValueError("ни в одной категории нет обоих классов")
    return float(np.mean(scores)), len(scores)


def load_truth(data: Path, items: Path | None) -> pl.DataFrame:
    """Ручные пары с истиной. Выведенные замыканием исключаются: обучение их не
    видело в правильной конфигурации, и оценка на них несравнима с ориентирами."""
    frame = pl.read_parquet(data, columns=["fold", "id1", "id2", "target", "source"])
    frame = frame.filter(pl.col("source") == "hand").drop("source")
    if items is not None:
        # Категорию берём по первому товару пары: кандидаты в дубли приходят из
        # одной категории, так что второй товар дал бы то же самое.
        catalog = pl.read_parquet(items, columns=["id", "category"])
        frame = frame.join(catalog.rename({"id": "id1"}), on="id1", how="left")
    return frame


def score_arm(run: Path, truth: pl.DataFrame) -> dict[str, object]:
    per_fold: dict[str, float] = {}
    pooled_target, pooled_score, pooled_category = [], [], []
    for fold in sorted(truth["fold"].unique().to_list()):
        path = run / f"{fold}.csv"
        if not path.is_file():
            continue
        predicted = pl.read_csv(path)
        held = truth.filter(pl.col("fold") == fold).join(
            predicted, on=["id1", "id2"], how="inner")
        if held.height == 0:
            raise ValueError(f"{path}: ни одна пара не сошлась с истиной")
        target = held["target"].to_numpy().astype(float)
        score = held["predict"].to_numpy().astype(float)
        per_fold[fold] = average_precision(target, score)
        pooled_target.append(target)
        pooled_score.append(score)
        if "category" in held.columns:
            pooled_category.append(held["category"].to_numpy())
    if not per_fold:
        raise FileNotFoundError(f"{run}: не найдено ни одного файла фолда")

    result: dict[str, object] = {
        "name": run.name,
        "per_fold": per_fold,
        "mean": float(np.mean(list(per_fold.values()))),
        "folds": len(per_fold),
    }
    if pooled_category:
        macro, used = macro_over_categories(
            np.concatenate(pooled_target), np.concatenate(pooled_score),
            np.concatenate(pooled_category))
        result["macro"] = macro
        result["categories"] = used
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=None,
                        help="каталог с колонкой category для макро-метрики")
    args = parser.parse_args()

    truth = load_truth(args.data, args.items)
    arms = [score_arm(run, truth) for run in args.runs]
    folds = sorted({f for arm in arms for f in arm["per_fold"]})

    width = max(len(str(arm["name"])) for arm in arms) + 2
    header = "ветка".ljust(width) + "".join(f.rjust(11) for f in folds)
    header += "среднее".rjust(12)
    if any("macro" in arm for arm in arms):
        header += "макро".rjust(11)
    print(header)
    print("-" * len(header))
    for arm in arms:
        line = str(arm["name"]).ljust(width)
        line += "".join(f"{arm['per_fold'].get(f, float('nan')):11.6f}" for f in folds)
        line += f"{arm['mean']:12.6f}"
        if "macro" in arm:
            line += f"{arm['macro']:11.6f}"
        print(line)

    used = {arm.get("categories") for arm in arms if "categories" in arm}
    if used:
        print(f"\nмакро посчитано по {max(used)} категориям "
              f"(остальные пропущены: в них один класс)")

    print("\nориентиры (среднее по фолдам, та же метрика):")
    for name, value in BASELINES.items():
        print(f"  {value:.6f}  {name}")

    best = max(arms, key=lambda a: a["mean"])
    print(f"\nлучшая ветка: {best['name']}  {best['mean']:.6f}")
    for name, value in BASELINES.items():
        delta = best["mean"] - value
        verdict = "лучше" if delta > 0 else "хуже"
        print(f"  против {name}: {delta:+.6f} ({verdict})")

    if len(arms) > 1:
        print("\nпарные разности по фолдам относительно лучшей ветки.")
        print("Устойчивый знак на всех фолдах весомее величины среднего:")
        for arm in arms:
            if arm["name"] == best["name"]:
                continue
            deltas = [best["per_fold"][f] - arm["per_fold"][f]
                      for f in folds
                      if f in best["per_fold"] and f in arm["per_fold"]]
            wins = sum(1 for d in deltas if d > 0)
            marks = " ".join(f"{d:+.6f}" for d in deltas)
            print(f"  {best['name']} - {arm['name']}: {marks}"
                  f"   среднее {np.mean(deltas):+.6f}, в пользу лучшей {wins}/{len(deltas)}")


if __name__ == "__main__":
    main()

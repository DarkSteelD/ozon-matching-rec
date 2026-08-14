"""Шаблон эксперимента. Скопируй каталог в members/<свой-ник>/experiments/<имя>/.

Делает минимально валидную вещь: на каждый фолд пишет константное предсказание
в правильном формате трека. Это не модель, а рыба — заменяй `predict_fold`.

Раннер (`validation/ops/train.py`) вызывает так:

    python train.py --out-dir <dir> --data-dir <repo>/data/raw --repo <repo> \
                    --folds fold_01,fold_02,fold_03,fold_04

Обязанность скрипта — записать <out-dir>/<fold>.csv на каждый фолд. Колонки
идентификаторов и их порядок обязаны совпадать с каноническим файлом фолда:
`validation/targets/<fold>.csv`, а где его нет — `validation/folds/fold_assignments.csv`.
Отсюда же берётся порядок строк, поэтому его не надо угадывать.

ГЛАВНОЕ ПРАВИЛО: предсказывая фолд K, не используй его данные ни для обучения,
ни для подбора порогов. Обучайся на остальных фолдах.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

def prediction_columns(repo: Path) -> tuple[list[str], str]:
    """Колонки файла предсказаний — объявлены в ops_config.json трека.

    Угадывать их по файлу целей нельзя: там есть служебные поля (category, fold),
    а `validation.evaluate` требует ровно заданный набор и порядок. Тип тоже важен:
    на quality вердикты жёсткие (int), на остальных треках — вещественный скор.
    """
    config = json.loads((repo / "validation" / "ops" / "ops_config.json").read_text(encoding="utf-8"))
    columns = config.get("prediction_columns")
    if not columns:
        raise SystemExit("в validation/ops/ops_config.json нет prediction_columns")
    return list(columns), config.get("prediction_dtype", "float")


def fold_reference(repo: Path, fold: str, id_columns: list[str]) -> list[dict]:
    """Канонический список объектов фолда в требуемом порядке строк."""
    targets = repo / "validation" / "targets" / f"{fold}.csv"
    if targets.is_file():
        with targets.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    assignments = repo / "validation" / "folds" / "fold_assignments.csv"
    if assignments.is_file():
        with assignments.open(newline="", encoding="utf-8") as handle:
            return [r for r in csv.DictReader(handle) if r.get("fold") == fold]

    raise SystemExit(
        f"не нашёл канонический файл фолда {fold}. Собери его: `make validation-targets` "
        "(или `make folds` на треке quality)")


def predict_fold(fold: str, rows: list[dict], data_dir: Path, repo: Path) -> list[float]:
    """ЗАМЕНИ ЭТО. Здесь обучение на остальных фолдах и предсказание текущего.

    Возвращает по одному числу на строку `rows`, в том же порядке.
    """
    return [0.0] * len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--folds", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns, dtype = prediction_columns(args.repo)
    id_columns = columns[:-1]
    cast = int if dtype == "int" else float
    for fold in args.folds.split(","):
        rows = fold_reference(args.repo, fold, id_columns)
        predictions = predict_fold(fold, rows, args.data_dir, args.repo)
        if len(predictions) != len(rows):
            raise SystemExit(f"{fold}: предсказаний {len(predictions)}, а объектов {len(rows)}")
        destination = args.out_dir / f"{fold}.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            for row, value in zip(rows, predictions):
                writer.writerow([*(row[c] for c in id_columns), cast(value)])
        print(f"[experiment] {fold}: {len(rows)} строк -> {destination.name}", flush=True)


if __name__ == "__main__":
    main()

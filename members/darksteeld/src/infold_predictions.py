"""In-fold предсказания: модель, обученная на ВСЕХ парах, предсказывает их же.

Очереди доразметки сейчас строятся по OOF, где модель пары не видела. Такая
ошибка смешивает две причины: метка неверна ИЛИ модель не обобщила. In-fold
ошибка отсекает вторую — если модель не смогла подогнаться под метку, которую ей
прямо показывали и по которой её штрафовали, значит метка спорит с остальными
данными.

Приём работает при одном условии: у модели должно хватать ёмкости, чтобы
подогнаться под обучающую выборку. Модель, которая на обучающих парах немногим
лучше, чем на отложенных, ничего нового не скажет — её in-fold ошибки будут те
же самые OOF-ошибки. Разрыв между in-fold и OOF печатается здесь именно для
того, чтобы это было видно, а не предполагалось.

Предсказания кладутся в ``validation/predictions_v2_infold/``, отдельно от
OOF: смешивать их в одном каталоге нельзя, иначе in-sample скор попадёт на
лидерборд как честный.

    .venv/bin/python members/darksteeld/src/infold_predictions.py \\
        --container lgbm_cheap_audit --name lgbm_cheap_audit
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTAINERS = REPOSITORY_ROOT / "members" / "darksteeld" / "container"
FOLDS = [f"fold_{k:02d}" for k in range(1, 5)]


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return 0.0
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True,
                        help="каталог под members/darksteeld/container/")
    parser.add_argument("--name", required=True, help="имя, под которым сохранить")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-root", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2_infold"
                        / "darksteeld")
    parser.add_argument("--oof-root", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2" / "darksteeld")
    args = parser.parse_args()

    import polars as pl

    container = CONTAINERS / args.container
    if not (container / "run.py").is_file():
        raise SystemExit(f"нет {container / 'run.py'}")

    # контейнеру нужен файл пар без целей — собираем во временный
    matches = pl.read_parquet(args.data_dir / "matches.parquet")
    with tempfile.TemporaryDirectory() as tmp:
        pairs_path = Path(tmp) / "pairs.parquet"
        matches.select("id1", "id2").write_parquet(pairs_path)
        output_path = Path(tmp) / "infold.csv"
        command = [
            sys.executable, "-u", str(container / "run.py"),
            "--items_path", str(args.data_dir / "items_human.parquet"),
            "--matches_path", str(pairs_path),
            "--output_path", str(output_path),
        ]
        print(f"запускаю {args.container}/run.py на всех {matches.height:,} парах", flush=True)
        completed = subprocess.run(command, cwd=str(container))
        if completed.returncode != 0:
            raise SystemExit(f"run.py вышел с кодом {completed.returncode}")
        predicted = pl.read_csv(output_path)

    if not predicted["id1"].equals(matches["id1"]) or not predicted["id2"].equals(matches["id2"]):
        raise SystemExit("порядок пар на выходе контейнера отличается от matches.parquet")
    score_of_pair = dict(zip(zip(predicted["id1"].to_list(), predicted["id2"].to_list()),
                             predicted["predict"].to_list()))

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    infold_all, oof_all, target_all = [], [], []
    for fold in FOLDS:
        rows = list(csv.DictReader((args.targets_dir / f"{fold}.csv").open(encoding="utf-8")))
        with (out_dir / f"{fold}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for row in rows:
                key = (int(row["id1"]), int(row["id2"]))
                writer.writerow([row["id1"], row["id2"], f"{score_of_pair[key]:.8f}"])
                infold_all.append(score_of_pair[key])
                target_all.append(float(row["target"]))
        oof_path = args.oof_root / args.name / f"{fold}.csv"
        if oof_path.is_file():
            oof_all += [float(r["predict"])
                        for r in csv.DictReader(oof_path.open(encoding="utf-8"))]

    target = np.asarray(target_all)
    infold = np.asarray(infold_all)
    print(f"\nin-fold AP  {average_precision(target, infold):.6f}   -> {out_dir}")
    if len(oof_all) == len(infold):
        oof = np.asarray(oof_all)
        gap = average_precision(target, infold) - average_precision(target, oof)
        print(f"OOF AP      {average_precision(target, oof):.6f}")
        print(f"разрыв      {gap:+.6f}   — насколько модель подогналась под обучающие пары")
        if gap < 0.02:
            print("  разрыв мал: модель почти не запоминает обучающую выборку, поэтому её "
                  "in-fold ошибки будут почти теми же, что OOF")
    else:
        print("OOF-предсказаний с таким именем нет — разрыв не посчитан")


if __name__ == "__main__":
    main()

"""Точка входа контейнера — взвешенное среднее четырёх моделей.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Состав и веса подобраны на замороженных фолдах spec-v2 честной процедурой:
веса выбирались на трёх фолдах и проверялись на четвёртом, и на всех четырёх
разбиениях выбор оказался один и тот же — 0.4 / 0.2 / 0.2 / 0.2. Поэтому
честная оценка совпала с оптимистичной, переобучения на подборе весов нет.

    lgbm   0.4   бустинг по 21 ручному признаку пары        OOF 0.638077
    joint  0.2   совместная KNRM: имя и атрибуты в одной сети OOF 0.619154
    attrs  0.2   KNRM только по словарю атрибутов            OOF 0.570335
    name   0.2   KNRM только по названию, предобучен на LLM  OOF 0.565710
                                                    бленд    OOF 0.688738

**Почему подпроцессы, а не один объединённый скрипт.** У каждой модели свой
``run.py`` — ровно тот файл, которым получен её измеренный скор, а у двух из них
ещё и тот, что уже проверен на публичном лидерборде. Переписать их скоринг здесь
значило бы завести вторую копию, которая может незаметно разойтись с проверенной.
Поэтому все четыре запускаются без изменений на одних и тех же входах, а
усредняются уже их CSV. Единственный новый код — усреднение, и оно проверяется:
каждая компонента обязана вернуть те же пары в том же порядке, что на входе.

Каждая половина сохраняет свой откат для пары, чьих товаров нет в файле товаров
(бустинг — обучающий prior, три KNRM — 0.5), так что смесь откатов постоянна:
для ранжирующей метрики это безразлично, зато ни одна компонента здесь не
переопределяется задним числом.

Временные CSV пишутся рядом с ``output_path`` — единственным каталогом, про
который контракт обещает, что он доступен на запись, — и удаляются после.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMPONENTS = (("lgbm", 0.4), ("joint", 0.2), ("attrs", 0.2), ("name", 0.2))


def log(message: str) -> None:
    print(f"[blend4] {message}", flush=True)


def run_component(name: str, items_path: str, matches_path: str, out_path: Path) -> None:
    command = [
        sys.executable, "-u", str(HERE / name / "run.py"),
        "--items_path", items_path,
        "--matches_path", matches_path,
        "--output_path", str(out_path),
    ]
    started = time.time()
    completed = subprocess.run(command, cwd=str(HERE / name))
    if completed.returncode != 0:
        raise SystemExit(f"{name}/run.py вышел с кодом {completed.returncode}")
    log(f"{name}: готово за {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--keep-parts", action="store_true",
                        help="не удалять покомпонентные CSV (для отладки)")
    args = parser.parse_args()

    started = time.time()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    total_weight = sum(w for _, w in COMPONENTS)
    log(f"пар на входе {len(matches):,}; компоненты "
        f"{', '.join(f'{n} {w}' for n, w in COMPONENTS)} (сумма весов {total_weight})")

    parts: dict[str, Path] = {}
    blended = None
    try:
        for name, weight in COMPONENTS:
            part_path = output_path.with_name(f"{output_path.stem}.{name}.csv")
            parts[name] = part_path
            run_component(name, args.items_path, args.matches_path, part_path)

            part = pd.read_csv(part_path)
            if len(part) != len(matches):
                raise SystemExit(f"{name}: {len(part):,} строк против {len(matches):,} пар")
            if not (part["id1"].to_numpy() == matches["id1"].to_numpy()).all() or \
               not (part["id2"].to_numpy() == matches["id2"].to_numpy()).all():
                raise SystemExit(f"{name}: порядок пар отличается от входного")
            contribution = (weight / total_weight) * part["predict"].to_numpy()
            blended = contribution if blended is None else blended + contribution
            log(f"{name}: вес {weight}, диапазон "
                f"{part['predict'].min():.6f}..{part['predict'].max():.6f}")
    finally:
        if not args.keep_parts:
            for path in parts.values():
                path.unlink(missing_ok=True)

    pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": blended}).to_csv(
        output_path, index=False)
    log(f"записан {output_path} ({len(matches):,} строк), диапазон "
        f"{blended.min():.6f}..{blended.max():.6f} — всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

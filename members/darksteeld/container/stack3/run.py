"""Container entry point — стак трёх компонент: KNRM по именам, KNRM по
атрибутам и LightGBM по дешёвым парным признакам.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Выход — CSV с колонками id1, id2, predict, по строке на каждую входную пару, в
порядке входа.

**Почему подпроцессы, а не один слитый скрипт.** Каждый компонент везёт свой
``run.py`` — ровно тот файл, который дал его локальный скор, а для LightGBM и
KNRM по именам ещё и тот, что уже отработал на публичном борде. Переписать
любой из путей скоринга здесь значило бы завести вторую копию, которая тихо
разъедется с проверенной. Поэтому скрипт запускает все три без изменений на
одних и тех же входах и складывает три CSV. Новый код тут только складывающий,
и он проверяется: у всех трёх выходов должны быть те же пары в том же порядке,
что во входе.

**Как складывает** — задаёт ``combiner.json``:

* ``{"mode": "stack"}`` — логистическая регрессия по логитам компонент,
  коэффициенты из ``meta.json``. Мета-модель обучена на всех четырёх фолдах;
  её честная оценка (leave-one-fold-out) лежит в том же ``meta.json``.
* ``{"mode": "weights", "weights": [...]}`` — фиксированные веса в
  вероятностном пространстве, без обучения.

Порядок компонент в ``meta.json``/``weights`` — это порядок ``COMPONENTS``
ниже, и он сверяется с именами из ``meta.json`` на старте, чтобы перестановка
коэффициентов не прошла молча.

У каждой половины свой fallback для пары, чьих товаров нет в items (LightGBM —
обучающий prior, KNRM — 0.5), так что комбинация трёх fallback-ов есть
константа: для ранговой метрики безвредно, и ни один компонент здесь не
переспрашивается.

Временные CSV компонентов пишутся рядом с ``output_path`` — это единственный
каталог, про который контракт обещает, что он записываемый, — и потом удаляются.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# (каталог компонента, имя эксперимента в meta.json) — порядок задаёт порядок
# коэффициентов и весов.
COMPONENTS = (
    ("knrm_name", "knrm_llm_pretrain"),
    ("knrm_attrs", "knrm_attrs_llm"),
    ("lgbm", "lgbm_cheap_v1"),
)


def log(message: str) -> None:
    print(f"[stack] {message}", flush=True)


def run_component(name: str, items_path: str, matches_path: str, out_path: Path) -> None:
    command = [
        sys.executable, "-u", str(HERE / name / "run.py"),
        "--items_path", items_path,
        "--matches_path", matches_path,
        "--output_path", str(out_path),
    ]
    log(f"{name}: {' '.join(command[2:])}")
    started = time.time()
    completed = subprocess.run(command, cwd=str(HERE / name))
    if completed.returncode != 0:
        raise SystemExit(f"{name}/run.py вышел с кодом {completed.returncode}")
    log(f"{name}: готово за {time.time() - started:.0f}s")


def load_combiner() -> dict:
    combiner = json.loads((HERE / "combiner.json").read_text(encoding="utf-8"))
    mode = combiner.get("mode")
    if mode == "stack":
        meta = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
        expected = [experiment for _, experiment in COMPONENTS]
        if meta["components"] != expected:
            raise SystemExit(
                f"meta.json перечисляет компоненты {meta['components']}, "
                f"а контейнер собран под {expected} — коэффициенты бы встали не на свои места")
        if meta["transform"] != "logit":
            raise SystemExit(f"поддержан только transform=logit, в meta.json {meta['transform']}")
        if len(meta["coefficients"]) != len(COMPONENTS):
            raise SystemExit("число коэффициентов не совпадает с числом компонент")
        combiner["meta"] = meta
        log("комбинатор: логит-стак, коэффициенты " + ", ".join(
            f"{experiment}={value:+.4f}"
            for experiment, value in zip(expected, meta["coefficients"], strict=True))
            + f", bias={meta['intercept']:+.4f}")
    elif mode == "weights":
        weights = combiner["weights"]
        if len(weights) != len(COMPONENTS):
            raise SystemExit("число весов не совпадает с числом компонент")
        total = float(sum(weights))
        if total <= 0:
            raise SystemExit("сумма весов должна быть положительной")
        combiner["weights"] = [w / total for w in weights]
        log("комбинатор: фиксированные веса " + ", ".join(
            f"{experiment}={weight:.3f}"
            for (_, experiment), weight in zip(COMPONENTS, combiner["weights"], strict=True)))
    else:
        raise SystemExit(f"неизвестный режим комбинатора: {mode!r}")
    return combiner


def combine(combiner: dict, columns: list[np.ndarray]) -> np.ndarray:
    if combiner["mode"] == "weights":
        stacked = np.zeros_like(columns[0])
        for weight, column in zip(combiner["weights"], columns, strict=True):
            stacked += weight * column
        return stacked

    meta = combiner["meta"]
    epsilon = float(meta.get("epsilon", 1e-6))
    score = np.full_like(columns[0], float(meta["intercept"]))
    for coefficient, column in zip(meta["coefficients"], columns, strict=True):
        clipped = np.clip(column, epsilon, 1.0 - epsilon)
        score += coefficient * np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--keep-parts", action="store_true",
                        help="не удалять покомпонентные CSV (для отладки)")
    args = parser.parse_args()

    started = time.time()
    combiner = load_combiner()
    # Компоненты запускаются со своим cwd (каждый читает модель рядом с собой),
    # поэтому относительный путь к данным, переданный нам, у них бы не открылся.
    # Резолвим здесь, один раз, а не полагаемся на то, что проверяющая система
    # всегда передаёт абсолютные пути.
    items_path = str(Path(args.items_path).resolve())
    matches_path = str(Path(args.matches_path).resolve())
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    log(f"пар на входе {len(matches):,}")

    parts: dict[str, Path] = {}
    columns: list[np.ndarray] = []
    try:
        for name, experiment in COMPONENTS:
            part_path = output_path.with_name(f"{output_path.stem}.{name}.csv")
            parts[name] = part_path
            run_component(name, items_path, matches_path, part_path)

            part = pd.read_csv(part_path)
            if len(part) != len(matches):
                raise SystemExit(f"{name}: {len(part):,} строк против {len(matches):,} пар")
            if not (part["id1"].to_numpy() == matches["id1"].to_numpy()).all() or \
               not (part["id2"].to_numpy() == matches["id2"].to_numpy()).all():
                raise SystemExit(f"{name}: порядок пар отличается от входного")
            column = part["predict"].to_numpy(dtype=np.float64)
            columns.append(column)
            log(f"{name} ({experiment}): диапазон {column.min():.6f}..{column.max():.6f}, "
                f"среднее {column.mean():.6f}")
    finally:
        if not args.keep_parts:
            for path in parts.values():
                path.unlink(missing_ok=True)

    stacked = combine(combiner, columns)
    pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": stacked}).to_csv(
        output_path, index=False
    )
    log(f"записан {output_path} ({len(matches):,} строк), диапазон "
        f"{stacked.min():.6f}..{stacked.max():.6f} — всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

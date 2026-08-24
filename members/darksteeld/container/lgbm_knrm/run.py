"""Container entry point — стекинг на уровне признаков: две KNRM дают колонки,
LightGBM решает.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Выход — CSV с колонками id1, id2, predict, по строке на каждую входную пару, в
порядке входа.

Схема отличается от бленда и от стека предсказаний: выходы KNRM по названию и
KNRM по атрибутам не усредняются с третьей моделью, а подаются в LightGBM двумя
признаками рядом с 21 дешёвым. На фолдах spec-v2 это 0.704896 против 0.683971 у
логит-стека тех же трёх компонент.

**Что здесь подпроцессы и что нет.** Обе KNRM запускаются своими неизменёнными
``run.py`` — теми же файлами, что дали их локальные скоры. LightGBM, наоборот,
не может переиспользовать ``lgbm_cheap/run.py``: у него на два признака больше,
и матрица собирается здесь. Это единственный новый путь скоринга в контейнере,
и он симметричен обучению — те же ``pair_features.build_features``, тот же
порядок признаков из ``artifact.json``.

**Чем обучались признаки KNRM.** Out-of-fold: на обучении LightGBM видел скоры
от KNRM, не видевших ту пару, потому что на тесте он ровно это и получит.
In-sample признак измерен и стоит 0.0388 PR-AUC.

Пара, чьих товаров нет в items, получает обучающий prior — как и в
``lgbm_cheap``; KNRM для таких пар отдают 0.5, но до LightGBM это не доходит,
потому что строка всё равно заменяется prior-ом.

Временные CSV компонентов пишутся рядом с ``output_path`` — единственным
каталогом, про который контракт обещает, что он записываемый, — и удаляются.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.append(str(HERE / "vendor"))

from pair_features import build_features  # noqa: E402

COMPONENTS = (("knrm_name", "knrm_llm_pretrain"), ("knrm_attrs", "knrm_attrs_llm"))


def log(message: str) -> None:
    print(f"[stack-features] {message}", flush=True)


def preload_openmp() -> str:
    """libgomp для lightgbm: образ его не несёт, torch и sklearn — несут.

    Копия из ``lgbm_cheap/run.py``: загрузка с RTLD_GLOBAL кладёт символы в
    глобальное пространство, и dlopen внутри lightgbm разрешается против них.
    Ставить LD_LIBRARY_PATH из Python поздно — glibc читает её при старте.
    """
    import ctypes
    from glob import glob

    candidates = ["libgomp.so.1"]
    candidates += sorted(glob(str(HERE / "vendor" / "libgomp.so*")))
    for package in ("torch/lib", "scikit_learn.libs", "sklearn/.libs"):
        for root in ("/usr/local/lib/python3.12/dist-packages", *sys.path):
            candidates += sorted(glob(f"{root}/{package}/libgomp*.so*"))
    for candidate in candidates:
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return candidate
        except OSError:
            continue
    return "none found — lightgbm import will probably fail"


def run_component(name: str, items_path: str, matches_path: str, out_path: Path) -> None:
    command = [
        sys.executable, "-u", str(HERE / name / "run.py"),
        "--items_path", items_path,
        "--matches_path", matches_path,
        "--output_path", str(out_path),
    ]
    log(f"{name}: старт")
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
    log(f"openmp: {preload_openmp()}")
    import lightgbm as lgb

    artifact = json.loads((HERE / "artifact.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(HERE / "model.txt"))
    log(f"модель: {booster.num_trees()} деревьев, lightgbm {lgb.__version__} "
        f"(обучалась на {artifact['lightgbm_version']}), признаков "
        f"{len(artifact['feature_names'])}")
    for key, name in (("knrm_name_experiment", "knrm_name"),
                      ("knrm_attrs_experiment", "knrm_attrs")):
        shipped = json.loads((HERE / name / "artifact.json").read_text(
            encoding="utf-8")).get("experiment")
        if artifact.get(key) and shipped != artifact[key]:
            raise SystemExit(
                f"{name}: в контейнере лежит {shipped!r}, а LightGBM обучался на "
                f"OOF от {artifact[key]!r} — распределение признака разъедется")

    # Компоненты запускаются со своим cwd, поэтому относительный путь к данным у
    # них не открылся бы: резолвим здесь, один раз.
    items_path = str(Path(args.items_path).resolve())
    matches_path = str(Path(args.matches_path).resolve())
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    log(f"пар на входе {len(matches):,}")

    parts: dict[str, Path] = {}
    knrm_columns: list[np.ndarray] = []
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
            knrm_columns.append(column)
            log(f"{name} ({experiment}): признак готов, диапазон "
                f"{column.min():.6f}..{column.max():.6f}, среднее {column.mean():.6f}")
    finally:
        if not args.keep_parts:
            for path in parts.values():
                path.unlink(missing_ok=True)

    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    log(f"товаров {len(items):,} | сборка дешёвых признаков")
    cheap, known = build_features(
        items["id"].tolist(), items["name"].tolist(), items["attributes"].tolist(),
        items["category"].tolist(),
        matches["id1"].to_numpy(), matches["id2"].to_numpy(),
        artifact["category_codes"],
        log=lambda message: log(message.strip()),
    )
    features = np.column_stack([cheap, *knrm_columns])
    if features.shape[1] != len(artifact["feature_names"]):
        raise SystemExit(f"собрано {features.shape[1]} признаков против "
                         f"{len(artifact['feature_names'])} в артефакте")
    log(f"матрица признаков {features.shape} за {time.time() - started:.0f}s")

    predictions = np.full(len(matches), artifact["prior"], dtype=np.float64)
    if known.any():
        predictions[known] = booster.predict(features[known])
    if not known.all():
        log(f"{int((~known).sum()):,} пар получили prior (товаров нет в items)")

    pd.DataFrame({
        "id1": matches["id1"].to_numpy(),
        "id2": matches["id2"].to_numpy(),
        "predict": predictions,
    }).to_csv(output_path, index=False)
    log(f"записан {output_path} ({len(matches):,} строк), диапазон "
        f"{predictions.min():.6f}..{predictions.max():.6f} — всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

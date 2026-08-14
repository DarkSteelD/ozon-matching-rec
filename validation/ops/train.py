"""Прогон эксперимента участника: обучение -> предсказания по фолдам -> скоринг.

Один и тот же файл во всех трёх репо трека; поведение задаётся соседним
`ops_config.json`. Ничего не знает про конкретную модель — только про контракт
запуска, поэтому работает на любой машине, где есть данные и фолды.

## Контракт эксперимента

    members/<member>/experiments/<experiment>/
        experiment.json      манифест (см. ниже)
        <любой код>

`experiment.json`:

    {
      "description": "LightGBM на recency-окнах 7/14/30",
      "entry": "python train.py",     обязательное: команда запуска
      "requirements": ["lightgbm"],   опционально: ставится в кэшируемый venv
      "gpus": 1,                      опционально: сколько карт нужно (по умолчанию 0)
      "timeout_min": 120              опционально: по умолчанию 180
    }

Скрипт эксперимента вызывается один раз и получает:

    <entry> --out-dir <куда писать> --data-dir <repo>/data/raw \
            --repo <repo> --folds fold_01,fold_02,fold_03,fold_04

Он обязан записать `<out-dir>/fold_0K.csv` на каждый фолд в формате трека
(см. validation/README.md). Если передан `--submission-dir`, туда же можно
положить готовый артефакт для отправки на ODS (CSV предсказаний на реальном
тесте или zip-контейнер) — CV-предсказания по фолдам таким артефактом не
являются. Флаг опциональный: эксперимент вправе его игнорировать. Обучать одну модель на все фолды или по модели на
фолд — дело эксперимента; правило одно: предсказывая фолд K, не использовать
его данные.

## Запуск

    python validation/ops/train.py --member kikjeck --experiment catboost_v5
    python validation/ops/train.py --member kikjeck --experiment catboost_v5 --dry-run
    python validation/ops/train.py --list

`--dry-run` пишет предсказания во временный каталог и НЕ регистрирует результат:
запись в `validation/predictions/` подхватывается общей джобой, которая скорит,
коммитит и пушит — для проверки кода это лишний шум.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent
REPO = OPS_DIR.parents[1]
CFG = json.loads((OPS_DIR / "ops_config.json").read_text(encoding="utf-8"))
PRIMARY = CFG["primary"]
REVERSED = bool(CFG.get("reversed", False))

EXPERIMENTS_GLOB = "members/*/experiments/*/experiment.json"
DEFAULT_TIMEOUT_MIN = 180
VENV_CACHE = Path(os.environ.get("ECUP_VENV_CACHE", Path.home() / ".cache" / "ecup-envs"))


def log(message: str) -> None:
    print(f"[train] {message}", flush=True)


def fold_ids() -> list[str]:
    """Идентификаторы фолдов из спецификации трека."""
    for name in ("folds.json", "spec.json"):
        path = REPO / "validation" / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            folds = payload.get("folds") or []
            ids = [f.get("id") for f in folds if isinstance(f, dict) and f.get("id")]
            if ids:
                return ids
    return ["fold_01", "fold_02", "fold_03", "fold_04"]


def manifest_path(member: str, experiment: str) -> Path:
    return REPO / "members" / member / "experiments" / experiment / "experiment.json"


def discover() -> list[tuple[str, str, dict]]:
    out = []
    for path in sorted(REPO.glob(EXPERIMENTS_GLOB)):
        member = path.parents[2].name
        experiment = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            log(f"пропускаю {member}/{experiment}: битый experiment.json ({error})")
            continue
        out.append((member, experiment, payload))
    return out


def ensure_env(requirements: list[str], base_python: str) -> str:
    """Venv под набор зависимостей, кэшируется по их хешу."""
    if not requirements:
        return base_python
    key = hashlib.sha256("\n".join(sorted(requirements)).encode()).hexdigest()[:12]
    env_dir = VENV_CACHE / key
    python = env_dir / "bin" / "python"
    if python.is_file():
        log(f"окружение из кэша: {env_dir}")
        return str(python)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"собираю окружение {key}: {', '.join(requirements)}")
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "venv", "--python", base_python, str(env_dir)], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(python), *requirements], check=True)
    else:
        subprocess.run([base_python, "-m", "venv", str(env_dir)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q", *requirements], check=True)
    return str(python)


def run_experiment(member: str, experiment: str, manifest: dict, out_dir: Path,
                   base_python: str, gpus: str | None,
                   submission_dir: Path | None = None) -> int:
    entry = manifest.get("entry")
    if not entry:
        raise SystemExit(f"{member}/{experiment}: в experiment.json нет обязательного поля entry")
    workdir = manifest_path(member, experiment).parent
    python = ensure_env(list(manifest.get("requirements") or []), base_python)

    command = shlex.split(entry)
    # подменяем голый `python` на интерпретатор подготовленного окружения
    if command and command[0] in ("python", "python3"):
        command[0] = python
    command += ["--out-dir", str(out_dir), "--data-dir", str(REPO / "data" / "raw"),
                "--repo", str(REPO), "--folds", ",".join(fold_ids())]
    if submission_dir is not None:
        command += ["--submission-dir", str(submission_dir)]

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH", "")]))
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus
        log(f"карты: {gpus or 'нет'}")

    timeout = int(manifest.get("timeout_min") or DEFAULT_TIMEOUT_MIN) * 60
    log(f"запускаю: {' '.join(command)}")
    started = time.time()
    try:
        process = subprocess.run(command, cwd=workdir, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"ПРЕВЫШЕН ЛИМИТ {timeout // 60} мин — снято")
        return 124
    log(f"код возврата {process.returncode}, {time.time() - started:.0f} с")
    return process.returncode


def check_outputs(out_dir: Path) -> list[str]:
    missing = [f for f in fold_ids() if not (out_dir / f"{f}.csv").is_file()]
    return missing


def register(member: str, experiment: str, out_dir: Path, notes: str, base_python: str) -> int:
    command = [base_python, "-m", "validation.evaluate",
               "--member", member, "--experiment", experiment,
               "--predictions-dir", str(out_dir), "--notes", notes]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH", "")]))
    return subprocess.run(command, cwd=REPO, env=env).returncode


def read_score(member: str, experiment: str) -> float | None:
    path = REPO / "validation" / "results" / member / f"{experiment}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(PRIMARY)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member")
    parser.add_argument("--experiment")
    parser.add_argument("--list", action="store_true", help="показать найденные эксперименты")
    parser.add_argument("--dry-run", action="store_true",
                        help="во временный каталог, без регистрации результата")
    parser.add_argument("--out-dir", type=Path, help="переопределить каталог предсказаний")
    parser.add_argument("--gpus", help="значение CUDA_VISIBLE_DEVICES, напр. 0 или 0,1")
    parser.add_argument("--submission-dir", type=Path,
                        help="куда эксперимент кладёт готовый артефакт решения "
                             "(CSV или zip); без флага артефакт не собирается")
    parser.add_argument("--python", default=sys.executable, help="базовый интерпретатор")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.list:
        found = discover()
        if not found:
            print("экспериментов не найдено (ждём members/<ник>/experiments/<имя>/experiment.json)")
            return
        for member, experiment, manifest in found:
            scored = read_score(member, experiment)
            mark = f"{scored:.5f}" if scored is not None else "не считан"
            print(f"  {member}/{experiment:<28} {mark:>12}   {manifest.get('description', '')[:60]}")
        return

    if not args.member or not args.experiment:
        raise SystemExit("нужны --member и --experiment (или --list)")
    path = manifest_path(args.member, args.experiment)
    if not path.is_file():
        raise SystemExit(f"нет манифеста: {path.relative_to(REPO)}")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if args.out_dir:
        out_dir = args.out_dir
        temporary = None
    elif args.dry_run:
        temporary = tempfile.mkdtemp(prefix=f"ecup-{args.member}-{args.experiment}-")
        out_dir = Path(temporary)
    else:
        temporary = None
        out_dir = REPO / "validation" / "predictions" / args.member / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"предсказания -> {out_dir}")

    if args.submission_dir:
        args.submission_dir.mkdir(parents=True, exist_ok=True)
        log(f"артефакт решения -> {args.submission_dir}")
    code = run_experiment(args.member, args.experiment, manifest, out_dir, args.python,
                          args.gpus, args.submission_dir)
    if code != 0:
        raise SystemExit(code)

    missing = check_outputs(out_dir)
    if missing:
        raise SystemExit(f"эксперимент не записал фолды: {', '.join(missing)}")
    log("все фолды на месте")

    if args.submission_dir:
        produced = [p.name for p in sorted(args.submission_dir.iterdir()) if p.is_file()]
        log(f"артефакт решения: {', '.join(produced)}" if produced
            else "артефакт решения не записан — эксперимент не поддерживает --submission-dir")

    if args.dry_run:
        log("dry-run: результат не регистрирую")
        if temporary:
            log(f"предсказания остались в {temporary}")
        return

    notes = args.notes or manifest.get("description", "")
    if register(args.member, args.experiment, out_dir, notes, args.python) != 0:
        raise SystemExit("скоринг не прошёл")
    scored = read_score(args.member, args.experiment)
    log(f"{PRIMARY} = {scored:.6f} ({'меньше лучше' if REVERSED else 'больше лучше'})"
        if scored is not None else "результат записан")


if __name__ == "__main__":
    main()

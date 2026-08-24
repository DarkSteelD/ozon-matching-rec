"""Container entry point — 0.5 * KNRM(LLM-pretrained) + 0.5 * LightGBM, both on
labels corrected by hand.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Output: CSV with columns id1, id2, predict — one row per input pair, in input
order.

**Why subprocesses and not one merged script.** Each component ships its own
``run.py`` — the exact file that produced its local score and, for the LightGBM
half, the exact file already scored on the public board. Re-implementing either
scoring path here would create a second copy that can silently drift from the
validated one. Instead this script runs both unchanged, on the same inputs, and
averages the two CSVs. The only new code is the averaging, and it is checked:
both outputs must carry the same pairs in the same order as the input.

Each half keeps its own fallback for a pair whose items are missing from the
items file (LightGBM: the training prior; KNRM: 0.5), so the blend of two
fallbacks is a constant — harmless for a ranking metric, and it means neither
component is second-guessed here.

Temporary per-component CSVs are written next to ``output_path``, which is the
one directory the contract guarantees is writable, and removed afterwards.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
COMPONENTS = (("knrm", 0.5), ("lgbm", 0.5))


def log(message: str) -> None:
    print(f"[blend] {message}", flush=True)


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
    log(f"пар на входе {len(matches):,}")

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
            contribution = weight * part["predict"].to_numpy()
            blended = contribution if blended is None else blended + contribution
            log(f"{name}: вес {weight}, диапазон "
                f"{part['predict'].min():.6f}..{part['predict'].max():.6f}")
    finally:
        if not args.keep_parts:
            for path in parts.values():
                path.unlink(missing_ok=True)

    pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"], "predict": blended}).to_csv(
        output_path, index=False
    )
    log(f"записан {output_path} ({len(matches):,} строк), диапазон "
        f"{blended.min():.6f}..{blended.max():.6f} — всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

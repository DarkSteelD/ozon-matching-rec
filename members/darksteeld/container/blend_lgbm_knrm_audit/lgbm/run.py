"""Container entry point — matching solution, LightGBM over cheap pair features.

Contract (taken from the organisers' matching-baseline-lightweight.zip):

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

``items_path`` has id, name, attributes, category; ``matches_path`` has id1, id2
and no target. The output is a CSV with exactly the columns id1, id2, predict —
one row per input pair, in input order.

Everything model-side is read from the artifact next to this file: model.txt
(LightGBM text format) and artifact.json (pinned category codes, feature order,
fallback prior). Features come from src/pair_features.py, the same module the
cross-validation harness imports, so the container and the local leaderboard
cannot drift apart.

No network is used and nothing is downloaded. The TF-IDF vectorizer is fitted on
the items file supplied at run time — transductive over the test items, which is
legal here because the whole items file is given before any prediction is made,
and it is exactly how the local score was produced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
# lightgbm is NOT in odsai/ecup26-matching-baseline:1.0 (verified by listing the
# image's distributions), so the manylinux wheel is unpacked into vendor/.
# Appended, not prepended: the image has no lightgbm so it falls through to the
# vendored copy anyway, while on a developer machine an installed build still
# wins — which keeps this exact file runnable outside the container. Everything
# else needed — pandas, numpy, scikit-learn, pyarrow — is already in the image.
sys.path.append(str(HERE / "vendor"))

from pair_features import build_features  # noqa: E402


def log(message: str) -> None:
    print(f"[run] {message}", flush=True)


def preload_openmp() -> str:
    """Make libgomp resolvable before lightgbm dlopens its native library.

    The image has no system libgomp — lightgbm's manylinux wheel fails with
    "libgomp.so.1: cannot open shared object file" — but torch and scikit-learn
    each ship a copy. Loading one with RTLD_GLOBAL puts its symbols in the global
    namespace, so lightgbm's own dlopen resolves against it. Setting
    LD_LIBRARY_PATH from inside Python would be too late: glibc reads it once at
    process start.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True, help="test items parquet")
    parser.add_argument("--matches_path", type=str, required=True, help="test pairs parquet")
    parser.add_argument("--output_path", type=str, required=True, help="output csv")
    args = parser.parse_args()

    started = time.time()
    log(f"openmp: {preload_openmp()}")
    import lightgbm as lgb

    artifact = json.loads((HERE / "artifact.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(HERE / "model.txt"))
    log(f"model: {booster.num_trees()} trees, lightgbm {lgb.__version__} "
        f"(trained with {artifact['lightgbm_version']})")

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    log(f"items {len(items):,} | pairs {len(matches):,} | loaded in {time.time() - started:.0f}s")

    features, known = build_features(
        items["id"].tolist(),
        items["name"].tolist(),
        items["attributes"].tolist(),
        items["category"].tolist(),
        matches["id1"].to_numpy(),
        matches["id2"].to_numpy(),
        artifact["category_codes"],
        log=lambda message: log(message.strip()),
    )
    log(f"features {features.shape} in {time.time() - started:.0f}s")

    predictions = np.full(len(matches), artifact["prior"], dtype=np.float64)
    if known.any():
        predictions[known] = booster.predict(features[known])
    if not known.all():
        log(f"{int((~known).sum()):,} pairs scored with the prior (items not in the items file)")

    pd.DataFrame({
        "id1": matches["id1"].to_numpy(),
        "id2": matches["id2"].to_numpy(),
        "predict": predictions,
    }).to_csv(args.output_path, index=False)
    log(f"wrote {args.output_path} ({len(matches):,} rows) — total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

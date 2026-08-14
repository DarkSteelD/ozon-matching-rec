"""Train the shipped LightGBM model and pin everything the container needs.

The CV harness (``members/darksteeld/src/lgbm_cheap.py``) trains four models,
one per fold, and throws them away after writing out-of-fold predictions — that
is what a validation harness is for and it produces nothing submittable. This
script trains **one** model on all 365,654 hand-labeled pairs and persists it.

What gets pinned, and why each matters:

* ``model.txt`` — LightGBM's own text format, not a pickle. It survives a
  different LightGBM build inside the container, which a pickle would not.
* ``category_codes`` — ``category`` is the top feature by gain. The CV code
  derived its codes with ``sorted(set(...))`` over whatever data was in front of
  it; doing that again inside the container would renumber the categories
  against the test file and quietly turn the strongest feature into noise. The
  training-time map ships with the model and is never rebuilt.
* ``prior`` — the score given to a pair whose items are missing from the test
  items file, so every input pair gets a row in the output.
* feature names, LightGBM params, seed, TF-IDF settings — recorded so the
  artifact can be audited against the code that produced it.

    .venv/bin/python members/darksteeld/container/lgbm_cheap/build_artifact.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from pair_features import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURE_NAMES, TFIDF_KWARGS, build_category_codes, build_features,
)

SEED = 20260813
NUM_BOOST_ROUND = 400
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    import lightgbm as lgb
    import polars as pl

    started = time.time()
    matches = pl.read_parquet(args.data_dir / "matches.parquet")
    items = pl.read_parquet(
        args.data_dir / "items_human.parquet", columns=["id", "name", "attributes", "category"]
    )
    print(f"pairs {matches.height:,} | items {items.height:,}")

    # category of a pair = category of its first item, exactly as the folds define it
    category_of_id = dict(zip(items["id"].to_list(), items["category"].to_list()))
    pair_categories = [category_of_id[i] for i in matches["id1"].to_list()]
    category_codes = build_category_codes(pair_categories)
    print(f"categories pinned: {len(category_codes)}")

    features, known = build_features(
        items["id"].to_list(),
        items["name"].to_list(),
        items["attributes"].to_list(),
        items["category"].to_list(),
        matches["id1"].to_numpy(),
        matches["id2"].to_numpy(),
        category_codes,
        log=lambda message: print(message, flush=True),
    )
    if not known.all():
        raise AssertionError("items_human must cover every hand pair")
    labels = matches["target"].to_numpy().astype(np.float64)
    prior = float(labels.mean())

    print(f"training on all {len(labels):,} pairs, {NUM_BOOST_ROUND} rounds...", flush=True)
    dataset = lgb.Dataset(
        features, label=labels, feature_name=FEATURE_NAMES,
        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True,
    )
    booster = lgb.train(PARAMS, dataset, num_boost_round=NUM_BOOST_ROUND)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.txt"
    booster.save_model(str(model_path))

    artifact = {
        "experiment": "lgbm_cheap_v1",
        "trained_on": "data/raw/matches.parquet, all pairs",
        "pairs": int(len(labels)),
        "positives": int(labels.sum()),
        "prior": prior,
        "seed": SEED,
        "num_boost_round": NUM_BOOST_ROUND,
        "params": PARAMS,
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "tfidf_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in TFIDF_KWARGS.items()},
        "category_codes": category_codes,
        "lightgbm_version": lgb.__version__,
        "repo_commit": git_commit(REPOSITORY_ROOT),
        "local_cv": {
            "spec_v1_mean_prauc": 0.63786621,
            "spec_v2_mean_prauc": 0.63817140,
            "note": "out-of-fold on the frozen folds; the shipped model is trained on all pairs",
        },
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    gains = booster.feature_importance("gain")
    order = np.argsort(-gains)[:8]
    print(f"model.txt {model_path.stat().st_size / 1e6:.1f} MB, {booster.num_trees()} trees")
    print("top gain:", [(FEATURE_NAMES[k], round(float(gains[k]), 1)) for k in order])
    print(f"prior {prior:.6f} | built in {time.time() - started:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()

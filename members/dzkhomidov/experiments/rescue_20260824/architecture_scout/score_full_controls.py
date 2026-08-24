from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent
HARD = Path("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")


def category_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = np.empty(len(scores), np.float64)
    for category in np.unique(categories):
        index = np.flatnonzero(categories == category)
        order = np.argsort(scores[index], kind="stable")
        ranks = np.empty(len(index), np.float64)
        ranks[order] = (np.arange(len(index)) + 0.5) / len(index)
        result[index] = ranks
    return result


def macro_ap(target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> float:
    return float(
        np.mean(
            [
                average_precision_score(target[categories == category], scores[categories == category])
                for category in np.unique(categories)
            ]
        )
    )


def main() -> None:
    hard = pl.read_parquet(HARD)
    result = {"weight": 0.1, "random_seed_base": 20260824, "folds": {}}
    for fold in ("fold_01", "fold_02"):
        frame = hard.filter(pl.col("fold") == fold).select("id1", "id2", "target", "category")
        for name, directory in (("baseline", "full_baseline"), ("minilm", "full_minilm")):
            predictions = pl.read_csv(ROOT / "preds" / directory / f"{fold}.csv").rename({"predict": name})
            frame = frame.join(predictions, on=["id1", "id2"], how="left", validate="1:1")
        target = frame["target"].to_numpy()
        categories = frame["category"].to_numpy()
        baseline = frame["baseline"].to_numpy()
        minilm = frame["minilm"].to_numpy()
        baseline_rank = category_ranks(baseline, categories)
        minilm_rank = category_ranks(minilm, categories)
        shuffle_rng = np.random.default_rng(20260824 + int(fold[-2:]))
        random_rng = np.random.default_rng(30260824 + int(fold[-2:]))
        shuffled_minilm = np.empty(len(minilm_rank), np.float64)
        random_uniform = np.empty(len(minilm_rank), np.float64)
        for category in np.unique(categories):
            index = np.flatnonzero(categories == category)
            shuffled_minilm[index] = minilm_rank[shuffle_rng.permutation(index)]
            random_uniform[index] = random_rng.random(len(index))
        variants = {
            "baseline": baseline,
            "minilm_standalone": minilm,
            "pretrained10": 0.9 * baseline_rank + 0.1 * minilm_rank,
            "shuffled_pretrained10": 0.9 * baseline_rank + 0.1 * shuffled_minilm,
            "uniform_random10": 0.9 * baseline_rank + 0.1 * random_uniform,
        }
        base_metric = macro_ap(target, baseline, categories)
        result["folds"][fold] = {
            name: {
                "macro_category_ap": macro_ap(target, scores, categories),
                "delta_vs_baseline": macro_ap(target, scores, categories) - base_metric,
            }
            for name, scores in variants.items()
        }
    deltas = [result["folds"][fold]["pretrained10"]["delta_vs_baseline"] for fold in ("fold_01", "fold_02")]
    result["aggregate"] = {
        "pretrained10_mean_delta": float(np.mean(deltas)),
        "pretrained10_sample_std_delta": float(np.std(deltas, ddof=1)),
        "gate_gt_0.001_each_fold": bool(all(delta > 0.001 for delta in deltas)),
    }
    output = ROOT / "metrics_full_controls.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

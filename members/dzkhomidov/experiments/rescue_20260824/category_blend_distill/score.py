#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parent
OOF = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet")
EXPS = {
    "baseline": "cbd_v3cal_baseline_s20260814",
    "global10": "cbd_v3cal_global10_s20260814",
    "category10": "cbd_v3cal_category10_s20260814",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["folds12", "all"], required=True)
    args = parser.parse_args()
    truth = pd.read_parquet(OOF, columns=["fold", "id1", "id2", "target", "category"])
    folds = ["fold_01", "fold_02"] if args.stage == "folds12" else sorted(truth.fold.unique())
    rows = []
    for fold in folds:
        ref = truth.loc[truth.fold.eq(fold)].reset_index(drop=True)
        scores = {}
        for variant, exp in EXPS.items():
            p = pd.read_csv(ROOT / "preds" / exp / f"{fold}.csv")
            assert np.array_equal(p[["id1", "id2"]].to_numpy(), ref[["id1", "id2"]].to_numpy())
            scores[variant] = p.predict.to_numpy()
        for cat, idx in ref.groupby("category").indices.items():
            idx = np.asarray(idx)
            values = {v: average_precision_score(ref.target.to_numpy()[idx], p[idx]) for v, p in scores.items()}
            rows.append({"fold": fold, "category": cat, **values,
                         "category_delta_baseline": values["category10"] - values["baseline"],
                         "category_delta_global": values["category10"] - values["global10"]})
    detail = pd.DataFrame(rows)
    detail.to_csv(ROOT / f"category_metrics_{args.stage}.csv", index=False)
    aggregate = detail.groupby("fold")[[*EXPS]].mean()
    aggregate["category_delta_baseline"] = aggregate.category10 - aggregate.baseline
    aggregate["category_delta_global"] = aggregate.category10 - aggregate.global10
    gate = bool((aggregate.category_delta_baseline > .001).all())
    pooled = []
    for fold in folds:
        ref = truth.loc[truth.fold.eq(fold)].reset_index(drop=True)
        row = {"fold": fold}
        for variant, exp in EXPS.items():
            p = pd.read_csv(ROOT / "preds" / exp / f"{fold}.csv").predict
            row[variant] = average_precision_score(ref.target, p)
        pooled.append(row)
    by_category = detail.groupby("category")[["category_delta_baseline", "category_delta_global"]].mean()
    rng = np.random.default_rng(20260824)
    boot = rng.choice(len(by_category), size=(10000, len(by_category)), replace=True)
    ci = {}
    for col in by_category:
        draws = by_category[col].to_numpy()[boot].mean(axis=1)
        ci[col] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]
    result = {"stage": args.stage, "gate_threshold": .001, "gate_pass": gate,
              "fold_metrics": aggregate.reset_index().to_dict(orient="records"),
              "mean_fold_metrics": aggregate.mean().to_dict(),
              "pooled_fold_prauc": pooled, "paired_category_bootstrap95": ci}
    (ROOT / f"metrics_{args.stage}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(aggregate.to_string())
    print("gate_pass", gate)


if __name__ == "__main__":
    main()

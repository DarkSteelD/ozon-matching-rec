from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path("/home/dzkhomidov/matching-work/rescue_20260824/fgm")
HARD = Path("/home/dzkhomidov/matching-work/data/hand_pairs.parquet")
FOLDS = ("fold_01", "fold_02")
VARIANTS = ("bce", "bce2x", "fgm05", "random05")


def main():
    data = pd.read_parquet(
        HARD, columns=["fold", "id1", "id2", "target", "category"]
    )
    threshold_rows = []
    distribution_rows = []
    category_rows = []

    for fold in FOLDS:
        ref = data[data.fold.eq(fold)].reset_index(drop=True)
        y = ref.target.to_numpy().astype(int)
        fold_predictions = {}
        for variant in VARIANTS:
            pred_frame = pd.read_csv(ROOT / "preds" / variant / f"{fold}.csv")
            assert np.array_equal(
                pred_frame[["id1", "id2"]].to_numpy(),
                ref[["id1", "id2"]].to_numpy(),
            )
            pred = pred_frame.predict.to_numpy()
            fold_predictions[variant] = pred
            hard_pred = pred >= 0.5
            threshold_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "tp": int(np.sum(hard_pred & (y == 1))),
                    "fp": int(np.sum(hard_pred & (y == 0))),
                    "fn": int(np.sum(~hard_pred & (y == 1))),
                    "tn": int(np.sum(~hard_pred & (y == 0))),
                }
            )
            distribution_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "mean_pred": float(pred.mean()),
                    "mean_pred_y0": float(pred[y == 0].mean()),
                    "mean_pred_y1": float(pred[y == 1].mean()),
                    "std_pred": float(pred.std()),
                    "q01": float(np.quantile(pred, 0.01)),
                    "q50": float(np.quantile(pred, 0.50)),
                    "q99": float(np.quantile(pred, 0.99)),
                }
            )
            scored = ref.assign(pred=pred)
            for category, group in scored.groupby("category", sort=True):
                category_rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "category": category,
                        "rows": len(group),
                        "ap": average_precision_score(group.target, group.pred),
                    }
                )

        base = fold_predictions["bce"]
        for variant in VARIANTS[1:]:
            candidate = fold_predictions[variant]
            distribution_rows.append(
                {
                    "fold": fold,
                    "variant": f"{variant}_vs_bce",
                    "mean_pred": float((candidate - base).mean()),
                    "mean_pred_y0": float((candidate[y == 0] - base[y == 0]).mean()),
                    "mean_pred_y1": float((candidate[y == 1] - base[y == 1]).mean()),
                    "std_pred": float((candidate - base).std()),
                    "q01": float(np.quantile(candidate - base, 0.01)),
                    "q50": float(np.quantile(candidate - base, 0.50)),
                    "q99": float(np.quantile(candidate - base, 0.99)),
                }
            )

    thresholds = pd.DataFrame(threshold_rows)
    base_thresholds = thresholds[thresholds.variant.eq("bce")].set_index("fold")
    for column in ("tp", "fp", "fn", "tn"):
        thresholds[f"delta_{column}"] = [
            int(getattr(row, column) - base_thresholds.loc[row.fold, column])
            for row in thresholds.itertuples()
        ]
    thresholds.to_csv(ROOT / "stage1_score" / "threshold05_counts.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(
        ROOT / "stage1_score" / "prediction_diagnostics.csv", index=False
    )

    categories = pd.DataFrame(category_rows)
    pivot = categories.pivot(
        index=["fold", "category"], columns="variant", values="ap"
    ).reset_index()
    for variant in VARIANTS[1:]:
        pivot[f"delta_{variant}"] = pivot[variant] - pivot.bce
    pivot.to_csv(ROOT / "stage1_score" / "category_deltas.csv", index=False)
    summary = pivot.groupby("category", as_index=False).agg(
        rows=("bce", "size"),
        bce_mean=("bce", "mean"),
        fgm05_delta_mean=("delta_fgm05", "mean"),
        fgm05_delta_min=("delta_fgm05", "min"),
        fgm05_delta_max=("delta_fgm05", "max"),
        random05_delta_mean=("delta_random05", "mean"),
        bce2x_delta_mean=("delta_bce2x", "mean"),
    )
    summary["fgm05_same_positive"] = summary.fgm05_delta_min > 0
    summary["fgm05_same_negative"] = summary.fgm05_delta_max < 0
    summary.sort_values("fgm05_delta_mean", ascending=False).to_csv(
        ROOT / "stage1_score" / "category_summary.csv", index=False
    )
    print(thresholds.to_string(index=False))
    print(summary.sort_values("fgm05_delta_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

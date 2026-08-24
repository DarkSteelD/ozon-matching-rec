#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


MODES = ("prefix", "headtail", "middle")


def metrics(df, pred):
    cats = []
    for _, g in df.assign(_p=pred).groupby("category", sort=True):
        if g.target.nunique() == 2:
            cats.append(average_precision_score(g.target, g._p))
    return float(np.mean(cats)), float(average_precision_score(df.target, pred)), len(cats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--coverage", required=True)
    ap.add_argument("--preds", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02")
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data, columns=["fold", "id1", "id2", "target", "category"])
    cov = pd.read_parquet(args.coverage)
    assert np.array_equal(data[["fold", "id1", "id2", "target", "category"]].to_numpy(),
                          cov[["fold", "id1", "id2", "target", "category"]].to_numpy())
    df = data.join(cov[["len1", "len2", "total_len", "any_trunc", "both_trunc"]])
    slices = {
        "all": np.ones(len(df), bool), "no_trunc": ~df.any_trunc.to_numpy(),
        "any_trunc": df.any_trunc.to_numpy(), "both_trunc": df.both_trunc.to_numpy(),
        "len_382_512": df.total_len.between(382, 512).to_numpy(),
        "len_513_768": df.total_len.between(513, 768).to_numpy(),
        "len_gt768": df.total_len.gt(768).to_numpy(),
    }
    folds = args.folds.split(","); pred = {}
    for mode in MODES:
        pred[mode] = np.full(len(df), np.nan)
        for fold in folds:
            mask = df.fold.eq(fold).to_numpy(); ref = df.loc[mask]
            p = pd.read_csv(Path(args.preds) / mode / f"{fold}.csv")
            assert np.array_equal(p[["id1", "id2"]].to_numpy(), ref[["id1", "id2"]].to_numpy())
            pred[mode][mask] = p.predict
    slice_rows, cat_rows, primary = [], [], []
    for fold in folds:
        fm = df.fold.eq(fold).to_numpy()
        for mode in MODES:
            for name, sm in slices.items():
                m = fm & sm
                macro, pooled, cats = metrics(df.loc[m], pred[mode][m])
                slice_rows.append({"fold": fold, "mode": mode, "slice": name, "rows": int(m.sum()),
                                   "macro_category_ap": macro, "pooled_ap": pooled, "categories_scored": cats})
                if name == "all": primary.append({"fold": fold, "mode": mode, "macro_category_ap": macro,
                                                   "pooled_ap": pooled})
            for cat, idx in df.loc[fm].groupby("category", sort=True).indices.items():
                absolute = df.loc[fm].iloc[np.asarray(idx)].index.to_numpy()
                for name, sm in {"all": slices["all"], "any_trunc": slices["any_trunc"]}.items():
                    use = absolute[sm[absolute]]
                    if len(use) and df.target.iloc[use].nunique() == 2:
                        cat_rows.append({"fold": fold, "mode": mode, "category": cat, "slice": name,
                                         "rows": len(use), "ap": average_precision_score(df.target.iloc[use], pred[mode][use])})
    primary = pd.DataFrame(primary)
    base = primary.loc[primary["mode"].eq("prefix")].set_index("fold").macro_category_ap
    primary["delta_vs_prefix"] = [r.macro_category_ap - base[r.fold] for r in primary.itertuples()]
    gate_delta = primary.loc[primary["mode"].eq("headtail")].set_index("fold").delta_vs_prefix
    gate = bool((gate_delta > .001).all())
    cats = pd.DataFrame(cat_rows)
    cat_all = cats.loc[cats["slice"].eq("all")].pivot(index=["fold", "category"], columns="mode", values="ap").dropna()
    delta = (cat_all.headtail - cat_all.prefix).groupby("category").mean().to_numpy()
    rng = np.random.default_rng(20260824)
    boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    result = {"gate_threshold_each_fold": .001, "gate_pass": gate,
              "bootstrap95_category_delta": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
              "primary": primary.to_dict(orient="records")}
    primary.to_csv(out / "primary_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out / "slice_metrics.csv", index=False)
    cats.to_csv(out / "category_metrics.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(primary.to_string(index=False)); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

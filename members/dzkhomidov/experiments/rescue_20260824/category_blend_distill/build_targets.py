#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parent
BASE = Path("/home/dzkhomidov/matching-work/data/hand_pairs_pd_v3cal.parquet")
OOF = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet")
TEACHER = Path("/home/dzkhomidov/matching-work/rescue_20260824/category_blend/artifacts_v2/heldout_predictions.parquet")
MIX = 0.10


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibrate(score, hard, folds):
    out = np.empty(len(score))
    for fold in sorted(np.unique(folds)):
        m = folds == fold
        out[m] = IsotonicRegression(out_of_bounds="clip", y_min=.001, y_max=.999).fit_transform(score[m], hard[m])
    return out


def stats(x):
    return {"min": float(x.min()), "q01": float(np.quantile(x, .01)),
            "mean": float(x.mean()), "std": float(x.std()),
            "q99": float(np.quantile(x, .99)), "max": float(x.max())}


def main():
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    base = pl.read_parquet(BASE)
    oof = pl.read_parquet(OOF, columns=["fold", "id1", "id2", "target", "category"])
    teacher = pl.read_parquet(TEACHER, columns=["fold", "target", "category", "global_grid_nested", "category_grid_shrink75"])
    for col in ["fold", "id1", "id2", "category"]:
        assert base[col].to_list() == oof[col].to_list(), f"row mismatch: {col}"
    for col in ["fold", "target", "category"]:
        assert teacher[col].to_list() == oof[col].to_list(), f"teacher mismatch: {col}"
    hard = oof["target"].to_numpy().astype(float)
    target = base["target"].to_numpy()
    folds = base["fold"].to_numpy()
    p_old = (target - .3 * hard) / .7
    reconstruction = .3 * hard + .7 * p_old
    assert np.max(np.abs(reconstruction - target)) < 1e-12
    assert p_old.min() >= .001 - 1e-10 and p_old.max() <= .999 + 1e-10
    p_global = calibrate(teacher["global_grid_nested"].to_numpy(), hard, folds)
    p_category = calibrate(teacher["category_grid_shrink75"].to_numpy(), hard, folds)
    global_target = .3 * hard + (.7 - MIX) * p_old + MIX * p_global
    category_target = .3 * hard + (.7 - MIX) * p_old + MIX * p_category
    # Input teacher ranks are float32; allow only their sub-2e-8 bound error.
    assert global_target.min() >= .0007 - 2e-8 and global_target.max() <= .9993 + 2e-8
    assert category_target.min() >= .0007 - 2e-8 and category_target.max() <= .9993 + 2e-8
    global_path = out / "hand_pairs_pd_v3cal_global10.parquet"
    category_path = out / "hand_pairs_pd_v3cal_category10.parquet"
    base.with_columns(pl.Series("target", global_target)).write_parquet(global_path)
    base.with_columns(pl.Series("target", category_target)).write_parquet(category_path)
    diagnostics = {
        "mix": MIX, "rows": base.height,
        "input_sha256": {str(p): sha256(p) for p in [BASE, OOF, TEACHER]},
        "base_target": stats(target), "recovered_old_teacher": stats(p_old),
        "global_teacher_calibrated": stats(p_global), "category_teacher_calibrated": stats(p_category),
        "global_target": stats(global_target), "category_target": stats(category_target),
        "category_minus_base": stats(category_target - target),
        "category_minus_global": stats(category_target - global_target),
        "target_correlations": {
            "base_global": float(np.corrcoef(target, global_target)[0, 1]),
            "base_category": float(np.corrcoef(target, category_target)[0, 1]),
            "global_category": float(np.corrcoef(global_target, category_target)[0, 1]),
        },
        "output_sha256": {str(p): sha256(p) for p in [global_path, category_path]},
    }
    (ROOT / "target_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

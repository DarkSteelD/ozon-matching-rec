#!/usr/bin/env python3
"""Nested LOFO residual-member screen over every available OOF channel."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path('/home/dzkhomidov/matching-work/rescue_20260824/residual_matrix')
MATRIX = Path('/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/members/dzkhomidov/preds/all_model_predictions_oof.parquet')
BASE = Path('/home/dzkhomidov/matching-work/rescue_20260824/category_blend/artifacts_v2/heldout_predictions.parquet')
USED = {'ce_rubase_len384', 'ce_e5_len288', 'ce_mdeb_len224', 'zs_llm_blend', 'ce_priodistill'}
WEIGHTS = np.array([0.0, 0.025, 0.05, 0.10, 0.20, 0.35])
SEED = 20260824


def ap(y, p):
    return float(average_precision_score(y, p))


def rank_by_fold_cat(df, values):
    out = np.empty(len(df), np.float32)
    for idx in df.groupby(['fold', 'category'], sort=False).indices.values():
        idx = np.asarray(idx)
        out[idx] = pd.Series(values[idx]).rank(method='average', pct=True).to_numpy(np.float32)
    return out


def macro(df, pred):
    return float(np.mean([ap(g.target, pred[g.index]) for _, g in df.groupby('category', sort=True)]))


def run_candidate(df, base, candidate):
    folds = sorted(df.fold.unique())
    cats = sorted(df.category.unique())
    out = np.empty(len(df), np.float32)
    selected = []
    for held in folds:
        tr_fold = df.fold.ne(held).to_numpy()
        va = df.fold.eq(held).to_numpy()
        global_scores = []
        for w in WEIGHTS:
            p = (1-w)*base + w*candidate
            global_scores.append(np.mean([ap(df.target[(tr_fold)&(df.category.eq(c))], p[(tr_fold)&(df.category.eq(c))]) for c in cats]))
        gw = WEIGHTS[int(np.argmax(global_scores))]
        weights = np.empty(len(df.loc[va]), np.float32)
        va_df = df.loc[va]
        for c in cats:
            tr = tr_fold & df.category.eq(c).to_numpy()
            scores = [ap(df.target[tr], ((1-w)*base+w*candidate)[tr]) for w in WEIGHTS]
            best = np.flatnonzero(np.isclose(scores, np.max(scores), atol=1e-12, rtol=0))
            cw = WEIGHTS[best[np.argmin(np.abs(WEIGHTS[best]-gw))]]
            sw = 0.75*cw + 0.25*gw
            weights[va_df.category.eq(c).to_numpy()] = sw
            selected.append({'held_fold': held, 'category': c, 'global_weight': gw, 'category_weight': cw, 'shrunk_weight': sw})
        out[va] = (1-weights)*base[va] + weights*candidate[va]
    return out, selected


def main():
    df = pd.read_parquet(MATRIX)
    b = pd.read_parquet(BASE)
    assert len(df) == len(b)
    assert np.array_equal(df.fold.to_numpy(), b.fold.to_numpy())
    assert np.array_equal(df.target.to_numpy(), b.target.to_numpy())
    assert np.array_equal(df.category.to_numpy(), b.category.to_numpy())
    model_cols = [c for c in df if c.startswith(('ce_', 'zs_', 'knrm_', 'final_')) and df[c].notna().all()]
    candidates = [c for c in model_cols if c not in USED]
    base = rank_by_fold_cat(df, b.category_grid_shrink75.to_numpy())
    base_macro = macro(df, base)
    ranks = {c: rank_by_fold_cat(df, df[c].to_numpy()) for c in model_cols}
    corr = pd.DataFrame({c: ranks[c] for c in model_cols}).corr(method='pearson')
    corr.to_csv(ROOT/'rank_correlation.csv')
    rows, selections = [], []
    rng = np.random.default_rng(SEED)
    for c in candidates:
        pred, sel = run_candidate(df, base, ranks[c])
        fold_d = {}
        for f in sorted(df.fold.unique()):
            m = df.fold.eq(f).to_numpy()
            fold_d[f] = macro(df.loc[m].reset_index(drop=True), pred[m]) - macro(df.loc[m].reset_index(drop=True), base[m])
        rows.append({'candidate': c, 'control': 'actual', 'metric': macro(df, pred), 'delta': macro(df, pred)-base_macro,
                     **{f'delta_{f}': v for f,v in fold_d.items()}})
        selections.extend([{'candidate': c, **x} for x in sel])
        shuffled = ranks[c].copy()
        for idx in df.groupby(['fold', 'category'], sort=False).indices.values():
            idx = np.asarray(idx); shuffled[idx] = shuffled[rng.permutation(idx)]
        cpred, _ = run_candidate(df, base, shuffled)
        rows.append({'candidate': c, 'control': 'shuffled', 'metric': macro(df, cpred), 'delta': macro(df, cpred)-base_macro})
    result = pd.DataFrame(rows).sort_values(['control','delta'], ascending=[True,False])
    result.to_csv(ROOT/'metrics.csv', index=False)
    pd.DataFrame(selections).to_csv(ROOT/'selected_weights.csv', index=False)
    (ROOT/'summary.json').write_text(json.dumps({'base_macro':base_macro,'candidates':candidates,'rows':len(df)}, indent=2))
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path('/home/dzkhomidov/matching-work/rescue_20260824/graph_matrices')
SEED = 20260824

d = pl.read_parquet(ROOT / 'oof_predictions.parquet')
g = pl.read_parquet(ROOT / 'graph_features.parquet')
y = d['target'].to_numpy()
cat = d['category'].to_numpy()
fold = d['fold'].to_numpy()
base = d['baseline'].to_numpy()
multi = g['comp_edges'].to_numpy() > 1


def ranks(x):
    out = np.zeros(len(x))
    for c in np.unique(cat):
        m = cat == c
        out[m] = rankdata(x[m], method='average') / m.sum()
    return out


def macro(pred, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    vals = []
    for c in np.unique(cat):
        m = mask & (cat == c)
        if m.any() and y[m].sum():
            vals.append(average_precision_score(y[m], pred[m]))
    return float(np.mean(vals))


def matched_shuffle(x):
    rng = np.random.default_rng(SEED)
    z = x.copy()
    deg_bin = np.minimum(5, np.floor(np.log2(np.maximum(1, g['deg_max'].to_numpy())))).astype(np.int8)
    keys = pl.DataFrame({'fold': fold, 'cat': cat, 'bin': deg_bin}).with_row_index('row')
    for part in keys.partition_by(['fold', 'cat', 'bin'], maintain_order=True):
        idx = part['row'].to_numpy()
        if len(idx) > 1:
            z[idx] = x[rng.permutation(idx)]
    return z


rb = ranks(base)
result = {'coverage': float(multi.mean()), 'baseline_macro': macro(base), 'arms': []}
for feature in ['other_mean_max', 'other_max_max', 'component_other_mean']:
    raw = g[feature].to_numpy().astype(float)
    raw = np.where(np.isfinite(raw), raw, base)
    for control, values in [('graph', raw), ('shuffled', matched_shuffle(raw))]:
        rv = ranks(values)
        for weight in [0.01, 0.025, 0.05, 0.10]:
            pred = rb.copy()
            pred[multi] = (1 - weight) * rb[multi] + weight * rv[multi]
            fold_delta = {}
            for f in np.unique(fold):
                m = fold == f
                fold_delta[str(f)] = macro(pred, m) - macro(rb, m)
            result['arms'].append({
                'feature': feature,
                'control': control,
                'weight': weight,
                'macro_delta': macro(pred) - macro(rb),
                'fold_delta': fold_delta,
            })

(ROOT / 'diffusion_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))

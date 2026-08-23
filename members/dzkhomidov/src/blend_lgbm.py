"""LightGBM over cheap pair features + optional CE OOF score columns.

Reuses the exact feature construction of members/darksteeld/src/lgbm_cheap.py
but reads/writes only scratch dirs. Extra features: for every --ce EXP given,
reads ~/matching-work/preds/EXP/fold_0K.csv and appends the predict column
(canonical order) as a feature.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "ozon-hack/repos/ozon-matching-rec"
WORK = Path.home() / "matching-work"

spec = importlib.util.spec_from_file_location(
    "lgbm_cheap", REPO / "members/darksteeld/src/lgbm_cheap.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def read_pred_column(exp: str, fold: str, expected_pairs) -> np.ndarray:
    path = WORK / "preds" / exp / f"{fold}.csv"
    vals = []
    with path.open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            assert (int(row["id1"]), int(row["id2"])) == expected_pairs[i]
            vals.append(float(row["predict"]))
    return np.asarray(vals)


def main() -> None:
    import json

    import lightgbm as lgb
    import polars as pl

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ce", action="append", default=[], help="CE experiment name(s)")
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--feat-cache", default="/dev/shm/mwork/base_features.npy")
    args = ap.parse_args()

    folds = lc.load_folds()
    fold_ids = list(folds)
    all_pairs = [p for f in fold_ids for p in folds[f][0]]
    y = np.concatenate([folds[f][1] for f in fold_ids])
    fold_of_row = np.concatenate(
        [np.full(len(folds[f][0]), i) for i, f in enumerate(fold_ids)])
    categories_all = [c for f in fold_ids for c in folds[f][2]]
    category_codes = {n: c for c, n in enumerate(sorted(set(categories_all)))}

    feature_names = [
        "name_cosine", "name_token_jaccard", "name_exact", "prefix_ratio",
        "len1", "len2", "len_absdiff", "len_ratio",
        "num_jaccard", "num_equal", "num_left_only", "num_right_only", "num_any",
        "kv_jaccard", "key_jaccard", "n_shared_keys", "n_agree_keys", "n_conflict_keys",
        "kv_size_min", "kv_size_absdiff", "category",
    ]
    cache = Path(args.feat_cache)
    if cache.exists():
        X_base = np.load(cache)
        print("feature cache loaded", X_base.shape, flush=True)
        X = np.zeros((len(all_pairs), len(feature_names) + len(args.ce)))
        X[:, :len(feature_names)] = X_base
        col = len(feature_names)
        for exp in args.ce:
            offset = 0
            for f in fold_ids:
                pairs = folds[f][0]
                X[offset:offset + len(pairs), col] = read_pred_column(exp, f, pairs)
                offset += len(pairs)
            feature_names.append(f"ce_{exp}")
            col += 1
        train_and_write(args, X, y, fold_of_row, folds, fold_ids, feature_names)
        return

    items = pl.read_parquet(lc.RAW_DIR / "items_human.parquet",
                            columns=["id", "name", "attributes"])
    row_of_id = {int(it): r for r, it in enumerate(items["id"].to_list())}
    index1 = np.fromiter((row_of_id[a] for a, _ in all_pairs), dtype=np.int64)
    index2 = np.fromiter((row_of_id[b] for _, b in all_pairs), dtype=np.int64)

    print("name features...", flush=True)
    names = items["name"].to_list()
    normalized = [lc.normalize_name(n) for n in names]
    name_tokens = [frozenset(n.split()) for n in normalized]
    name_numbers = [lc.number_tokens(n) for n in normalized]

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                          sublinear_tf=True, dtype=np.float32)
    matrix = vec.fit_transform(names)
    cosine = np.zeros(len(all_pairs))
    for s in range(0, len(all_pairs), 200_000):
        e = min(s + 200_000, len(all_pairs))
        cosine[s:e] = np.asarray(
            matrix[index1[s:e]].multiply(matrix[index2[s:e]]).sum(axis=1)).ravel()
    del matrix, vec

    print("attr features...", flush=True)
    kv_interner, key_interner = {}, {}
    kv_key_of, kv_sets, key_sets = [], [], []
    for raw in items["attributes"].to_list():
        try:
            attrs = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            attrs = None
        kv_t, key_t = set(), set()
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                kl = str(k).lower()
                kid = key_interner.setdefault(kl, len(key_interner))
                key_t.add(kid)
                for el in v if isinstance(v, list) else [v]:
                    kv = f"{kl}={str(el).lower()}"
                    kvid = kv_interner.get(kv)
                    if kvid is None:
                        kvid = len(kv_interner)
                        kv_interner[kv] = kvid
                        kv_key_of.append(kid)
                    kv_t.add(kvid)
        kv_sets.append(frozenset(kv_t))
        key_sets.append(frozenset(key_t))
    kv_key_array = np.asarray(kv_key_of, dtype=np.int64)

    n = len(all_pairs)
    X = np.zeros((n, len(feature_names) + len(args.ce)))
    X[:, 0] = cosine
    print("pair loop...", flush=True)
    for p in range(n):
        i, j = index1[p], index2[p]
        n1, n2 = normalized[i], normalized[j]
        X[p, 1] = lc.jaccard(name_tokens[i], name_tokens[j])
        X[p, 2] = 1.0 if n1 == n2 and n1 else 0.0
        lim = min(len(n1), len(n2)); c = 0
        while c < lim and n1[c] == n2[c]:
            c += 1
        X[p, 3] = c / max(len(n1), len(n2), 1)
        X[p, 4] = len(n1); X[p, 5] = len(n2)
        X[p, 6] = abs(len(n1) - len(n2))
        X[p, 7] = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
        u1, u2 = name_numbers[i], name_numbers[j]
        X[p, 8] = lc.jaccard(u1, u2)
        X[p, 9] = 1.0 if u1 == u2 else 0.0
        X[p, 10] = len(u1 - u2); X[p, 11] = len(u2 - u1)
        X[p, 12] = 1.0 if (u1 or u2) else 0.0
        kv1, kv2 = kv_sets[i], kv_sets[j]
        k1, k2 = key_sets[i], key_sets[j]
        X[p, 13] = lc.jaccard(kv1, kv2)
        X[p, 14] = lc.jaccard(k1, k2)
        shared = k1 & k2
        agree = {int(kv_key_array[t]) for t in kv1 & kv2}
        X[p, 15] = len(shared); X[p, 16] = len(agree)
        X[p, 17] = len(shared) - len(agree)
        X[p, 18] = min(len(kv1), len(kv2))
        X[p, 19] = abs(len(kv1) - len(kv2))
        X[p, 20] = category_codes[categories_all[p]]

    np.save(cache, X[:, :len(feature_names)])
    print("feature cache saved", flush=True)
    col = len(feature_names)
    for exp in args.ce:
        offset = 0
        for f in fold_ids:
            pairs = folds[f][0]
            X[offset:offset + len(pairs), col] = read_pred_column(exp, f, pairs)
            offset += len(pairs)
        feature_names.append(f"ce_{exp}")
        col += 1
    train_and_write(args, X, y, fold_of_row, folds, fold_ids, feature_names)


def train_and_write(args, X, y, fold_of_row, folds, fold_ids, feature_names):
    import lightgbm as lgb
    import csv

    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, num_threads=16,
                  min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
                  bagging_freq=1, seed=20260813, deterministic=True,
                  force_row_wise=True, verbosity=-1)
    outdir = WORK / "preds" / args.exp
    outdir.mkdir(parents=True, exist_ok=True)
    for fi, f in enumerate(fold_ids):
        train = fold_of_row != fi
        ds = lgb.Dataset(X[train], label=y[train], feature_name=feature_names,
                         categorical_feature=["category"], free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=args.rounds)
        scores = booster.predict(X[~train])
        with (outdir / f"{f}.csv").open("w", newline="", encoding="utf-8") as sink:
            w = csv.writer(sink, lineterminator="\n")
            w.writerow(["id1", "id2", "predict"])
            for (a, b), s in zip(folds[f][0], scores.tolist(), strict=True):
                w.writerow([a, b, f"{s:.8f}"])
        print(f, "done", flush=True)
        if fi == 0:
            gains = booster.feature_importance("gain")
            order = np.argsort(-gains)[:8]
            print("top gain:", [(feature_names[k], round(float(gains[k]), 1)) for k in order])


if __name__ == "__main__":
    main()

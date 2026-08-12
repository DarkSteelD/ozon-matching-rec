"""First trained reference: LightGBM over cheap pair features, OOF by folds.

For every frozen fold K the model is trained on the labels of the other three
folds only and predicts fold K, so the out-of-fold predictions respect the
grouped split (no item of fold K appears in its training pairs by
construction of the folds). Fixed hyperparameters, fixed seed, no early
stopping on the evaluation fold: this is a reference point, not a tuned model.

Features per pair (items_human texts only, label-free):
  name char_wb 3-5gram TF-IDF cosine; normalized-name token Jaccard; exact
  normalized-name equality; common-prefix ratio; name lengths and diffs;
  number-token sets from names (Jaccard, equality, one-sided counts);
  attributes key=value Jaccard; key-set Jaccard; shared keys; agreeing keys;
  conflicting keys (same key, no common value); attribute set sizes; category.

Writes ``validation/predictions/darksteeld/lgbm_cheap_v1/fold_0K.csv``.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
TARGETS_DIR = REPOSITORY_ROOT / "validation" / "targets"
EXPERIMENT = "lgbm_cheap_v1"
PREDICTIONS_DIR = REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld" / EXPERIMENT
SEED = 20260813

NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).lower().replace("ё", "е")
    return NON_ALNUM.sub(" ", text).strip()


def number_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token.replace(",", ".").lstrip("0") or "0" for token in NUMBER.findall(text)
    )


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / (len(left) + len(right) - intersection)


def load_folds() -> dict[str, tuple[list[tuple[int, int]], np.ndarray, list[str]]]:
    folds: dict[str, tuple[list[tuple[int, int]], np.ndarray, list[str]]] = {}
    paths = sorted(TARGETS_DIR.glob("fold_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold targets in {TARGETS_DIR}; run make validation-targets")
    for path in paths:
        pairs: list[tuple[int, int]] = []
        targets: list[int] = []
        categories: list[str] = []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                pairs.append((int(row["id1"]), int(row["id2"])))
                targets.append(int(row["target"]))
                categories.append(row["category"])
        folds[path.stem] = (pairs, np.asarray(targets, dtype=np.float64), categories)
    return folds


def main() -> None:
    import lightgbm as lgb

    folds = load_folds()
    fold_ids = list(folds)
    all_pairs = [pair for fold_id in fold_ids for pair in folds[fold_id][0]]
    y = np.concatenate([folds[fold_id][1] for fold_id in fold_ids])
    fold_of_row = np.concatenate(
        [np.full(len(folds[fold_id][0]), index) for index, fold_id in enumerate(fold_ids)]
    )
    categories_all = [category for fold_id in fold_ids for category in folds[fold_id][2]]
    category_codes = {name: code for code, name in enumerate(sorted(set(categories_all)))}
    print(f"pairs={len(all_pairs)} folds={ {f: len(folds[f][0]) for f in fold_ids} }")

    items = pl.read_parquet(RAW_DIR / "items_human.parquet", columns=["id", "name", "attributes"])
    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_list())}
    index1 = np.fromiter((row_of_id[a] for a, _ in all_pairs), dtype=np.int64, count=len(all_pairs))
    index2 = np.fromiter((row_of_id[b] for _, b in all_pairs), dtype=np.int64, count=len(all_pairs))

    print("name features...")
    names = items["name"].to_list()
    normalized = [normalize_name(name) for name in names]
    name_tokens = [frozenset(name.split()) for name in normalized]
    name_numbers = [number_tokens(name) for name in normalized]

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, dtype=np.float32
    )
    matrix = vectorizer.fit_transform(names)
    cosine = np.zeros(len(all_pairs), dtype=np.float64)
    for start in range(0, len(all_pairs), 200_000):
        stop = min(start + 200_000, len(all_pairs))
        cosine[start:stop] = np.asarray(
            matrix[index1[start:stop]].multiply(matrix[index2[start:stop]]).sum(axis=1)
        ).ravel()
    del matrix, vectorizer

    print("attribute features...")
    kv_interner: dict[str, int] = {}
    key_interner: dict[str, int] = {}
    kv_key_of: list[int] = []
    kv_sets: list[frozenset[int]] = []
    key_sets: list[frozenset[int]] = []
    for raw in items["attributes"].to_list():
        try:
            attributes = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            attributes = None
        kv_tokens: set[int] = set()
        key_tokens: set[int] = set()
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                key_lower = str(key).lower()
                key_id = key_interner.setdefault(key_lower, len(key_interner))
                key_tokens.add(key_id)
                for element in value if isinstance(value, list) else [value]:
                    kv = f"{key_lower}={str(element).lower()}"
                    kv_id = kv_interner.get(kv)
                    if kv_id is None:
                        kv_id = len(kv_interner)
                        kv_interner[kv] = kv_id
                        kv_key_of.append(key_id)
                    kv_tokens.add(kv_id)
        kv_sets.append(frozenset(kv_tokens))
        key_sets.append(frozenset(key_tokens))
    kv_key_array = np.asarray(kv_key_of, dtype=np.int64)
    del kv_interner, key_interner, kv_key_of

    print("pair feature matrix...")
    n_pairs = len(all_pairs)
    feature_names = [
        "name_cosine", "name_token_jaccard", "name_exact", "prefix_ratio",
        "len1", "len2", "len_absdiff", "len_ratio",
        "num_jaccard", "num_equal", "num_left_only", "num_right_only", "num_any",
        "kv_jaccard", "key_jaccard", "n_shared_keys", "n_agree_keys", "n_conflict_keys",
        "kv_size_min", "kv_size_absdiff", "category",
    ]
    features = np.zeros((n_pairs, len(feature_names)), dtype=np.float64)
    features[:, 0] = cosine
    for position in range(n_pairs):
        i, j = index1[position], index2[position]
        n1, n2 = normalized[i], normalized[j]
        features[position, 1] = jaccard(name_tokens[i], name_tokens[j])
        features[position, 2] = 1.0 if n1 == n2 and n1 else 0.0
        limit = min(len(n1), len(n2))
        common = 0
        while common < limit and n1[common] == n2[common]:
            common += 1
        features[position, 3] = common / max(len(n1), len(n2), 1)
        features[position, 4] = len(n1)
        features[position, 5] = len(n2)
        features[position, 6] = abs(len(n1) - len(n2))
        features[position, 7] = min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
        numbers1, numbers2 = name_numbers[i], name_numbers[j]
        features[position, 8] = jaccard(numbers1, numbers2)
        features[position, 9] = 1.0 if numbers1 == numbers2 else 0.0
        features[position, 10] = len(numbers1 - numbers2)
        features[position, 11] = len(numbers2 - numbers1)
        features[position, 12] = 1.0 if (numbers1 or numbers2) else 0.0
        kv1, kv2 = kv_sets[i], kv_sets[j]
        keys1, keys2 = key_sets[i], key_sets[j]
        features[position, 13] = jaccard(kv1, kv2)
        features[position, 14] = jaccard(keys1, keys2)
        shared_keys = keys1 & keys2
        agree_keys = {int(kv_key_array[token]) for token in kv1 & kv2}
        features[position, 15] = len(shared_keys)
        features[position, 16] = len(agree_keys)
        features[position, 17] = len(shared_keys) - len(agree_keys)
        features[position, 18] = min(len(kv1), len(kv2))
        features[position, 19] = abs(len(kv1) - len(kv2))
        features[position, 20] = category_codes[categories_all[position]]

    params = {
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
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for fold_index, fold_id in enumerate(fold_ids):
        train_mask = fold_of_row != fold_index
        dataset = lgb.Dataset(
            features[train_mask],
            label=y[train_mask],
            feature_name=feature_names,
            categorical_feature=["category"],
            free_raw_data=True,
        )
        booster = lgb.train(params, dataset, num_boost_round=400)
        scores = booster.predict(features[~train_mask])
        pairs = folds[fold_id][0]
        with (PREDICTIONS_DIR / f"{fold_id}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for (id1, id2), score in zip(pairs, scores.tolist(), strict=True):
                writer.writerow([id1, id2, f"{score:.8f}"])
        print(f"{fold_id}: trained on {int(train_mask.sum())} pairs, predicted {len(pairs)}")
        if fold_index == 0:
            gains = booster.feature_importance("gain")
            order = np.argsort(-gains)[:10]
            print("  top gain:", [(feature_names[k], round(float(gains[k]), 1)) for k in order])


if __name__ == "__main__":
    main()

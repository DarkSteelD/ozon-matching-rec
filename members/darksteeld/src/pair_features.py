"""The 21 cheap pair features, shared by cross-validation and the container.

Both the CV harness (``lgbm_cheap.py``) and the submission container import
this module, so the features cannot drift between the number on the local
leaderboard and the number the platform scores. Nothing here reads labels,
folds or the repository layout: it takes item texts and a pair list and returns
a dense matrix, so it works identically on ``items_human.parquet`` locally and
on the test items file inside the container.

The category code map is an explicit input/output rather than something derived
on the fly. ``category`` is the top feature by gain, and deriving its codes from
whatever data happens to be at hand would silently renumber the categories
between training and inference — the strongest feature would become noise. The
map is built once at training time and pinned into the artifact.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

FEATURE_NAMES = [
    "name_cosine", "name_token_jaccard", "name_exact", "prefix_ratio",
    "len1", "len2", "len_absdiff", "len_ratio",
    "num_jaccard", "num_equal", "num_left_only", "num_right_only", "num_any",
    "kv_jaccard", "key_jaccard", "n_shared_keys", "n_agree_keys", "n_conflict_keys",
    "kv_size_min", "kv_size_absdiff", "category",
]
CATEGORICAL_FEATURES = ["category"]
UNKNOWN_CATEGORY = -1  # a category absent from the pinned map; LightGBM reads it as unseen

# char_wb 3-5gram TF-IDF, fitted on the item names actually being scored.
# Transductive but test-legal: the container receives the whole test items file
# before predicting, so the identical fit is available there.
TFIDF_KWARGS = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name)).lower().replace("ё", "е")
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


def build_category_codes(categories: list[str]) -> dict[str, int]:
    """Training-time map. Pinned into the artifact and never rebuilt at inference."""
    return {name: code for code, name in enumerate(sorted(set(categories)))}


def fit_name_vectorizer(names: list[str]):
    """Fit the char n-gram vectorizer once so two item universes can share it.

    Needed when training mixes the hand-labeled items with the LLM-labeled ones:
    fitting a separate vectorizer per universe would put the cosine feature on
    two different scales and the model would see one column meaning two things.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(dtype=np.float32, **TFIDF_KWARGS).fit(names)


def build_features(
    item_ids: list[int],
    names: list[str],
    attributes: list[str],
    categories: list[str],
    pair_id1: np.ndarray,
    pair_id2: np.ndarray,
    category_codes: dict[str, int],
    *,
    vectorizer=None,
    log=print,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for every pair, plus a mask of pairs whose items were found.

    Pairs referencing an unknown item get an all-zero row and ``False`` in the
    mask; the caller decides what to score them with. Nothing is dropped, so the
    output always lines up row-for-row with the input pair list.
    """
    row_of_id = {int(item): row for row, item in enumerate(item_ids)}
    pair_count = len(pair_id1)

    known = np.fromiter(
        ((int(a) in row_of_id and int(b) in row_of_id) for a, b in zip(pair_id1, pair_id2)),
        dtype=bool,
        count=pair_count,
    )
    missing = int((~known).sum())
    if missing:
        log(f"  {missing:,} of {pair_count:,} pairs reference an item outside the items file")

    index1 = np.fromiter((row_of_id.get(int(a), 0) for a in pair_id1), dtype=np.int64, count=pair_count)
    index2 = np.fromiter((row_of_id.get(int(b), 0) for b in pair_id2), dtype=np.int64, count=pair_count)

    log("  name features...")
    normalized = [normalize_name(name) for name in names]
    name_tokens = [frozenset(name.split()) for name in normalized]
    name_numbers = [number_tokens(name) for name in normalized]

    if vectorizer is None:
        matrix = fit_name_vectorizer(names).transform(names)  # rows are L2-normalized
    else:
        matrix = vectorizer.transform(names)  # shared fit across universes
    # Канонизируем порядок колонок. ``fit_transform`` возвращает матрицу с
    # НЕсортированными индексами, ``fit().transform()`` — с сортированными;
    # значения при этом совпадают, а порядок хранения различается почти везде.
    # Сумма ниже накапливается именно в этом порядке, поэтому в float32 косинус
    # расходится на ~2e-07 у пятой части пар. Само по себе это ничто, но
    # LightGBM раскладывает признак по гистограммным корзинам, значение у
    # границы уходит в соседнюю, и итоговый PR-AUC гуляет на 0.0004 — столько же,
    # сколько даёт смена сида. Сортировка убирает зависимость от того, каким
    # путём получена матрица.
    matrix.sort_indices()
    cosine = np.zeros(pair_count, dtype=np.float64)
    for start in range(0, pair_count, 200_000):
        stop = min(start + 200_000, pair_count)
        cosine[start:stop] = np.asarray(
            matrix[index1[start:stop]].multiply(matrix[index2[start:stop]]).sum(axis=1)
        ).ravel()
    del matrix  # the caller may still need the vectorizer for another universe

    log("  attribute features...")
    kv_interner: dict[str, int] = {}
    key_interner: dict[str, int] = {}
    kv_key_of: list[int] = []
    kv_sets: list[frozenset[int]] = []
    key_sets: list[frozenset[int]] = []
    for raw in attributes:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        kv_tokens: set[int] = set()
        key_tokens: set[int] = set()
        if isinstance(parsed, dict):
            for key, value in parsed.items():
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

    log("  pair feature matrix...")
    features = np.zeros((pair_count, len(FEATURE_NAMES)), dtype=np.float64)
    features[:, 0] = cosine
    for position in range(pair_count):
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
        # category of the pair = category of its first item; every hand pair is
        # intra-category, which build_folds asserts
        features[position, 20] = category_codes.get(categories[i], UNKNOWN_CATEGORY)

    features[~known] = 0.0
    return features, known

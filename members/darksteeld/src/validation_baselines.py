"""Cheap reference baselines for the shared validation harness.

Writes ``validation/predictions/darksteeld/<experiment>/fold_0K.csv`` for every
shared fold, in the canonical target pair order. No baseline reads the target
column of any fold it predicts: const_prior uses the global prior (PR-AUC of a
constant equals fold prevalence regardless of the constant), the other
baselines are label-free text similarities.

TF-IDF / Jaccard are fitted on ``items_human`` texts only. That is transductive
over the evaluation items but test-legal: at submit time the container receives
the full test items file before predicting, so the same fit is available there.

Baselines:
  const_prior            constant 0.2568 (global hand-label prior)
  name_exact             1.0 if normalized names equal else 0.0
  name_tfidf_cos         char_wb 3-5gram TF-IDF cosine between names
  attr_jaccard           Jaccard over attributes key=value token sets
  name_tfidf_attr_blend  0.5 * name_tfidf_cos + 0.5 * attr_jaccard
"""

from __future__ import annotations

import argparse
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
PREDICTIONS_DIR = REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld"

GLOBAL_PRIOR = 0.2567727961406138  # positives / pairs in matches.parquet
NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")
BASELINES = (
    "const_prior",
    "name_exact",
    "name_tfidf_cos",
    "attr_jaccard",
    "name_tfidf_attr_blend",
)


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).lower().replace("ё", "е")
    return NON_ALNUM.sub(" ", text).strip()


def load_fold_pairs(targets_dir: Path = TARGETS_DIR) -> dict[str, list[tuple[int, int]]]:
    folds: dict[str, list[tuple[int, int]]] = {}
    paths = sorted(targets_dir.glob("fold_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No fold targets in {targets_dir}; run: make validation-targets"
        )
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            folds[path.stem] = [(int(row["id1"]), int(row["id2"])) for row in reader]
    return folds


def write_fold(experiment: str, fold_id: str, pairs: list[tuple[int, int]], scores: np.ndarray,
               predictions_dir: Path = PREDICTIONS_DIR) -> None:
    destination = predictions_dir / experiment / f"{fold_id}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        for (id1, id2), score in zip(pairs, scores.tolist(), strict=True):
            writer.writerow([id1, id2, f"{score:.8f}"])


def attribute_token_set(raw: str, interner: dict[str, int]) -> frozenset[int]:
    try:
        attributes = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    if not isinstance(attributes, dict):
        return frozenset()
    tokens: set[int] = set()
    for key, value in attributes.items():
        values = value if isinstance(value, list) else [value]
        for element in values:
            token = f"{key}={element}".lower()
            token_id = interner.get(token)
            if token_id is None:
                token_id = len(interner)
                interner[token] = token_id
            tokens.add(token_id)
    return frozenset(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=[*BASELINES, "all"], default="all")
    parser.add_argument("--targets-dir", type=Path, default=TARGETS_DIR,
                        help="fold targets to predict; use validation/targets_v2 for spec v2")
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    args = parser.parse_args()
    wanted = set(BASELINES) if args.baseline == "all" else {args.baseline}

    folds = load_fold_pairs(args.targets_dir)
    all_pairs = [pair for pairs in folds.values() for pair in pairs]
    print(f"Folds: { {fold_id: len(pairs) for fold_id, pairs in folds.items()} }")

    if "const_prior" in wanted:
        for fold_id, pairs in folds.items():
            write_fold("const_prior", fold_id, pairs, np.full(len(pairs), GLOBAL_PRIOR),
                       args.predictions_dir)
        print("const_prior written")
        wanted.discard("const_prior")
    if not wanted:
        return

    needed_columns = ["id", "name"]
    if wanted & {"attr_jaccard", "name_tfidf_attr_blend"}:
        needed_columns.append("attributes")
    items = pl.read_parquet(RAW_DIR / "items_human.parquet", columns=needed_columns)
    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_list())}
    index1 = np.fromiter((row_of_id[a] for a, _ in all_pairs), dtype=np.int64, count=len(all_pairs))
    index2 = np.fromiter((row_of_id[b] for _, b in all_pairs), dtype=np.int64, count=len(all_pairs))

    def distribute(experiment: str, scores: np.ndarray) -> None:
        offset = 0
        for fold_id, pairs in folds.items():
            write_fold(experiment, fold_id, pairs, scores[offset : offset + len(pairs)],
                       args.predictions_dir)
            offset += len(pairs)
        print(f"{experiment} written")

    cosine: np.ndarray | None = None
    if wanted & {"name_exact", "name_tfidf_cos", "name_tfidf_attr_blend"}:
        names = items["name"].to_list()
        if "name_exact" in wanted:
            normalized = [normalize_name(name) for name in names]
            equal = np.fromiter(
                (normalized[i] == normalized[j] and normalized[i] != "" for i, j in zip(index1, index2)),
                dtype=np.float64,
                count=len(all_pairs),
            )
            distribute("name_exact", equal)
        if wanted & {"name_tfidf_cos", "name_tfidf_attr_blend"}:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(names)  # L2-normalized rows
            print(f"TF-IDF matrix: {matrix.shape}, nnz={matrix.nnz}")
            cosine = np.zeros(len(all_pairs), dtype=np.float64)
            chunk = 200_000
            for start in range(0, len(all_pairs), chunk):
                stop = min(start + chunk, len(all_pairs))
                left = matrix[index1[start:stop]]
                right = matrix[index2[start:stop]]
                cosine[start:stop] = np.asarray(left.multiply(right).sum(axis=1)).ravel()
            if "name_tfidf_cos" in wanted:
                distribute("name_tfidf_cos", cosine)

    if wanted & {"attr_jaccard", "name_tfidf_attr_blend"}:
        interner: dict[str, int] = {}
        token_sets = [attribute_token_set(raw, interner) for raw in items["attributes"].to_list()]
        print(f"Attribute tokens interned: {len(interner)}")
        jaccard = np.zeros(len(all_pairs), dtype=np.float64)
        for position, (i, j) in enumerate(zip(index1.tolist(), index2.tolist())):
            left, right = token_sets[i], token_sets[j]
            if left and right:
                intersection = len(left & right)
                if intersection:
                    jaccard[position] = intersection / (len(left) + len(right) - intersection)
        if "attr_jaccard" in wanted:
            distribute("attr_jaccard", jaccard)
        if "name_tfidf_attr_blend" in wanted:
            if cosine is None:
                raise RuntimeError("blend requires the TF-IDF cosine to be computed")
            distribute("name_tfidf_attr_blend", 0.5 * cosine + 0.5 * jaccard)


if __name__ == "__main__":
    main()

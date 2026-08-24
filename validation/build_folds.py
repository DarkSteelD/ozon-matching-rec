"""Build the frozen grouped folds for hand-labeled matching pairs.

Deterministic, RNG-free construction:

1. Read ``data/raw/matches.parquet`` (id1, id2, target in {0, 1}) and
   ``data/raw/items_human.parquet`` (id, category).
2. Union-find over all hand pairs; every connected component is one leakage
   group (an item must never appear in two folds).
3. Component key = the minimum item id inside the component. The key is a
   property of the data, independent of union-find implementation details.
4. Fold index = ``sha256(f"{seed}:{component_key}") mod k``: stable under any
   row order and stable if pairs are ever appended.
5. Write ``validation/targets/fold_0K.csv`` with columns
   ``id1,id2,target,category`` sorted by (id1, id2), and print the SHA256 of
   every file. The hashes are pinned in ``validation/folds.json`` and verified
   by ``validation/evaluate.py`` before any scoring.

Targets stay local (gitignored): only the builder and the pinned hashes are
committed, so labels are never pushed while every member can reproduce
byte-identical folds from ``data/raw``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from validation.spec import HASH_ASSIGNMENT, STRATIFIED_ASSIGNMENT, load_spec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_SPEC = REPOSITORY_ROOT / "validation" / "folds.json"
DEFAULT_TARGETS = REPOSITORY_ROOT / "validation" / "targets"


def connected_component_keys(id1: np.ndarray, id2: np.ndarray) -> dict[int, int]:
    """Map every item id to its component key (min item id in the component)."""
    unique_ids = np.unique(np.concatenate([id1, id2]))
    index = {int(item): position for position, item in enumerate(unique_ids)}
    parent = np.arange(len(unique_ids), dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    for left, right in zip(id1.tolist(), id2.tolist(), strict=True):
        root_left, root_right = find(index[left]), find(index[right])
        if root_left != root_right:
            parent[root_right] = root_left

    component_min: dict[int, int] = {}
    for item in unique_ids.tolist():
        root = find(index[item])
        current = component_min.get(root)
        if current is None or item < current:
            component_min[root] = item
    return {int(item): component_min[find(index[int(item)])] for item in unique_ids}


def fold_of_component(seed: str, component_key: int, k: int) -> int:
    digest = hashlib.sha256(f"{seed}:{component_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % k


def stratified_fold_of_component(
    seed: str,
    k: int,
    component_of_item: dict[int, int],
    id1: np.ndarray,
    target: np.ndarray,
    category: list[str],
) -> dict[int, int]:
    """Assign whole components to folds, balancing each category separately.

    Spec version 2. Every pair is intra-category (the builder rejects
    cross-category pairs), so a connected component lies entirely inside one
    category and stratification never has to split a leakage group.

    Stratification is two-dimensional, on category *and* label: PR-AUC depends
    mechanically on prevalence, so equalising pairs per fold while letting the
    positive rate drift would trade one source of fold-to-fold variance for
    another. Balancing pairs alone measurably does that — it drives the
    per-category count spread to <=1 pair but leaves the per-category positive
    rate no slack to settle.

    Deterministic and RNG-free, like the v1 hash: within a category components
    are processed largest-first, ties broken by ``sha256(seed:key)``, and each
    goes to the fold whose *worst* relative load after the assignment would be
    smallest, where load is measured against the per-fold target on both axes
    (pairs and positives), ties by lowest fold index. Largest-first is the
    standard greedy that keeps residual imbalance below one component.
    """
    components: dict[int, dict[str, object]] = {}
    for item, label, name in zip(id1.tolist(), target.tolist(), category, strict=True):
        key = component_of_item[item]
        entry = components.get(key)
        if entry is None:
            entry = components[key] = {"pairs": 0, "positives": 0, "category": name}
        entry["pairs"] = int(entry["pairs"]) + 1
        entry["positives"] = int(entry["positives"]) + int(label)
        if entry["category"] != name:
            raise AssertionError(
                f"component {key} spans categories {entry['category']!r} and {name!r}"
            )

    by_category: dict[str, list[int]] = {}
    for key, entry in components.items():
        by_category.setdefault(str(entry["category"]), []).append(key)

    fold_of_key: dict[int, int] = {}
    for name in sorted(by_category):
        keys = sorted(
            by_category[name],
            key=lambda key: (
                -int(components[key]["pairs"]),
                hashlib.sha256(f"{seed}:{key}".encode()).digest(),
            ),
        )
        pairs_target = max(sum(int(components[key]["pairs"]) for key in keys) / k, 1.0)
        positives_target = max(sum(int(components[key]["positives"]) for key in keys) / k, 1.0)
        pairs_load = [0] * k
        positives_load = [0] * k

        def cost(index: int, added_pairs: int, added_positives: int) -> tuple[float, int]:
            return (
                max(
                    (pairs_load[index] + added_pairs) / pairs_target,
                    (positives_load[index] + added_positives) / positives_target,
                ),
                index,
            )

        for key in keys:
            entry = components[key]
            added_pairs = int(entry["pairs"])
            added_positives = int(entry["positives"])
            fold = min(range(k), key=lambda index: cost(index, added_pairs, added_positives))
            fold_of_key[key] = fold
            pairs_load[fold] += added_pairs
            positives_load[fold] += added_positives
    return fold_of_key


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    raw_dir: Path,
    targets_dir: Path,
    seed: str,
    k: int,
    assignment: str = HASH_ASSIGNMENT,
) -> dict[str, dict[str, object]]:
    matches = pl.read_parquet(raw_dir / "matches.parquet")
    if matches.columns != ["id1", "id2", "target"]:
        raise ValueError(f"matches.parquet: unexpected columns {matches.columns}")
    if not matches["target"].is_in([0.0, 1.0]).all():
        raise ValueError("matches.parquet: target must be strictly 0/1")

    categories = pl.read_parquet(raw_dir / "items_human.parquet", columns=["id", "category"])
    matches = (
        matches.join(categories.rename({"id": "id1", "category": "cat1"}), on="id1", how="left")
        .join(categories.rename({"id": "id2", "category": "cat2"}), on="id2", how="left")
    )
    if matches["cat1"].null_count() or matches["cat2"].null_count():
        raise ValueError("items_human.parquet does not cover every paired item")
    if (matches["cat1"] != matches["cat2"]).any():
        raise ValueError("cross-category pair found; the category column is ambiguous")

    id1 = matches["id1"].to_numpy()
    id2 = matches["id2"].to_numpy()
    component_of_item = connected_component_keys(id1, id2)

    if assignment == STRATIFIED_ASSIGNMENT:
        fold_of_key = stratified_fold_of_component(
            seed,
            k,
            component_of_item,
            id1,
            matches["target"].to_numpy(),
            matches["cat1"].to_list(),
        )
    else:
        component_keys = sorted(set(component_of_item.values()))
        fold_of_key = {key: fold_of_component(seed, key, k) for key in component_keys}
    pair_fold = np.fromiter(
        (fold_of_key[component_of_item[int(item)]] for item in id1),
        dtype=np.int64,
        count=len(id1),
    )

    frame = matches.select(
        "id1",
        "id2",
        pl.col("target").cast(pl.Int8),
        pl.col("cat1").alias("category"),
    ).with_columns(pl.Series("fold", pair_fold)).sort(["id1", "id2"])

    targets_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}
    for fold_index in range(k):
        fold_id = f"fold_{fold_index + 1:02d}"
        fold_frame = frame.filter(pl.col("fold") == fold_index)
        destination = targets_dir / f"{fold_id}.csv"
        with destination.open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "target", "category"])
            for row in fold_frame.select(["id1", "id2", "target", "category"]).iter_rows():
                writer.writerow(row)
        items_in_fold = set(fold_frame["id1"].to_list()) | set(fold_frame["id2"].to_list())
        components_in_fold = {component_of_item[item] for item in items_in_fold}
        summary[fold_id] = {
            "pairs": fold_frame.height,
            "positives": int(fold_frame["target"].sum()),
            "positive_rate": round(float(fold_frame["target"].mean()), 6),
            "items": len(items_in_fold),
            "components": len(components_in_fold),
            "sha256": sha256_of_file(destination),
        }

    # no item may appear in two folds: components are assigned atomically, so
    # verifying the counts is enough to prove the partition
    if sum(int(fold["pairs"]) for fold in summary.values()) != frame.height:
        raise AssertionError("folds do not partition the pairs")
    if sum(int(fold["items"]) for fold in summary.values()) != len(component_of_item):
        raise AssertionError("an item leaked into more than one fold")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--write-pins",
        action="store_true",
        help="Write the freshly built SHA256 hashes into folds.json "
        "(bootstrap only; changing pinned folds is a team decision)",
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    print(f"spec v{spec.version}, assignment={spec.assignment}, targets -> {args.targets_dir}")
    summary = build(args.raw_dir, args.targets_dir, spec.seed, spec.k, spec.assignment)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    pinned = {fold.id: fold.sha256 for fold in spec.folds}
    built = {fold_id: str(stats["sha256"]) for fold_id, stats in summary.items()}
    if args.write_pins:
        payload = json.loads(args.spec.read_text(encoding="utf-8"))
        for item in payload["folds"]:
            item["sha256"] = built[item["id"]]
        args.spec.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Pinned SHA256 hashes written to {args.spec}")
        return

    if any(sha is None for sha in pinned.values()):
        print("folds.json has no pinned hashes yet; rerun with --write-pins to pin them.")
        return
    mismatched = [fold_id for fold_id, sha in built.items() if pinned[fold_id] != sha]
    if mismatched:
        print(f"ERROR: built folds do not match pinned hashes: {mismatched}", file=sys.stderr)
        raise SystemExit(1)
    print("Built folds match the pinned SHA256 hashes.")


if __name__ == "__main__":
    main()

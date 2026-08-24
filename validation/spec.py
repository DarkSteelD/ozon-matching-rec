"""Load and validate the shared grouped-fold specification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPEC_VERSION = 1

# Every fold specification the harness can read. A new version is a new
# validation contract and a team decision (see validation/README.md); old
# versions stay loadable so previously registered results remain verifiable.
#   1 — components hashed into folds, no stratification
#   2 — components assigned per category, balanced on pairs then positives
SUPPORTED_SPEC_VERSIONS = (1, 2)

HASH_ASSIGNMENT = "hash-mod"
STRATIFIED_ASSIGNMENT = "stratified-by-category"
ASSIGNMENTS = (HASH_ASSIGNMENT, STRATIFIED_ASSIGNMENT)


@dataclass(frozen=True)
class Fold:
    id: str
    sha256: str | None


@dataclass(frozen=True)
class FoldSpec:
    version: int
    task: str
    metric: str
    k: int
    seed: str
    folds: list[Fold]
    assignment: str = HASH_ASSIGNMENT


def load_spec(path: Path) -> FoldSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in SUPPORTED_SPEC_VERSIONS:
        raise ValueError(
            f"Unsupported fold specification version: {payload.get('version')}; "
            f"supported: {list(SUPPORTED_SPEC_VERSIONS)}"
        )
    assignment = str(payload.get("assignment", HASH_ASSIGNMENT))
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"Unknown fold assignment {assignment!r}; supported: {list(ASSIGNMENTS)}")

    folds = [Fold(id=item["id"], sha256=item.get("sha256")) for item in payload["folds"]]
    if not folds or len({fold.id for fold in folds}) != len(folds):
        raise ValueError("Fold ids must be non-empty and unique")
    if len(folds) != int(payload["k"]):
        raise ValueError(f"folds.json lists {len(folds)} folds but k={payload['k']}")
    expected_ids = [f"fold_{index:02d}" for index in range(1, len(folds) + 1)]
    if [fold.id for fold in folds] != expected_ids:
        raise ValueError(f"Fold ids must be exactly {expected_ids}")

    return FoldSpec(
        version=int(payload["version"]),
        task=str(payload["task"]),
        metric=str(payload["metric"]),
        k=int(payload["k"]),
        seed=str(payload["seed"]),
        folds=folds,
        assignment=assignment,
    )

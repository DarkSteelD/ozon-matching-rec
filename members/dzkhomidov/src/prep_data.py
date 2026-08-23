"""Build training-ready parquets: hand fold pairs with texts, and an LLM pair sample.

Outputs (in ~/matching-work/data):
  hand_pairs.parquet: fold, id1, id2, target, category, name1, attrs1, name2, attrs2
    -- row order within each fold == canonical target CSV order (prediction contract).
  llm_pairs_<N>.parquet: id1, id2, target, category1, name1, attrs1, name2, attrs2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import polars as pl

REPO = Path.home() / "ozon-hack/repos/ozon-matching-rec"
RAW = REPO / "data/raw"
TARGETS = REPO / "validation/targets"
OUT = Path.home() / "matching-work/data"


def compact_attrs(raw: str | None, limit: int = 800) -> str:
    if not raw:
        return ""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(d, dict):
        return ""
    parts = []
    for k in sorted(d, key=str.lower):
        v = d[k]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v[:6])
        parts.append(f"{k}:{v}")
    s = "; ".join(parts)
    return s[:limit]


def build_hand() -> None:
    rows = []
    for path in sorted(TARGETS.glob("fold_*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for order, r in enumerate(csv.DictReader(f)):
                rows.append((path.stem, order, int(r["id1"]), int(r["id2"]),
                             int(r["target"]), r["category"]))
    pairs = pl.DataFrame(
        rows, schema=["fold", "order", "id1", "id2", "target", "category"], orient="row"
    )
    items = pl.read_parquet(RAW / "items_human.parquet", columns=["id", "name", "attributes"])
    items = items.with_columns(
        pl.col("attributes").map_elements(compact_attrs, return_dtype=pl.String).alias("attrs")
    ).drop("attributes")
    out = (
        pairs
        .join(items.rename({"id": "id1", "name": "name1", "attrs": "attrs1"}), on="id1", how="left")
        .join(items.rename({"id": "id2", "name": "name2", "attrs": "attrs2"}), on="id2", how="left")
        .sort(["fold", "order"])
        .drop("order")
    )
    assert out["name1"].null_count() == 0 and out["name2"].null_count() == 0
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "hand_pairs.parquet")
    print("hand_pairs:", out.shape)


def build_llm(n: int, seed: int = 20260814) -> None:
    llm = pl.read_parquet(RAW / "matches_llm.parquet")
    if n < llm.height:
        llm = llm.sample(n=n, seed=seed)
    ids = pl.concat([llm["id1"], llm["id2"]]).unique().rename("id").to_frame()
    items = (
        pl.scan_parquet(RAW / "items.parquet")
        .select(["id", "name", "attributes", "category"])
        .join(ids.lazy(), on="id", how="semi")
        .collect(engine="streaming")
    )
    print("items joined:", items.shape)
    items = items.with_columns(
        pl.col("attributes").map_elements(compact_attrs, return_dtype=pl.String).alias("attrs")
    ).drop("attributes")
    out = (
        llm
        .join(items.rename({"id": "id1", "name": "name1", "attrs": "attrs1",
                            "category": "category1"}), on="id1", how="left")
        .join(items.select(["id", "name", "attrs"]).rename(
            {"id": "id2", "name": "name2", "attrs": "attrs2"}), on="id2", how="left")
    )
    null1, null2 = out["name1"].null_count(), out["name2"].null_count()
    print("null names:", null1, null2)
    out = out.filter(pl.col("name1").is_not_null() & pl.col("name2").is_not_null())
    out.write_parquet(OUT / f"llm_pairs_{n}.parquet")
    print(f"llm_pairs_{n}:", out.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", action="store_true")
    ap.add_argument("--llm", type=int, default=0, help="sample size; 0 = skip, -1 = all")
    args = ap.parse_args()
    if args.hand:
        build_hand()
    if args.llm:
        build_llm(args.llm if args.llm > 0 else 10**9)

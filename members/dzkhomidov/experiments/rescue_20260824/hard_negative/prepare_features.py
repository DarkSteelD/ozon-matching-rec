from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def words(text: str | None) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def jaccard(a: str | None, b: str | None) -> float:
    x, y = words(a), words(b)
    return len(x & y) / len(x | y) if x or y else 0.0


def attrs(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for field in (text or "").split(";"):
        if ":" not in field:
            continue
        key, value = field.split(":", 1)
        key = " ".join(TOKEN_RE.findall(key.lower()))
        value = " ".join(TOKEN_RE.findall(value.lower()))
        if key and value:
            out[key] = value
    return out


def conflict(a: str | None, b: str | None) -> bool:
    x, y = attrs(a), attrs(b)
    return any(x[k] != y[k] for k in x.keys() & y.keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pl.read_parquet(args.data)
    pred_parts = []
    for path in sorted(Path(args.preds).glob("fold_*.csv")):
        pred_parts.append(pl.read_csv(path).select("id1", "id2", pl.col("predict").alias("ce_oof")))
    pred = pl.concat(pred_parts)
    df = df.join(pred, on=["id1", "id2"], how="left", validate="1:1")
    assert df["ce_oof"].null_count() == 0

    ns = np.fromiter((jaccard(a, b) for a, b in zip(df["name1"], df["name2"])),
                     dtype=np.float32, count=df.height)
    ac = np.fromiter((conflict(a, b) for a, b in zip(df["attrs1"], df["attrs2"])),
                     dtype=np.bool_, count=df.height)
    df = df.with_columns(pl.Series("name_jaccard", ns), pl.Series("attr_conflict", ac))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    summary = {
        "rows": df.height,
        "negatives": int((df["target"] == 0).sum()),
        "ce_oof_mean_neg": float(df.filter(pl.col("target") == 0)["ce_oof"].mean()),
        "name_jaccard_mean_neg": float(df.filter(pl.col("target") == 0)["name_jaccard"].mean()),
        "attr_conflict_rate_neg": float(df.filter(pl.col("target") == 0)["attr_conflict"].mean()),
    }
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

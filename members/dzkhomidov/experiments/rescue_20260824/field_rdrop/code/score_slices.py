import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score


def score(rows, mask):
    part = rows.filter(mask)
    positives = int(part["target"].sum())
    if not part.height or positives in (0, part.height):
        return {"rows": part.height, "positives": positives, "ap": None}
    return {"rows": part.height, "positives": positives,
            "ap": float(average_precision_score(part["target"], part["predict"]))}


def field_keys(text):
    return {part.split(":", 1)[0].strip() for part in (text or "").split(";") if ":" in part}


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--pred-root", required=True)
parser.add_argument("--variants", required=True)
parser.add_argument("--folds", default="fold_01,fold_02")
parser.add_argument("--output", required=True)
args = parser.parse_args()
data = pl.read_parquet(args.data)
parts = []
for fold in args.folds.split(","):
    base = data.filter(pl.col("fold") == fold)
    for variant in args.variants.split(","):
        pred = pl.read_csv(Path(args.pred_root) / variant / f"{fold}.csv")
        joined = base.join(pred, on=["id1", "id2"], validate="1:1")
        keys1 = [field_keys(x) for x in joined["attrs1"].to_list()]
        keys2 = [field_keys(x) for x in joined["attrs2"].to_list()]
        joined = joined.with_columns(
            pl.lit(variant).alias("variant"), pl.lit(fold).alias("eval_fold"),
            (pl.col("attrs1").str.len_chars() + pl.col("name1").str.len_chars()).alias("len1"),
            (pl.col("attrs2").str.len_chars() + pl.col("name2").str.len_chars()).alias("len2"),
            pl.Series("field_count1", [len(x) for x in keys1]),
            pl.Series("field_count2", [len(x) for x in keys2]),
            pl.Series("field_key_jaccard", [len(a & b) / max(1, len(a | b)) for a, b in zip(keys1, keys2)]),
        )
        parts.append(joined)
rows = pl.concat(parts)
result = []
for variant in args.variants.split(","):
    arm = rows.filter(pl.col("variant") == variant)
    slices = {
        "all": pl.lit(True),
        "either_attrs_empty": (pl.col("attrs1").str.len_chars() == 0) | (pl.col("attrs2").str.len_chars() == 0),
        "one_side_attrs_empty": (pl.col("attrs1").str.len_chars() == 0) ^ (pl.col("attrs2").str.len_chars() == 0),
        "field_key_overlap_lt025": (pl.col("field_count1") > 0) & (pl.col("field_count2") > 0) & (pl.col("field_key_jaccard") < 0.25),
        "field_count_asymmetry_ge2": pl.max_horizontal("field_count1", "field_count2") >= 2 * pl.max_horizontal(pl.min_horizontal("field_count1", "field_count2"), pl.lit(1)),
        "text_length_asymmetry_ge2": pl.max_horizontal("len1", "len2") >= 2 * pl.max_horizontal(pl.min_horizontal("len1", "len2"), pl.lit(1)),
    }
    for name, mask in slices.items():
        result.append({"variant": variant, "slice": name, **score(arm, mask)})
    for category in sorted(arm["category"].unique().to_list()):
        result.append({"variant": variant, "slice": f"category:{category}",
                       **score(arm, pl.col("category") == category)})
Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))

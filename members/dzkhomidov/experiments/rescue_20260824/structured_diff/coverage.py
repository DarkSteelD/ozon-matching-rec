from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

from train import pair_token


STATE_RE = re.compile(r"([а-я_]+)=(совпал|различен|неизвестно)(?:_\d+)?")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--output", required=True)
    args = ap.parse_args(); df = pl.read_parquet(args.data, columns=["attrs1", "attrs2", "category"])
    overall = defaultdict(Counter); by_category = defaultdict(lambda: defaultdict(Counter))
    for left, right, category in zip(df["attrs1"], df["attrs2"], df["category"]):
        for key, value in STATE_RE.findall(pair_token(left, right)):
            overall[key][value] += 1; by_category[category][key][value] += 1
    def finish(counts):
        return {key: {**dict(values), "known_coverage": (values["совпал"] + values["различен"]) / sum(values.values())}
                for key, values in counts.items()}
    result = {"rows": df.height, "overall": finish(overall),
              "by_category": {category: finish(counts) for category, counts in by_category.items()}}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__": main()

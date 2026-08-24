import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pl.read_parquet(args.data)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = np.empty(df.height, dtype=np.int32)
    for s in range(0, df.height, 10000):
        d = df.slice(s, min(10000, df.height - s))
        a = [f"{n} | {c} | {x}" if x else f"{n} | {c}"
             for n, c, x in zip(d["name1"], d["category"], d["attrs1"])]
        b = [f"{n} | {c} | {x}" if x else f"{n} | {c}"
             for n, c, x in zip(d["name2"], d["category"], d["attrs2"])]
        lengths[s:s + len(d)] = tok(a, b, truncation=False, padding=False,
                                      return_length=True)["length"]
    thresholds = [224, 384, 448, 512]

    def stats(mask):
        x = lengths[mask]
        return {"rows": int(len(x)), "median": float(np.median(x)),
                "p90": float(np.quantile(x, .9)), "p95": float(np.quantile(x, .95)),
                "p99": float(np.quantile(x, .99)), "max": int(x.max()),
                **{f"gt_{t}": int((x > t).sum()) for t in thresholds},
                **{f"pct_gt_{t}": float((x > t).mean()) for t in thresholds}}

    folds = df["fold"].to_numpy()
    cats = df["category"].to_numpy()
    payload = {"overall": stats(np.ones(df.height, dtype=bool)),
               "folds": {str(v): stats(folds == v) for v in sorted(set(folds))},
               "categories": {str(v): stats(cats == v) for v in sorted(set(cats))}}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

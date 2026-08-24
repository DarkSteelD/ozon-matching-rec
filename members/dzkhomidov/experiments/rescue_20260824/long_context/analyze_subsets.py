import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pred-root", required=True)
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
    out = {}
    for variant in ["len384", "len448", "len512"]:
        out[variant] = {}
        for fold in ["fold_01", "fold_02"]:
            idx = np.flatnonzero(df["fold"].to_numpy() == fold)
            pred = pl.read_csv(Path(args.pred_root) / variant / f"{fold}.csv")["predict"].to_numpy()
            y = df["target"].to_numpy()[idx]
            groups = {"le384": lengths[idx] <= 384, "gt384": lengths[idx] > 384,
                      "385_448": (lengths[idx] > 384) & (lengths[idx] <= 448),
                      "gt448": lengths[idx] > 448}
            out[variant][fold] = {
                name: {"rows": int(mask.sum()),
                       "positives": int(y[mask].sum()),
                       "prauc": float(average_precision_score(y[mask], pred[mask]))}
                for name, mask in groups.items()
            }
    base = out["len384"]
    for variant in ["len448", "len512"]:
        for fold, groups in out[variant].items():
            for name, values in groups.items():
                values["delta_vs_len384"] = values["prauc"] - base[fold][name]["prauc"]
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

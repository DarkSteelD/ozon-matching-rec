#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from transformers import AutoTokenizer


MODES = ("prefix", "headtail", "middle")


def allocation(a, b, budget=381):
    if a + b <= budget:
        return a, b
    half = budget // 2
    if a > b:
        kb = min(b, half)
        return budget - kb, kb
    ka = min(a, half)
    return ka, budget - ka


def take(ids, k, mode):
    if len(ids) <= k or mode == "prefix":
        return ids[:k]
    if mode == "middle":
        start = (len(ids) - k) // 2
        return ids[start:start + k]
    head = (k + 1) // 2
    return ids[:head] + ids[-(k - head):]


def text(name, attrs, category):
    value = f"{name} | {category}" if category else str(name)
    return f"{value} | {attrs}" if attrs else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(args.data)
    tok = AutoTokenizer.from_pretrained(args.model)
    n, max_len = df.height, 384
    ids = {m: np.lib.format.open_memmap(out / f"ids_{m}.npy", "w+", np.int32, (n, max_len)) for m in MODES}
    tt = {m: np.lib.format.open_memmap(out / f"tt_{m}.npy", "w+", np.uint8, (n, max_len)) for m in MODES}
    lengths = np.zeros((n, 6), np.int32)
    names1, attrs1, names2, attrs2, cats = (df[c].to_list() for c in ["name1", "attrs1", "name2", "attrs2", "category"])
    cls, sep = tok.cls_token_id, tok.sep_token_id
    first_texts = second_texts = None
    for start in range(0, n, args.batch):
        end = min(start + args.batch, n)
        atext = [text(x, y, c) for x, y, c in zip(names1[start:end], attrs1[start:end], cats[start:end])]
        btext = [text(x, y, c) for x, y, c in zip(names2[start:end], attrs2[start:end], cats[start:end])]
        if first_texts is None:
            first_texts, second_texts = atext[:128], btext[:128]
        aid = tok(atext, add_special_tokens=False)["input_ids"]
        bid = tok(btext, add_special_tokens=False)["input_ids"]
        for local, (a, b) in enumerate(zip(aid, bid)):
            row = start + local
            ka, kb = allocation(len(a), len(b))
            lengths[row] = [len(a), len(b), ka, kb, int(len(a) > ka), int(len(b) > kb)]
            for mode in MODES:
                aa, bb = take(a, ka, mode), take(b, kb, mode)
                packed = [cls, *aa, sep, *bb, sep]
                ids[mode][row, :len(packed)] = packed
                tt[mode][row, len(aa) + 2:len(packed)] = 1
        if end % 50000 < args.batch:
            print(f"packed {end}/{n}", flush=True)
    # Runnable correctness check: custom prefix must equal native tokenizer.
    native = tok(first_texts, second_texts, truncation=True, max_length=max_len,
                 padding="max_length", return_tensors="np")
    assert np.array_equal(ids["prefix"][:128], native["input_ids"].astype(np.int32))
    assert np.array_equal(tt["prefix"][:128], native["token_type_ids"].astype(np.uint8))
    fit = np.flatnonzero((lengths[:, 4] + lengths[:, 5]) == 0)[:1000]
    assert np.array_equal(ids["prefix"][fit], ids["headtail"][fit])
    assert np.array_equal(ids["prefix"][fit], ids["middle"][fit])
    np.save(out / "lengths.npy", lengths)
    diag = pl.DataFrame({"fold": df["fold"], "id1": df["id1"], "id2": df["id2"],
                         "target": df["target"], "category": df["category"],
                         "len1": lengths[:, 0], "len2": lengths[:, 1],
                         "keep1": lengths[:, 2], "keep2": lengths[:, 3],
                         "trunc1": lengths[:, 4], "trunc2": lengths[:, 5]}).with_columns(
        (pl.col("len1") + pl.col("len2")).alias("total_len"),
        ((pl.col("trunc1") + pl.col("trunc2")) > 0).alias("any_trunc"),
        ((pl.col("trunc1") + pl.col("trunc2")) == 2).alias("both_trunc"),
    )
    diag.write_parquet(out / "coverage.parquet")
    summary = {"rows": n, "runtime_seconds": time.time() - started,
               "any_trunc": int(diag["any_trunc"].sum()), "both_trunc": int(diag["both_trunc"].sum()),
               "any_trunc_fraction": float(diag["any_trunc"].mean()),
               "both_trunc_fraction": float(diag["both_trunc"].mean()),
               "total_len_quantiles": {str(q): float(np.quantile(lengths[:, 0] + lengths[:, 1], q))
                                       for q in [.5, .75, .9, .95, .99]},
               "prefix_native_check_rows": 128}
    (out / "coverage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

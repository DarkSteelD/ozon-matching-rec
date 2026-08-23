"""Resumable, memory-lean parallel pre-tokenization into a memmapped ids cache.

Slices are aligned to parquet row groups; each worker reads only its row
groups, builds texts, tokenizes, writes ids into the shared memmap, then
touches a done-marker. token_type_ids are NOT stored (recomputed on GPU from
SEP positions). Rerun to resume after crashes.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ARGS = None


def make_texts(tbl, catcol):
    names1 = tbl.column("name1").to_pylist()
    attrs1 = tbl.column("attrs1").to_pylist()
    names2 = tbl.column("name2").to_pylist()
    attrs2 = tbl.column("attrs2").to_pylist()
    cats = tbl.column(catcol).to_pylist()

    def mk(names, attrs):
        out = []
        for n, a, c in zip(names, attrs, cats):
            t = n
            if ARGS.cat and c:
                t = f"{n} | {c}"
            if ARGS.attrs and a:
                t = f"{t} | {a}"
            out.append(t)
        return out
    return mk(names1, attrs1), mk(names2, attrs2)


def worker(job):
    idx, start, rgs = job
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tok = AutoTokenizer.from_pretrained(ARGS.model)
    pf = pq.ParquetFile(ARGS.file)
    catcol = "category1" if "category1" in pf.schema_arrow.names else "category"
    tbl = pf.read_row_groups(rgs, columns=["name1", "attrs1", "name2", "attrs2", catcol])
    t1, t2 = make_texts(tbl, catcol)
    del tbl
    count = len(t1)
    ids_mm = np.lib.format.open_memmap(ARGS.cache + ".ids.npy", mode="r+")
    for s in range(0, count, 20_000):
        e = min(s + 20_000, count)
        enc = tok(t1[s:e], t2[s:e], truncation=True, max_length=ARGS.max_len,
                  padding="max_length", return_tensors="np")
        ids_mm[start + s:start + e] = enc["input_ids"].astype(np.int32)
    ids_mm.flush()
    Path(ARGS.cache + f".done/slice_{idx}").touch()
    return idx, count


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--cat", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--groups-per-slice", type=int, default=2)
    ARGS = ap.parse_args()

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ARGS.file)
    nrg = pf.num_row_groups
    counts = [pf.metadata.row_group(i).num_rows for i in range(nrg)]
    n = sum(counts)
    print("rows:", n, "row groups:", nrg, flush=True)

    if not Path(ARGS.cache + ".y.npy").exists():
        y = pf.read(columns=["target"]).column("target").to_numpy()
        np.save(ARGS.cache + ".y.npy", y.astype(np.float32))
        del y
    if not Path(ARGS.cache + ".ids.npy").exists():
        np.lib.format.open_memmap(ARGS.cache + ".ids.npy", mode="w+",
                                  dtype=np.int32, shape=(n, ARGS.max_len))
    Path(ARGS.cache + ".done").mkdir(exist_ok=True)

    jobs, start = [], 0
    for i in range(0, nrg, ARGS.groups_per_slice):
        rgs = list(range(i, min(i + ARGS.groups_per_slice, nrg)))
        cnt = sum(counts[r] for r in rgs)
        idx = i // ARGS.groups_per_slice
        if not Path(ARGS.cache + f".done/slice_{idx}").exists():
            jobs.append((idx, start, rgs))
        start += cnt
    print(f"{len(jobs)} slices to do", flush=True)
    t0, done = time.time(), 0
    for attempt in range(6):
        if not jobs:
            break
        failed = []
        with ProcessPoolExecutor(max_workers=ARGS.workers) as ex:
            futs = {ex.submit(worker, j): j for j in jobs}
            for f in as_completed(futs):
                j = futs[f]
                try:
                    idx, count = f.result()
                    done += count
                    rate = done / (time.time() - t0)
                    print(f"slice {idx} done ({done} rows, {rate:.0f}/s)", flush=True)
                except Exception as err:  # noqa: BLE001
                    print(f"slice {j[0]} FAILED: {type(err).__name__}", flush=True)
                    failed.append(j)
        jobs = failed
        if failed:
            print(f"retry round {attempt+1}: {len(failed)} slices", flush=True)
            time.sleep(20)
    print("cache complete" if not jobs else f"INCOMPLETE: {len(jobs)} slices",
          flush=True)


if __name__ == "__main__":
    main()

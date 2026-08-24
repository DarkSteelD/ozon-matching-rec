"""Resumable row-group-aligned tokenization into a numpy memmap cache."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ARGS = None


def make_texts(table, category_column):
    names1 = table.column("name1").to_pylist()
    attrs1 = table.column("attrs1").to_pylist()
    names2 = table.column("name2").to_pylist()
    attrs2 = table.column("attrs2").to_pylist()
    categories = table.column(category_column).to_pylist()

    def build(names, attrs):
        result = []
        for name, attr, category in zip(names, attrs, categories, strict=True):
            text = f"{name} | {category}" if ARGS.cat and category else name
            result.append(f"{text} | {attr}" if ARGS.attrs and attr else text)
        return result

    return build(names1, attrs1), build(names2, attrs2)


def worker(job):
    index, start, row_groups = job
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(ARGS.model)
    parquet = pq.ParquetFile(ARGS.file)
    category_column = "category1" if "category1" in parquet.schema_arrow.names else "category"
    table = parquet.read_row_groups(
        row_groups, columns=["name1", "attrs1", "name2", "attrs2", category_column]
    )
    left, right = make_texts(table, category_column)
    cache = np.lib.format.open_memmap(ARGS.cache + ".ids.npy", mode="r+")
    for offset in range(0, len(left), 20_000):
        stop = min(offset + 20_000, len(left))
        encoded = tokenizer(
            left[offset:stop],
            right[offset:stop],
            truncation=True,
            max_length=ARGS.max_len,
            padding="max_length",
            return_tensors="np",
        )
        cache[start + offset : start + stop] = encoded["input_ids"].astype(np.int32)
    cache.flush()
    Path(ARGS.cache + f".done/slice_{index}").touch()
    return index, len(left)


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--attrs", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--groups-per-slice", type=int, default=2)
    ARGS = parser.parse_args()

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(ARGS.file)
    counts = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
    rows = sum(counts)
    print(f"rows: {rows} row groups: {parquet.num_row_groups}", flush=True)
    Path(ARGS.cache).parent.mkdir(parents=True, exist_ok=True)
    if not Path(ARGS.cache + ".y.npy").exists():
        targets = parquet.read(columns=["target"]).column("target").to_numpy()
        np.save(ARGS.cache + ".y.npy", targets.astype(np.float32))
    if not Path(ARGS.cache + ".ids.npy").exists():
        np.lib.format.open_memmap(
            ARGS.cache + ".ids.npy", mode="w+", dtype=np.int32, shape=(rows, ARGS.max_len)
        )
    Path(ARGS.cache + ".done").mkdir(exist_ok=True)

    jobs = []
    start = 0
    for first in range(0, parquet.num_row_groups, ARGS.groups_per_slice):
        groups = list(range(first, min(first + ARGS.groups_per_slice, parquet.num_row_groups)))
        count = sum(counts[group] for group in groups)
        index = first // ARGS.groups_per_slice
        if not Path(ARGS.cache + f".done/slice_{index}").exists():
            jobs.append((index, start, groups))
        start += count
    print(f"{len(jobs)} slices to do", flush=True)
    started = time.time()
    done = 0
    for attempt in range(6):
        if not jobs:
            break
        failed = []
        with ProcessPoolExecutor(max_workers=ARGS.workers) as executor:
            futures = {executor.submit(worker, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    index, count = future.result()
                    done += count
                    rate = done / (time.time() - started)
                    print(f"slice {index} done ({done} rows, {rate:.0f}/s)", flush=True)
                except Exception as error:
                    print(f"slice {job[0]} FAILED: {type(error).__name__}: {error}", flush=True)
                    failed.append(job)
        jobs = failed
        if jobs:
            print(f"retry round {attempt + 1}: {len(jobs)} slices", flush=True)
            time.sleep(20)
    if jobs:
        raise RuntimeError(f"incomplete token cache: {len(jobs)} slices")
    print("cache complete", flush=True)


if __name__ == "__main__":
    main()

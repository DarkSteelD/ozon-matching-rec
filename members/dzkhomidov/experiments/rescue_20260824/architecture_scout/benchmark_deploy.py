from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=224)
    args = parser.parse_args()

    frame = pl.read_parquet(args.data).head(args.rows)
    category = frame["category"].fill_null("").to_list()
    name1 = frame["name1"].fill_null("").to_list()
    name2 = frame["name2"].fill_null("").to_list()
    attrs1 = frame["attrs1"].fill_null("").to_list()
    attrs2 = frame["attrs2"].fill_null("").to_list()
    text1 = [f"{n} | {c} | {a}" for n, c, a in zip(name1, category, attrs1)]
    text2 = [f"{n} | {c} | {a}" for n, c, a in zip(name2, category, attrs2)]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True, local_files_only=True
    ).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats()

    token_seconds = 0.0
    infer_seconds = 0.0
    for left, right in ((text1, text2), (text2, text1)):
        for start in range(0, len(frame), args.batch_size):
            end = min(start + args.batch_size, len(frame))
            before = time.perf_counter()
            batch = tokenizer(
                left[start:end],
                right[start:end],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            token_seconds += time.perf_counter() - before
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            before = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**batch).logits
            torch.cuda.synchronize()
            infer_seconds += time.perf_counter() - before

    passes = 2 * len(frame)
    total_seconds = token_seconds + infer_seconds
    result = {
        "host": "avi-ix-devbox02",
        "gpu": torch.cuda.get_device_name(0),
        "rows": len(frame),
        "directional_passes": passes,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "tokenize_seconds": token_seconds,
        "inference_seconds": infer_seconds,
        "total_seconds": total_seconds,
        "pairs_per_second_two_direction": len(frame) / total_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": Path(args.model, "model.safetensors").stat().st_size,
        "estimated_seconds_365654_pairs_two_direction": 365_654 * total_seconds / len(frame),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

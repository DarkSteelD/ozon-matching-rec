"""Two-fold hand-only scale screen for ruRoBERTa-large.

Writes only OOF predictions and per-fold runtime metadata. Existing fold CSVs
are skipped, so the exact command can be rerun safely after interruption.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def tokenize(data: Path, tokenizer, max_len: int):
    df = pl.read_parquet(data)
    cats = df["category"].to_list()

    def texts(name_col: str, attrs_col: str) -> list[str]:
        return [
            f"{name} | {cat} | {attrs}" if attrs else f"{name} | {cat}"
            for name, attrs, cat in zip(
                df[name_col].to_list(), df[attrs_col].to_list(), cats, strict=True
            )
        ]

    left, right = texts("name1", "attrs1"), texts("name2", "attrs2")
    ids = np.empty((df.height, max_len), dtype=np.int32)
    started = time.time()
    for start in range(0, df.height, 20_000):
        stop = min(start + 20_000, df.height)
        encoded = tokenizer(
            left[start:stop],
            right[start:stop],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="np",
        )
        ids[start:stop] = encoded["input_ids"].astype(np.int32)
    print(f"tokenized {df.height} rows in {time.time() - started:.1f}s", flush=True)
    return (
        df["fold"].to_numpy(),
        df["id1"].to_numpy(),
        df["id2"].to_numpy(),
        df["target"].to_numpy().astype(np.float32),
        ids,
    )


def run_fold(args, tokenizer, fold: str, fold_col, y, ids):
    train_idx = np.flatnonzero(fold_col != fold)
    eval_idx = np.flatnonzero(fold_col == fold)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates_per_epoch = len(train_idx) // args.effective_batch
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_updates,
        pct_start=0.06,
        anneal_strategy="linear",
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    accum = args.effective_batch // args.micro_batch
    assert args.effective_batch % args.micro_batch == 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.time()
    update = 0
    for _ in range(args.epochs):
        permutation = rng.permutation(len(train_idx))[: updates_per_epoch * args.effective_batch]
        ordered = train_idx[permutation]
        for start in range(0, len(ordered), args.micro_batch):
            idx = ordered[start : start + args.micro_batch]
            batch_ids = torch.from_numpy(ids[idx].astype(np.int64)).cuda(non_blocking=True)
            targets = torch.from_numpy(y[idx]).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch_ids,
                    attention_mask=(batch_ids != tokenizer.pad_token_id).long(),
                ).logits.squeeze(-1)
                loss = loss_fn(logits, targets) / accum
            loss.backward()
            if ((start // args.micro_batch) + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % 100 == 0:
                    rate = update / (time.time() - started)
                    eta = (total_updates - update) / rate / 60
                    print(
                        f"{fold} update {update}/{total_updates} "
                        f"loss {loss.item() * accum:.4f} {rate:.2f} upd/s eta {eta:.1f}m",
                        flush=True,
                    )

    model.eval()
    predictions = np.empty(len(eval_idx), dtype=np.float32)
    eval_batch = args.micro_batch * 2
    with torch.no_grad():
        for start in range(0, len(eval_idx), eval_batch):
            idx = eval_idx[start : start + eval_batch]
            batch_ids = torch.from_numpy(ids[idx].astype(np.int64)).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch_ids,
                    attention_mask=(batch_ids != tokenizer.pad_token_id).long(),
                ).logits.squeeze(-1)
            predictions[start : start + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    runtime = time.time() - started
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return eval_idx, predictions, runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", default="fold_01,fold_02")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fold_col, id1, id2, y, ids = tokenize(args.data, tokenizer, args.max_len)
    for fold in args.folds.split(","):
        destination = args.output / f"{fold}.csv"
        if destination.is_file():
            print(f"{fold}: existing output skipped", flush=True)
            continue
        eval_idx, predictions, runtime = run_fold(args, tokenizer, fold, fold_col, y, ids)
        temporary = destination.with_suffix(".csv.partial")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            writer.writerows(
                (id1[i], id2[i], f"{score:.8f}")
                for i, score in zip(eval_idx, predictions, strict=True)
            )
        temporary.replace(destination)
        metadata = {
            "fold": fold,
            "rows_train": int(np.sum(fold_col != fold)),
            "rows_eval": int(len(eval_idx)),
            "runtime_seconds": runtime,
            "epochs": args.epochs,
            "micro_batch": args.micro_batch,
            "effective_batch": args.effective_batch,
            "lr": args.lr,
            "max_len": args.max_len,
            "seed": args.seed,
        }
        (args.output / f"{fold}.runtime.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{fold}: written in {runtime / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()

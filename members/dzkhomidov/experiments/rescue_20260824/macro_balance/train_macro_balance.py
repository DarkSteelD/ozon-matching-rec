from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def build_tokens(args, tok):
    df = pl.read_parquet(args.data)

    def texts(names, attrs, categories):
        return [f"{n} | {c} | {a}" if a else f"{n} | {c}"
                for n, a, c in zip(names, attrs, categories)]

    left = texts(df["name1"].to_list(), df["attrs1"].to_list(), df["category"].to_list())
    right = texts(df["name2"].to_list(), df["attrs2"].to_list(), df["category"].to_list())
    n = df.height

    def encode(a, b):
        ids = np.zeros((n, args.max_len), dtype=np.int32)
        token_types = np.zeros((n, args.max_len), dtype=np.uint8)
        for start in range(0, n, 20000):
            stop = min(start + 20000, n)
            enc = tok(a[start:stop], b[start:stop], truncation=True,
                      max_length=args.max_len, padding="max_length", return_tensors="np")
            ids[start:stop] = enc["input_ids"].astype(np.int32)
            if "token_type_ids" in enc:
                token_types[start:stop] = enc["token_type_ids"].astype(np.uint8)
        return ids, token_types

    started = time.time()
    ids, token_types = encode(left, right)
    ids_rev, token_types_rev = encode(right, left)
    print(f"tokenized {n} rows in {time.time() - started:.1f}s", flush=True)
    return df, ids, token_types, ids_rev, token_types_rev


def category_weights(categories, train_idx):
    train_categories = categories[train_idx]
    names, counts = np.unique(train_categories, return_counts=True)
    per_category = {name: len(train_idx) / (len(names) * count)
                    for name, count in zip(names, counts)}
    weights = np.array([per_category[name] for name in categories], dtype=np.float32)
    weights /= weights[train_idx].mean()
    return weights, per_category


def train_fold(args, fold, train_idx, eval_idx, arrays, targets, weights, pad_id):
    ids, token_types, ids_rev, token_types_rev = arrays
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_token_types = getattr(model.config, "type_vocab_size", 0) > 1
    steps = len(train_idx) // args.batch_size * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear")
    rng = np.random.default_rng(args.seed)
    started = time.time()
    model.train()
    step = 0
    for _ in range(args.epochs):
        permutation = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx) - args.batch_size + 1, args.batch_size):
            idx = train_idx[np.sort(permutation[start:start + args.batch_size])]
            reverse = rng.random(len(idx)) < 0.5
            batch_ids = np.where(reverse[:, None], ids_rev[idx], ids[idx])
            batch_tt = np.where(reverse[:, None], token_types_rev[idx], token_types[idx])
            batch_ids = torch.from_numpy(batch_ids.astype(np.int64)).cuda()
            batch_tt = torch.from_numpy(batch_tt.astype(np.int64)).cuda()
            batch_targets = torch.from_numpy(targets[idx]).cuda()
            batch_weights = torch.from_numpy(weights[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch_ids,
                               attention_mask=(batch_ids != pad_id).long(),
                               token_type_ids=batch_tt if use_token_types else None).logits.squeeze(-1)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, batch_targets, reduction="none")
                loss = (losses * batch_weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{fold} step {step}/{steps} loss={loss.item():.5f} "
                      f"rate={rate:.2f}/s eta={(steps-step)/rate/60:.1f}m", flush=True)

    model.eval()
    predictions = np.zeros(len(eval_idx), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(eval_idx), args.batch_size * 4):
            idx = eval_idx[start:start + args.batch_size * 4]
            total = 0.0
            for variant_ids, variant_tt in ((ids, token_types), (ids_rev, token_types_rev)):
                batch_ids = torch.from_numpy(variant_ids[idx].astype(np.int64)).cuda()
                batch_tt = torch.from_numpy(variant_tt[idx].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch_ids,
                                   attention_mask=(batch_ids != pad_id).long(),
                                   token_type_ids=batch_tt if use_token_types else None).logits.squeeze(-1)
                total = total + torch.sigmoid(logits.float()).cpu().numpy()
            predictions[start:start + len(idx)] = total / 2
    runtime = time.time() - started
    del model
    torch.cuda.empty_cache()
    return predictions, runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "category_balanced"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", default="fold_01,fold_02")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    df, ids, tt, ids_rev, tt_rev = build_tokens(args, tokenizer)
    folds = df["fold"].to_numpy()
    categories = df["category"].to_numpy()
    targets = df["target"].to_numpy().astype(np.float32)
    for fold in args.folds.split(","):
        eval_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        if args.variant == "category_balanced":
            weights, per_category = category_weights(categories, train_idx)
        else:
            weights = np.ones(len(df), dtype=np.float32)
            per_category = {name: 1.0 for name in np.unique(categories)}
        print(f"{args.variant} {fold}: train={len(train_idx)} eval={len(eval_idx)} "
              f"weight_range={weights[train_idx].min():.4f}..{weights[train_idx].max():.4f}", flush=True)
        predictions, runtime = train_fold(
            args, fold, train_idx, eval_idx, (ids, tt, ids_rev, tt_rev),
            targets, weights, tokenizer.pad_token_id)
        pl.DataFrame({"id1": df["id1"][eval_idx], "id2": df["id2"][eval_idx],
                      "predict": predictions}).write_csv(output / f"{fold}.csv")
        (output / f"{fold}.meta.json").write_text(json.dumps({
            "variant": args.variant, "fold": fold, "seed": args.seed,
            "runtime_seconds": runtime, "category_weights": per_category,
            "train_rows": len(train_idx), "eval_rows": len(eval_idx)
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"{fold} written runtime={runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()

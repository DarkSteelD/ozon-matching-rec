"""Minimal rank-loss arm over matching-work/scripts/train_hand_fast.py."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def build_tokens(args, tok):
    df = pl.read_parquet(args.data)

    def mk(names, attrs, cats):
        out = []
        for name, attr, cat in zip(names, attrs, cats):
            text = f"{name} | {cat}" if args.cat and cat else name
            out.append(f"{text} | {attr}" if args.attrs and attr else text)
        return out

    left = mk(df["name1"].to_list(), df["attrs1"].to_list(), df["category"].to_list())
    right = mk(df["name2"].to_list(), df["attrs2"].to_list(), df["category"].to_list())
    n = df.height
    ids = np.zeros((n, args.max_len), dtype=np.int32)
    token_types = np.zeros((n, args.max_len), dtype=np.uint8)
    started = time.time()
    for start in range(0, n, 20_000):
        end = min(start + 20_000, n)
        encoded = tok(left[start:end], right[start:end], truncation=True,
                      max_length=args.max_len, padding="max_length", return_tensors="np")
        ids[start:end] = encoded["input_ids"].astype(np.int32)
        if "token_type_ids" in encoded:
            token_types[start:end] = encoded["token_type_ids"].astype(np.uint8)
    category_names = sorted(df["category"].unique().to_list())
    category_to_id = {name: i for i, name in enumerate(category_names)}
    categories = np.asarray([category_to_id[x] for x in df["category"].to_list()], dtype=np.int16)
    print(f"tokenized {n} in {time.time() - started:.0f}s", flush=True)
    return (df["fold"].to_numpy(), df["id1"].to_numpy(), df["id2"].to_numpy(),
            df["target"].to_numpy().astype(np.float32), categories, ids, token_types)


def rank_loss(logits, labels, categories, mode, rng):
    """Return RankNet loss and pair count; pairing never crosses folds or batches."""
    positive = np.flatnonzero(labels > 0.5)
    negative = np.flatnonzero(labels <= 0.5)
    pairs = []
    if mode == "random":
        count = min(len(positive), len(negative))
        if count:
            pairs.append((rng.permutation(positive)[:count], rng.permutation(negative)[:count]))
    elif mode == "within":
        for category in np.unique(categories):
            in_category = categories == category
            pos = np.flatnonzero(in_category & (labels > 0.5))
            neg = np.flatnonzero(in_category & (labels <= 0.5))
            count = min(len(pos), len(neg))
            if count:
                pairs.append((rng.permutation(pos)[:count], rng.permutation(neg)[:count]))
    if not pairs:
        return logits.sum() * 0.0, 0
    pos_idx = torch.as_tensor(np.concatenate([p for p, _ in pairs]), device=logits.device)
    neg_idx = torch.as_tensor(np.concatenate([n for _, n in pairs]), device=logits.device)
    return F.softplus(-(logits[pos_idx] - logits[neg_idx])).mean(), len(pos_idx)


def run_fold(args, fold, fold_number, tr_idx, ev_idx, ids, token_types, labels,
             categories, pad_id):
    torch.manual_seed(args.seed + fold_number)
    torch.cuda.manual_seed_all(args.seed + fold_number)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_token_types = getattr(model.config, "type_vocab_size", 0) > 1
    steps = len(tr_idx) // args.bs * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear")
    batch_rng = np.random.default_rng(args.seed + fold_number)
    pair_rng = np.random.default_rng(args.seed + fold_number + 10_000)
    pair_counts = []
    started = time.time()
    model.train()
    for _ in range(args.epochs):
        permutation = batch_rng.permutation(len(tr_idx))
        for start in range(0, len(tr_idx) - args.bs + 1, args.bs):
            idx = tr_idx[np.sort(permutation[start:start + args.bs])]
            input_ids = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            tt = torch.from_numpy(token_types[idx].astype(np.int64)).cuda()
            targets = torch.from_numpy(labels[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=(input_ids != pad_id).long(),
                               token_type_ids=tt if use_token_types else None).logits.squeeze(-1)
                bce = F.binary_cross_entropy_with_logits(logits, targets)
                if args.rank_lambda:
                    ranking, pair_count = rank_loss(logits, labels[idx], categories[idx],
                                                    args.rank_mode, pair_rng)
                    loss = bce + args.rank_lambda * ranking
                else:
                    ranking, pair_count, loss = logits.sum() * 0.0, 0, bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            pair_counts.append(pair_count)
            step = len(pair_counts)
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"  {fold} step {step}/{steps} bce {bce.item():.4f} "
                      f"rank {ranking.item():.4f} pairs {pair_count} {rate:.1f} it/s",
                      flush=True)

    model.eval()
    output = np.zeros(len(ev_idx))
    with torch.no_grad():
        for start in range(0, len(ev_idx), args.bs * 4):
            idx = ev_idx[start:start + args.bs * 4]
            input_ids = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            tt = torch.from_numpy(token_types[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=(input_ids != pad_id).long(),
                               token_type_ids=tt if use_token_types else None).logits.squeeze(-1)
            output[start:start + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    audit = {
        "fold": fold,
        "train_rows": len(tr_idx),
        "eval_rows": len(ev_idx),
        "batches": len(pair_counts),
        "valid_pair_batches": int(np.count_nonzero(pair_counts)),
        "pairs_total": int(np.sum(pair_counts)),
        "pairs_per_batch_mean": float(np.mean(pair_counts)),
        "pairs_per_batch_min": int(np.min(pair_counts)),
        "pairs_per_batch_max": int(np.max(pair_counts)),
        "runtime_s": time.time() - started,
    }
    del model
    torch.cuda.empty_cache()
    return output, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=224)
    parser.add_argument("--attrs", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--folds", default="fold_01,fold_02")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--rank-mode", choices=["within", "random"], default="within")
    parser.add_argument("--rank-lambda", type=float, default=0.0)
    args = parser.parse_args()
    if args.rank_lambda == 0:
        args.rank_mode = "within"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fold_col, id1, id2, labels, categories, ids, token_types = build_tokens(args, tokenizer)
    root = Path(args.outdir)
    pred_dir = root / "preds" / args.exp
    pred_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for fold in args.folds.split(","):
        if fold not in FOLDS:
            raise ValueError(f"unknown fold: {fold}")
        ev_idx = np.flatnonzero(fold_col == fold)
        tr_idx = np.flatnonzero(fold_col != fold)
        assert not np.intersect1d(tr_idx, ev_idx).size
        scores, audit = run_fold(args, fold, FOLDS.index(fold), tr_idx, ev_idx,
                                 ids, token_types, labels, categories,
                                 tokenizer.pad_token_id)
        with (pred_dir / f"{fold}.csv").open("w", encoding="utf-8") as output:
            output.write("id1,id2,predict\n")
            for left, right, score in zip(id1[ev_idx], id2[ev_idx], scores):
                output.write(f"{left},{right},{score:.8f}\n")
        audits.append(audit)
        print(json.dumps(audit), flush=True)
    (root / "metrics" / f"{args.exp}_training.json").write_text(
        json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    main()

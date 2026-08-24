"""Matched two-view semantic field-dropout/R-Drop hand fine-tune."""
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


def drop_fields(text, rate, rng):
    text = text or ""
    fields = [field for field in text.split(";") if field]
    kept = [field for field in fields if rng.random() >= rate]
    return ";".join(kept) + (";" if kept and text.endswith(";") else "")


def corrupt_span(text, rate, rng):
    """Equal-rate corruption without respecting field boundaries."""
    text = text or ""
    width = min(len(text), round(len(text) * rate))
    if not width:
        return text
    start = int(rng.integers(0, len(text) - width + 1))
    return text[:start] + text[start + width:]


def build_tokens(args, tok):
    df = pl.read_parquet(args.data)

    def mk(names, attrs, cats):
        out = []
        for name, attr, cat in zip(names, attrs, cats):
            text = f"{name} | {cat}" if args.cat and cat else name
            out.append(f"{text} | {attr}" if args.attrs and attr else text)
        return out

    attrs1 = df["attrs1"].to_list()
    attrs2 = df["attrs2"].to_list()
    names1 = df["name1"].to_list()
    names2 = df["name2"].to_list()
    cats = df["category"].to_list()
    left = mk(names1, attrs1, cats)
    right = mk(names2, attrs2, cats)
    n = df.height

    def encode(a, b):
        ids = np.zeros((n, args.max_len), dtype=np.int32)
        token_types = np.zeros((n, args.max_len), dtype=np.uint8)
        for start in range(0, n, 20_000):
            end = min(start + 20_000, n)
            encoded = tok(a[start:end], b[start:end], truncation=True,
                          max_length=args.max_len, padding="max_length", return_tensors="np")
            ids[start:end] = encoded["input_ids"].astype(np.int32)
            if "token_type_ids" in encoded:
                token_types[start:end] = encoded["token_type_ids"].astype(np.uint8)
        return ids, token_types

    started = time.time()
    clean_ids, clean_tt = encode(left, right)
    if args.mode == "baseline":
        view1_ids, view1_tt = clean_ids, clean_tt
        view2_ids, view2_tt = clean_ids, clean_tt
        retained = 1.0
    else:
        transform = drop_fields if args.mode in ("field", "mismatch") else corrupt_span
        rng1 = np.random.default_rng(args.seed + 30_001)
        rng2 = np.random.default_rng(args.seed + 30_002)
        a11 = [transform(x, args.drop_rate, rng1) for x in attrs1]
        a12 = [transform(x, args.drop_rate, rng1) for x in attrs2]
        a21 = [transform(x, args.drop_rate, rng2) for x in attrs1]
        a22 = [transform(x, args.drop_rate, rng2) for x in attrs2]
        view1_ids, view1_tt = encode(mk(names1, a11, cats), mk(names2, a12, cats))
        view2_ids, view2_tt = encode(mk(names1, a21, cats), mk(names2, a22, cats))
        original_chars = sum(map(len, attrs1)) + sum(map(len, attrs2))
        retained = (sum(map(len, a11)) + sum(map(len, a12))) / original_chars
    print(f"tokenized {n} in {time.time() - started:.0f}s retained_chars={retained:.4f}", flush=True)
    return (df["fold"].to_numpy(), df["id1"].to_numpy(), df["id2"].to_numpy(),
            df["target"].to_numpy().astype(np.float32), clean_ids, clean_tt,
            view1_ids, view1_tt, view2_ids, view2_tt, retained)


def symmetric_bernoulli_kl(logits1, logits2):
    p = torch.sigmoid(logits1.float()).clamp(1e-5, 1 - 1e-5)
    q = torch.sigmoid(logits2.float()).clamp(1e-5, 1 - 1e-5)
    kl_pq = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    kl_qp = q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()
    return 0.5 * (kl_pq + kl_qp).mean()


def run_fold(args, fold, fold_number, tr_idx, ev_idx, clean_ids, clean_tt,
             view1_ids, view1_tt, view2_ids, view2_tt, labels, pad_id):
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
    losses = []
    started = time.time()
    model.train()
    for _ in range(args.epochs):
        permutation = batch_rng.permutation(len(tr_idx))
        for start in range(0, len(tr_idx) - args.bs + 1, args.bs):
            idx = tr_idx[np.sort(permutation[start:start + args.bs])]
            input_ids = torch.from_numpy(np.concatenate([view1_ids[idx], view2_ids[idx]]).astype(np.int64)).cuda()
            tt = torch.from_numpy(np.concatenate([view1_tt[idx], view2_tt[idx]]).astype(np.int64)).cuda()
            targets = torch.from_numpy(labels[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=(input_ids != pad_id).long(),
                               token_type_ids=tt if use_token_types else None).logits.squeeze(-1)
                logits1, logits2 = logits.chunk(2)
                bce = 0.5 * (F.binary_cross_entropy_with_logits(logits1, targets) +
                             F.binary_cross_entropy_with_logits(logits2, targets))
                consistency_logits = logits2.roll(1) if args.mode in ("negative", "mismatch") else logits2
                consistency = symmetric_bernoulli_kl(logits1, consistency_logits)
                loss = bce + args.consistency_weight * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(float(consistency.detach()))
            step = len(losses)
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"  {fold} step {step}/{steps} bce {bce.item():.4f} "
                      f"consistency {consistency.item():.4f} {rate:.1f} it/s",
                      flush=True)

    model.eval()
    output = np.zeros(len(ev_idx))
    with torch.no_grad():
        for start in range(0, len(ev_idx), args.bs * 4):
            idx = ev_idx[start:start + args.bs * 4]
            input_ids = torch.from_numpy(clean_ids[idx].astype(np.int64)).cuda()
            tt = torch.from_numpy(clean_tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=(input_ids != pad_id).long(),
                               token_type_ids=tt if use_token_types else None).logits.squeeze(-1)
            output[start:start + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    audit = {
        "fold": fold,
        "train_rows": len(tr_idx),
        "eval_rows": len(ev_idx),
        "batches": len(losses),
        "consistency_mean": float(np.mean(losses)),
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
    parser.add_argument("--mode", choices=["baseline", "field", "negative", "span", "mismatch"], required=True)
    parser.add_argument("--drop-rate", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.mode == "baseline":
        args.consistency_weight = 0.0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    (fold_col, id1, id2, labels, clean_ids, clean_tt, view1_ids, view1_tt,
     view2_ids, view2_tt, retained) = build_tokens(args, tokenizer)
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
                                 clean_ids, clean_tt, view1_ids, view1_tt,
                                 view2_ids, view2_tt, labels,
                                 tokenizer.pad_token_id)
        audit["retained_chars"] = retained
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

"""LLM-stage cross-encoder pretraining with a pre-tokenized numpy cache.

Tokenizes pairs once (fast tokenizer, rayon-parallel, no fork) into int32/uint8
arrays on disk; training then slices arrays directly — no dataloader workers,
no per-item Python string traffic. Saves checkpoint for hand-stage FT.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORK = Path.home() / "matching-work"


def make_texts(df, use_attrs, use_cat):
    def mk(names, attrs, cats):
        out = []
        for n, a, c in zip(names, attrs, cats):
            t = n
            if use_cat and c:
                t = f"{n} | {c}"
            if use_attrs and a:
                t = f"{t} | {a}"
            out.append(t)
        return out
    t1 = mk(df["name1"].to_list(), df["attrs1"].to_list(), df["category1"].to_list())
    t2 = mk(df["name2"].to_list(), df["attrs2"].to_list(), df["category1"].to_list())
    return t1, t2


def build_cache(args, tok):
    df = pl.read_parquet(args.llm_file)
    n = df.height
    print(f"tokenizing {n} pairs...", flush=True)
    t1, t2 = make_texts(df, args.attrs, args.cat)
    y = df["target"].to_numpy().astype(np.float32)
    del df
    ids = np.zeros((n, args.max_len), dtype=np.int32)
    tt = np.zeros((n, args.max_len), dtype=np.uint8)
    chunk = 100_000
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        enc = tok(t1[s:e], t2[s:e], truncation=True, max_length=args.max_len,
                  padding="max_length", return_tensors="np")
        ids[s:e] = enc["input_ids"].astype(np.int32)
        tt[s:e] = enc["token_type_ids"].astype(np.uint8)
        if (s // chunk) % 10 == 0:
            rate = e / (time.time() - t0)
            print(f"  {e}/{n} tok {rate:.0f}/s eta {(n-e)/rate/60:.1f}m", flush=True)
    np.save(args.cache + ".ids.npy", ids)
    np.save(args.cache + ".tt.npy", tt)
    np.save(args.cache + ".y.npy", y)
    print("cache saved", flush=True)
    return ids, tt, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--llm-file", required=True)
    ap.add_argument("--cache", required=True, help="cache path prefix")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--cat", action="store_true")
    ap.add_argument("--save", required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = np.load(args.cache + ".ids.npy")
    y = np.load(args.cache + ".y.npy")
    sep_id = tok.sep_token_id
    pad_id = tok.pad_token_id
    print("cache loaded", ids.shape, "sep", sep_id, flush=True)

    n = len(y)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init or args.model, num_labels=1).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    steps_per_epoch = n // args.bs
    total = steps_per_epoch * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total, pct_start=0.06,
        anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    rng = np.random.default_rng(20260814)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        for s in range(0, steps_per_epoch * args.bs, args.bs):
            idx = np.sort(perm[s:s + args.bs])
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda(non_blocking=True)
            by = torch.from_numpy(y[idx]).cuda(non_blocking=True)
            mask = (bi != pad_id).long()
            sep = (bi == sep_id).long()
            bt = ((sep.cumsum(1) - sep) >= 1).long() * mask
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=mask,
                               token_type_ids=bt if use_tt else None
                               ).logits.squeeze(-1)
                loss = lossf(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - t0)
                print(f"  step {step}/{total} loss {loss.item():.4f} "
                      f"{rate:.2f} it/s eta {(total-step)/rate/60:.0f}m", flush=True)
            if step % 8000 == 0:
                Path(args.save + f"_s{step}").mkdir(parents=True, exist_ok=True)
                model.save_pretrained(args.save + f"_s{step}")
                tok.save_pretrained(args.save + f"_s{step}")
                print(f"  interim ckpt saved at {step}", flush=True)
    Path(args.save).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.save)
    tok.save_pretrained(args.save)
    print("saved", args.save, flush=True)


if __name__ == "__main__":
    main()

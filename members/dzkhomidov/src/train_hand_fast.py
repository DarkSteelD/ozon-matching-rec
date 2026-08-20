"""Hand-stage 4-fold OOF cross-encoder FT with in-process pre-tokenization.

Tokenizes the 365K hand pairs once (fast tokenizer, in-process), then runs
GPU-bound fold training. Writes preds/<exp>/fold_0K.csv in canonical order.
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
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def build_tokens(args, tok):
    df = pl.read_parquet(getattr(args, "data", None) or WORK / "data/hand_pairs.parquet")
    def mk(names, attrs, cats):
        out = []
        for n, a, c in zip(names, attrs, cats):
            t = n
            if args.cat and c:
                t = f"{n} | {c}"
            if args.attrs and a:
                t = f"{t} | {a}"
            out.append(t)
        return out
    t1 = mk(df["name1"].to_list(), df["attrs1"].to_list(), df["category"].to_list())
    t2 = mk(df["name2"].to_list(), df["attrs2"].to_list(), df["category"].to_list())
    n = df.height
    ids = np.zeros((n, args.max_len), dtype=np.int32)
    tt = np.zeros((n, args.max_len), dtype=np.uint8)
    t0 = time.time()
    for s in range(0, n, 20000):
        e = min(s + 20000, n)
        enc = tok(t1[s:e], t2[s:e], truncation=True, max_length=args.max_len,
                  padding="max_length", return_tensors="np")
        ids[s:e] = enc["input_ids"].astype(np.int32)
        if "token_type_ids" in enc:
            tt[s:e] = enc["token_type_ids"].astype(np.uint8)
    print(f"tokenized {n} in {time.time()-t0:.0f}s", flush=True)
    return (df["fold"].to_numpy(), df["id1"].to_numpy(), df["id2"].to_numpy(),
            df["target"].to_numpy().astype(np.float32), ids, tt)


def run_fold(args, fold, tr_idx, ev_idx, ids, tt, y, pad_id=0):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init or args.model, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    n = len(tr_idx)
    steps = n // args.bs * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    rng = np.random.default_rng(args.seed)
    step, t0 = 0, time.time()
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for s in range(0, n - args.bs + 1, args.bs):
            idx = tr_idx[np.sort(perm[s:s + args.bs])]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            by = torch.from_numpy(y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lossf(model(input_ids=bi, attention_mask=(bi != pad_id).long(),
                                   token_type_ids=bt if use_tt else None
                                   ).logits.squeeze(-1), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                r = step / (time.time() - t0)
                print(f"  {fold} step {step}/{steps} loss {loss.item():.4f} "
                      f"{r:.1f} it/s eta {(steps-step)/r/60:.0f}m", flush=True)
    model.eval()
    out = np.zeros(len(ev_idx))
    with torch.no_grad():
        for s in range(0, len(ev_idx), args.bs * 4):
            idx = ev_idx[s:s + args.bs * 4]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != pad_id).long(),
                               token_type_ids=bt if use_tt else None
                               ).logits.squeeze(-1)
            out[s:s + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--cat", action="store_true")
    ap.add_argument("--folds", default="")
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tok = AutoTokenizer.from_pretrained(args.model)
    fold_col, id1, id2, y, ids, tt = build_tokens(args, tok)
    outdir = WORK / "preds" / args.exp
    outdir.mkdir(parents=True, exist_ok=True)
    for fold in (args.folds.split(",") if args.folds else FOLDS):
        ev = np.flatnonzero(fold_col == fold)
        tr = np.flatnonzero(fold_col != fold)
        print(f"{fold}: train {len(tr)}", flush=True)
        scores = run_fold(args, fold, tr, ev, ids, tt, y, tok.pad_token_id)
        with (outdir / f"{fold}.csv").open("w", newline="", encoding="utf-8") as f:
            f.write("id1,id2,predict\n")
            for a, b, s in zip(id1[ev], id2[ev], scores.tolist()):
                f.write(f"{a},{b},{s:.8f}\n")
        print(fold, "written", flush=True)


if __name__ == "__main__":
    main()

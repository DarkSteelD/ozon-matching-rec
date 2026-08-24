"""Minimal long-context variant of the recovered hand-FT trainer.

The effective batch remains 256 while micro-batching avoids a len512 OOM.
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

FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def build_tokens(args, tok):
    df = pl.read_parquet(args.data)

    def mk(names, attrs, cats):
        return [f"{n} | {c} | {a}" if a else f"{n} | {c}"
                for n, a, c in zip(names, attrs, cats)]

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
    print(f"tokenized {n} at len={args.max_len} in {time.time()-t0:.0f}s", flush=True)
    return (df["fold"].to_numpy(), df["id1"].to_numpy(), df["id2"].to_numpy(),
            df["target"].to_numpy().astype(np.float32), ids, tt)


def run_fold(args, fold, tr_idx, ev_idx, ids, tt, y, pad_id):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    n = len(tr_idx)
    steps_per_epoch = n // args.effective_bs
    steps = steps_per_epoch * args.epochs
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
        for s in range(0, steps_per_epoch * args.effective_bs, args.effective_bs):
            batch_idx = tr_idx[np.sort(perm[s:s + args.effective_bs])]
            optim.zero_grad(set_to_none=True)
            for m in range(0, args.effective_bs, args.micro_bs):
                idx = batch_idx[m:m + args.micro_bs]
                bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
                bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
                by = torch.from_numpy(y[idx]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi, attention_mask=(bi != pad_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                    loss = lossf(logits, by) * (len(idx) / args.effective_bs)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % 250 == 0:
                rate = step / (time.time() - t0)
                print(f"  {fold} step {step}/{steps} loss {loss.item():.4f} "
                      f"{rate:.2f} it/s eta {(steps-step)/rate/60:.0f}m", flush=True)
    model.eval()
    out = np.zeros(len(ev_idx))
    with torch.no_grad():
        for s in range(0, len(ev_idx), args.eval_bs):
            idx = ev_idx[s:s + args.eval_bs]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != pad_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
            out[s:s + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--effective-bs", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    assert args.effective_bs % args.micro_bs == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = AutoTokenizer.from_pretrained(args.model)
    fold_col, id1, id2, y, ids, tt = build_tokens(args, tok)
    outdir = Path(args.output_root) / args.exp
    outdir.mkdir(parents=True, exist_ok=True)
    for fold in args.folds.split(","):
        ev = np.flatnonzero(fold_col == fold)
        tr = np.flatnonzero(fold_col != fold)
        print(f"{fold}: train {len(tr)} eval {len(ev)}", flush=True)
        scores = run_fold(args, fold, tr, ev, ids, tt, y, tok.pad_token_id)
        with (outdir / f"{fold}.csv").open("w", encoding="utf-8") as f:
            f.write("id1,id2,predict\n")
            for a, b, score in zip(id1[ev], id2[ev], scores.tolist()):
                f.write(f"{a},{b},{score:.8f}\n")
        print(f"{fold} written", flush=True)


if __name__ == "__main__":
    main()

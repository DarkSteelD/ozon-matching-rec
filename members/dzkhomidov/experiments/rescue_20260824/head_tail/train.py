#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run_fold(args, fold, fold_no, tr, ev, ids, tt, y, pad_id):
    torch.manual_seed(args.seed + fold_no)
    torch.cuda.manual_seed_all(args.seed + fold_no)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).cuda()
    n = len(tr)
    steps = n // args.bs * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, args.lr, total_steps=steps,
                                                 pct_start=.06, anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed + fold_no)
    model.train(); step = 0; started = time.time()
    for _ in range(args.epochs):
        perm = rng.permutation(n)
        for start in range(0, n - args.bs + 1, args.bs):
            idx = tr[np.sort(perm[start:start + args.bs])]
            bi = torch.from_numpy(np.asarray(ids[idx], dtype=np.int64)).cuda()
            bt = torch.from_numpy(np.asarray(tt[idx], dtype=np.int64)).cuda()
            by = torch.from_numpy(y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != pad_id).long(), token_type_ids=bt).logits.squeeze(-1)
                loss = lossf(logits, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{fold} step {step}/{steps} loss {loss.item():.4f} {rate:.2f}it/s eta {(steps-step)/rate/60:.1f}m", flush=True)
    model.eval(); pred = np.zeros(len(ev), np.float32)
    with torch.no_grad():
        for start in range(0, len(ev), args.bs * 2):
            idx = ev[start:start + args.bs * 2]
            bi = torch.from_numpy(np.asarray(ids[idx], dtype=np.int64)).cuda()
            bt = torch.from_numpy(np.asarray(tt[idx], dtype=np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(input_ids=bi, attention_mask=(bi != pad_id).long(), token_type_ids=bt).logits.squeeze(-1).float())
            pred[start:start + len(idx)] = p.cpu().numpy()
    del model; torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prefix", "headtail", "middle"], required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--tokens", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--folds", required=True); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    df = pl.read_parquet(args.data, columns=["fold", "id1", "id2", "target"])
    ids = np.load(Path(args.tokens) / f"ids_{args.mode}.npy", mmap_mode="r")
    tt = np.load(Path(args.tokens) / f"tt_{args.mode}.npy", mmap_mode="r")
    assert len(df) == len(ids) and ids.shape[1] == 384
    y = df["target"].to_numpy().astype(np.float32); fold_col = df["fold"].to_numpy()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.output) / args.mode; out.mkdir(parents=True, exist_ok=True)
    run_meta = {"mode": args.mode, "args": vars(args), "pid": __import__("os").getpid(), "started": time.time()}
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    for fold in args.folds.split(","):
        ev = np.flatnonzero(fold_col == fold); tr = np.flatnonzero(fold_col != fold)
        print(f"{args.mode} {fold}: train={len(tr)} eval={len(ev)}", flush=True)
        pred = run_fold(args, fold, int(fold[-2:]), tr, ev, ids, tt, y, tok.pad_token_id)
        with (out / f"{fold}.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["id1", "id2", "predict"])
            w.writerows(zip(df["id1"].to_numpy()[ev], df["id2"].to_numpy()[ev], pred))
        print(f"{fold} written", flush=True)
    run_meta["finished"] = time.time(); run_meta["runtime_seconds"] = run_meta["finished"] - run_meta["started"]
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

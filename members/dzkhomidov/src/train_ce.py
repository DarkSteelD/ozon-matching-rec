"""Cross-encoder trainer for Ozon matching pairs.

Modes:
  --stage llm   : train one model on an LLM pair parquet (soft BCE), save checkpoint,
                  optionally zero-shot predict all 4 hand folds.
  --stage hand  : 4-fold OOF — for each fold, train on the other three (optionally
                  starting from --init checkpoint), predict the held-out fold.

Predictions land in ~/matching-work/preds/<exp>/fold_0K.csv in canonical order
(hand_pairs.parquet preserves target-file order per fold).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORK = Path.home() / "matching-work"
FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def make_text(name: str, attrs: str, cat: str | None, use_attrs: bool, use_cat: bool) -> str:
    t = name
    if use_cat and cat:
        t = f"{name} | {cat}"
    if use_attrs and attrs:
        t = f"{t} | {attrs}"
    return t


class PairDS(Dataset):
    def __init__(self, t1, t2, y):
        self.t1, self.t2, self.y = t1, t2, y

    def __len__(self):
        return len(self.t1)

    def __getitem__(self, i):
        return self.t1[i], self.t2[i], self.y[i]


def collate(batch, tok, max_len):
    t1, t2, y = zip(*batch)
    enc = tok(list(t1), list(t2), truncation=True, max_length=max_len,
              padding=True, return_tensors="pt")
    return enc, torch.tensor(y, dtype=torch.float32)


def train_one(model, tok, ds, args, steps_per_epoch=None):
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8,
                        collate_fn=lambda b: collate(b, tok, args.max_len),
                        pin_memory=True, drop_last=True, persistent_workers=False)
    total = (steps_per_epoch or len(loader)) * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=args.lr, total_steps=total, pct_start=0.06, anneal_strategy="linear")
    model.train()
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t0 = 0, time.time()
    for _ in range(args.epochs):
        for enc, y in loader:
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
                loss = lossf(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - t0)
                print(f"  step {step}/{total} loss {loss.item():.4f} "
                      f"{rate:.1f} it/s eta {(total-step)/rate/60:.0f}m", flush=True)
            if step >= total:
                break
    return model


@torch.no_grad()
def predict(model, tok, t1, t2, args):
    model.eval()
    out = np.zeros(len(t1), dtype=np.float64)
    bs = args.bs * 4
    for s in range(0, len(t1), bs):
        e = tok(list(t1[s:s + bs]), list(t2[s:s + bs]), truncation=True,
                max_length=args.max_len, padding=True, return_tensors="pt")
        e = {k: v.cuda() for k, v in e.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**e).logits.squeeze(-1)
        out[s:s + bs] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def write_fold(exp: str, fold: str, ids1, ids2, scores):
    d = WORK / "preds" / exp
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{fold}.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("id1,id2,predict\n")
        for a, b, s in zip(ids1, ids2, scores):
            f.write(f"{a},{b},{s:.8f}\n")


def load_hand(args):
    df = pl.read_parquet(WORK / "data/hand_pairs.parquet")
    t1 = [make_text(n, a, c, args.attrs, args.cat)
          for n, a, c in zip(df["name1"], df["attrs1"], df["category"])]
    t2 = [make_text(n, a, c, args.attrs, args.cat)
          for n, a, c in zip(df["name2"], df["attrs2"], df["category"])]
    return df, np.array(t1, dtype=object), np.array(t2, dtype=object)


def fresh_model(args):
    src = args.init or args.model
    m = AutoModelForSequenceClassification.from_pretrained(
        src, num_labels=1, ignore_mismatched_sizes=True)
    return m.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--stage", choices=["llm", "hand"], required=True)
    ap.add_argument("--model", default="cointegrated/rubert-tiny2")
    ap.add_argument("--init", default=None, help="checkpoint dir to start from")
    ap.add_argument("--llm-file", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--cat", action="store_true")
    ap.add_argument("--save", default=None, help="checkpoint output dir")
    ap.add_argument("--predict-folds", action="store_true",
                    help="(llm stage) zero-shot predict hand folds")
    ap.add_argument("--folds", default="", help="comma list, default all (hand stage)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(args.model)

    if args.stage == "llm":
        df = pl.read_parquet(args.llm_file)
        t1 = [make_text(n, a, c, args.attrs, args.cat)
              for n, a, c in zip(df["name1"], df["attrs1"], df["category1"])]
        t2 = [make_text(n, a, c, args.attrs, args.cat)
              for n, a, c in zip(df["name2"], df["attrs2"], df["category1"])]
        y = df["target"].to_numpy().astype(np.float32)
        print(f"LLM train: {len(t1)} pairs", flush=True)
        model = fresh_model(args)
        model = train_one(model, tok, PairDS(t1, t2, y), args)
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.save)
            tok.save_pretrained(args.save)
            print("saved", args.save)
        if args.predict_folds:
            hd, h1, h2 = load_hand(args)
            for fold in FOLDS:
                mask = (hd["fold"] == fold).to_numpy()
                scores = predict(model, tok, h1[mask], h2[mask], args)
                write_fold(args.exp, fold, hd["id1"].to_numpy()[mask],
                           hd["id2"].to_numpy()[mask], scores)
                print(fold, "written", flush=True)
    else:
        hd, h1, h2 = load_hand(args)
        y = hd["target"].to_numpy().astype(np.float32)
        fold_col = hd["fold"].to_numpy()
        folds = args.folds.split(",") if args.folds else FOLDS
        for fold in folds:
            mask = fold_col == fold
            model = fresh_model(args)
            print(f"{fold}: train {int((~mask).sum())} pairs", flush=True)
            model = train_one(model, tok, PairDS(h1[~mask], h2[~mask], y[~mask]), args)
            scores = predict(model, tok, h1[mask], h2[mask], args)
            write_fold(args.exp, fold, hd["id1"].to_numpy()[mask],
                       hd["id2"].to_numpy()[mask], scores)
            print(fold, "written", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

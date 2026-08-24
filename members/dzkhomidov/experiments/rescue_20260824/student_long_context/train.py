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


def build_tokens(args, tok):
    df = pl.read_parquet(args.data)
    def mk(side):
        out = []
        for n, a, c in zip(df[f"name{side}"], df[f"attrs{side}"], df["category"]):
            text = f"{n} | {c}"
            if a:
                text += f" | {a}"
            out.append(text)
        return out
    left, right = mk(1), mk(2)
    n = df.height
    def enc(a, b):
        ids = np.zeros((n, args.max_len), dtype=np.int32)
        tt = np.zeros((n, args.max_len), dtype=np.uint8)
        for start in range(0, n, 20_000):
            end = min(start + 20_000, n)
            batch = tok(a[start:end], b[start:end], truncation=True,
                        max_length=args.max_len, padding="max_length", return_tensors="np")
            ids[start:end] = batch["input_ids"].astype(np.int32)
            if "token_type_ids" in batch:
                tt[start:end] = batch["token_type_ids"].astype(np.uint8)
        return ids, tt
    t0 = time.time()
    ids, tt = enc(left, right)
    ids_r, tt_r = enc(right, left)
    print(f"tokenized {n} both directions len={args.max_len} in {time.time()-t0:.1f}s", flush=True)
    return df, ids, tt, ids_r, tt_r


def run_fold(args, fold, df, ids, tt, ids_r, tt_r, tok):
    f = df["fold"].to_numpy()
    train_idx = np.flatnonzero(f != fold)
    eval_idx = np.flatnonzero(f == fold)
    y = df["target"].to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    steps_per_epoch = len(train_idx) // args.effective_bs
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.06,
        anneal_strategy="linear")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    step, t0 = 0, time.time()
    model.train()
    for _ in range(args.epochs):
        perm = rng.permutation(len(train_idx))
        for start in range(0, steps_per_epoch * args.effective_bs, args.effective_bs):
            rows = train_idx[np.sort(perm[start:start + args.effective_bs])]
            reverse = rng.random(len(rows)) < 0.5
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, args.effective_bs, args.micro_bs):
                part = rows[micro:micro + args.micro_bs]
                rev = reverse[micro:micro + args.micro_bs]
                bi_np = np.where(rev[:, None], ids_r[part], ids[part])
                bt_np = np.where(rev[:, None], tt_r[part], tt[part])
                bi = torch.from_numpy(bi_np.astype(np.int64)).cuda()
                bt = torch.from_numpy(bt_np.astype(np.int64)).cuda()
                by = torch.from_numpy(y[part]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                    loss = loss_fn(logits, by) * (len(part) / args.effective_bs)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - t0)
                print(f"{fold} {step}/{total_steps} loss={loss.item():.5f} "
                      f"rate={rate:.2f}/s eta={(total_steps-step)/rate/60:.1f}m", flush=True)
    checkpoint_dir = args.output / "checkpoints" / args.variant / fold
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    model.eval()
    out = np.empty(len(eval_idx), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(eval_idx), args.eval_bs):
            rows = eval_idx[start:start + args.eval_bs]
            prob = 0.0
            for xids, xtt in ((ids, tt), (ids_r, tt_r)):
                bi = torch.from_numpy(xids[rows].astype(np.int64)).cuda()
                bt = torch.from_numpy(xtt[rows].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                prob = prob + torch.sigmoid(logits.float()).cpu().numpy()
            out[start:start + len(rows)] = prob / 2
    outdir = args.output / "preds" / args.variant
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"{fold}.csv").open("w", encoding="utf-8") as handle:
        handle.write("id1,id2,predict\n")
        for a, b, p in zip(df["id1"].to_numpy()[eval_idx],
                           df["id2"].to_numpy()[eval_idx], out):
            handle.write(f"{a},{b},{p:.9f}\n")
    meta = {"variant": args.variant, "fold": fold, "steps": total_steps,
            "train_rows": len(train_idx), "eval_rows": len(eval_idx),
            "max_len": args.max_len, "runtime_seconds": time.time()-t0}
    (args.output / f"runtime_{args.variant}_{fold}.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--effective-bs", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    assert args.effective_bs % args.micro_bs == 0
    args.output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = AutoTokenizer.from_pretrained(args.model)
    df, ids, tt, ids_r, tt_r = build_tokens(args, tok)
    for fold in args.folds.split(","):
        run_fold(args, fold, df, ids, tt, ids_r, tt_r, tok)


if __name__ == "__main__":
    main()

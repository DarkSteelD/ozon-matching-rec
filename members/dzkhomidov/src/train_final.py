"""Final-submission FT: train a CE on ALL hand pairs and save the model (fp16).

Same recipe as train_hand_fast.py fold training, but no eval split and the
model is saved for the ODS container.

Usage: train_final.py --exp <name> --model <tok_src> --init <ckpt> --max-len 224 --cat --attrs
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import train_hand_fast as thf

WORK = Path.home() / "matching-work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--cat", action="store_true")
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--data", default=None)
    ap.add_argument("--sym", action="store_true")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tok = AutoTokenizer.from_pretrained(args.model)
    _, _, _, y, ids, tt, ids_r, tt_r = thf.build_tokens(args, tok)
    pad_id = tok.pad_token_id
    n = len(y)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.init or args.model, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
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
            idx = np.sort(perm[s:s + args.bs])
            if ids_r is not None:
                mask = rng.random(len(idx)) < 0.5
                bi_np = np.where(mask[:, None], ids_r[idx], ids[idx])
                bt_np = np.where(mask[:, None], tt_r[idx], tt[idx])
            else:
                bi_np, bt_np = ids[idx], tt[idx]
            bi = torch.from_numpy(bi_np.astype(np.int64)).cuda()
            bt = torch.from_numpy(bt_np.astype(np.int64)).cuda()
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
                print(f"step {step}/{steps} loss {loss.item():.4f} "
                      f"{r:.1f} it/s eta {(steps-step)/r/60:.0f}m", flush=True)

    outdir = WORK / "ckpt" / args.exp
    model.half()
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)
    print("saved", outdir, flush=True)


if __name__ == "__main__":
    main()

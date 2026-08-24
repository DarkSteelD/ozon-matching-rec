from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EPS = {"bce": 0.0, "poly05": 0.5, "poly1": 1.0, "polyneg05": -0.5}


def encode(tok, left, right, max_len):
    n = len(left); ids = np.zeros((n, max_len), np.int32); tt = np.zeros((n, max_len), np.uint8)
    for s in range(0, n, 20_000):
        e = min(s + 20_000, n)
        x = tok(left[s:e], right[s:e], truncation=True, max_length=max_len,
                padding="max_length", return_tensors="np")
        ids[s:e] = x["input_ids"].astype(np.int32)
        if "token_type_ids" in x: tt[s:e] = x["token_type_ids"].astype(np.uint8)
    return ids, tt


def tokens(data, tok, max_len):
    df = pl.read_parquet(data)
    def side(k):
        return [f"{n} | {c}" + (f" | {a}" if a else "")
                for n, c, a in zip(df[f"name{k}"], df["category"], df[f"attrs{k}"])]
    a, b = side(1), side(2); t = time.time()
    ids, tt = encode(tok, a, b, max_len); ids_r, tt_r = encode(tok, b, a, max_len)
    print(f"tokenized {len(a)} both directions in {time.time()-t:.1f}s", flush=True)
    return df, ids, tt, ids_r, tt_r


def predict(model, rows, ids, tt, ids_r, tt_r, pad, bs):
    model.eval(); use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    out = np.empty(len(rows), np.float64)
    with torch.no_grad():
        for s in range(0, len(rows), bs * 4):
            idx = rows[s:s + bs * 4]; p = 0.0
            for vi, vt in ((ids, tt), (ids_r, tt_r)):
                bi = torch.from_numpy(vi[idx].astype(np.int64)).cuda()
                bt = torch.from_numpy(vt[idx].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(input_ids=bi, attention_mask=(bi != pad).long(),
                              token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                p = p + torch.sigmoid(z.float()).cpu().numpy()
            out[s:s+len(idx)] = p / 2
    return out


def run(args, variant, fold, df, ids, tt, ids_r, tt_r, tok):
    eps = EPS[variant]; torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    fc = df["fold"].to_numpy(); tr = np.flatnonzero(fc != fold); ev = np.flatnonzero(fc == fold)
    y = df["target"].to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    steps = len(tr) // args.bs * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                               pct_start=.06, anneal_strategy="linear")
    model.train(); step = 0; t0 = time.time()
    for _ in range(args.epochs):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr) - args.bs + 1, args.bs):
            idx = tr[np.sort(perm[s:s+args.bs])]; rev = rng.random(len(idx)) < .5
            bi = torch.from_numpy(np.where(rev[:, None], ids_r[idx], ids[idx]).astype(np.int64)).cuda()
            bt = torch.from_numpy(np.where(rev[:, None], tt_r[idx], tt[idx]).astype(np.int64)).cuda()
            by = torch.from_numpy(y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                          token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                bce = F.binary_cross_entropy_with_logits(z, by)
                p = torch.sigmoid(z); pt = by * p + (1 - by) * (1 - p)
                loss = bce + eps * (1 - pt).mean()
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time()-t0)
                print(f"{variant} {fold} {step}/{steps} loss={loss.item():.5f} bce={bce.item():.5f} "
                      f"rate={rate:.2f}/s eta={(steps-step)/rate/60:.1f}m", flush=True)
    pred = predict(model, ev, ids, tt, ids_r, tt_r, tok.pad_token_id, args.bs)
    out = args.output / "preds" / variant; out.mkdir(parents=True, exist_ok=True)
    with (out / f"{fold}.csv").open("w") as f:
        f.write("id1,id2,predict\n")
        for a, b, p in zip(df["id1"].to_numpy()[ev], df["id2"].to_numpy()[ev], pred):
            f.write(f"{a},{b},{p:.9f}\n")
    meta = {"variant": variant, "epsilon": eps, "fold": fold, "seed": args.seed,
            "train_rows": len(tr), "eval_rows": len(ev), "steps": steps,
            "runtime_seconds": time.time()-t0, "pid": os.getpid()}
    (out / f"{fold}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{variant} {fold} written", flush=True); del model; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--variants", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02"); ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    unknown = set(args.variants.split(",")) - EPS.keys(); assert not unknown, unknown
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = AutoTokenizer.from_pretrained(args.model); df, ids, tt, ids_r, tt_r = tokens(args.data, tok, args.max_len)
    for variant in args.variants.split(","):
        for fold in args.folds.split(","): run(args, variant, fold, df, ids, tt, ids_r, tt_r, tok)


if __name__ == "__main__": main()

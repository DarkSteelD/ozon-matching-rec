from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RoutedClassifier(nn.Module):
    def __init__(self, init, categories, variant, strength=0.75):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1)
        hidden = self.model.config.hidden_size
        self.experts = nn.Linear(hidden, len(categories))
        nn.init.zeros_(self.experts.weight)
        nn.init.zeros_(self.experts.bias)
        self.variant, self.strength = variant, strength

    def forward(self, ids, tt, pad_id, routes):
        use_tt = getattr(self.model.config, "type_vocab_size", 0) > 1
        out = self.model.base_model(input_ids=ids, attention_mask=(ids != pad_id).long(),
                                    token_type_ids=tt if use_tt else None, return_dict=True)
        h = self.model.dropout(out.pooler_output)
        shared = self.model.classifier(h).squeeze(-1)
        if self.variant == "shared":
            return shared
        residual = self.experts(h).gather(1, routes.clamp_min(0)[:, None]).squeeze(1)
        return shared + self.strength * residual * (routes >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["shared", "category", "random"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    df = pl.read_parquet(args.data)
    cats = sorted(df["category"].unique().to_list())
    cat_to_id = {c: i for i, c in enumerate(cats)}
    cat_ids = np.array([cat_to_id[c] for c in df["category"]], dtype=np.int64)
    # Stable, balanced-ish partition independent of category.
    id1 = df["id1"].to_numpy().astype(np.uint64)
    id2 = df["id2"].to_numpy().astype(np.uint64)
    random_ids = ((id1 * np.uint64(11400714819323198485) ^ id2 * np.uint64(7046029254386353131))
                  % np.uint64(len(cats))).astype(np.int64)
    tok = AutoTokenizer.from_pretrained(args.init)

    def text(n, c, a):
        return [f"{x} | {z} | {q}" if q else f"{x} | {z}" for x, z, q in zip(n, c, a)]

    a = text(df["name1"], df["category"], df["attrs1"])
    b = text(df["name2"], df["category"], df["attrs2"])
    n = df.height

    def encode(x, z):
        ids = np.zeros((n, args.max_len), np.int32)
        tt = np.zeros((n, args.max_len), np.uint8)
        for s in range(0, n, 20000):
            e = min(s + 20000, n)
            enc = tok(x[s:e], z[s:e], truncation=True, max_length=args.max_len,
                      padding="max_length", return_tensors="np")
            ids[s:e] = enc["input_ids"]
            if "token_type_ids" in enc:
                tt[s:e] = enc["token_type_ids"]
        return ids, tt

    t0 = time.time()
    ids, tt = encode(a, b)
    ids_r, tt_r = encode(b, a)
    print(f"tokenized {n} both directions in {time.time()-t0:.0f}s", flush=True)
    fold_col = df["fold"].to_numpy()
    y = df["target"].to_numpy().astype(np.float32)
    outdir = Path(args.output) / "preds" / args.variant
    outdir.mkdir(parents=True, exist_ok=True)

    for fold in args.folds.split(","):
        ev = np.flatnonzero(fold_col == fold)
        tr = np.flatnonzero(fold_col != fold)
        base_routes = cat_ids if args.variant == "category" else random_ids
        counts = np.bincount(base_routes[tr], minlength=len(cats))
        active = counts >= 5000
        routes = np.where(active[base_routes], base_routes, -1)
        print(f"{fold} train={len(tr)} active_heads={active.sum()} counts={counts.tolist()}", flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = RoutedClassifier(args.init, cats, args.variant).cuda()
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(tr) // args.bs * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr,
            total_steps=steps, pct_start=.06, anneal_strategy="linear")
        lossf = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(args.seed)
        model.train(); step = 0; start = time.time()
        for _ in range(args.epochs):
            perm = rng.permutation(len(tr))
            for s in range(0, len(tr) - args.bs + 1, args.bs):
                idx = tr[np.sort(perm[s:s + args.bs])]
                swap = rng.random(len(idx)) < .5
                bi = torch.from_numpy(np.where(swap[:,None], ids_r[idx], ids[idx]).astype(np.int64)).cuda()
                bt = torch.from_numpy(np.where(swap[:,None], tt_r[idx], tt[idx]).astype(np.int64)).cuda()
                by = torch.from_numpy(y[idx]).cuda()
                br = torch.from_numpy(routes[idx]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(bi, bt, tok.pad_token_id, br)
                    loss = lossf(logits, by)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step += 1
                if step % 500 == 0:
                    rate = step / (time.time() - start)
                    print(f"{fold} step {step}/{steps} loss={loss.item():.4f} rate={rate:.2f} eta={(steps-step)/rate/60:.1f}m", flush=True)
        model.eval(); pred = np.zeros(len(ev))
        with torch.no_grad():
            for s in range(0, len(ev), args.bs * 2):
                idx = ev[s:s + args.bs * 2]
                br = torch.from_numpy(routes[idx]).cuda()
                total = 0
                for vi, vt in [(ids,tt),(ids_r,tt_r)]:
                    bi = torch.from_numpy(vi[idx].astype(np.int64)).cuda()
                    bt = torch.from_numpy(vt[idx].astype(np.int64)).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        total = total + torch.sigmoid(model(bi, bt, tok.pad_token_id, br).float())
                pred[s:s+len(idx)] = (total / 2).cpu().numpy()
        with (outdir / f"{fold}.csv").open("w") as f:
            f.write("id1,id2,predict\n")
            for x,z,p in zip(df["id1"].to_numpy()[ev], df["id2"].to_numpy()[ev], pred):
                f.write(f"{x},{z},{p:.8f}\n")
        print(f"{fold} written runtime={time.time()-start:.1f}s", flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

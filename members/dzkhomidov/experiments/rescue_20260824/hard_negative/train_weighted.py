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

FOLDS = ["fold_01", "fold_02", "fold_03", "fold_04"]


def select_masks(df: pl.DataFrame, heldout: str, coverage: float, seed: int):
    train = df.with_row_index("row").filter((pl.col("fold") != heldout) & (pl.col("target") == 0))
    hard_rows, random_rows, by_cat = [], [], []
    rng = np.random.default_rng(seed + int(heldout[-2:]))
    for cat, group in train.group_by("category", maintain_order=True):
        g = group.with_columns(
            pl.col("ce_oof").rank("average").truediv(pl.len()).alias("ce_pct"),
            pl.col("name_jaccard").rank("average").truediv(pl.len()).alias("name_pct"),
        ).with_columns(
            (pl.col("ce_pct") + pl.col("name_pct") + 0.5 * pl.col("attr_conflict").cast(pl.Float32)).alias("hardness")
        ).sort(["hardness", "ce_oof", "row"], descending=[True, True, False])
        n = max(1, round(g.height * coverage))
        h = g.head(n)["row"].to_numpy()
        r = rng.choice(g["row"].to_numpy(), size=n, replace=False)
        hard_rows.extend(h.tolist()); random_rows.extend(r.tolist())
        by_cat.append({"category": cat[0], "negatives": g.height, "selected": n})
    hard = np.zeros(df.height, dtype=bool); hard[np.asarray(hard_rows)] = True
    random = np.zeros(df.height, dtype=bool); random[np.asarray(random_rows)] = True
    return hard, random, by_cat


def tokenize(df, tok, max_len):
    def mk(n, a, c): return f"{n} | {c} | {a}" if a else f"{n} | {c}"
    a = [mk(*x) for x in zip(df["name1"], df["attrs1"], df["category"])]
    b = [mk(*x) for x in zip(df["name2"], df["attrs2"], df["category"])]
    n = df.height
    ids = np.zeros((n, max_len), dtype=np.int32)
    tt = np.zeros((n, max_len), dtype=np.uint8)
    for s in range(0, n, 20000):
        e = min(s + 20000, n)
        enc = tok(a[s:e], b[s:e], truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="np")
        ids[s:e] = enc["input_ids"].astype(np.int32)
        if "token_type_ids" in enc: tt[s:e] = enc["token_type_ids"].astype(np.uint8)
    return ids, tt


def train_one(args, fold, mode, weight, df, ids, tt, y, selected, tok):
    ev = np.flatnonzero(df["fold"].to_numpy() == fold)
    tr = np.flatnonzero(df["fold"].to_numpy() != fold)
    model = AutoModelForSequenceClassification.from_pretrained(args.init, num_labels=1).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(tr) // args.bs * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.06, anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(args.seed)
    model.train(); step = 0; started = time.time()
    for _ in range(args.epochs):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr) - args.bs + 1, args.bs):
            idx = tr[np.sort(perm[s:s + args.bs])]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            by = torch.from_numpy(y[idx]).cuda()
            bw = torch.from_numpy(np.where(selected[idx], weight, 1.0).astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                loss = (lossf(logits, by) * bw).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{mode}{weight:g} {fold} step {step}/{steps} loss={loss.item():.4f} rate={rate:.2f}/s", flush=True)
    model.eval(); out = np.zeros(len(ev), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(ev), args.bs * 4):
            idx = ev[s:s + args.bs * 4]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
            out[s:s + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    exp = f"{mode}{int(weight)}x"
    outdir = Path(args.output) / "preds" / exp; outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id1": df["id1"][ev], "id2": df["id2"][ev], "predict": out}).write_csv(outdir / f"{fold}.csv")
    runtime = time.time() - started
    del model; torch.cuda.empty_cache()
    return runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--variants", default="hard2,hard4,random2,random4")
    ap.add_argument("--coverage", type=float, default=0.10); ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    df = pl.read_parquet(args.data); y = df["target"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(args.init)
    t0 = time.time(); ids, tt = tokenize(df, tok, args.max_len)
    print(f"tokenized rows={df.height} seconds={time.time()-t0:.1f}", flush=True)
    manifest = {"args": vars(args), "runs": []}
    for fold in args.folds.split(","):
        hard, random, by_cat = select_masks(df, fold, args.coverage, args.seed)
        manifest.setdefault("selection", {})[fold] = {
            "hard": int(hard.sum()), "random": int(random.sum()), "by_category": by_cat,
            "hard_ce_mean": float(df.filter(pl.Series(hard))["ce_oof"].mean()),
            "random_ce_mean": float(df.filter(pl.Series(random))["ce_oof"].mean()),
            "hard_name_mean": float(df.filter(pl.Series(hard))["name_jaccard"].mean()),
            "random_name_mean": float(df.filter(pl.Series(random))["name_jaccard"].mean()),
            "hard_conflict_rate": float(df.filter(pl.Series(hard))["attr_conflict"].mean()),
            "random_conflict_rate": float(df.filter(pl.Series(random))["attr_conflict"].mean()),
        }
        for spec in args.variants.split(","):
            mode = "hard" if spec.startswith("hard") else "random"
            weight = float(spec.removeprefix(mode))
            runtime = train_one(args, fold, mode, weight, df, ids, tt, y,
                                hard if mode == "hard" else random, tok)
            manifest["runs"].append({"fold": fold, "variant": spec, "runtime_seconds": runtime})
            Path(args.output, "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()

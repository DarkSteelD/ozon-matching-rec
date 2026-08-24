from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from normalize_units import normalize, self_check


def prepare(df):
    variants = {v: {f: [] for f in ("name1", "attrs1", "name2", "attrs2")}
                for v in ("baseline", "normalized", "corrupt")}
    families = {k: np.zeros(df.height, dtype=bool) for k in
                ("mass", "volume", "dimension", "count", "fashion_size")}
    examples, edits = [], 0
    for field in ("name1", "attrs1", "name2", "attrs2"):
        for row, source in enumerate(df[field]):
            source = source or ""
            good, bad, found = normalize(source)
            variants["baseline"][field].append(source)
            variants["normalized"][field].append(good)
            variants["corrupt"][field].append(bad)
            for edit in found:
                families[edit.family][row] = True; edits += 1
                if len(examples) < 100:
                    examples.append({"field": field, "source": edit.source,
                                     "normalized": edit.normalized, "corrupted": edit.corrupted,
                                     "family": edit.family})
            assert len(source) == len(good) == len(bad)
    row_any = np.logical_or.reduce(list(families.values()))
    audit = {"rows": df.height, "edited_rows": int(row_any.sum()), "edits": edits,
             "family_rows": {k: int(v.sum()) for k, v in families.items()}, "examples": examples}
    return variants, families, row_any, audit


def tokenize(df, fields, tok, max_len):
    cats = df["category"].to_list()
    def side(n, a):
        return [f"{name} | {cat} | {attr}" if attr else f"{name} | {cat}"
                for name, attr, cat in zip(fields[n], fields[a], cats)]
    a, b = side("name1", "attrs1"), side("name2", "attrs2")
    n = df.height; ids = np.zeros((n, max_len), np.int32); tt = np.zeros((n, max_len), np.uint8)
    for s in range(0, n, 20000):
        e = min(s + 20000, n)
        enc = tok(a[s:e], b[s:e], truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="np")
        ids[s:e] = enc["input_ids"].astype(np.int32)
        if "token_type_ids" in enc: tt[s:e] = enc["token_type_ids"].astype(np.uint8)
    return ids, tt


def run(args, variant, fold, df, ids, tt, tok):
    fold_col, y = df["fold"].to_numpy(), df["target"].to_numpy().astype(np.float32)
    tr, ev = np.flatnonzero(fold_col != fold), np.flatnonzero(fold_col == fold)
    model = AutoModelForSequenceClassification.from_pretrained(args.init, num_labels=1).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    steps = len(tr) // args.bs * args.epochs
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.06, anneal_strategy="linear")
    lossf = torch.nn.BCEWithLogitsLoss(); rng = np.random.default_rng(args.seed)
    started = time.time(); step = 0; model.train()
    for _ in range(args.epochs):
        perm = rng.permutation(len(tr))
        for s in range(0, len(tr) - args.bs + 1, args.bs):
            idx = tr[np.sort(perm[s:s + args.bs])]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            by = torch.from_numpy(y[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                loss = lossf(logits, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{variant} {fold} step {step}/{steps} loss={loss.item():.4f} rate={rate:.2f}/s", flush=True)
    model.eval(); out = np.zeros(len(ev), np.float32)
    with torch.no_grad():
        for s in range(0, len(ev), args.bs * 4):
            idx = ev[s:s + args.bs * 4]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                          token_type_ids=bt if use_tt else None).logits.squeeze(-1)
            out[s:s+len(idx)] = torch.sigmoid(z.float()).cpu().numpy()
    outdir = Path(args.output, "preds", variant); outdir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id1": df["id1"][ev], "id2": df["id2"][ev], "predict": out}).write_csv(outdir/f"{fold}.csv")
    runtime = time.time()-started; del model; torch.cuda.empty_cache(); return runtime


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--init",required=True)
    ap.add_argument("--output",required=True); ap.add_argument("--folds",default="fold_01,fold_02")
    ap.add_argument("--variants",default="baseline,normalized,corrupt"); ap.add_argument("--max-len",type=int,default=224)
    ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--bs",type=int,default=192)
    ap.add_argument("--lr",type=float,default=2e-5); ap.add_argument("--seed",type=int,default=20260814); args=ap.parse_args()
    self_check(); os.environ["TOKENIZERS_PARALLELISM"]="true"
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    df=pl.read_parquet(args.data); fields,families,row_any,audit=prepare(df)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    (root/"normalization_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    mask_cols={"unit_any":row_any}; mask_cols.update({f"unit_{k}":v for k,v in families.items()})
    pl.DataFrame({"id1":df["id1"],"id2":df["id2"],**mask_cols}).write_parquet(root/"slice_masks.parquet")
    tok=AutoTokenizer.from_pretrained(args.init); manifest={"args":vars(args),"runs":[]}
    for variant in args.variants.split(","):
        t=time.time(); ids,tt=tokenize(df,fields[variant],tok,args.max_len)
        print(f"tokenized {variant} in {time.time()-t:.1f}s",flush=True)
        for fold in args.folds.split(","):
            runtime=run(args,variant,fold,df,ids,tt,tok)
            manifest["runs"].append({"variant":variant,"fold":fold,"runtime_seconds":runtime})
            (root/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__": main()

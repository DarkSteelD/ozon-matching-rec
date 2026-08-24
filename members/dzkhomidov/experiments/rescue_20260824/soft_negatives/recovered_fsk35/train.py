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


def select_masks(df: pl.DataFrame, heldout: str, coverage: float, seed: int):
    train = df.with_row_index("row").filter((pl.col("fold") != heldout) & (pl.col("target") == 0))
    hard_rows, random_rows, categories = [], [], []
    rng = np.random.default_rng(seed + int(heldout[-2:]))
    for cat, group in train.group_by("category", maintain_order=True):
        group = group.sort(["ce_oof", "row"], descending=[True, False])
        n = max(1, round(group.height * coverage))
        hard = group.head(n)["row"].to_numpy()
        random = rng.choice(group["row"].to_numpy(), size=n, replace=False)
        hard_rows.extend(hard); random_rows.extend(random)
        categories.append({"category": cat[0], "train_negatives": group.height, "selected": n,
                           "hard_oof_mean": float(group.head(n)["ce_oof"].mean()),
                           "hard_oof_min": float(group.head(n)["ce_oof"].min()),
                           "random_oof_mean": float(group.filter(pl.col("row").is_in(random))["ce_oof"].mean())})
    hard = np.zeros(df.height, dtype=bool); hard[np.asarray(hard_rows)] = True
    random = np.zeros(df.height, dtype=bool); random[np.asarray(random_rows)] = True
    assert hard.sum() == random.sum()
    return hard, random, categories


def self_check():
    frame = pl.DataFrame({"fold": ["fold_02"] * 10 + ["fold_01"], "target": [0] * 11,
                          "category": ["x"] * 11, "ce_oof": list(range(10)) + [999]})
    hard, random, _ = select_masks(frame, "fold_01", 0.2, 7)
    assert hard.sum() == random.sum() == 2 and hard[8] and hard[9] and not hard[10]


def tokenize(df, tokenizer, max_len):
    def text(n, a, c): return f"{n} | {c} | {a}" if a else f"{n} | {c}"
    left = [text(*x) for x in zip(df["name1"], df["attrs1"], df["category"])]
    right = [text(*x) for x in zip(df["name2"], df["attrs2"], df["category"])]
    n = df.height; ids = np.zeros((n, max_len), np.int32); tt = np.zeros((n, max_len), np.uint8)
    for start in range(0, n, 20000):
        stop = min(start + 20000, n)
        enc = tokenizer(left[start:stop], right[start:stop], truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="np")
        ids[start:stop] = enc["input_ids"].astype(np.int32)
        if "token_type_ids" in enc: tt[start:stop] = enc["token_type_ids"].astype(np.uint8)
    return ids, tt


def train_fold(args, fold, variant, weight, selected, df, ids, tt, target, tokenizer):
    folds = df["fold"].to_numpy(); train_idx = np.flatnonzero(folds != fold); eval_idx = np.flatnonzero(folds == fold)
    model = AutoModelForSequenceClassification.from_pretrained(args.init, num_labels=1).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_idx) // args.batch_size * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps,
                                                    pct_start=0.06, anneal_strategy="linear")
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(args.seed); started = time.time(); step = 0; model.train()
    for _ in range(args.epochs):
        permutation = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx) - args.batch_size + 1, args.batch_size):
            idx = train_idx[np.sort(permutation[start:start + args.batch_size])]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda()
            bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            by = torch.from_numpy(target[idx]).cuda()
            bw = torch.from_numpy(np.where(selected[idx], weight, 1.0).astype(np.float32)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tokenizer.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                loss = (loss_fn(logits, by) * bw).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{variant} {fold} {step}/{steps} loss={loss.item():.5f} rate={rate:.2f}/s", flush=True)
    model.eval(); predictions = np.zeros(len(eval_idx), np.float32)
    with torch.no_grad():
        for start in range(0, len(eval_idx), args.batch_size * 4):
            idx = eval_idx[start:start + args.batch_size * 4]
            bi = torch.from_numpy(ids[idx].astype(np.int64)).cuda(); bt = torch.from_numpy(tt[idx].astype(np.int64)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tokenizer.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
            predictions[start:start + len(idx)] = torch.sigmoid(logits.float()).cpu().numpy()
    runtime = time.time() - started
    out = Path(args.output, "preds", variant); out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id1": df["id1"][eval_idx], "id2": df["id2"][eval_idx],
                  "predict": predictions}).write_csv(out / f"{fold}.csv")
    (out / f"{fold}.meta.json").write_text(json.dumps({"variant": variant, "fold": fold,
        "weight": weight, "host": os.uname().nodename, "pid": os.getpid(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "runtime_seconds": runtime,
        "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    del model; torch.cuda.empty_cache(); return runtime


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--oof-root", required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--folds", default="fold_01,fold_02"); ap.add_argument("--variants", default="baseline,hard075,random075,hard050,random050")
    ap.add_argument("--coverage", type=float, default=0.10); ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args(); os.environ["TOKENIZERS_PARALLELISM"] = "true"
    self_check()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    df = pl.read_parquet(args.data)
    parts = [pl.read_csv(path).select("id1", "id2", pl.col("predict").alias("ce_oof"))
             for path in sorted(Path(args.oof_root).glob("fold_*.csv"))]
    assert len(parts) == 4
    df = df.join(pl.concat(parts), on=["id1", "id2"], validate="1:1")
    assert df["ce_oof"].null_count() == 0 and df["target"].n_unique() == 2
    tokenizer = AutoTokenizer.from_pretrained(args.init); started = time.time()
    ids, tt = tokenize(df, tokenizer, args.max_len)
    print(f"host={os.uname().nodename} pid={os.getpid()} tokenized={df.height} seconds={time.time()-started:.1f}", flush=True)
    target = df["target"].to_numpy().astype(np.float32); manifest = {"args": vars(args), "selection": {}, "runs": []}
    for fold in args.folds.split(","):
        hard, random, categories = select_masks(df, fold, args.coverage, args.seed)
        manifest["selection"][fold] = {"hard_count": int(hard.sum()), "random_count": int(random.sum()),
            "categories": categories, "validation_rows_used_in_selection": 0}
        for variant in args.variants.split(","):
            if variant == "baseline": selected, weight = np.zeros(df.height, bool), 1.0
            else:
                selected = hard if variant.startswith("hard") else random
                weight = 0.75 if variant.endswith("075") else 0.50
            runtime = train_fold(args, fold, variant, weight, selected, df, ids, tt, target, tokenizer)
            manifest["runs"].append({"fold": fold, "variant": variant, "runtime_seconds": runtime})
            Path(args.output, "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__": main()

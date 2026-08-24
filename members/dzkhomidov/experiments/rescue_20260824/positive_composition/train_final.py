from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--effective-bs", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    assert args.effective_bs % args.micro_bs == 0
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    started = time.time()
    df = pl.read_parquet(args.data)
    n = df.height
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = []
    for side in (1, 2):
        values = []
        for name, attrs, category in zip(df[f"name{side}"], df[f"attrs{side}"], df["category"]):
            text = f"{name} | {category}"
            if attrs:
                text += f" | {attrs}"
            values.append(text)
        texts.append(values)

    def encode(first, second):
        ids = np.zeros((n, args.max_len), dtype=np.int32)
        token_types = np.zeros((n, args.max_len), dtype=np.uint8)
        for start in range(0, n, 20_000):
            stop = min(start + 20_000, n)
            batch = tokenizer(first[start:stop], second[start:stop], truncation=True,
                              max_length=args.max_len, padding="max_length", return_tensors="np")
            ids[start:stop] = batch["input_ids"].astype(np.int32)
            if "token_type_ids" in batch:
                token_types[start:stop] = batch["token_type_ids"].astype(np.uint8)
        return ids, token_types

    ids, tt = encode(texts[0], texts[1])
    ids_r, tt_r = encode(texts[1], texts[0])
    print(f"tokenized {n} both directions len={args.max_len} in {time.time()-started:.1f}s", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    labels = df["target"].to_numpy().astype(np.float32)
    steps_per_epoch = n // args.effective_bs
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.06,
        anneal_strategy="linear")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    step = 0
    train_started = time.time()
    model.train()
    for _ in range(args.epochs):
        permutation = rng.permutation(n)
        for start in range(0, steps_per_epoch * args.effective_bs, args.effective_bs):
            rows = permutation[start:start + args.effective_bs]
            reverse = rng.random(len(rows)) < 0.5
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, args.effective_bs, args.micro_bs):
                part = rows[micro:micro + args.micro_bs]
                rev = reverse[micro:micro + args.micro_bs]
                bi = torch.from_numpy(np.where(rev[:, None], ids_r[part], ids[part]).astype(np.int64)).cuda()
                bt = torch.from_numpy(np.where(rev[:, None], tt_r[part], tt[part]).astype(np.int64)).cuda()
                by = torch.from_numpy(labels[part]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi,
                                   attention_mask=(bi != tokenizer.pad_token_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                    loss = loss_fn(logits, by) * (len(part) / args.effective_bs)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - train_started)
                print(f"{step}/{total_steps} loss={loss.item():.5f} rate={rate:.2f}/s "+
                      f"eta={(total_steps-step)/rate/60:.1f}m", flush=True)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    manifest = {
        "status": "trained_on_all_rows",
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
        "init": str(args.init),
        "init_model_sha256": file_sha256(args.init / "model.safetensors"),
        "rows": n,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "effective_batch_size": args.effective_bs,
        "micro_batch_size": args.micro_bs,
        "learning_rate": args.lr,
        "seed": args.seed,
        "total_steps": total_steps,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    hashes = {path.name: file_sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()}
    (args.output / "HASHES.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

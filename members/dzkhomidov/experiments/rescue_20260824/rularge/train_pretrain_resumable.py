"""Exact-resume LLM-stage pretraining from an existing memmap token cache."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def save_state(path: Path, model, optimizer, scheduler, epoch: int, update: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "update": update,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=64)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    assert args.effective_batch % args.micro_batch == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = np.load(str(args.cache) + ".ids.npy", mmap_mode="r")
    targets = np.load(str(args.cache) + ".y.npy", mmap_mode="r")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    updates_per_epoch = len(targets) // args.effective_batch
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_updates,
        pct_start=0.06,
        anneal_strategy="linear",
    )
    state_path = args.save / "training_state.pt"
    start_epoch = start_update = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_update = int(state["epoch"]), int(state["update"])
        print(f"resumed epoch={start_epoch} update={start_update}", flush=True)

    accumulation = args.effective_batch // args.micro_batch
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    global_update = start_epoch * updates_per_epoch + start_update
    for epoch in range(start_epoch, args.epochs):
        permutation = np.random.default_rng(args.seed + epoch).permutation(len(targets))
        permutation = permutation[: updates_per_epoch * args.effective_batch]
        first_micro = start_update * accumulation if epoch == start_epoch else 0
        for micro_index in range(first_micro, updates_per_epoch * accumulation):
            start = micro_index * args.micro_batch
            idx = np.sort(permutation[start : start + args.micro_batch])
            batch_ids = torch.from_numpy(ids[idx].astype(np.int64)).cuda(non_blocking=True)
            batch_targets = torch.from_numpy(targets[idx].copy()).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch_ids,
                    attention_mask=(batch_ids != tokenizer.pad_token_id).long(),
                ).logits.squeeze(-1)
                loss = loss_fn(logits, batch_targets) / accumulation
            loss.backward()
            if (micro_index + 1) % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_update = micro_index // accumulation + 1
            global_update += 1
            if global_update % 100 == 0:
                rate = (global_update - (start_epoch * updates_per_epoch + start_update)) / (
                    time.time() - started
                )
                eta = (total_updates - global_update) / rate / 3600
                print(
                    f"update {global_update}/{total_updates} loss {loss.item() * accumulation:.4f} "
                    f"{rate:.3f} upd/s eta {eta:.2f}h",
                    flush=True,
                )
            if global_update % args.save_every == 0:
                save_state(state_path, model, optimizer, scheduler, epoch, epoch_update)
                print(f"exact state saved at update {global_update}", flush=True)
            if args.max_updates and global_update >= args.max_updates:
                elapsed = time.time() - started
                completed = global_update - (start_epoch * updates_per_epoch + start_update)
                rate = completed / elapsed
                projected_hours = total_updates / rate / 3600
                save_state(state_path, model, optimizer, scheduler, epoch, epoch_update)
                print(
                    f"benchmark stop at update {global_update}: {rate:.4f} upd/s, "
                    f"projected full ETA {projected_hours:.2f}h",
                    flush=True,
                )
                return
        start_update = 0
        save_state(state_path, model, optimizer, scheduler, epoch + 1, 0)

    final_dir = args.save / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved {final_dir}", flush=True)


if __name__ == "__main__":
    main()

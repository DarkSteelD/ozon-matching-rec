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


def build_tokens(data: str, tokenizer, max_length: int):
    frame = pl.read_parquet(data)

    def texts(names, attrs, categories):
        return [f"{name} | {category} | {attr}" if attr else f"{name} | {category}"
                for name, attr, category in zip(names, attrs, categories)]

    left = texts(frame["name1"].to_list(), frame["attrs1"].to_list(), frame["category"].to_list())
    right = texts(frame["name2"].to_list(), frame["attrs2"].to_list(), frame["category"].to_list())

    def encode(first, second):
        ids = np.zeros((frame.height, max_length), np.int32)
        token_types = np.zeros((frame.height, max_length), np.uint8)
        for start in range(0, frame.height, 20_000):
            end = min(start + 20_000, frame.height)
            batch = tokenizer(first[start:end], second[start:end], truncation=True,
                              max_length=max_length, padding="max_length", return_tensors="np")
            ids[start:end] = batch["input_ids"].astype(np.int32)
            if "token_type_ids" in batch:
                token_types[start:end] = batch["token_type_ids"].astype(np.uint8)
        return ids, token_types

    started = time.perf_counter()
    ids, token_types = encode(left, right)
    reverse_ids, reverse_types = encode(right, left)
    return frame, ids, token_types, reverse_ids, reverse_types, time.perf_counter() - started


def predict(model, indices, ids, token_types, reverse_ids, reverse_types, pad_id, batch_size):
    model.eval()
    output = np.zeros(len(indices), np.float32)
    use_token_types = getattr(model.config, "type_vocab_size", 0) > 1
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size * 4):
            index = indices[start:start + batch_size * 4]
            probability = None
            for values, types in ((ids, token_types), (reverse_ids, reverse_types)):
                batch_ids = torch.from_numpy(values[index].astype(np.int64)).cuda()
                batch_types = torch.from_numpy(types[index].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch_ids,
                                   attention_mask=(batch_ids != pad_id).long(),
                                   token_type_ids=batch_types if use_token_types else None).logits.squeeze(-1)
                current = torch.sigmoid(logits.float()).cpu().numpy()
                probability = current if probability is None else probability + current
            output[start:start + len(index)] = probability / 2
    model.train()
    return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--checkpoints", default="250,500,1000,full")
    parser.add_argument("--stop-step", default="full")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    frame, ids, types, reverse_ids, reverse_types, token_seconds = build_tokens(
        args.data, tokenizer, args.max_length
    )
    folds = frame["fold"].to_numpy()
    target = frame["target"].to_numpy().astype(np.float32)
    train_index = np.flatnonzero(folds != args.fold)
    eval_index = np.flatnonzero(folds == args.fold)
    full_steps = len(train_index) // args.batch_size * 2
    requested = [full_steps if value == "full" else int(value) for value in args.checkpoints.split(",")]
    stop_step = full_steps if args.stop_step == "full" else int(args.stop_step)
    requested = sorted(set(step for step in requested if step <= stop_step))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, ignore_mismatched_sizes=True, local_files_only=True
    ).cuda()
    if args.head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("classifier.")
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("classifier head parameters were not found")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    schedule_steps = stop_step if args.head_only else full_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=schedule_steps, pct_start=0.06, anneal_strategy="linear"
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    train_started = time.perf_counter()
    evaluation_seconds = 0.0
    written = []
    step = 0
    for _ in range(2):
        permutation = rng.permutation(len(train_index))
        for start in range(0, len(train_index) - args.batch_size + 1, args.batch_size):
            index = train_index[np.sort(permutation[start:start + args.batch_size])]
            mask = rng.random(len(index)) < 0.5
            batch_np = np.where(mask[:, None], reverse_ids[index], ids[index])
            type_np = np.where(mask[:, None], reverse_types[index], types[index])
            batch_ids = torch.from_numpy(batch_np.astype(np.int64)).cuda()
            batch_types = torch.from_numpy(type_np.astype(np.int64)).cuda()
            batch_target = torch.from_numpy(target[index]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch_ids,
                               attention_mask=(batch_ids != tokenizer.pad_token_id).long(),
                               token_type_ids=None).logits.squeeze(-1)
                loss = loss_fn(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step in requested:
                scores, eval_seconds = predict(model, eval_index, ids, types, reverse_ids,
                                                reverse_types, tokenizer.pad_token_id, args.batch_size)
                evaluation_seconds += eval_seconds
                label = "full" if step == full_steps else str(step)
                destination = output / f"{args.fold}_step_{label}.csv"
                pl.DataFrame({"id1": frame["id1"][eval_index], "id2": frame["id2"][eval_index],
                              "predict": scores}).write_csv(destination)
                written.append({"step": step, "label": label, "prediction": str(destination),
                                "eval_seconds": eval_seconds})
                print(f"checkpoint {label} written after {step} updates", flush=True)
            if step % 250 == 0:
                elapsed = time.perf_counter() - train_started
                print(f"step {step}/{stop_step} loss={loss.item():.4f} rate={step/elapsed:.2f} updates/s", flush=True)
            if step >= stop_step:
                break
        if step >= stop_step:
            break

    manifest = {
        "args": vars(args),
        "fold": args.fold,
        "train_rows": len(train_index),
        "eval_rows": len(eval_index),
        "full_steps": full_steps,
        "stop_step": stop_step,
        "tokenize_seconds": token_seconds,
        "train_plus_eval_seconds": time.perf_counter() - train_started,
        "evaluation_seconds": evaluation_seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "written": written,
    }
    suffix = "head_only" if args.head_only else "trajectory"
    (output / f"manifest_{args.fold}_{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

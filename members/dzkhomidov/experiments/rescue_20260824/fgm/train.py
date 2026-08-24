from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


VARIANTS = {"bce", "bce2x", "fgm05", "random05", "fgm1"}


def encode(tokenizer, left, right, max_len):
    n = len(left)
    ids = np.zeros((n, max_len), np.int32)
    token_types = np.zeros((n, max_len), np.uint8)
    for start in range(0, n, 20_000):
        end = min(start + 20_000, n)
        encoded = tokenizer(
            left[start:end], right[start:end], truncation=True, max_length=max_len,
            padding="max_length", return_tensors="np"
        )
        ids[start:end] = encoded["input_ids"].astype(np.int32)
        if "token_type_ids" in encoded:
            token_types[start:end] = encoded["token_type_ids"].astype(np.uint8)
    return ids, token_types


def tokenize(data, tokenizer, max_len):
    frame = pl.read_parquet(data)

    def side(number):
        return [
            f"{name} | {category}" + (f" | {attrs}" if attrs else "")
            for name, category, attrs in zip(
                frame[f"name{number}"], frame["category"], frame[f"attrs{number}"]
            )
        ]

    left, right = side(1), side(2)
    started = time.time()
    ids, token_types = encode(tokenizer, left, right, max_len)
    ids_reverse, token_types_reverse = encode(tokenizer, right, left, max_len)
    print(f"tokenized {len(left)} both directions in {time.time()-started:.1f}s", flush=True)
    return frame, ids, token_types, ids_reverse, token_types_reverse


def model_loss(model, batch_ids, batch_types, labels, pad_token_id, use_types):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=batch_ids,
            attention_mask=(batch_ids != pad_token_id).long(),
            token_type_ids=batch_types if use_types else None,
        ).logits.squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels)


def active_delta(embedding, epsilon, mode, fixed_random=None):
    gradient = embedding.grad
    active = torch.nonzero(gradient.abs().sum(dim=1) > 0, as_tuple=False).squeeze(1)
    direction = gradient[active] if mode == "fgm" else fixed_random[active]
    norm = direction.float().norm()
    assert active.numel() and torch.isfinite(norm) and norm > 0
    delta = direction * (epsilon / norm).to(direction.dtype)
    with torch.no_grad():
        embedding[active].add_(delta)
    return active, delta, float(norm)


def predict(model, rows, ids, types, ids_reverse, types_reverse, pad, batch_size):
    model.eval()
    use_types = getattr(model.config, "type_vocab_size", 0) > 1
    output = np.empty(len(rows), np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size * 4):
            index = rows[start:start + batch_size * 4]
            probability = 0.0
            for all_ids, all_types in ((ids, types), (ids_reverse, types_reverse)):
                batch_ids = torch.from_numpy(all_ids[index].astype(np.int64)).cuda()
                batch_types = torch.from_numpy(all_types[index].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        input_ids=batch_ids,
                        attention_mask=(batch_ids != pad).long(),
                        token_type_ids=batch_types if use_types else None,
                    ).logits.squeeze(-1)
                probability += torch.sigmoid(logits.float()).cpu().numpy()
            output[start:start + len(index)] = probability / 2
    return output


def run(args, variant, fold, frame, ids, types, ids_reverse, types_reverse, tokenizer):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    fold_column = frame["fold"].to_numpy()
    train_rows = np.flatnonzero(fold_column != fold)
    eval_rows = np.flatnonzero(fold_column == fold)
    targets = frame["target"].to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True
    ).cuda()
    use_types = getattr(model.config, "type_vocab_size", 0) > 1
    embedding = model.get_input_embeddings().weight
    fixed_random = None
    if variant == "random05":
        generator = torch.Generator(device=embedding.device).manual_seed(args.random_seed)
        fixed_random = torch.randn(
            embedding.shape, dtype=embedding.dtype, device=embedding.device, generator=generator
        )
    steps = len(train_rows) // args.batch_size * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear"
    )
    model.train()
    step = 0
    started = time.time()
    epsilon = 1.0 if variant == "fgm1" else 0.5
    two_pass = variant != "bce"
    perturb_norms = []
    for _ in range(args.epochs):
        permutation = rng.permutation(len(train_rows))
        for start in range(0, len(train_rows) - args.batch_size + 1, args.batch_size):
            index = train_rows[np.sort(permutation[start:start + args.batch_size])]
            reverse = rng.random(len(index)) < 0.5
            batch_ids = torch.from_numpy(
                np.where(reverse[:, None], ids_reverse[index], ids[index]).astype(np.int64)
            ).cuda()
            batch_types = torch.from_numpy(
                np.where(reverse[:, None], types_reverse[index], types[index]).astype(np.int64)
            ).cuda()
            labels = torch.from_numpy(targets[index]).cuda()
            clean_loss = model_loss(
                model, batch_ids, batch_types, labels, tokenizer.pad_token_id, use_types
            )
            (clean_loss * (0.5 if two_pass else 1.0)).backward()
            second_loss = None
            if two_pass:
                active = delta = None
                if variant.startswith("fgm"):
                    active, delta, raw_norm = active_delta(embedding, epsilon, "fgm")
                    perturb_norms.append(epsilon)
                elif variant == "random05":
                    active, delta, raw_norm = active_delta(
                        embedding, epsilon, "random", fixed_random
                    )
                    perturb_norms.append(epsilon)
                second_loss = model_loss(
                    model, batch_ids, batch_types, labels, tokenizer.pad_token_id, use_types
                )
                (second_loss * 0.5).backward()
                if active is not None:
                    with torch.no_grad():
                        embedding[active].sub_(delta)
            assert torch.isfinite(clean_loss) and (
                second_loss is None or torch.isfinite(second_loss)
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                second = "" if second_loss is None else f" second={second_loss.item():.5f}"
                print(
                    f"{variant} {fold} {step}/{steps} clean={clean_loss.item():.5f}{second} "
                    f"rate={rate:.2f}/s eta={(steps-step)/rate/60:.1f}m",
                    flush=True,
                )
    prediction = predict(
        model, eval_rows, ids, types, ids_reverse, types_reverse,
        tokenizer.pad_token_id, args.batch_size
    )
    output = args.output / "preds" / variant
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{fold}.csv").open("w") as handle:
        handle.write("id1,id2,predict\n")
        for left, right, probability in zip(
            frame["id1"].to_numpy()[eval_rows],
            frame["id2"].to_numpy()[eval_rows], prediction
        ):
            handle.write(f"{left},{right},{probability:.9f}\n")
    metadata = {
        "variant": variant, "fold": fold, "seed": args.seed,
        "random_seed": args.random_seed, "epsilon": epsilon if perturb_norms else 0.0,
        "passes_per_step": 2 if two_pass else 1,
        "loss_scale_per_pass": 0.5 if two_pass else 1.0,
        "train_rows": len(train_rows), "eval_rows": len(eval_rows), "steps": steps,
        "runtime_seconds": time.time() - started, "pid": os.getpid(),
        "observed_perturb_norm_min": min(perturb_norms) if perturb_norms else None,
        "observed_perturb_norm_max": max(perturb_norms) if perturb_norms else None,
    }
    (output / f"{fold}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{variant} {fold} written", flush=True)
    del model, fixed_random
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--folds", default="fold_01,fold_02")
    parser.add_argument("--max-len", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--random-seed", type=int, default=20260815)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    unknown = set(args.variants.split(",")) - VARIANTS
    assert not unknown, unknown
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model or args.init)
    values = tokenize(args.data, tokenizer, args.max_len)
    for variant in args.variants.split(","):
        for fold in args.folds.split(","):
            run(args, variant, fold, *values, tokenizer)


if __name__ == "__main__":
    main()

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


FOLDS = ["fold_01", "fold_02"]
FRACTIONS = (0.25, 0.5, 0.75, 0.875, 1.0)


def encode(tok, left, right, max_len):
    n = len(left)
    ids = np.zeros((n, max_len), dtype=np.int32)
    tt = np.zeros((n, max_len), dtype=np.uint8)
    for start in range(0, n, 20_000):
        end = min(start + 20_000, n)
        batch = tok(left[start:end], right[start:end], truncation=True,
                    max_length=max_len, padding="max_length", return_tensors="np")
        ids[start:end] = batch["input_ids"].astype(np.int32)
        if "token_type_ids" in batch:
            tt[start:end] = batch["token_type_ids"].astype(np.uint8)
    return ids, tt


def make_tokens(data, tokenizer, max_len):
    df = pl.read_parquet(data)
    def texts(side):
        out = []
        for name, attrs, cat in zip(df[f"name{side}"], df[f"attrs{side}"], df["category"]):
            text = f"{name} | {cat}"
            if attrs:
                text += f" | {attrs}"
            out.append(text)
        return out
    left, right = texts(1), texts(2)
    t0 = time.time()
    ids, tt = encode(tokenizer, left, right, max_len)
    ids_r, tt_r = encode(tokenizer, right, left, max_len)
    print(f"tokenized {len(left)} rows both directions in {time.time()-t0:.1f}s", flush=True)
    return df, ids, tt, ids_r, tt_r


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def mean_state(states):
    out = {}
    for key in states[0]:
        vals = [s[key] for s in states]
        if vals[0].is_floating_point():
            out[key] = torch.stack(vals).mean(0)
        else:
            out[key] = vals[-1].clone()
    return out


def predict(model, idx, ids, tt, ids_r, tt_r, pad_id, batch_size):
    model.eval()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    out = np.empty(len(idx), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size * 4):
            rows = idx[start:start + batch_size * 4]
            probs = 0.0
            for xids, xtt in ((ids, tt), (ids_r, tt_r)):
                bi = torch.from_numpy(xids[rows].astype(np.int64)).cuda()
                bt = torch.from_numpy(xtt[rows].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi, attention_mask=(bi != pad_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                probs = probs + torch.sigmoid(logits.float()).cpu().numpy()
            out[start:start + len(rows)] = probs / 2
    return out


def write_preds(path, id1, id2, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("id1,id2,predict\n")
        for a, b, p in zip(id1, id2, pred):
            f.write(f"{a},{b},{p:.9f}\n")


def run_fold(args, fold, df, ids, tt, ids_r, tt_r, tokenizer):
    fold_col = df["fold"].to_numpy()
    train_idx = np.flatnonzero(fold_col != fold)
    eval_idx = np.flatnonzero(fold_col == fold)
    y = df["target"].to_numpy().astype(np.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init, num_labels=1, ignore_mismatched_sizes=True).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    steps = len(train_idx) // args.batch_size * args.epochs
    save_steps = {max(1, round(frac * steps)): frac for frac in FRACTIONS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    checkpoints = {}
    ema = None
    ema_decay = args.ema_decay
    model.train()
    step = 0
    t0 = time.time()
    for _ in range(args.epochs):
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx) - args.batch_size + 1, args.batch_size):
            rows = train_idx[np.sort(perm[start:start + args.batch_size])]
            reverse = rng.random(len(rows)) < 0.5
            bi_np = np.where(reverse[:, None], ids_r[rows], ids[rows])
            bt_np = np.where(reverse[:, None], tt_r[rows], tt[rows])
            bi = torch.from_numpy(bi_np.astype(np.int64)).cuda()
            bt = torch.from_numpy(bt_np.astype(np.int64)).cuda()
            by = torch.from_numpy(y[rows]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tokenizer.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                loss = loss_fn(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if step >= round(0.5 * steps):
                params = [p.detach() for p in model.parameters()]
                if ema is None:
                    ema = [p.clone() for p in params]
                else:
                    torch._foreach_lerp_(ema, params, 1.0 - ema_decay)
            if step in save_steps:
                frac = save_steps[step]
                state = clone_state(model)
                checkpoints[frac] = state
                torch.save(state, args.output / "checkpoints" / f"{fold}_q{frac:g}.pt")
                print(f"{fold} saved fraction={frac:g} step={step}/{steps}", flush=True)
            if step % 500 == 0:
                rate = step / (time.time() - t0)
                print(f"{fold} step {step}/{steps} loss={loss.item():.5f} "
                      f"rate={rate:.2f}/s eta={(steps-step)/rate/60:.1f}m", flush=True)

    assert set(checkpoints) == set(FRACTIONS), (checkpoints.keys(), save_steps, steps)
    final_state = checkpoints[1.0]
    late_state = mean_state([checkpoints[0.75], checkpoints[0.875], checkpoints[1.0]])
    early_state = mean_state([checkpoints[0.25], checkpoints[0.5], checkpoints[0.75]])
    ema_state = clone_state(model)
    assert ema is not None
    for (name, _), avg in zip(model.named_parameters(), ema):
        ema_state[name] = avg.cpu().clone()
    derived = {"final": final_state, "late_avg": late_state,
               "early_avg": early_state, "ema": ema_state}
    for name, state in derived.items():
        torch.save(state, args.output / "checkpoints" / f"{fold}_{name}.pt")

    predictions = {}
    for name, state in derived.items():
        model.load_state_dict(state)
        pred = predict(model, eval_idx, ids, tt, ids_r, tt_r,
                       tokenizer.pad_token_id, args.batch_size)
        predictions[name] = pred
        write_preds(args.output / "preds" / name / f"{fold}.csv",
                    df["id1"].to_numpy()[eval_idx], df["id2"].to_numpy()[eval_idx], pred)
        print(f"{fold} evaluated {name}", flush=True)
    late_preds = []
    for frac in (0.75, 0.875, 1.0):
        model.load_state_dict(checkpoints[frac])
        late_preds.append(predict(model, eval_idx, ids, tt, ids_r, tt_r,
                                  tokenizer.pad_token_id, args.batch_size))
    predictions["late_pred_avg"] = np.mean(late_preds, axis=0)
    write_preds(args.output / "preds" / "late_pred_avg" / f"{fold}.csv",
                df["id1"].to_numpy()[eval_idx], df["id2"].to_numpy()[eval_idx],
                predictions["late_pred_avg"])
    np.savez_compressed(args.output / "diagnostics" / f"{fold}_late_preds.npz",
                        q75=late_preds[0], q875=late_preds[1], q100=late_preds[2])
    meta = {"fold": fold, "train_rows": len(train_idx), "eval_rows": len(eval_idx),
            "steps": steps, "save_steps": {str(k): v for k, v in save_steps.items()},
            "runtime_seconds": time.time() - t0, "ema_decay": ema_decay}
    (args.output / "diagnostics" / f"{fold}_run.json").write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    args = ap.parse_args()
    for sub in ("checkpoints", "preds", "diagnostics"):
        (args.output / sub).mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    df, ids, tt, ids_r, tt_r = make_tokens(args.data, tokenizer, args.max_len)
    for fold in args.folds.split(","):
        run_fold(args, fold, df, ids, tt, ids_r, tt_r, tokenizer)


if __name__ == "__main__":
    main()

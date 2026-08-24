from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


KEY_GROUPS = {
    "бренд": ("бренд", "brand", "производитель"),
    "модель": ("артикул", "модель", "партномер", "part number", "model", "sku", "код товара"),
    "цвет": ("цвет", "color"),
    "материал": ("материал", "material"),
    "количество": ("количество", "комплектация", "число предметов", "штук"),
    "объем": ("объем", "объём", "volume"),
    "вес": ("вес", "масса", "weight"),
}
NUM_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?:\s*[xх×]\s*\d+(?:[.,]\d+)?){0,2}\s*(?:мл|ml|л|l|г|гр|g|кг|kg|мм|mm|см|cm|м|m|шт)?", re.I)
PAIR_SUFFIX_RE = re.compile(r"\s*@@\s*сравнение:.*$", re.I | re.S)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower().replace("ё", "е")
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    return " ".join(re.sub(r"[^0-9a-zа-я.]+", " ", s).split())


def norm_number(token: str) -> str:
    t = norm(token).replace(" ", "")
    m = re.fullmatch(r"([0-9.]+)(мл|ml|л|l|г|гр|g|кг|kg)?", t)
    if not m:
        return t
    value, unit = float(m.group(1)), m.group(2) or ""
    scale = {"кг": 1000, "kg": 1000, "л": 1000, "l": 1000}.get(unit, 1)
    unit = {"кг": "g", "kg": "g", "гр": "g", "г": "g", "g": "g",
            "л": "ml", "l": "ml", "мл": "ml", "ml": "ml"}.get(unit, unit)
    value *= scale
    return f"{value:g}{unit}"


def parse_attrs(raw: str) -> dict[str, list[str]]:
    raw = PAIR_SUFFIX_RE.sub("", str(raw or ""))
    out: dict[str, list[str]] = {}
    for part in raw.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key, value = norm(key), norm(value)
        if key and value:
            out.setdefault(key, []).append(value)
    return out


def values_for(attrs: dict[str, list[str]], aliases: tuple[str, ...]) -> list[str]:
    vals = []
    for key, value in attrs.items():
        if any(alias in key for alias in aliases) and not ("бренд" in key and "страна" in key):
            vals.extend(value)
    return vals


def state(a: list[str], b: list[str]) -> str:
    if not a or not b:
        return "неизвестно"
    aa, bb = {norm(x) for x in a}, {norm(x) for x in b}
    return "совпал" if aa & bb else "различен"


def numeric_set(values: list[str]) -> set[str]:
    return {norm_number(x.group()) for value in values for x in NUM_RE.finditer(value)}


def pair_token(attrs1: str, attrs2: str) -> str:
    a, b = parse_attrs(attrs1), parse_attrs(attrs2)
    states = []
    grouped_keys = set()
    for label, aliases in KEY_GROUPS.items():
        av, bv = values_for(a, aliases), values_for(b, aliases)
        if label in {"количество", "объем", "вес"} and av and bv:
            an, bn = numeric_set(av), numeric_set(bv)
            value_state = "совпал" if an and bn and an & bn else "различен"
        else:
            value_state = state(av, bv)
        states.append(f"{label}={value_state}")
        grouped_keys.update(k for k in (*a.keys(), *b.keys()) if any(x in k for x in aliases))
    numeric_conflicts = 0
    numeric_matches = 0
    for key in sorted(set(a) & set(b) - grouped_keys):
        av, bv = numeric_set(a[key]), numeric_set(b[key])
        if av and bv:
            if av & bv:
                numeric_matches += 1
            else:
                numeric_conflicts += 1
    other = "неизвестно"
    if numeric_conflicts:
        other = f"различен_{min(numeric_conflicts, 9)}"
    elif numeric_matches:
        other = f"совпал_{min(numeric_matches, 9)}"
    states.append(f"прочие_числа={other}")
    return "@@ различия: " + "; ".join(states)


def self_check():
    assert norm("  Ёлка-12,5 ") == "елка 12.5"
    token = pair_token("бренд:Ёлка; вес:1,5 кг; цвет:красный",
                       "brand: елка; масса:1500 г; цвет товара:синий")
    assert "бренд=совпал" in token and "вес=совпал" in token and "цвет=различен" in token
    assert "модель=неизвестно" in pair_token("артикул:AB-12", "цвет:черный")


def build_texts(df: pl.DataFrame, variant: str, seed: int):
    attrs1, attrs2 = df["attrs1"].to_list(), df["attrs2"].to_list()
    blocks = [pair_token(a, b) for a, b in zip(attrs1, attrs2)]
    if variant == "shuffled":
        blocks = np.asarray(blocks, dtype=object)[np.random.default_rng(seed).permutation(len(blocks))].tolist()
    def one(name, attrs, cat, block):
        base = f"{name} | {cat} | {attrs}" if attrs else f"{name} | {cat}"
        return base if variant == "baseline" else f"{block} | {base}"
    left = [one(*x, block) for x, block in zip(zip(df["name1"], attrs1, df["category"]), blocks)]
    right = [one(*x, block) for x, block in zip(zip(df["name2"], attrs2, df["category"]), blocks)]
    return left, right, blocks


def tokenize(tok, left, right, max_len):
    n = len(left)
    ids = np.zeros((n, max_len), dtype=np.int32)
    tt = np.zeros((n, max_len), dtype=np.uint8)
    for start in range(0, n, 20000):
        stop = min(start + 20000, n)
        enc = tok(left[start:stop], right[start:stop], truncation=True, max_length=max_len,
                  padding="max_length", return_tensors="np")
        ids[start:stop] = enc["input_ids"].astype(np.int32)
        if "token_type_ids" in enc:
            tt[start:stop] = enc["token_type_ids"].astype(np.uint8)
    return ids, tt


def train_fold(args, fold, df, arrays, target, tok):
    ids, tt, ids_r, tt_r = arrays
    folds = df["fold"].to_numpy()
    train_idx, eval_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    model = AutoModelForSequenceClassification.from_pretrained(args.init, num_labels=1).cuda()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_idx) // args.batch_size * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.06, anneal_strategy="linear")
    rng = np.random.default_rng(args.seed)
    started, step = time.time(), 0
    model.train()
    for _ in range(args.epochs):
        perm = rng.permutation(len(train_idx))
        for start in range(0, len(train_idx) - args.batch_size + 1, args.batch_size):
            idx = train_idx[np.sort(perm[start:start + args.batch_size])]
            rev = rng.random(len(idx)) < 0.5
            bi = torch.from_numpy(np.where(rev[:, None], ids_r[idx], ids[idx]).astype(np.int64)).cuda()
            bt = torch.from_numpy(np.where(rev[:, None], tt_r[idx], tt[idx]).astype(np.int64)).cuda()
            by = torch.from_numpy(target[idx]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                               token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad(set_to_none=True); step += 1
            if step % 500 == 0:
                rate = step / (time.time() - started)
                print(f"{args.variant} {fold} {step}/{steps} loss={loss.item():.5f} rate={rate:.2f}/s", flush=True)
    model.eval(); pred = np.zeros(len(eval_idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(eval_idx), args.batch_size * 4):
            idx = eval_idx[start:start + args.batch_size * 4]
            total = 0
            for vi, vt in ((ids, tt), (ids_r, tt_r)):
                bi = torch.from_numpy(vi[idx].astype(np.int64)).cuda()
                bt = torch.from_numpy(vt[idx].astype(np.int64)).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=bi, attention_mask=(bi != tok.pad_token_id).long(),
                                   token_type_ids=bt if use_tt else None).logits.squeeze(-1)
                total += torch.sigmoid(logits.float()).cpu().numpy()
            pred[start:start + len(idx)] = total / 2
    runtime = time.time() - started
    del model; torch.cuda.empty_cache()
    return eval_idx, pred, runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("baseline", "structured", "shuffled"), required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--folds", default="fold_01,fold_02")
    ap.add_argument("--max-len", type=int, default=224); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=192); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=20260814)
    args = ap.parse_args()
    self_check()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    requested = [out / f"{fold}.csv" for fold in args.folds.split(",")]
    if any(path.exists() for path in requested):
        deadline = time.time() + 1800
        while not all(path.exists() for path in requested) and time.time() < deadline:
            time.sleep(10)
        if all(path.exists() for path in requested):
            print(f"all requested predictions already exist in {out}; skip", flush=True)
            return
    df = pl.read_parquet(args.data); tok = AutoTokenizer.from_pretrained(args.init)
    started = time.time(); left, right, blocks = build_texts(df, args.variant, args.seed)
    ids, tt = tokenize(tok, left, right, args.max_len)
    ids_r, tt_r = tokenize(tok, right, left, args.max_len)
    counts = Counter(x for block in blocks for x in re.findall(r"=(совпал|различен|неизвестно)", block))
    (out / "coverage.json").write_text(json.dumps({"rows": df.height, "state_counts": counts,
        "examples": blocks[:10]}, ensure_ascii=False, indent=2) + "\n")
    print(f"host={os.uname().nodename} pid={os.getpid()} variant={args.variant} tokenized={df.height} seconds={time.time()-started:.1f}", flush=True)
    target = df["target"].to_numpy().astype(np.float32)
    for fold in args.folds.split(","):
        idx, pred, runtime = train_fold(args, fold, df, (ids, tt, ids_r, tt_r), target, tok)
        pl.DataFrame({"id1": df["id1"][idx], "id2": df["id2"][idx], "predict": pred}).write_csv(out / f"{fold}.csv")
        (out / f"{fold}.meta.json").write_text(json.dumps({"args": vars(args), "fold": fold,
            "host": os.uname().nodename, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime_seconds": runtime}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

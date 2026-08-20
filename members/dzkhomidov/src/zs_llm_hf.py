"""Zero-shot LLM pair scoring, HF transformers backend (no generation).

One forward pass per batch; score = softmax over full vocab at the last
position, P(match) = p("1"-variants) / (p("1") + p("0")).

Usage:
  zs_llm_hf.py --model <dir> --data <pairs.parquet> --out <preds.csv>
               [--attrs-limit 500] [--batch-tokens 24000] [--limit N]
"""
import argparse
import time
from pathlib import Path

import polars as pl
import torch

from zs_llm import SYSTEM, USER_TMPL


def load_model(path: str):
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    last_err = None
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                path, dtype=torch.bfloat16, trust_remote_code=True,
                attn_implementation="sdpa",
            )
            print("loaded with", cls_name, flush=True)
            return tok, model
        except Exception as e:
            last_err = e
            print(f"{cls_name} failed: {str(e)[:200]}", flush=True)
    raise last_err


def answer_token_ids(tok) -> tuple[list[int], list[int]]:
    ones, zeros = set(), set()
    for variant, bucket in [("1", ones), (" 1", ones), ("0", zeros), (" 0", zeros)]:
        ids = tok.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            bucket.add(ids[0])
    if not ones or not zeros:
        raise RuntimeError("no single-token 1/0 ids")
    return sorted(ones), sorted(zeros)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attrs-limit", type=int, default=500)
    ap.add_argument("--batch-tokens", type=int, default=24000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--prompt-json", default=None,
                    help="json file {'system':..., 'user':...} overriding built-ins")
    args = ap.parse_args()

    global SYSTEM, USER_TMPL
    if args.prompt_json:
        import json as _json
        p = _json.load(open(args.prompt_json))
        SYSTEM, USER_TMPL = p["system"], p["user"]

    df = pl.read_parquet(args.data)
    if args.offset:
        df = df.slice(args.offset)
    if args.limit:
        df = df.head(args.limit)

    tok, model = load_model(args.model)
    model = model.to("cuda").eval()
    ones, zeros = answer_token_ids(tok)
    print("answer ids:", ones, zeros, flush=True)

    prompts = []
    for r in df.iter_rows(named=True):
        user = USER_TMPL.format(
            category=r["category"], name1=r["name1"] or "",
            attrs1=(r["attrs1"] or "")[:args.attrs_limit],
            name2=r["name2"] or "", attrs2=(r["attrs2"] or "")[:args.attrs_limit],
        )
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except Exception:
            try:
                text = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            except Exception:  # no system role support (gemma)
                merged = [{"role": "user",
                           "content": SYSTEM + "\n\n" + user}]
                text = tok.apply_chat_template(merged, tokenize=False,
                                               add_generation_prompt=True)
        prompts.append(text)

    enc_all = tok(prompts, add_special_tokens=False)["input_ids"]
    lens = [len(x) for x in enc_all]
    order = sorted(range(len(prompts)), key=lambda i: lens[i])
    print(f"{len(prompts)} prompts, tok len p50={sorted(lens)[len(lens)//2]} "
          f"max={max(lens)}", flush=True)

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    preds = [0.5] * len(prompts)
    t0 = time.time()
    i = 0
    done = 0
    ids_1 = torch.tensor(ones, device="cuda")
    ids_0 = torch.tensor(zeros, device="cuda")
    while i < len(order):
        j = i + 1
        maxlen = lens[order[i]]
        while j < len(order):
            cand = max(maxlen, lens[order[j]])
            if cand * (j - i + 1) > args.batch_tokens:
                break
            maxlen = cand
            j += 1
        idx = order[i:j]
        batch = tok([prompts[k] for k in idx], return_tensors="pt",
                    padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            try:
                # only last-position logits: full [B,T,V] OOMs on 256K vocabs
                logits = model(**batch, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**batch).logits[:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        p1 = probs[:, ids_1].sum(dim=-1)
        p0 = probs[:, ids_0].sum(dim=-1)
        score = (p1 / (p1 + p0 + 1e-9)).cpu().tolist()
        for k, s in zip(idx, score):
            preds[k] = s
        done += len(idx)
        i = j
        if done % 2000 < len(idx):
            dt = time.time() - t0
            print(f"{done}/{len(prompts)} {done/dt:.1f} pairs/s", flush=True)

    dt = time.time() - t0
    print(f"inference {dt:.0f}s ({len(prompts)/dt:.1f} pairs/s)", flush=True)
    out = df.select("fold", "id1", "id2", "target", "category").with_columns(
        pl.Series("predict", preds))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

"""Zero-shot LLM pair scoring with vLLM: P(same product) from first-token logprobs.

Prompt asks for a strict 1/0 answer; score = softmax(lp("1"), lp("0")) restricted
to the two answer tokens, read from top-20 logprobs of the first generated token.

Usage:
  zs_llm.py --model <hf_dir> --data <pairs.parquet> --out <preds.csv>
            [--attrs-limit 500] [--gpu-mem 0.85] [--max-model-len 2048]

Output CSV: fold,id1,id2,target,predict (plus category for metric slicing).
"""
import argparse
import math
import time
from pathlib import Path

import polars as pl

SYSTEM = (
    "Ты — эксперт по матчингу товаров маркетплейса. Твоя задача: определить, "
    "являются ли два предложения ОДНИМ И ТЕМ ЖЕ товаром (совпадают модель, "
    "цвет, размер, объём, комплектация). Разные продавцы, цены и формулировки "
    "названия не важны. Отвечай строго одной цифрой без пояснений: "
    "1 — один и тот же товар, 0 — разные товары."
)

USER_TMPL = (
    "Категория: {category}\n"
    "Товар A: {name1}\n"
    "Атрибуты A: {attrs1}\n"
    "Товар B: {name2}\n"
    "Атрибуты B: {attrs2}\n"
    "Это один и тот же товар? Ответ (1 или 0):"
)


def build_messages(df: pl.DataFrame, attrs_limit: int) -> list[list[dict]]:
    msgs = []
    for r in df.iter_rows(named=True):
        user = USER_TMPL.format(
            category=r["category"],
            name1=r["name1"] or "",
            attrs1=(r["attrs1"] or "")[:attrs_limit],
            name2=r["name2"] or "",
            attrs2=(r["attrs2"] or "")[:attrs_limit],
        )
        msgs.append([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ])
    return msgs


def prob_from_logprobs(logprobs: dict | None) -> float:
    """P(match) from the first generated token's top-k logprobs."""
    if not logprobs:
        return 0.5
    lp1 = lp0 = None
    floor = min(lp.logprob for lp in logprobs.values()) - 2.0
    for lp in logprobs.values():
        tok = (lp.decoded_token or "").strip()
        if tok == "1" and (lp1 is None or lp.logprob > lp1):
            lp1 = lp.logprob
        elif tok == "0" and (lp0 is None or lp.logprob > lp0):
            lp0 = lp.logprob
    if lp1 is None and lp0 is None:
        return 0.5
    if lp1 is None:
        lp1 = floor
    if lp0 is None:
        lp0 = floor
    return math.exp(lp1) / (math.exp(lp1) + math.exp(lp0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attrs-limit", type=int, default=500)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    df = pl.read_parquet(args.data)
    messages = build_messages(df, args.attrs_limit)
    print(f"{len(messages)} pairs, model={args.model}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    t0 = time.time()
    try:
        outs = llm.chat(messages, sp, chat_template_kwargs={"enable_thinking": False})
    except Exception as e:  # template without enable_thinking support
        print(f"retry without chat_template_kwargs: {e}", flush=True)
        outs = llm.chat(messages, sp)
    dt = time.time() - t0
    print(f"inference {dt:.0f}s ({len(messages) / dt:.1f} pairs/s)", flush=True)

    preds = [prob_from_logprobs(o.outputs[0].logprobs[0] if o.outputs[0].logprobs else None)
             for o in outs]
    out = df.select("fold", "id1", "id2", "target", "category").with_columns(
        pl.Series("predict", preds)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

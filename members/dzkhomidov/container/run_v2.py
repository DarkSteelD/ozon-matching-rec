"""E-CUP 2026 matching submission: cross-encoder pair scoring.

Reads models.json: [{"path": "models/<dir>", "max_len": 224, "weight": 1.0}, ...].
Scores every pair with each model (fp16 on CUDA when available); if several
models are listed, blends by weighted mean of per-model rank. A time guard
skips trailing models when the 20-minute budget would be exceeded.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

T0 = time.time()
TIME_BUDGET = 15.5 * 60  # leave margin inside the 20-min limit
ATTRS_LIMIT = 800


def elapsed() -> float:
    return time.time() - T0


def compact_attrs(raw) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(d, dict):
        return ""
    parts = []
    for k in sorted(d, key=str.lower):
        v = d[k]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v[:6])
        parts.append(f"{k}:{v}")
    return "; ".join(parts)[:ATTRS_LIMIT]


def build_texts(items: pd.DataFrame) -> dict:
    id2text = {}
    for iid, name, cat, attrs in zip(items["id"], items["name"],
                                     items["category"], items["attributes"]):
        t = str(name) if name is not None else ""
        if cat:
            t = f"{t} | {cat}"
        a = compact_attrs(attrs)
        if a:
            t = f"{t} | {a}"
        id2text[iid] = t
    return id2text


def score_model(spec, t1, t2, device):
    tok = AutoTokenizer.from_pretrained(spec["path"])
    model = AutoModelForSequenceClassification.from_pretrained(spec["path"])
    model = model.half().to(device) if device.type == "cuda" else model.to(device)
    model.eval()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    max_len = spec["max_len"]

    n = len(t1)
    # order by rough length for tight dynamic padding
    rough = np.array([len(a) + len(b) for a, b in zip(t1, t2)])
    order = np.argsort(rough, kind="stable")
    preds = np.zeros(n, dtype=np.float64)
    bs = 512 if device.type == "cuda" else 64
    done = 0
    t_model = time.time()
    with torch.inference_mode():
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            # degrade length if the running rate projects past the budget:
            # batches are length-sorted, so only the expensive tail is capped
            if done > 20000 and max_len > 144:
                rate = done / (time.time() - t_model)
                if elapsed() + (n - done) / rate > TIME_BUDGET * 0.95:
                    max_len = max(144, int(max_len * 0.75))
                    print(f"degrading max_len -> {max_len} "
                          f"(elapsed {elapsed():.0f}s)", flush=True)
            enc = tok([t1[i] for i in idx], [t2[i] for i in idx],
                      truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            if not use_tt:
                enc.pop("token_type_ids", None)
            logits = model(**enc).logits.squeeze(-1)
            preds[idx] = torch.sigmoid(logits.float()).cpu().numpy()
            done += len(idx)
            if done % (bs * 50) < bs:
                print(f"[{spec['path']}] {done}/{n} elapsed {elapsed():.0f}s",
                      flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items_path", required=True)
    ap.add_argument("--matches_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} elapsed {elapsed():.0f}s", flush=True)

    items = pd.read_parquet(args.items_path,
                            columns=["id", "name", "category", "attributes"])
    matches = pd.read_parquet(args.matches_path)[["id1", "id2"]]
    id2text = build_texts(items)
    t1 = [id2text.get(i, "") for i in matches["id1"]]
    t2 = [id2text.get(i, "") for i in matches["id2"]]
    print(f"{len(t1)} pairs, texts built, elapsed {elapsed():.0f}s", flush=True)

    specs = json.load(open("models.json"))
    ranks, weights = [], []
    per_model_cost = None
    for k, spec in enumerate(specs):
        if k > 0 and per_model_cost is not None \
                and elapsed() + per_model_cost * spec.get('cost', 1.0) * 1.1 > TIME_BUDGET:
            print(f"time guard: skipping {spec['path']} "
                  f"(elapsed {elapsed():.0f}s)", flush=True)
            continue
        t_start = time.time()
        preds = score_model(spec, t1, t2, device)
        per_model_cost = time.time() - t_start
        r = pd.Series(preds).rank().to_numpy() / len(preds)
        ranks.append(r * spec.get("weight", 1.0))
        weights.append(spec.get("weight", 1.0))
        print(f"{spec['path']} done in {per_model_cost:.0f}s", flush=True)

    final = np.sum(ranks, axis=0) / np.sum(weights) if len(ranks) > 1 else \
        (ranks[0] / weights[0])
    out = pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"],
                        "predict": final})
    out.to_csv(args.output_path, index=False)
    print(f"wrote {args.output_path}, total {elapsed():.0f}s", flush=True)


if __name__ == "__main__":
    main()

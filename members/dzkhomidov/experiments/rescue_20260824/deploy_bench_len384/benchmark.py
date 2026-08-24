from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).parent
MODEL = ROOT / "model"
ITEMS = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/items.parquet")
MATCHES = Path("/home/dzkhomidov/ozon-hack/repos/ozon-matching-rec/data/raw/matches.parquet")
MAX_LEN = 384
BATCH_SIZE = 512
ATTRS_LIMIT = 800
DIRECTIONS = int(os.environ.get("DIRECTIONS", "2"))


def compact_attrs(raw) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(data, dict):
        return ""
    parts = []
    for key in sorted(data, key=str.lower):
        value = data[key]
        if isinstance(value, list):
            value = ",".join(str(x) for x in value[:6])
        parts.append(f"{key}:{value}")
    return "; ".join(parts)[:ATTRS_LIMIT]


def main() -> None:
    started = time.perf_counter()
    items = pd.read_parquet(ITEMS, columns=["id", "name", "category", "attributes"])
    matches = pd.read_parquet(MATCHES, columns=["id1", "id2"])
    texts = {}
    for iid, name, category, attrs in zip(
            items["id"], items["name"], items["category"], items["attributes"]):
        text = str(name) if name is not None else ""
        if category:
            text += f" | {category}"
        attrs_text = compact_attrs(attrs)
        if attrs_text:
            text += f" | {attrs_text}"
        texts[iid] = text
    left = [texts.get(i, "") for i in matches["id1"]]
    right = [texts.get(i, "") for i in matches["id2"]]
    build_seconds = time.perf_counter() - started

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, local_files_only=True)
    model = model.half().cuda().eval()
    use_tt = getattr(model.config, "type_vocab_size", 0) > 1
    rough = np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int64)
    order = np.argsort(rough, kind="stable")
    preds = np.empty(len(order), dtype=np.float32)

    # Test the most expensive batch before committing to the full benchmark.
    preflight = order[-BATCH_SIZE:]
    with torch.inference_mode():
        for first, second in ((left, right), (right, left))[:DIRECTIONS]:
            enc = tokenizer([first[i] for i in preflight], [second[i] for i in preflight],
                            truncation=True, max_length=MAX_LEN, padding=True,
                            return_tensors="pt").to("cuda")
            if not use_tt:
                enc.pop("token_type_ids", None)
            model(**enc).logits
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    infer_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            avg = None
            for first, second in ((left, right), (right, left))[:DIRECTIONS]:
                enc = tokenizer([first[i] for i in idx], [second[i] for i in idx],
                                truncation=True, max_length=MAX_LEN, padding=True,
                                return_tensors="pt").to("cuda")
                if not use_tt:
                    enc.pop("token_type_ids", None)
                prob = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
                avg = prob if avg is None else avg + prob
            preds[idx] = avg / DIRECTIONS
            if start and start % (BATCH_SIZE * 100) == 0:
                print(f"{start}/{len(order)}", flush=True)
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_started
    total_seconds = time.perf_counter() - started
    result = {
        "host": "avi-ling-gpu03",
        "physical_gpu": 1,
        "model": str(MODEL),
        "pairs": len(order),
        "directions": DIRECTIONS,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "build_seconds": build_seconds,
        "inference_seconds": infer_seconds,
        "total_seconds": total_seconds,
        "pairs_per_second": len(order) / infer_seconds,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "prediction_sha256_float32": hashlib.sha256(preds.tobytes()).hexdigest(),
    }
    (ROOT / f"metrics_{DIRECTIONS}dir.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

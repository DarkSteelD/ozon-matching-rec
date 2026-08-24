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
import pyarrow.parquet as pq
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



PRIO = ['бренд','модель','артикул','код товара','партномер','цвет','название цвета','тип','материал']
DROP_FASHION = ['размер','российский размер','длина стельки']
FASHION = {'Обувь','Одежда','Галантерея и аксессуары','Ювелирные изделия'}


def parse_kv(attr):
    out = []
    if attr:
        for part in attr.split('; '):
            k, _, v = part.partition(':')
            if v:
                out.append((k.strip().lower(), v.strip()))
    return out


def prio_attrs(attr, fashion):
    kv = parse_kv(attr)
    if fashion:
        kv = [(k, v) for k, v in kv if not any(d in k for d in DROP_FASHION)]

    def idx(k):
        for i, p in enumerate(PRIO):
            if p in k:
                return i
        return len(PRIO)
    kv.sort(key=lambda x: (idx(x[0]), x[0]))
    return '; '.join(f'{k}:{v}' for k, v in kv)[:700]


def getv(kv, keys):
    for k, v in kv:
        if any(kk in k for kk in keys):
            return v.lower()
    return None


def cmp_tok(a, b):
    if a is None or b is None:
        return 'неизвестно'
    if a == b or (len(a) > 4 and (a in b or b in a)):
        return 'совпал'
    return 'различен'



import re as _re
_NUM = _re.compile(r"\d+(?:[.,]\d+)?")


def size_sets(kv):
    out = {}
    for k, v in kv:
        if 'размер' not in k or 'упаков' in k:
            continue
        vl = v.lower()
        if 'росс' in k or ' ru' in vl or k.endswith('ru'):
            sysn = 'ru'
        elif 'производител' in k or 'eu' in vl or 'us' in vl:
            sysn = 'mk'
        else:
            sysn = 'plain'
        nums = set(x.replace(',', '.') for x in _NUM.findall(v))
        if nums:
            out.setdefault(sysn, set()).update(nums)
    return out


def size_mismatch(s1, s2):
    for sysn in ('ru', 'plain', 'mk'):
        if sysn in s1 and sysn in s2:
            return not (s1[sysn] & s2[sysn])
    return False

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


def read_needed_items(path, needed_ids) -> pd.DataFrame:
    """Read all parquet row groups, retain only IDs that occur in matches."""
    chunks = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
            columns=["id", "name", "category", "attributes"], batch_size=500_000):
        frame = batch.to_pandas()
        frame = frame[frame["id"].isin(needed_ids)]
        if not frame.empty:
            chunks.append(frame)
    items = pd.concat(chunks, ignore_index=True)
    if items["id"].duplicated().any():
        raise ValueError("duplicate item IDs")
    missing = len(needed_ids) - items["id"].isin(needed_ids).sum()
    if missing:
        print(f"warning: {missing} needed item IDs absent", flush=True)
    return items


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
    if device.type == "cuda":
        # The grader's T4 has a much smaller memory envelope than our A100s.
        # Keep the inherited 512 batch for short experts, but cap long-context
        # batches before the first forward rather than relying on an OOM retry.
        bs = 128 if max_len > 288 else (256 if max_len > 192 else 512)
    else:
        bs = 64
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

    matches = pd.read_parquet(args.matches_path)[["id1", "id2"]]
    needed_ids = np.unique(np.concatenate((matches["id1"].to_numpy(),
                                           matches["id2"].to_numpy())))
    items = read_needed_items(args.items_path, needed_ids)
    print(f"retained {len(items)}/{len(needed_ids)} needed items, "
          f"elapsed {elapsed():.0f}s", flush=True)
    id2text = build_texts(items)
    id2cat = dict(zip(items["id"], items["category"]))
    id2attr = dict(zip(items["id"], items["attributes"]))
    t1 = [id2text.get(i, "") for i in matches["id1"]]
    t2 = [id2text.get(i, "") for i in matches["id2"]]
    specs_pre = json.load(open("models.json"))
    need_prio = any(s.get("texts") == "prio" for s in specs_pre)
    p1 = p2 = None
    penalty_flags = None
    if need_prio:
        p1, p2 = [], []
        penalty_flags = []
        cache = {}
        for i1, i2 in zip(matches["id1"], matches["id2"]):
            for iid in (i1, i2):
                if iid not in cache:
                    name = id2text.get(iid, "").split(" | ")[0]
                    cat = id2cat.get(iid) or ""
                    fashion = cat in FASHION
                    kv = parse_kv(compact_attrs(id2attr.get(iid)))
                    cache[iid] = (name, cat, kv,
                                  prio_attrs(compact_attrs(id2attr.get(iid)), fashion))
            n1, c1, kv1, a1 = cache[i1]
            n2, c2, kv2, a2 = cache[i2]
            diff = (" @@ сравнение: цвет=" + cmp_tok(getv(kv1, ["цвет"]), getv(kv2, ["цвет"]))
                    + "; артикул=" + cmp_tok(
                        getv(kv1, ["артикул", "модель", "код товара", "партномер"]),
                        getv(kv2, ["артикул", "модель", "код товара", "партномер"])))
            p1.append(f"{n1} | {c1} | {a1}{diff}")
            p2.append(f"{n2} | {c2} | {a2}{diff}")
            penalty_flags.append(
                c1 in FASHION and size_mismatch(size_sets(kv1), size_sets(kv2)))
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
        use_prio = spec.get("texts") == "prio"
        preds = score_model(spec, p1 if use_prio else t1, p2 if use_prio else t2, device)
        per_model_cost = time.time() - t_start
        r = pd.Series(preds).rank().to_numpy() / len(preds)
        ranks.append(r * spec.get("weight", 1.0))
        weights.append(spec.get("weight", 1.0))
        print(f"{spec['path']} done in {per_model_cost:.0f}s", flush=True)

    final = np.sum(ranks, axis=0) / np.sum(weights) if len(ranks) > 1 else \
        (ranks[0] / weights[0])
    if penalty_flags is not None:
        final = np.asarray(final, dtype=np.float64)
        mask = np.array(penalty_flags)
        final[mask] *= 0.25
        print(f"size penalty applied to {int(mask.sum())} fashion pairs", flush=True)
    out = pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"],
                        "predict": final})
    out.to_csv(args.output_path, index=False)
    print(f"wrote {args.output_path}, total {elapsed():.0f}s", flush=True)


if __name__ == "__main__":
    main()

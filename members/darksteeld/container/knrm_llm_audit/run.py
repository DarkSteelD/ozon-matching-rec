"""Container entry point — matching solution, KNRM over product names.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Output: CSV with columns id1, id2, predict — one row per input pair, in input
order.

The artifact ships vectors addressed by **token string**, because the test items
are different products and so carry a different vocabulary than training did.
This script builds an index table over whatever tokens the test file contains:
a token seen in training gets its trained vector, a token never seen gets the
deterministic vector derived from the string itself. That keeps the exact-match
kernel correct for unseen article codes — the same code on both sides of a pair
resolves to the same vector, cosine exactly 1.0 — which a single shared <unk>
row would destroy.

torch, pandas, numpy and pyarrow all ship in the image, so nothing is vendored
and nothing is downloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from knrm_model import KNRM, PAD_ID, encode_names, tokenize, vector_for_unknown  # noqa: E402


def log(message: str) -> None:
    print(f"[run] {message}", flush=True)


def build_embedding(names: list[str], shipped: dict[str, int], vectors: np.ndarray,
                    dim: int) -> tuple[dict[str, int], torch.Tensor, int, int]:
    """Index space over the test vocabulary, filled from the artifact or by hash."""
    test_tokens: dict[str, int] = {}
    for name in names:
        for token in tokenize(name):
            if token not in test_tokens:
                test_tokens[token] = len(test_tokens) + 1  # 0 stays PAD

    weight = np.zeros((len(test_tokens) + 1, dim), dtype=np.float32)
    seen = 0
    for token, index in test_tokens.items():
        row = shipped.get(token)
        if row is None:
            weight[index] = vector_for_unknown(token, dim)
        else:
            weight[index] = vectors[row]
            seen += 1
    return test_tokens, torch.from_numpy(weight), seen, len(test_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    started = time.time()
    torch.set_grad_enabled(False)
    artifact = json.loads((HERE / "artifact.json").read_text(encoding="utf-8"))
    blob = np.load(HERE / "model.npz", allow_pickle=True)
    shipped = {token: row for row, token in enumerate(blob["vocabulary"].tolist())}
    vectors = blob["vectors"].astype(np.float32)
    log(f"artifact: {len(shipped):,} trained tokens, dim {artifact['dim']}, "
        f"trained with torch {artifact['torch_version']} (running {torch.__version__})")

    items = pd.read_parquet(args.items_path, columns=["id", "name"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    names = items["name"].tolist()
    log(f"items {len(items):,} | pairs {len(matches):,} | loaded in {time.time() - started:.0f}s")

    test_tokens, weight, seen, total = build_embedding(
        names, shipped, vectors, artifact["dim"]
    )
    log(f"test vocabulary {total:,} tokens — {seen:,} trained ({100 * seen / max(total, 1):.1f}%), "
        f"{total - seen:,} unseen (deterministic vector from the token string)")

    model = KNRM(weight, sparse=False)
    model.head.weight.copy_(torch.from_numpy(blob["head_weight"]))
    model.head.bias.copy_(torch.from_numpy(blob["head_bias"]))
    model.norm.weight.copy_(torch.from_numpy(blob["bn_weight"]))
    model.norm.bias.copy_(torch.from_numpy(blob["bn_bias"]))
    model.norm.running_mean.copy_(torch.from_numpy(blob["bn_mean"]))
    model.norm.running_var.copy_(torch.from_numpy(blob["bn_var"]))
    model.norm.eps = float(blob["bn_eps"])
    model.eval()

    encoded = torch.from_numpy(encode_names(names, test_tokens, artifact["max_len"]))
    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_numpy())}
    id1 = matches["id1"].to_numpy()
    id2 = matches["id2"].to_numpy()
    known = np.fromiter(((int(a) in row_of_id and int(b) in row_of_id) for a, b in zip(id1, id2)),
                        dtype=bool, count=len(id1))
    rows1 = np.array([row_of_id.get(int(a), 0) for a in id1], dtype=np.int64)
    rows2 = np.array([row_of_id.get(int(b), 0) for b in id2], dtype=np.int64)

    predictions = np.full(len(id1), 0.5, dtype=np.float64)
    order = np.flatnonzero(known)
    for start in range(0, len(order), args.batch_size):
        pick = order[start : start + args.batch_size]
        predictions[pick] = torch.sigmoid(model(
            encoded[torch.from_numpy(rows1[pick])],
            encoded[torch.from_numpy(rows2[pick])],
        )).numpy()
    if not known.all():
        log(f"{int((~known).sum()):,} pairs scored 0.5 (items not in the items file)")

    pd.DataFrame({"id1": id1, "id2": id2, "predict": predictions}).to_csv(
        args.output_path, index=False
    )
    log(f"wrote {args.output_path} ({len(id1):,} rows) — total {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

"""Train the shipped KNRM model and export it in a token-addressed form.

The CV experiment trains four models and keeps none. This trains one on all
365,654 hand-labeled pairs, with the same early stopping (patience 1) on a
component-grouped slice held out of that pool, then exports everything the
container needs.

Exported as **token -> vector**, not as the index-addressed embedding table the
model trains with: at submit time the items are different products, so the
container builds its own index space over the test vocabulary and looks each
token up by string. Vectors are float16 (halves 316 MB to 158 MB; the cosines
that drive the kernels are unaffected at that precision).

navec is NOT shipped — the pretrained vectors are already baked into the trained
embeddings, so the container needs no pretrained file and no navec package.

    .venv/bin/python members/darksteeld/container/knrm_name_v2/build_artifact.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_model import (  # noqa: E402
    DIM, KERNEL_MU, KERNEL_SIGMA, KNRM, MAX_LEN, PAD_ID, encode_names, tokenize,
)

SEED = 20260814
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20
PATIENCE = 1
VALIDATION_BUCKETS = 10
NAVEC = REPOSITORY_ROOT / "members" / "darksteeld" / "models" / "navec_hudlit_v1_12B_500K_300d_100q.tar"


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return 0.0
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def predict(model, encoded, rows1, rows2, batch_size=2048):
    model.eval()
    out = np.empty(len(rows1), dtype=np.float64)
    for start in range(0, len(rows1), batch_size):
        stop = min(start + batch_size, len(rows1))
        out[start:stop] = torch.sigmoid(model(
            encoded[torch.from_numpy(rows1[start:stop])],
            encoded[torch.from_numpy(rows2[start:stop])],
        )).numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--navec", type=Path, default=NAVEC)
    args = parser.parse_args()

    import polars as pl
    from validation.build_folds import connected_component_keys

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()

    matches = pl.read_parquet(args.data_dir / "matches.parquet")
    items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "name"])
    names = items["name"].to_list()
    row_of_id = {int(i): r for r, i in enumerate(items["id"].to_list())}
    print(f"pairs {matches.height:,} | items {len(names):,}")

    # vocabulary: every distinct token, no OOV hashing (see knrm_model docstring)
    token_id: dict[str, int] = {}
    for name in names:
        for token in tokenize(name):
            if token not in token_id:
                token_id[token] = len(token_id) + 1  # 0 stays PAD
    print(f"vocabulary {len(token_id):,} tokens")

    weight = torch.empty(len(token_id) + 1, DIM)
    nn.init.normal_(weight, 0.0, 1.0, generator=torch.Generator().manual_seed(SEED))
    weight = F.normalize(weight, p=2, dim=-1)
    covered = 0
    if args.navec.is_file():
        from navec import Navec
        navec = Navec.load(str(args.navec))
        known = set(navec.vocab.words)
        rows = [(i, navec[t]) for t, i in token_id.items() if t in known]
        if rows:
            idx = torch.tensor([r for r, _ in rows], dtype=torch.long)
            vec = torch.from_numpy(np.asarray([v for _, v in rows], dtype=np.float32))
            weight[idx] = F.normalize(vec, p=2, dim=-1)
        covered = len(rows)
    print(f"pretrained rows {covered:,} ({100 * covered / len(token_id):.1f}%)")
    weight[PAD_ID] = 0.0

    encoded = torch.from_numpy(encode_names(names, token_id, MAX_LEN))
    id1 = matches["id1"].to_numpy()
    id2 = matches["id2"].to_numpy()
    rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int64)
    rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int64)
    labels = matches["target"].to_numpy().astype(np.float32)

    # early-stopping slice: whole connected components, same grouping unit as the folds
    component_of_item = connected_component_keys(id1, id2)
    bucket: dict[int, int] = {}
    is_validation = np.zeros(len(id1), dtype=bool)
    for position, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        if key not in bucket:
            digest = hashlib.sha256(f"{SEED}:submit:{key}".encode()).digest()
            bucket[key] = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
        is_validation[position] = bucket[key] == 0
    print(f"train {int((~is_validation).sum()):,} / val {int(is_validation.sum()):,} "
          f"(component-grouped)")

    model = KNRM(weight.clone(), sparse=True)
    embedding_optimizer = torch.optim.SparseAdam(model.embedding.parameters(), lr=LEARNING_RATE)
    dense_optimizer = torch.optim.Adam(
        list(model.norm.parameters()) + list(model.head.parameters()), lr=LEARNING_RATE
    )
    loss_function = nn.BCEWithLogitsLoss()
    train_rows1, train_rows2 = rows1[~is_validation], rows2[~is_validation]
    train_labels = torch.from_numpy(labels[~is_validation])
    generator = torch.Generator().manual_seed(SEED)

    best_score, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(train_labels), generator=generator)
        running, batches, epoch_started = 0.0, 0, time.time()
        for start in range(0, len(train_labels), BATCH_SIZE):
            pick = permutation[start : start + BATCH_SIZE]
            loss = loss_function(
                model(encoded[torch.from_numpy(train_rows1)[pick]],
                      encoded[torch.from_numpy(train_rows2)[pick]]),
                train_labels[pick],
            )
            embedding_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            embedding_optimizer.step()
            dense_optimizer.step()
            running += float(loss.detach())
            batches += 1
        score = average_precision(
            labels[is_validation],
            predict(model, encoded, rows1[is_validation], rows2[is_validation]),
        )
        improved = score > best_score
        print(f"  epoch {epoch:>2}  loss {running / batches:.5f}  val PR-AUC {score:.6f}"
              f"{'  *' if improved else ''}  {time.time() - epoch_started:.0f}s", flush=True)
        if improved:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"  early stop, restoring epoch {best_epoch}")
                break
    model.load_state_dict(best_state)
    model.eval()

    # export token -> vector plus the ranking layer
    trained = model.embedding.weight.detach().numpy()
    vocabulary = [""] * (len(token_id) + 1)
    for token, index in token_id.items():
        vocabulary[index] = token
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "model.npz",
        vectors=trained[1:].astype(np.float16),          # row i-1 belongs to vocabulary[i]
        vocabulary=np.array(vocabulary[1:], dtype=object),
        head_weight=model.head.weight.detach().numpy().astype(np.float32),
        head_bias=model.head.bias.detach().numpy().astype(np.float32),
        bn_weight=model.norm.weight.detach().numpy().astype(np.float32),
        bn_bias=model.norm.bias.detach().numpy().astype(np.float32),
        bn_mean=model.norm.running_mean.numpy().astype(np.float32),
        bn_var=model.norm.running_var.numpy().astype(np.float32),
        bn_eps=np.float32(model.norm.eps),
    )
    artifact = {
        "experiment": "knrm_name_v2",
        "pairs": int(len(labels)),
        "vocabulary": len(token_id),
        "dim": DIM,
        "max_len": MAX_LEN,
        "kernel_mu": list(KERNEL_MU),
        "kernel_sigma": list(KERNEL_SIGMA),
        "seed": SEED,
        "best_epoch": best_epoch,
        "validation_prauc": best_score,
        "pretrained": "navec_hudlit_v1_12B_500K_300d_100q (MIT), baked into the exported vectors",
        "pretrained_rows": covered,
        "torch_version": torch.__version__,
        "repo_commit": git_commit(REPOSITORY_ROOT),
        "local_cv": {"spec_v1_mean_prauc": 0.52987093, "spec_v2_mean_prauc": 0.53077533},
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    size = (args.out_dir / "model.npz").stat().st_size / 1e6
    print(f"\nmodel.npz {size:.1f} MB | best epoch {best_epoch} (val {best_score:.6f}) "
          f"| built in {time.time() - started:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()

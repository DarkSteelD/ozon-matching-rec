"""Train the shipped LLM-pretrained KNRM and export it token-addressed.

Two stages, matching the validated experiment
(``members/darksteeld/experiments/knrm_llm_pretrain``, mean PR-AUC 0.56557 on
the spec-v2 folds against 0.53078 without pretraining):

1. pretrain one epoch on all 11,187,780 ``matches_llm`` pairs with soft targets,
   over a vocabulary spanning the whole catalogue;
2. fine-tune on **all** 365,654 hand pairs — early stopping (patience 1) on a
   component-grouped slice held out of them — and export.

The pretrained state is cached, so re-exporting does not repeat the 11-minute
first stage.

**Vocabulary pruning.** Pretraining needs all 1,603,850 catalogue tokens, but
shipping them is 962 MB of fp16 vectors. Measured against a simulated test set
(770,582 catalogue items outside the hand universe), keeping hand tokens plus
catalogue tokens seen >= ``--ship-min-count`` times covers 96.1% of token
*occurrences* at 271 MB. What is dropped are tokens occurring under five times
in a 13.4M-item catalogue: they receive almost no gradient during pretraining,
so their vectors stay at initialisation — and initialisation is exactly what the
container recomputes for an unseen token, from the token string. Dropping them
therefore changes nothing except archive size. The export verifies this rather
than asserting it: it reports how far the dropped vectors actually moved.

    .venv/bin/python members/darksteeld/container/knrm_llm_pretrain/build_artifact.py \
        --cache-dir <scratch>/llmcache
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

EXPERIMENT = REPOSITORY_ROOT / "members" / "darksteeld" / "experiments" / "knrm_llm_pretrain"
sys.path.insert(0, str(EXPERIMENT))

from knrm_model import DIM, KNRM, MAX_LEN, PAD_ID, tokenize, vector_for_unknown  # noqa: E402
from train import (  # noqa: E402  — reuse the validated experiment's own code
    average_precision, build_vocabulary, encode_stream, initial_weight, predict, run_epoch,
)

SEED = 20260814
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
VALIDATION_BUCKETS = 10
NAVEC = REPOSITORY_ROOT / "members" / "darksteeld" / "models" / "navec_hudlit_v1_12B_500K_300d_100q.tar"


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--navec", type=Path, default=NAVEC)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--ship-min-count", type=int, default=5,
                        help="catalogue-frequency cut for shipped vectors; hand tokens always ship")
    args = parser.parse_args()

    import polars as pl
    from validation.build_folds import connected_component_keys

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- vocabulary over the whole catalogue --------------------------------
    counts_path = args.cache_dir / "big_token_counts.npz"
    if not counts_path.is_file():
        raise SystemExit(f"missing {counts_path}; run the experiment once first to build the cache")
    blob = np.load(counts_path, allow_pickle=True)
    counts_big = Counter(dict(zip(blob["tokens"].tolist(), blob["counts"].tolist())))

    hand_items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "name"])
    hand_names = hand_items["name"].to_list()
    tokens_hand = {t for name in hand_names for t in tokenize(name)}
    token_id = build_vocabulary(counts_big, tokens_hand, 1)
    print(f"vocabulary {len(token_id):,} | hand tokens {len(tokens_hand):,}", flush=True)

    # ---- stage 1: pretrain on matches_llm (cached) --------------------------
    checkpoint = args.cache_dir / f"pretrained_e{args.pretrain_epochs}.pt"
    weight, covered = initial_weight(token_id, DIM, args.navec)
    model = KNRM(weight, sparse=True)
    del weight
    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"pretrained weights from cache: {checkpoint.name}", flush=True)
    else:
        llm = pl.read_parquet(args.data_dir / "matches_llm.parquet")
        llm_ids = np.unique(np.concatenate([llm["id1"].to_numpy(), llm["id2"].to_numpy()]))
        if np.intersect1d(llm_ids, hand_items["id"].to_numpy()).size:
            raise AssertionError("LLM and hand universes overlap — pretraining would leak")
        encoded_path = args.cache_dir / "llm_encoded_mc1.npy"
        llm_encoded = (np.load(encoded_path) if encoded_path.is_file()
                       else encode_stream(args.data_dir / "items.parquet", llm_ids, token_id, MAX_LEN))
        if not encoded_path.is_file():
            np.save(encoded_path, llm_encoded)
        rows1 = np.searchsorted(llm_ids, llm["id1"].to_numpy()).astype(np.int32)
        rows2 = np.searchsorted(llm_ids, llm["id2"].to_numpy()).astype(np.int32)
        target = llm["target"].to_numpy().astype(np.float32)
        del llm
        optimizers = [
            torch.optim.SparseAdam(model.embedding.parameters(), lr=LEARNING_RATE),
            torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                             lr=LEARNING_RATE),
        ]
        encoded_t = torch.from_numpy(llm_encoded)
        generator = torch.Generator().manual_seed(SEED)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = run_epoch(model, optimizers, nn.BCEWithLogitsLoss(), encoded_t,
                             rows1, rows2, target, BATCH_SIZE, generator, f"pretrain e{epoch}")
            print(f"  pretrain epoch {epoch}: loss {loss:.5f}", flush=True)
        torch.save(model.state_dict(), checkpoint)
        del encoded_t, llm_encoded, rows1, rows2, target
    init_weight = model.embedding.weight.detach().clone()  # to measure how far tokens moved

    # ---- stage 2: fine-tune on every hand pair ------------------------------
    matches = pl.read_parquet(args.data_dir / "matches.parquet")
    hand_encoded = np.zeros((len(hand_names), MAX_LEN), dtype=np.int32)
    for row, name in enumerate(hand_names):
        for column, token in enumerate(tokenize(name)[:MAX_LEN]):
            hand_encoded[row, column] = token_id.get(token, PAD_ID)
    hand_encoded_t = torch.from_numpy(hand_encoded)
    row_of_id = {int(i): r for r, i in enumerate(hand_items["id"].to_list())}
    id1 = matches["id1"].to_numpy()
    id2 = matches["id2"].to_numpy()
    rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int32)
    rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int32)
    labels = matches["target"].to_numpy().astype(np.float32)

    component_of_item = connected_component_keys(id1, id2)
    bucket: dict[int, int] = {}
    is_validation = np.zeros(len(id1), dtype=bool)
    for position, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        if key not in bucket:
            digest = hashlib.sha256(f"{SEED}:submit:{key}".encode()).digest()
            bucket[key] = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
        is_validation[position] = bucket[key] == 0
    print(f"fine-tune: train {int((~is_validation).sum()):,} / val {int(is_validation.sum()):,}",
          flush=True)

    optimizers = [
        torch.optim.SparseAdam(model.embedding.parameters(), lr=LEARNING_RATE),
        torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                         lr=LEARNING_RATE),
    ]
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    tr1, tr2, trt = rows1[~is_validation], rows2[~is_validation], labels[~is_validation]
    best, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        permutation = torch.randperm(len(trt), generator=generator)
        running, batches = 0.0, 0
        for start in range(0, len(trt), BATCH_SIZE):
            pick = permutation[start : start + BATCH_SIZE].numpy()
            loss = loss_function(
                model(hand_encoded_t[torch.from_numpy(tr1[pick].astype(np.int64))].long(),
                      hand_encoded_t[torch.from_numpy(tr2[pick].astype(np.int64))].long()),
                torch.from_numpy(trt[pick]))
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            running += float(loss.detach()); batches += 1
        score = average_precision(
            labels[is_validation],
            predict(model, hand_encoded_t, rows1[is_validation], rows2[is_validation]))
        improved = score > best
        print(f"  finetune e{epoch}  loss {running / batches:.5f}  val {score:.6f}"
              f"{'  *' if improved else ''}", flush=True)
        if improved:
            best, best_epoch, waited = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    # ---- export: prune, and verify the pruning is free ----------------------
    trained = model.embedding.weight.detach().numpy()
    ship = tokens_hand | {t for t, c in counts_big.items() if c >= args.ship_min_count}
    dropped = [t for t in token_id if t not in ship]
    if dropped:
        sample = dropped[:: max(1, len(dropped) // 20000)][:20000]
        idx = np.array([token_id[t] for t in sample])
        a = trained[idx]
        b = init_weight.numpy()[idx]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        print(f"\ndropped {len(dropped):,} low-frequency tokens; on a {len(sample):,} sample their "
              f"vectors moved by cosine-to-init: mean {cos.mean():.6f}, min {cos.min():.6f}, "
              f"{100 * (cos > 0.999).mean():.1f}% unchanged beyond 0.999")

    shipped_tokens = sorted(ship)
    rows = np.array([token_id[t] for t in shipped_tokens])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "model.npz",
        vectors=trained[rows].astype(np.float16),
        vocabulary=np.array(shipped_tokens, dtype=object),
        head_weight=model.head.weight.detach().numpy().astype(np.float32),
        head_bias=model.head.bias.detach().numpy().astype(np.float32),
        bn_weight=model.norm.weight.detach().numpy().astype(np.float32),
        bn_bias=model.norm.bias.detach().numpy().astype(np.float32),
        bn_mean=model.norm.running_mean.numpy().astype(np.float32),
        bn_var=model.norm.running_var.numpy().astype(np.float32),
        bn_eps=np.float32(model.norm.eps),
    )
    artifact = {
        "experiment": "knrm_llm_pretrain",
        "pairs_pretrain": 11_187_780,
        "pairs_finetune": int(len(labels)),
        "vocabulary_trained": len(token_id),
        "vocabulary_shipped": len(shipped_tokens),
        "ship_min_count": args.ship_min_count,
        "dim": DIM,
        "max_len": MAX_LEN,
        "seed": SEED,
        "pretrain_epochs": args.pretrain_epochs,
        "best_finetune_epoch": best_epoch,
        "validation_prauc": best,
        "navec_rows": covered,
        "torch_version": torch.__version__,
        "repo_commit": git_commit(REPOSITORY_ROOT),
        "local_cv": {
            "spec_v2_mean_prauc": 0.56556775,
            "spec_v2_zero_shot_mean_prauc": 0.474411,
            "control_knrm_name_v2": 0.53077533,
        },
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    size = (args.out_dir / "model.npz").stat().st_size / 1e6
    print(f"model.npz {size:.0f} MB | shipped {len(shipped_tokens):,} of {len(token_id):,} tokens "
          f"| best epoch {best_epoch} (val {best:.6f}) | {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

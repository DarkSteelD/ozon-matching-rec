"""KNRM pretrained on the LLM-labeled pairs, then fine-tuned on the hand folds.

`matches_llm.parquet` holds 11,187,780 pairs over 12,384,610 items — 30x the
pairs and 17x the items of the hand-labeled set — with soft targets in ten steps
from 0.0 to 1.0 (2,779,162 of them strictly between the extremes, so BCE
consumes them as distillation targets directly).

**No leakage.** The LLM item universe and the hand item universe are disjoint:
0 of 12,384,610 LLM items appear in `items_human.parquet` (re-measured here, not
taken on trust). Pretraining on every LLM pair therefore cannot touch any hand
fold, and the frozen folds stay valid.

Three numbers come out of one run, because the interesting one is free:

* ``zero_shot`` — pretrained on LLM only, evaluated on a hand fold with no hand
  label ever seen. This is the *train-on-LLM -> eval-on-hand* experiment the
  repository lists as the only way to measure the LLM label shift
  (README "Статус вопросов первого дня" §4).
* ``finetuned`` — the same weights then fine-tuned on the three training folds,
  early stopping (patience 1) on a component-grouped slice, predicting the
  held-out fold. This is what gets registered.
* the control is ``knrm_name_v2`` at mean 0.53078 on the same spec-v2 folds.

Vocabulary spans both universes so pretrained vectors transfer by index: every
hand token is kept (they are what the model finally predicts on), plus big-file
tokens above ``--min-count``. Rare LLM-only tokens are dropped on purpose — they
never appear at hand-evaluation time and the embedding table has to fit
alongside SparseAdam, which keeps two dense moment tensors its size.

    .venv/bin/python members/darksteeld/experiments/knrm_llm_pretrain/train.py \
        --targets-dir validation/targets_v2 \
        --out-dir validation/predictions_v2/darksteeld/knrm_llm_pretrain
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_model import DIM, KNRM, MAX_LEN, PAD_ID, tokenize, vector_for_unknown  # noqa: E402

SEED = 20260814
VALIDATION_BUCKETS = 10


def log(message: str) -> None:
    print(message, flush=True)


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


def build_vocabulary(counts_big: Counter, tokens_hand: set[str], min_count: int) -> dict[str, int]:
    """Every hand token survives; big-file tokens need min_count occurrences."""
    keep = {token for token, count in counts_big.items() if count >= min_count}
    keep |= tokens_hand
    return {token: index for index, token in enumerate(sorted(keep), start=1)}  # 0 = PAD


def encode_stream(parquet_path: Path, wanted_ids: np.ndarray, token_id: dict[str, int],
                  max_len: int, batch_size: int = 200_000) -> np.ndarray:
    """(len(wanted_ids), max_len) int32, filled by streaming the items file.

    ``wanted_ids`` must be sorted; membership and row lookup are one searchsorted
    each, which keeps 12.4M items addressable without a Python dict.
    """
    import pyarrow.parquet as pq

    encoded = np.zeros((len(wanted_ids), max_len), dtype=np.int32)
    seen, scanned, started = 0, 0, time.time()
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=batch_size,
                                                           columns=["id", "name"]):
        ids = np.asarray(batch.column("id"), dtype=np.int64)
        position = np.searchsorted(wanted_ids, ids)
        position[position >= len(wanted_ids)] = 0
        hit = wanted_ids[position] == ids
        if hit.any():
            names = batch.column("name").to_pylist()
            for local in np.flatnonzero(hit).tolist():
                row = position[local]
                for column, token in enumerate(tokenize(names[local])[:max_len]):
                    encoded[row, column] = token_id.get(token, PAD_ID)
                seen += 1
        scanned += batch.num_rows
        if scanned % 2_000_000 == 0:
            log(f"    scanned {scanned:,}, matched {seen:,}, {time.time() - started:.0f}s")
    log(f"  encoded {seen:,}/{len(wanted_ids):,} items in {time.time() - started:.0f}s")
    return encoded


def initial_weight(token_id: dict[str, int], dim: int, navec_path: Path | None) -> tuple[torch.Tensor, int]:
    """navec where available, otherwise the deterministic per-token vector.

    Using ``vector_for_unknown`` for the rest (instead of a torch RNG) makes a
    token's initial vector a pure function of its string in every context —
    training here, and the container's handling of tokens it has never seen.
    """
    weight = np.zeros((len(token_id) + 1, dim), dtype=np.float32)
    covered = 0
    pretrained: dict[str, np.ndarray] = {}
    if navec_path is not None and navec_path.is_file():
        from navec import Navec

        navec = Navec.load(str(navec_path))
        known = set(navec.vocab.words)
        for token in token_id:
            if token in known:
                pretrained[token] = navec[token]
        covered = len(pretrained)
    for token, index in token_id.items():
        vector = pretrained.get(token)
        if vector is None:
            weight[index] = vector_for_unknown(token, dim)
        else:
            norm = max(float(np.linalg.norm(vector)), 1e-12)
            weight[index] = np.asarray(vector, dtype=np.float32) / norm
    weight[PAD_ID] = 0.0
    return torch.from_numpy(weight), covered


@torch.no_grad()
def predict(model: KNRM, encoded: torch.Tensor, rows1: np.ndarray, rows2: np.ndarray,
            batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = np.empty(len(rows1), dtype=np.float64)
    for start in range(0, len(rows1), batch_size):
        stop = min(start + batch_size, len(rows1))
        out[start:stop] = torch.sigmoid(model(
            encoded[torch.from_numpy(rows1[start:stop].astype(np.int64))].long(),
            encoded[torch.from_numpy(rows2[start:stop].astype(np.int64))].long(),
        )).numpy()
    return out


def run_epoch(model, optimizers, loss_function, encoded, rows1, rows2, target,
              batch_size, generator, label) -> float:
    model.train()
    permutation = torch.randperm(len(target), generator=generator)
    running, batches, started = 0.0, 0, time.time()
    report = max(1, len(target) // batch_size // 10)
    for step, start in enumerate(range(0, len(target), batch_size)):
        pick = permutation[start : start + batch_size].numpy()
        loss = loss_function(
            model(encoded[torch.from_numpy(rows1[pick].astype(np.int64))].long(),
                  encoded[torch.from_numpy(rows2[pick].astype(np.int64))].long()),
            torch.from_numpy(target[pick]),
        )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        running += float(loss.detach())
        batches += 1
        if step and step % report == 0:
            log(f"    {label} {100 * step * batch_size / len(target):3.0f}%  "
                f"loss {running / batches:.5f}  {time.time() - started:.0f}s")
    return running / max(batches, 1)


def load_folds(targets_dir: Path, fold_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    folds = {}
    for fold_id in fold_ids:
        path = targets_dir / f"{fold_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        id1, id2, target = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                id1.append(int(row["id1"]))
                id2.append(int(row["id2"]))
                target.append(float(row["target"]))
        folds[fold_id] = {"id1": np.asarray(id1, dtype=np.int64),
                          "id2": np.asarray(id2, dtype=np.int64),
                          "target": np.asarray(target, dtype=np.float32)}
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2" / "darksteeld" / "knrm_llm_pretrain")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="where the encoded LLM items are cached between runs")
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04")
    parser.add_argument("--min-count", type=int, default=5,
                        help="occurrences required for a big-file token; hand tokens always kept")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--pretrain-pairs", type=int, default=0,
                        help="subsample the LLM pairs (0 = all 11.19M)")
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--navec", type=Path,
                        default=REPOSITORY_ROOT / "members" / "darksteeld" / "models"
                        / "navec_hudlit_v1_12B_500K_300d_100q.tar")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = [f.strip() for f in args.folds.split(",") if f.strip()]
    import polars as pl
    from validation.build_folds import connected_component_keys

    # ---------------- vocabulary over both universes -------------------------
    started = time.time()
    counts_path = args.cache_dir / "big_token_counts.npz"
    hand_items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "name"])
    hand_names = hand_items["name"].to_list()
    tokens_hand = {t for name in hand_names for t in tokenize(name)}
    if counts_path.is_file():
        blob = np.load(counts_path, allow_pickle=True)
        counts_big = Counter(dict(zip(blob["tokens"].tolist(), blob["counts"].tolist())))
        log(f"big-file token counts from cache: {len(counts_big):,}")
    else:
        import pyarrow.parquet as pq
        counts_big = Counter()
        scanned = 0
        for batch in pq.ParquetFile(args.data_dir / "items.parquet").iter_batches(
                batch_size=200_000, columns=["name"]):
            for name in batch.column("name").to_pylist():
                counts_big.update(tokenize(name))
            scanned += batch.num_rows
            if scanned % 2_000_000 == 0:
                log(f"  counting: {scanned:,} names, {len(counts_big):,} tokens, "
                    f"{time.time() - started:.0f}s")
        np.savez_compressed(counts_path,
                            tokens=np.array(list(counts_big), dtype=object),
                            counts=np.array(list(counts_big.values()), dtype=np.int64))
        log(f"counted {scanned:,} names -> {len(counts_big):,} tokens, cached")

    token_id = build_vocabulary(counts_big, tokens_hand, args.min_count)
    log(f"vocabulary {len(token_id):,} (hand {len(tokens_hand):,} all kept, "
        f"big-file min_count={args.min_count}) | table "
        f"{(len(token_id) + 1) * DIM * 4 / 1e9:.2f} GB, x3 with SparseAdam")

    weight, covered = initial_weight(token_id, DIM, args.navec)
    log(f"navec-initialised rows {covered:,} ({100 * covered / len(token_id):.1f}%); "
        f"rest deterministic from the token string | setup {time.time() - started:.0f}s")

    # ---------------- phase 1: pretrain on matches_llm -----------------------
    llm = pl.read_parquet(args.data_dir / "matches_llm.parquet")
    llm_ids = np.unique(np.concatenate([llm["id1"].to_numpy(), llm["id2"].to_numpy()]))
    overlap = np.intersect1d(llm_ids, hand_items["id"].to_numpy()).size
    if overlap:
        raise AssertionError(f"LLM and hand universes overlap in {overlap} items — pretraining would leak")
    log(f"LLM pairs {llm.height:,} over {len(llm_ids):,} items | "
        f"overlap with the hand universe: {overlap} (verified, not assumed)")

    encoded_path = args.cache_dir / f"llm_encoded_mc{args.min_count}.npy"
    if encoded_path.is_file():
        llm_encoded = np.load(encoded_path)
        log(f"encoded LLM items from cache {llm_encoded.shape}")
    else:
        log("encoding LLM item names (streaming items.parquet)...")
        llm_encoded = encode_stream(args.data_dir / "items.parquet", llm_ids, token_id, MAX_LEN)
        np.save(encoded_path, llm_encoded)
    llm_rows1 = np.searchsorted(llm_ids, llm["id1"].to_numpy()).astype(np.int32)
    llm_rows2 = np.searchsorted(llm_ids, llm["id2"].to_numpy()).astype(np.int32)
    llm_target = llm["target"].to_numpy().astype(np.float32)
    if args.pretrain_pairs and args.pretrain_pairs < len(llm_target):
        pick = np.random.default_rng(SEED).choice(len(llm_target), args.pretrain_pairs, replace=False)
        llm_rows1, llm_rows2, llm_target = llm_rows1[pick], llm_rows2[pick], llm_target[pick]
        log(f"subsampled to {len(llm_target):,} LLM pairs")
    del llm

    model = KNRM(weight, sparse=True)
    del weight  # the table is ~2 GB; SparseAdam adds two dense moments its size
    optimizers = [
        torch.optim.SparseAdam(model.embedding.parameters(), lr=args.learning_rate),
        torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                         lr=args.learning_rate),
    ]
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    encoded_llm_t = torch.from_numpy(llm_encoded)
    for epoch in range(1, args.pretrain_epochs + 1):
        loss = run_epoch(model, optimizers, loss_function, encoded_llm_t,
                         llm_rows1, llm_rows2, llm_target, args.batch_size, generator,
                         f"pretrain e{epoch}")
        log(f"  pretrain epoch {epoch}: loss {loss:.5f}")
    pretrained_state = copy.deepcopy(model.state_dict())
    del encoded_llm_t, llm_encoded, llm_rows1, llm_rows2, llm_target

    # ---------------- phase 2: hand folds ------------------------------------
    hand_encoded = np.zeros((len(hand_names), MAX_LEN), dtype=np.int32)
    for row, name in enumerate(hand_names):
        for column, token in enumerate(tokenize(name)[:MAX_LEN]):
            hand_encoded[row, column] = token_id.get(token, PAD_ID)
    hand_encoded_t = torch.from_numpy(hand_encoded)
    row_of_id = {int(i): r for r, i in enumerate(hand_items["id"].to_list())}

    folds = load_folds(args.targets_dir, fold_ids)
    rows1 = {f: np.array([row_of_id[i] for i in folds[f]["id1"].tolist()], dtype=np.int32) for f in fold_ids}
    rows2 = {f: np.array([row_of_id[i] for i in folds[f]["id2"].tolist()], dtype=np.int32) for f in fold_ids}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zero_shot, finetuned = {}, {}
    for held_out in fold_ids:
        model.load_state_dict(pretrained_state)
        zero_shot[held_out] = average_precision(
            folds[held_out]["target"],
            predict(model, hand_encoded_t, rows1[held_out], rows2[held_out]),
        )
        log(f"\n  {held_out} zero-shot (LLM only, no hand label seen): "
            f"PR-AUC {zero_shot[held_out]:.6f}")

        train_ids = [f for f in fold_ids if f != held_out]
        pool_id1 = np.concatenate([folds[f]["id1"] for f in train_ids])
        pool_rows1 = np.concatenate([rows1[f] for f in train_ids])
        pool_rows2 = np.concatenate([rows2[f] for f in train_ids])
        pool_target = np.concatenate([folds[f]["target"] for f in train_ids])
        component_of_item = connected_component_keys(
            pool_id1, np.concatenate([folds[f]["id2"] for f in train_ids]))
        bucket: dict[int, int] = {}
        is_validation = np.zeros(len(pool_id1), dtype=bool)
        for position, item in enumerate(pool_id1.tolist()):
            key = component_of_item[item]
            if key not in bucket:
                digest = hashlib.sha256(f"{SEED}:{held_out}:{key}".encode()).digest()
                bucket[key] = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
            is_validation[position] = bucket[key] == 0

        optimizers = [
            torch.optim.SparseAdam(model.embedding.parameters(), lr=args.finetune_lr),
            torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                             lr=args.finetune_lr),
        ]
        generator = torch.Generator().manual_seed(SEED + fold_ids.index(held_out))
        best, best_epoch, best_state, waited = -1.0, 0, None, 0
        for epoch in range(1, args.finetune_epochs + 1):
            model.train()
            permutation = torch.randperm(int((~is_validation).sum()), generator=generator)
            tr1, tr2 = pool_rows1[~is_validation], pool_rows2[~is_validation]
            trt = pool_target[~is_validation]
            running, batches = 0.0, 0
            for start in range(0, len(trt), args.batch_size):
                pick = permutation[start : start + args.batch_size].numpy()
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
                pool_target[is_validation],
                predict(model, hand_encoded_t, pool_rows1[is_validation], pool_rows2[is_validation]))
            improved = score > best
            log(f"    finetune e{epoch}  loss {running / batches:.5f}  val {score:.6f}"
                f"{'  *' if improved else ''}")
            if improved:
                best, best_epoch, waited = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= args.patience:
                    break
        model.load_state_dict(best_state)
        scores = predict(model, hand_encoded_t, rows1[held_out], rows2[held_out])
        finetuned[held_out] = average_precision(folds[held_out]["target"], scores)
        with (args.out_dir / f"{held_out}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for a, b, s in zip(folds[held_out]["id1"].tolist(), folds[held_out]["id2"].tolist(),
                               scores.tolist(), strict=True):
                writer.writerow([a, b, f"{s:.8f}"])
        log(f"  {held_out} fine-tuned (best epoch {best_epoch}): PR-AUC {finetuned[held_out]:.6f}")

    log("\n" + "=" * 62)
    log(f"zero-shot  mean {np.mean(list(zero_shot.values())):.6f}  "
        f"({', '.join(f'{k}={v:.4f}' for k, v in zero_shot.items())})")
    log(f"fine-tuned mean {np.mean(list(finetuned.values())):.6f}  "
        f"({', '.join(f'{k}={v:.4f}' for k, v in finetuned.items())})")
    log("control knrm_name_v2 mean 0.530775 (no pretraining, same spec-v2 folds)")
    log(f"predictions -> {args.out_dir}")


if __name__ == "__main__":
    main()

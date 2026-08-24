"""KNRM over product names — trained baseline on the shared grouped folds.

Kernel-based Neural Ranking Model (Xiong et al., SIGIR 2017) applied to pair
classification: embed the tokens of both names, build the token-by-token cosine
matrix, pool it with RBF kernels into soft term-frequency features, and score
the pair with a linear learning-to-rank layer.

Three deliberate deviations from the paper, all forced by the task:

1. **Symmetric scoring.** KNRM is asymmetric (soft-TF sums over query terms
   only), product matching is not. The kernel tensor is pooled in both
   directions and the two feature vectors are averaged, so the score of
   (A, B) equals the score of (B, A) exactly. The cosine matrix is computed
   once and reused for both directions.
2. **Binary cross-entropy instead of pairwise hinge.** There are no ranked
   lists here, only labeled pairs, and the metric is PR-AUC. The final layer
   emits a logit; the paper's `tanh` squashing exists to bound a ranking score
   under hinge loss and is dropped.
3. **Full-vocabulary embeddings, no OOV hashing.** Article numbers and model
   codes carry most of the matching signal and they are overwhelmingly rare:
   52.4% of the name vocabulary occurs exactly once. Hashing rare tokens into
   buckets would make two *different* article numbers collide onto one
   embedding row, and a collision lands in the exact-match kernel
   (mu=1.0, sigma=1e-3) — the model reads it as maximal evidence of a match.
   Every distinct token therefore gets its own row.

   This also means rare tokens work without being trained: the same article
   number in both names is the same row, so its cosine is exactly 1.0, while
   two distinct random rows in 64 dimensions are near-orthogonal (cosine
   std ~ 1/sqrt(dim) = 0.125). The exact-match kernel is correct at
   initialization and training can only refine the soft kernels around it.

The vocabulary is fitted on ``items_human`` names only — transductive over the
evaluation items but test-legal, the same argument as ``name_tfidf_cos``: at
submit time the container receives the full test items file before predicting,
so the identical fit is available there.

Out-of-fold protocol: for each fold K a fresh model is trained on the pairs of
the other three folds and predicts fold K. Folds are grouped by connected
component, so no item seen in training reappears in the predicted fold.

Run through the ops harness (contract: validation/ops/train.py):

    make train MEMBER=darksteeld EXPERIMENT=knrm_name

or directly, then register the result:

    .venv/bin/python members/darksteeld/experiments/knrm_name/train.py \
        --out-dir validation/predictions/darksteeld/knrm_name
    make score MEMBER=darksteeld EXPERIMENT=knrm_name NOTES="..."
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")

PAD_ID = 0  # embedding row 0, masked out everywhere and never updated
UNK_ID = 1  # unreachable when the vocabulary is fitted on the scored items

# Standard KNRM kernel bank: one exact-match kernel plus ten soft kernels
# evenly spaced over the cosine range.
KERNEL_MU = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9)
KERNEL_SIGMA = (1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)


def normalize_name(name: str) -> str:
    """Identical normalization to the cheap baselines, so inputs stay comparable."""
    text = unicodedata.normalize("NFKC", name).lower().replace("ё", "е")
    return NON_ALNUM.sub(" ", text).strip()


def build_vocabulary(names: list[str]) -> tuple[dict[str, int], list[list[str]]]:
    """Every distinct token gets its own id; rare article codes are the point."""
    tokenized = [normalize_name(name).split() for name in names]
    vocabulary: dict[str, int] = {}
    next_id = UNK_ID + 1
    for tokens in tokenized:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = next_id
                next_id += 1
    return vocabulary, tokenized


def encode(tokenized: list[list[str]], vocabulary: dict[str, int], max_len: int) -> np.ndarray:
    """(items, max_len) int32 matrix of token ids, zero-padded on the right."""
    encoded = np.zeros((len(tokenized), max_len), dtype=np.int32)
    for row, tokens in enumerate(tokenized):
        for column, token in enumerate(tokens[:max_len]):
            encoded[row, column] = vocabulary.get(token, UNK_ID)
    return encoded


class KNRM(nn.Module):
    """Kernel pooling over the cosine matrix of two token sequences."""

    def __init__(self, vocabulary_size: int, dim: int) -> None:
        super().__init__()
        # sparse gradients: a dense grad over the full table would be 16.9M
        # values per step, while a batch touches at most 2 * batch * max_len rows
        self.embedding = nn.Embedding(vocabulary_size, dim, padding_idx=PAD_ID, sparse=True)
        self.head = nn.Linear(len(KERNEL_MU), 1)
        self.register_buffer("mu", torch.tensor(KERNEL_MU).view(1, 1, 1, -1))
        self.register_buffer("sigma", torch.tensor(KERNEL_SIGMA).view(1, 1, 1, -1))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_mask = (left != PAD_ID).float()
        right_mask = (right != PAD_ID).float()

        left_vectors = F.normalize(self.embedding(left), p=2, dim=-1)
        right_vectors = F.normalize(self.embedding(right), p=2, dim=-1)
        similarity = torch.bmm(left_vectors, right_vectors.transpose(1, 2))

        pair_mask = left_mask.unsqueeze(2) * right_mask.unsqueeze(1)
        kernels = torch.exp(
            -((similarity.unsqueeze(-1) - self.mu) ** 2) / (2 * self.sigma**2)
        ) * pair_mask.unsqueeze(-1)

        # soft-TF in both directions off the same kernel tensor, then averaged:
        # the resulting score is exactly symmetric in (left, right)
        left_to_right = torch.log(kernels.sum(dim=2).clamp_min(1e-10)) * left_mask.unsqueeze(-1)
        right_to_left = torch.log(kernels.sum(dim=1).clamp_min(1e-10)) * right_mask.unsqueeze(-1)
        features = 0.5 * (left_to_right.sum(dim=1) + right_to_left.sum(dim=1))
        return self.head(features).squeeze(-1)


def load_folds(targets_dir: Path, fold_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    folds: dict[str, dict[str, np.ndarray]] = {}
    for fold_id in fold_ids:
        path = targets_dir / f"{fold_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; run: make validation-targets")
        id1, id2, target = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                id1.append(int(row["id1"]))
                id2.append(int(row["id2"]))
                target.append(float(row["target"]))
        folds[fold_id] = {
            "id1": np.asarray(id1, dtype=np.int64),
            "id2": np.asarray(id2, dtype=np.int64),
            "target": np.asarray(target, dtype=np.float32),
        }
    return folds


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """Same definition as validation/evaluate.py, for in-run progress only."""
    order = np.argsort(-score, kind="mergesort")
    labels = target[order]
    ranked = score[order]
    cumulative_true = np.cumsum(labels)
    total_positive = cumulative_true[-1]
    if total_positive == 0:
        return 0.0
    # collapse ties: only the last index of each equal-score run is a threshold
    last_of_run = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative_true[last_of_run] / (np.arange(len(labels))[last_of_run] + 1)
    recall = cumulative_true[last_of_run] / total_positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def train_fold(
    model: KNRM,
    encoded: torch.Tensor,
    rows1: np.ndarray,
    rows2: np.ndarray,
    target: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    embedding_optimizer = torch.optim.SparseAdam(model.embedding.parameters(), lr=learning_rate)
    head_optimizer = torch.optim.Adam(model.head.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    rows1_tensor = torch.from_numpy(rows1)
    rows2_tensor = torch.from_numpy(rows2)
    target_tensor = torch.from_numpy(target)
    pair_count = len(target)

    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(pair_count, generator=generator)
        running, batches, started = 0.0, 0, time.time()
        for start in range(0, pair_count, batch_size):
            selection = permutation[start : start + batch_size]
            left = encoded[rows1_tensor[selection]].to(device)
            right = encoded[rows2_tensor[selection]].to(device)
            labels = target_tensor[selection].to(device)

            logits = model(left, right)
            loss = loss_function(logits, labels)

            embedding_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            embedding_optimizer.step()
            head_optimizer.step()

            running += float(loss.detach())
            batches += 1
        print(
            f"      epoch {epoch}/{epochs}  loss {running / batches:.5f}  "
            f"{time.time() - started:.0f}s",
            flush=True,
        )


@torch.no_grad()
def predict(
    model: KNRM,
    encoded: torch.Tensor,
    rows1: np.ndarray,
    rows2: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows1_tensor = torch.from_numpy(rows1)
    rows2_tensor = torch.from_numpy(rows2)
    scores = np.empty(len(rows1), dtype=np.float64)
    for start in range(0, len(rows1), batch_size):
        stop = min(start + batch_size, len(rows1))
        left = encoded[rows1_tensor[start:stop]].to(device)
        right = encoded[rows2_tensor[start:stop]].to(device)
        scores[start:stop] = torch.sigmoid(model(left, right)).cpu().numpy()
    return scores


def write_predictions(out_dir: Path, fold_id: str, fold: dict[str, np.ndarray], scores: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{fold_id}.csv"
    with destination.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        for id1, id2, score in zip(
            fold["id1"].tolist(), fold["id2"].tolist(), scores.tolist(), strict=True
        ):
            writer.writerow([id1, id2, f"{score:.8f}"])


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    # sparse embedding gradients are a CPU-solid path; MPS/CUDA support for
    # SparseAdam is uneven, and the model is small enough that CPU is not the
    # bottleneck. Opt in explicitly with --device if the machine has a GPU.
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # contract arguments passed by validation/ops/train.py
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld" / "knrm_name")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04")
    parser.add_argument("--targets-dir", type=Path, default=None,
                        help="default <repo>/validation/targets; pass validation/targets_v2 "
                             "for the stratified spec-v2 folds")
    parser.add_argument("--submission-dir", type=Path, default=None, help="ignored by this experiment")
    # model / optimisation
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=20, help="96% of names fit whole")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    fold_ids = [fold.strip() for fold in args.folds.split(",") if fold.strip()]
    print(f"device={device}  folds={fold_ids}  seed={args.seed}", flush=True)

    import polars as pl

    items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "name"])
    names = items["name"].to_list()
    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_list())}

    started = time.time()
    vocabulary, tokenized = build_vocabulary(names)
    vocabulary_size = UNK_ID + 1 + len(vocabulary)
    encoded = torch.from_numpy(encode(tokenized, vocabulary, args.max_len).astype(np.int64))
    print(
        f"vocabulary {len(vocabulary):,} tokens (+pad,unk) -> embedding table "
        f"{vocabulary_size:,} x {args.dim} = {vocabulary_size * args.dim / 1e6:.1f}M params, "
        f"built in {time.time() - started:.0f}s",
        flush=True,
    )

    folds = load_folds(args.targets_dir or (args.repo / "validation" / "targets"), fold_ids)
    rows1 = {f: np.array([row_of_id[i] for i in folds[f]["id1"].tolist()], dtype=np.int64) for f in fold_ids}
    rows2 = {f: np.array([row_of_id[i] for i in folds[f]["id2"].tolist()], dtype=np.int64) for f in fold_ids}

    fold_scores: dict[str, float] = {}
    for held_out in fold_ids:
        train_ids = [f for f in fold_ids if f != held_out]
        train_rows1 = np.concatenate([rows1[f] for f in train_ids])
        train_rows2 = np.concatenate([rows2[f] for f in train_ids])
        train_target = np.concatenate([folds[f]["target"] for f in train_ids])
        print(
            f"  {held_out}: train on {', '.join(train_ids)} "
            f"({len(train_target):,} pairs) -> predict {len(folds[held_out]['target']):,}",
            flush=True,
        )

        generator = torch.Generator().manual_seed(args.seed + fold_ids.index(held_out))
        model = KNRM(vocabulary_size, args.dim).to(device)
        train_fold(
            model, encoded, train_rows1, train_rows2, train_target,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, device=device, generator=generator,
        )
        scores = predict(
            model, encoded, rows1[held_out], rows2[held_out],
            batch_size=args.batch_size * 4, device=device,
        )
        write_predictions(args.out_dir, held_out, folds[held_out], scores)
        fold_scores[held_out] = average_precision(folds[held_out]["target"], scores)
        weights = model.head.weight.detach().cpu().numpy().ravel()
        print(
            f"      PR-AUC {fold_scores[held_out]:.6f}   "
            f"exact-match kernel weight {weights[0]:+.3f}",
            flush=True,
        )

    print(f"\npredictions -> {args.out_dir}")
    print(f"mean PR-AUC {np.mean(list(fold_scores.values())):.6f}  "
          f"({', '.join(f'{f}={s:.6f}' for f, s in fold_scores.items())})")
    print("register with: make score MEMBER=darksteeld EXPERIMENT=knrm_name NOTES=\"...\"")


if __name__ == "__main__":
    main()

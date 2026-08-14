"""KNRM v2 over product names: scaled features, early stopping, pretrained init.

Three changes over ``knrm_name`` (v1, mean PR-AUC 0.40543), each targeting a
defect measured in that run.

1. **Feature scaling.** v1 fed the linear head raw KNRM soft-TF sums. For the
   exact-match kernel (sigma=1e-3) a token with no exact counterpart has
   soft-TF exactly 0, so the clamp fired on every such token and contributed
   ``log(1e-10) = -23``. The feature was therefore ``-23 * (unmatched tokens)``
   — a token-mismatch counter entangled with name length, on a scale of +-460
   into a linear layer. The head could not settle: epoch-1 loss ranged
   0.67-2.37 across folds, the exact-match kernel weight flipped sign
   (+0.211/-0.100/-0.089/+0.008) and fold spread was 0.0515 against 0.003-0.007
   for the label-free baselines.

   Fixed three ways: soft-TF is averaged over real tokens rather than summed
   (turning "how many tokens went unmatched" into "what fraction matched",
   which is what the feature should have measured), the clamp floor is raised
   to 1e-4, and a BatchNorm over the kernel features standardises whatever
   scale is left before the head.

2. **Early stopping, patience 1**, on a leakage-free validation slice. The
   validation pairs are carved out of the *training* folds by connected
   component — reusing ``validation.build_folds.connected_component_keys``, the
   same grouping the shared folds use — so no item of the validation slice
   appears in the training pairs, and the held-out fold is never touched.
   Roughly 10% of the training components are held out; the epoch with the best
   validation PR-AUC is restored before predicting. v1's loss was still falling
   at its fixed third epoch, so it was simply undertrained.

3. **Pretrained embeddings** instead of random init, which is what KNRM does in
   the paper and the main source of its reported gain. Vectors: **navec**
   ``hudlit_v1_12B_500K_300d_100q``, MIT licensed (both the loader package and
   the natasha/navec repository), numpy-only loader, no gensim, 50 MB on disk —
   small enough to bundle in the solution container, which has no network.

   Note this is GloVe, not word2vec: the same family of static word vectors,
   a different training objective.

   Measured coverage of the 263,740-token name vocabulary: 19.0% of types but
   65.8% of occurrences, and the split is exactly complementary to the task —
   92.9% of Cyrillic word occurrences and 53.7% of Latin ones are covered,
   while pure digits and alphanumeric article codes are covered 0%. Meaning
   comes from the pretrained side; identity of article codes keeps the
   random-orthogonal rows, where the exact-match kernel is already correct by
   construction (the same code is the same row, so its cosine is exactly 1.0).
   Every row is unit-normalised at init so pretrained and random rows enter
   training on the same footing.

Unchanged from v1: symmetric scoring, full 263,740-token vocabulary with no OOV
hashing, BCE, sparse embedding gradients, per-fold OOF over the frozen folds.

    make train MEMBER=darksteeld EXPERIMENT=knrm_name_v2

or directly, then register:

    .venv/bin/python members/darksteeld/experiments/knrm_name_v2/train.py \
        --out-dir validation/predictions/darksteeld/knrm_name_v2
    make score MEMBER=darksteeld EXPERIMENT=knrm_name_v2 NOTES="..."
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")
PAD_ID = 0
UNK_ID = 1

KERNEL_MU = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9)
KERNEL_SIGMA = (1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

# navec hudlit: MIT (natasha/navec), GloVe vectors, 500K vocab, 300d, 50MB.
# Not in Git (members/*/models/ is ignored); fetch once with:
#   curl -L -o members/darksteeld/models/navec_hudlit_v1_12B_500K_300d_100q.tar \
#     https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
NAVEC_URL = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"
DEFAULT_NAVEC = REPOSITORY_ROOT / "members" / "darksteeld" / "models" / "navec_hudlit_v1_12B_500K_300d_100q.tar"
VALIDATION_BUCKETS = 10  # one bucket of training components becomes the early-stopping slice


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).lower().replace("ё", "е")
    return NON_ALNUM.sub(" ", text).strip()


def build_vocabulary(names: list[str]) -> tuple[dict[str, int], list[list[str]]]:
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
    encoded = np.zeros((len(tokenized), max_len), dtype=np.int32)
    for row, tokens in enumerate(tokenized):
        for column, token in enumerate(tokens[:max_len]):
            encoded[row, column] = vocabulary.get(token, UNK_ID)
    return encoded


def initial_embeddings(
    vocabulary: dict[str, int], dim: int, navec_path: Path | None, seed: int
) -> tuple[torch.Tensor, int]:
    """Unit-norm random rows, overwritten by pretrained vectors where available."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.empty(UNK_ID + 1 + len(vocabulary), dim)
    nn.init.normal_(weight, mean=0.0, std=1.0, generator=generator)
    weight = F.normalize(weight, p=2, dim=-1)

    covered = 0
    if navec_path is not None:
        from navec import Navec

        navec = Navec.load(str(navec_path))
        if navec.pq.dim != dim:
            raise ValueError(f"--dim {dim} must equal the pretrained dimension {navec.pq.dim}")
        known = set(navec.vocab.words)
        rows, vectors = [], []
        for token, token_id in vocabulary.items():
            if token in known:
                rows.append(token_id)
                vectors.append(navec[token])
        if rows:
            pretrained = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
            weight[torch.tensor(rows, dtype=torch.long)] = F.normalize(pretrained, p=2, dim=-1)
        covered = len(rows)

    weight[PAD_ID] = 0.0
    return weight, covered


class KNRM(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=False, sparse=True, padding_idx=PAD_ID
        )
        # change 1: standardise the kernel features before the ranking layer
        self.norm = nn.BatchNorm1d(len(KERNEL_MU))
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

        # change 1: mean over real tokens, not sum, and a 1e-4 clamp floor
        left_length = left_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        right_length = right_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        left_to_right = torch.log(kernels.sum(dim=2).clamp_min(1e-4)) * left_mask.unsqueeze(-1)
        right_to_left = torch.log(kernels.sum(dim=1).clamp_min(1e-4)) * right_mask.unsqueeze(-1)
        features = 0.5 * (
            left_to_right.sum(dim=1) / left_length + right_to_left.sum(dim=1) / right_length
        )
        return self.head(self.norm(features)).squeeze(-1)


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


def validation_mask(id1: np.ndarray, id2: np.ndarray, seed: str) -> np.ndarray:
    """Hold out whole connected components, so no item spans train and validation.

    Same grouping unit and the same RNG-free hashing style as the shared fold
    builder; only the bucket count differs.
    """
    from validation.build_folds import connected_component_keys

    component_of_item = connected_component_keys(id1, id2)
    is_validation = np.zeros(len(id1), dtype=bool)
    bucket_of_key: dict[int, int] = {}
    for position, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        bucket = bucket_of_key.get(key)
        if bucket is None:
            digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
            bucket_of_key[key] = bucket
        is_validation[position] = bucket == 0
    return is_validation


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels = target[order]
    ranked = score[order]
    cumulative_true = np.cumsum(labels)
    total_positive = cumulative_true[-1]
    if total_positive == 0:
        return 0.0
    last_of_run = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative_true[last_of_run] / (np.arange(len(labels))[last_of_run] + 1)
    recall = cumulative_true[last_of_run] / total_positive
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@torch.no_grad()
def predict(model: KNRM, encoded: torch.Tensor, rows1: np.ndarray, rows2: np.ndarray,
            *, batch_size: int, device: torch.device) -> np.ndarray:
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


def train_fold(
    model: KNRM, encoded: torch.Tensor,
    train_rows1: np.ndarray, train_rows2: np.ndarray, train_target: np.ndarray,
    validation_rows1: np.ndarray, validation_rows2: np.ndarray, validation_target: np.ndarray,
    *, max_epochs: int, patience: int, batch_size: int, learning_rate: float,
    device: torch.device, generator: torch.Generator,
) -> tuple[int, float]:
    embedding_optimizer = torch.optim.SparseAdam(model.embedding.parameters(), lr=learning_rate)
    dense_parameters = list(model.norm.parameters()) + list(model.head.parameters())
    dense_optimizer = torch.optim.Adam(dense_parameters, lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    rows1_tensor = torch.from_numpy(train_rows1)
    rows2_tensor = torch.from_numpy(train_rows2)
    target_tensor = torch.from_numpy(train_target)
    pair_count = len(train_target)

    best_score, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(pair_count, generator=generator)
        running, batches, started = 0.0, 0, time.time()
        for start in range(0, pair_count, batch_size):
            selection = permutation[start : start + batch_size]
            left = encoded[rows1_tensor[selection]].to(device)
            right = encoded[rows2_tensor[selection]].to(device)
            labels = target_tensor[selection].to(device)

            loss = loss_function(model(left, right), labels)
            embedding_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            embedding_optimizer.step()
            dense_optimizer.step()
            running += float(loss.detach())
            batches += 1

        scores = predict(model, encoded, validation_rows1, validation_rows2,
                         batch_size=batch_size * 4, device=device)
        validation_score = average_precision(validation_target, scores)
        improved = validation_score > best_score
        print(f"      epoch {epoch:>2}  loss {running / batches:.5f}  "
              f"val PR-AUC {validation_score:.6f}{'  *' if improved else ''}  "
              f"{time.time() - started:.0f}s", flush=True)

        if improved:
            best_score, best_epoch, waited = validation_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                print(f"      early stop: no gain for {waited} epoch(s), "
                      f"restoring epoch {best_epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_score


def write_predictions(out_dir: Path, fold_id: str, fold: dict[str, np.ndarray], scores: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{fold_id}.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        for id1, id2, score in zip(fold["id1"].tolist(), fold["id2"].tolist(),
                                   scores.tolist(), strict=True):
            writer.writerow([id1, id2, f"{score:.8f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions" / "darksteeld" / "knrm_name_v2")
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04")
    parser.add_argument("--targets-dir", type=Path, default=None,
                        help="default <repo>/validation/targets; pass validation/targets_v2 "
                             "to train and predict on the stratified spec-v2 folds")
    parser.add_argument("--submission-dir", type=Path, default=None, help="ignored by this experiment")
    parser.add_argument("--navec", type=Path, default=DEFAULT_NAVEC,
                        help="pretrained vectors; pass 'none' for the v1 random init")
    parser.add_argument("--dim", type=int, default=300, help="must match the pretrained dimension")
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    fold_ids = [fold.strip() for fold in args.folds.split(",") if fold.strip()]
    navec_path = None if str(args.navec).lower() == "none" else args.navec
    if navec_path is not None and not navec_path.is_file():
        raise SystemExit(
            f"pretrained vectors not found: {navec_path}\n"
            f"fetch once:  curl -L -o {navec_path} {NAVEC_URL}\n"
            f"or pass --navec none to fall back to the v1 random init"
        )
    print(f"device={device}  folds={fold_ids}  seed={args.seed}  dim={args.dim}  "
          f"patience={args.patience}  pretrained={navec_path.name if navec_path else 'none'}",
          flush=True)

    import polars as pl

    items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "name"])
    names = items["name"].to_list()
    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_list())}

    started = time.time()
    vocabulary, tokenized = build_vocabulary(names)
    encoded = torch.from_numpy(encode(tokenized, vocabulary, args.max_len).astype(np.int64))
    weight, covered = initial_embeddings(vocabulary, args.dim, navec_path, args.seed)
    print(f"vocabulary {len(vocabulary):,} tokens -> table {tuple(weight.shape)} "
          f"({weight.numel() / 1e6:.1f}M params); pretrained rows {covered:,} "
          f"({100 * covered / max(len(vocabulary), 1):.1f}% of types); {time.time() - started:.0f}s",
          flush=True)

    targets_dir = args.targets_dir or (args.repo / "validation" / "targets")
    print(f"targets from {targets_dir}", flush=True)
    folds = load_folds(targets_dir, fold_ids)
    rows1 = {f: np.array([row_of_id[i] for i in folds[f]["id1"].tolist()], dtype=np.int64) for f in fold_ids}
    rows2 = {f: np.array([row_of_id[i] for i in folds[f]["id2"].tolist()], dtype=np.int64) for f in fold_ids}

    fold_scores: dict[str, float] = {}
    for held_out in fold_ids:
        train_ids = [f for f in fold_ids if f != held_out]
        pool_id1 = np.concatenate([folds[f]["id1"] for f in train_ids])
        pool_id2 = np.concatenate([folds[f]["id2"] for f in train_ids])
        pool_rows1 = np.concatenate([rows1[f] for f in train_ids])
        pool_rows2 = np.concatenate([rows2[f] for f in train_ids])
        pool_target = np.concatenate([folds[f]["target"] for f in train_ids])

        is_validation = validation_mask(pool_id1, pool_id2, f"{args.seed}:{held_out}")
        print(f"  {held_out}: pool {len(pool_target):,} pairs -> "
              f"train {int((~is_validation).sum()):,} / val {int(is_validation.sum()):,} "
              f"(component-grouped) -> predict {len(folds[held_out]['target']):,}", flush=True)

        generator = torch.Generator().manual_seed(args.seed + fold_ids.index(held_out))
        model = KNRM(weight.clone()).to(device)
        best_epoch, best_validation = train_fold(
            model, encoded,
            pool_rows1[~is_validation], pool_rows2[~is_validation], pool_target[~is_validation],
            pool_rows1[is_validation], pool_rows2[is_validation], pool_target[is_validation],
            max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, device=device, generator=generator,
        )
        scores = predict(model, encoded, rows1[held_out], rows2[held_out],
                         batch_size=args.batch_size * 4, device=device)
        write_predictions(args.out_dir, held_out, folds[held_out], scores)
        fold_scores[held_out] = average_precision(folds[held_out]["target"], scores)
        weights = model.head.weight.detach().cpu().numpy().ravel()
        print(f"      best epoch {best_epoch} (val {best_validation:.6f})  "
              f"-> fold PR-AUC {fold_scores[held_out]:.6f}   "
              f"exact-match kernel weight {weights[0]:+.3f}", flush=True)

    values = list(fold_scores.values())
    print(f"\npredictions -> {args.out_dir}")
    print(f"mean PR-AUC {np.mean(values):.6f}  spread {max(values) - min(values):.6f}  "
          f"({', '.join(f'{f}={s:.6f}' for f, s in fold_scores.items())})")


if __name__ == "__main__":
    main()

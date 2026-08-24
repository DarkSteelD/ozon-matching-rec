"""KNRM model and tokenisation, shared by the experiment and the container.

Both ``members/darksteeld/experiments/knrm_name_v2/train.py`` and the submission
container import this module, so the scored model and the cross-validated model
are the same code by construction.

The one thing the container cannot reuse from training is the *index* space: the
embedding table is indexed by the training vocabulary, and at submit time the
items are different products with a different vocabulary. The artifact therefore
ships a **token -> vector** map, and the container rebuilds an index table for
whatever tokens the test file contains.

Tokens absent from the training vocabulary get a deterministic pseudo-random
unit vector derived from the token string itself (``vector_for_unknown``). This
is not a detail: article codes are the highest-precision matching signal and are
overwhelmingly rare, so many will be unseen. Seeding by the string means the
same code in both names of a pair maps to the *same* vector — cosine exactly
1.0, the exact-match kernel fires — while two different codes get near-orthogonal
vectors. A single shared <unk> row would instead make every unseen token look
identical to every other, which is the worst possible failure here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NON_ALNUM = re.compile(r"[^0-9a-zа-я]+")
PAD_ID = 0

KERNEL_MU = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9)
KERNEL_SIGMA = (1e-3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
MAX_LEN = 20
DIM = 300


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name)).lower().replace("ё", "е")
    return NON_ALNUM.sub(" ", text).strip()


def tokenize(name: str) -> list[str]:
    return normalize_name(name).split()


def vector_for_unknown(token: str, dim: int) -> np.ndarray:
    """Deterministic unit vector for a token unseen during training.

    Derived from the token string, so it is identical in every process and on
    every machine — which is what makes an unseen article code still match
    itself exactly across the two sides of a pair.
    """
    seed = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def encode_names(names: list[str], token_id: dict[str, int], max_len: int = MAX_LEN) -> np.ndarray:
    encoded = np.zeros((len(names), max_len), dtype=np.int64)
    for row, name in enumerate(names):
        for column, token in enumerate(tokenize(name)[:max_len]):
            encoded[row, column] = token_id.get(token, PAD_ID)
    return encoded


class KNRM(nn.Module):
    """Kernel pooling over the token cosine matrix of two names.

    Symmetric by construction: the kernel tensor is pooled in both directions
    and averaged, so score(A, B) == score(B, A) exactly. Features are the mean
    over real tokens (not the sum) with a 1e-4 clamp floor, then standardised by
    a BatchNorm before the linear ranking layer — see the v2 experiment for why
    the raw KNRM sums were unusable here.
    """

    def __init__(self, weight: torch.Tensor, sparse: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=False, sparse=sparse, padding_idx=PAD_ID
        )
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

        left_length = left_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        right_length = right_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        left_to_right = torch.log(kernels.sum(dim=2).clamp_min(1e-4)) * left_mask.unsqueeze(-1)
        right_to_left = torch.log(kernels.sum(dim=1).clamp_min(1e-4)) * right_mask.unsqueeze(-1)
        features = 0.5 * (
            left_to_right.sum(dim=1) / left_length + right_to_left.sum(dim=1) / right_length
        )
        return self.head(self.norm(features)).squeeze(-1)

"""KNRM over the attribute dictionary: one embedding layer for keys, one for
values, combined per attribute before the kernel pooling.

The name-based KNRM treats a product as a bag of name tokens. This model treats
it as a bag of **attributes**, and an attribute is not a bag of words — it is a
pair. ``цвет = красный`` and ``цвет = синий`` share a token and mean opposite
things; ``цвет = красный`` and ``оттенок = красный`` share the other token and
mean nearly the same thing. Embedding the flattened text would smear both cases
into the same representation.

So each attribute gets two vectors from two independent tables — the key from
``E_key``, the value from ``E_value`` — and the attribute's representation is
their **element-wise product**. The product is what makes the pair inseparable:
it is unchanged only when *both* factors match, and two attributes agreeing on
just one side land somewhere between match and mismatch rather than at either
end. Both factors are L2-normalised before multiplying, so an attribute whose
value happens to be five words long does not outweigh a one-word one.

From there the model is ordinary KNRM, with attributes where tokens used to be:
cosine matrix between the two products' attribute rows, RBF kernel pooling,
BatchNorm, linear ranking layer. Kernel constants, padding id and the unseen-token
vector are imported from ``knrm_model`` rather than redefined, so the exact-match
kernel behaves identically in both models.

Names are deliberately not used here. This is meant to be a *decorrelated*
member: the whole reason the 0.5/0.5 blend gained +0.056 on the public board is
that its two members fail in different places, and a third model that reads the
same field as the first two would not add that.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from knrm_model import KERNEL_MU, KERNEL_SIGMA, PAD_ID, tokenize, vector_for_unknown

MAX_ATTRS = 24        # p90 товаров имеет <= 23 атрибутов
MAX_KEY_TOKENS = 4    # "толщина тормозного диска, мм" — 4 токена
MAX_VALUE_TOKENS = 6  # p90 значения — 3 токена
DIM = 300


def parse_attributes(raw: str) -> list[tuple[list[str], list[str]]]:
    """JSON-строка атрибутов -> список пар (токены ключа, токены значения).

    Слоты без ключа или без значения отбрасываются здесь, а не маскируются
    позже: пустое значение ("артикул производителя": "") не несёт информации,
    но занимало бы один из MAX_ATTRS слотов и вытесняло реальный атрибут.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, dict):
        return []
    out: list[tuple[list[str], list[str]]] = []
    for key, value in parsed.items():
        key_tokens = tokenize(str(key))[:MAX_KEY_TOKENS]
        value_tokens = tokenize(str(value))[:MAX_VALUE_TOKENS]
        if key_tokens and value_tokens:
            out.append((key_tokens, value_tokens))
        if len(out) >= MAX_ATTRS:
            break
    return out


def build_attribute_vocabularies(
    attributes: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Два независимых словаря: токены ключей и токены значений.

    Раздельные словари — не косметика: один и тот же токен ("состав") значит
    разное в позиции ключа и в позиции значения, и учить ему один вектор
    означало бы склеивать эти роли.

    Ничего не отсекается по частоте. Токены значений, встретившиеся один раз, —
    это в основном артикулы, то есть самый точный сигнал совпадения из всех,
    что здесь есть.
    """
    key_ids: dict[str, int] = {}
    value_ids: dict[str, int] = {}
    for raw in attributes:
        for key_tokens, value_tokens in parse_attributes(raw):
            for token in key_tokens:
                if token not in key_ids:
                    key_ids[token] = len(key_ids) + 1  # 0 остаётся под PAD
            for token in value_tokens:
                if token not in value_ids:
                    value_ids[token] = len(value_ids) + 1
    return key_ids, value_ids


def encode_attributes(
    attributes: list[str], key_ids: dict[str, int], value_ids: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """-> (ключи [N, MAX_ATTRS, MAX_KEY_TOKENS], значения [N, MAX_ATTRS, MAX_VALUE_TOKENS])."""
    keys = np.zeros((len(attributes), MAX_ATTRS, MAX_KEY_TOKENS), dtype=np.int32)
    values = np.zeros((len(attributes), MAX_ATTRS, MAX_VALUE_TOKENS), dtype=np.int32)
    for row, raw in enumerate(attributes):
        for slot, (key_tokens, value_tokens) in enumerate(parse_attributes(raw)):
            for column, token in enumerate(key_tokens):
                keys[row, slot, column] = key_ids.get(token, PAD_ID)
            for column, token in enumerate(value_tokens):
                values[row, slot, column] = value_ids.get(token, PAD_ID)
    return keys, values


def initial_weight(vocabulary: dict[str, int], dim: int, navec_path=None) -> tuple[torch.Tensor, int]:
    """Строка на токен: вектор navec, если он там есть, иначе детерминированный из строки.

    Детерминированный запасной вариант — то же самое, что в модели по названиям:
    неизвестный артикул обязан совпадать сам с собой на обеих сторонах пары.
    """
    weight = np.zeros((len(vocabulary) + 1, dim), dtype=np.float32)
    pretrained: dict[str, np.ndarray] = {}
    if navec_path is not None:
        try:
            from navec import Navec

            navec = Navec.load(str(navec_path))
            for token in vocabulary:
                if token in navec:
                    pretrained[token] = navec[token]
        except Exception as failure:  # noqa: BLE001 — предобучение необязательно
            print(f"  navec не загружен ({failure}); все векторы детерминированные")
    for token, index in vocabulary.items():
        vector = pretrained.get(token)
        if vector is None:
            weight[index] = vector_for_unknown(token, dim)
        else:
            vector = np.asarray(vector, dtype=np.float32)
            weight[index] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return torch.from_numpy(weight), len(pretrained)


class AttributeKNRM(nn.Module):
    """Ядровое пулирование по косинусной матрице атрибутов двух товаров.

    Атрибут = L2-нормированное среднее векторов токенов ключа, поэлементно
    умноженное на такое же среднее по токенам значения. Слот считается реальным,
    если у него есть хотя бы один токен ключа и хотя бы один токен значения.

    Симметрия по построению: тензор ядер пулится в обе стороны и усредняется,
    поэтому score(A, B) == score(B, A) точно.
    """

    def __init__(self, key_weight: torch.Tensor, value_weight: torch.Tensor,
                 sparse: bool = True) -> None:
        super().__init__()
        self.key_embedding = nn.Embedding.from_pretrained(
            key_weight, freeze=False, sparse=sparse, padding_idx=PAD_ID
        )
        self.value_embedding = nn.Embedding.from_pretrained(
            value_weight, freeze=False, sparse=sparse, padding_idx=PAD_ID
        )
        self.norm = nn.BatchNorm1d(len(KERNEL_MU))
        self.head = nn.Linear(len(KERNEL_MU), 1)
        self.register_buffer("mu", torch.tensor(KERNEL_MU).view(1, 1, 1, -1))
        self.register_buffer("sigma", torch.tensor(KERNEL_SIGMA).view(1, 1, 1, -1))

    @staticmethod
    def _pool_tokens(embedded: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """[B, A, T, D] -> [B, A, D]: среднее по непадовым токенам, затем L2."""
        mask = (ids != PAD_ID).float().unsqueeze(-1)
        summed = (embedded * mask).sum(dim=2)
        count = mask.sum(dim=2).clamp_min(1.0)
        return F.normalize(summed / count, p=2, dim=-1)

    def encode_side(self, keys: torch.Tensor, values: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_vectors = self._pool_tokens(self.key_embedding(keys), keys)
        value_vectors = self._pool_tokens(self.value_embedding(values), values)
        slots = F.normalize(key_vectors * value_vectors, p=2, dim=-1)
        real = ((keys != PAD_ID).any(dim=2) & (values != PAD_ID).any(dim=2)).float()
        return slots * real.unsqueeze(-1), real

    def forward(self, left_keys: torch.Tensor, left_values: torch.Tensor,
                right_keys: torch.Tensor, right_values: torch.Tensor) -> torch.Tensor:
        left_slots, left_mask = self.encode_side(left_keys, left_values)
        right_slots, right_mask = self.encode_side(right_keys, right_values)

        similarity = torch.bmm(left_slots, right_slots.transpose(1, 2))
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

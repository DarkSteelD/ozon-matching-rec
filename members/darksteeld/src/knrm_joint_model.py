"""Симметричная KNRM-подобная сеть: название и атрибуты в одном пространстве.

Две существующие сети репозитория читают по одному полю каждая: ``knrm_model``
сравнивает названия, ``knrm_attrs_model`` — атрибуты. Ни одна не видит случая,
когда свидетельство лежит в названии одного товара и в значении атрибута
другого: ``шуруповерт bosch gsr 12v`` против ``бренд = bosch``. Эта сеть считает
все четыре взаимодействия сразу, поверх **одной** таблицы эмбеддингов.

**Почему таблица обязана быть общей.** Косинус между токеном названия и токеном
значения имеет смысл, только если оба вектора взяты из одного пространства. С
раздельными таблицами (или с обучаемой проекцией между ними) кросс-каналы
измеряли бы шум. Поэтому названия, ключи и значения индексируются одним
``nn.Embedding``.

**Почему ключ не умножается на косинус значения.** Предыдущая атрибутная модель
представляла атрибут как ``normalize(k ⊙ v)`` — поэлементное произведение
векторов ключа и значения. Замер на отгруженных таблицах показал, что это
вырождается: ``slot(−k, −v) == slot(k, v)`` тождественно (знаки сокращаются до
косинуса), 48.9% положительной массы числителя набирается из координат, где
расходятся **обе** стороны, и в результате из одиннадцати ядер срабатывают два.
Здесь ключ управляет не косинусом, а уже неотрицательным откликом ядер:

    вес = sigmoid(MLP(признаки ключей))  ∈ (0, 1)
    вклад = вес × K_r(cos(значение, значение))   ≥ 0

Гейт может только ослабить свидетельство, но не поменять его знак, поэтому два
отрицательных косинуса ложного совпадения не создают.

**Симметрия.** ``score(A, B) == score(B, A)`` по построению: канал название-название
и канал атрибуты-атрибуты пулятся в обе стороны и усредняются, а кросс-каналы
входят в признаки только через ``0.5·(z_AB + z_BA)`` и ``|z_AB − z_BA|`` — обе
величины не меняются при перестановке. В сети нет BatchNorm (только LayerNorm),
поэтому предсказание одного примера не зависит от остальных в батче — это нужно и
для симметрии, и для того, чтобы добивка паддингом ничего не сдвигала.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


def default_kernels(num_kernels: int = 11, exact_sigma: float = 1e-3
                    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Одно ядро точного совпадения плюс равномерное покрытие [-1, 1].

    Ядро при ``mu = 1`` с крошечной sigma ловит буквальное совпадение токенов —
    артикулы, модели, бренды, размеры; остальные ядра садятся в центры равных
    корзин, а sigma берётся в половину ширины корзины, чтобы соседние ядра
    перекрывались, но не сливались.
    """
    if num_kernels < 2:
        raise ValueError("нужно хотя бы два ядра: точное и одно мягкое")
    soft = num_kernels - 1
    width = 2.0 / soft
    mu = [1.0] + [-1.0 + width * (index + 0.5) for index in range(soft)]
    sigma = [exact_sigma] + [width / 2.0] * soft
    return tuple(mu), tuple(sigma)


@dataclass(frozen=True)
class KNRMConfig:
    """Всё настраиваемое собрано здесь, чтобы конфиг эксперимента был одним объектом."""

    embedding_dim: int = 300
    num_kernels: int = 11
    mu: tuple[float, ...] | None = None      # None -> default_kernels(num_kernels)
    sigma: tuple[float, ...] | None = None
    exact_sigma: float = 1e-3
    hidden_dim: int = 64
    dropout: float = 0.1
    gate_hidden_dim: int = 64
    gate_bias_init: float = 2.0              # sigmoid(2) ≈ 0.88: гейты стартуют открытыми
    freeze_embeddings: bool = False
    sparse_embedding: bool = False           # True требует SparseAdam для таблицы
    attribute_chunk_size: int | None = None  # None -> без чанкования по атрибутам A
    eps: float = 1e-12

    def kernels(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if self.mu is None or self.sigma is None:
            return default_kernels(self.num_kernels, self.exact_sigma)
        if len(self.mu) != len(self.sigma):
            raise ValueError("длины mu и sigma должны совпадать")
        return tuple(self.mu), tuple(self.sigma)


@dataclass
class ItemTensors:
    """Батч одного товара пары. Формы — в комментариях справа."""

    title_ids: torch.Tensor          # [B, Lt]
    title_mask: torch.Tensor         # [B, Lt]      1.0 = реальный токен
    key_ids: torch.Tensor            # [B, Na, Lk]
    key_token_mask: torch.Tensor     # [B, Na, Lk]
    value_ids: torch.Tensor          # [B, Na, Lv]
    value_token_mask: torch.Tensor   # [B, Na, Lv]
    attribute_mask: torch.Tensor     # [B, Na]      1.0 = реальный атрибут

    def to(self, *args, **kwargs) -> "ItemTensors":
        return replace(self, **{
            field: getattr(self, field).to(*args, **kwargs)
            for field in ("title_ids", "title_mask", "key_ids", "key_token_mask",
                          "value_ids", "value_token_mask", "attribute_mask")
        })

    def validate(self) -> None:
        batch, _ = self.title_ids.shape
        if self.title_mask.shape != self.title_ids.shape:
            raise ValueError("title_mask и title_ids должны совпадать по форме")
        if self.key_ids.shape[:2] != self.value_ids.shape[:2]:
            raise ValueError("ключи и значения должны иметь одинаковые [B, Na]")
        if self.attribute_mask.shape != self.key_ids.shape[:2]:
            raise ValueError("attribute_mask должен быть [B, Na]")
        if self.key_ids.shape[0] != batch:
            raise ValueError("размер батча расходится между полями")


@dataclass
class EncodedItem:
    """Результат lookup: всё уже L2-нормировано, маски согласованы."""

    title_vectors: torch.Tensor      # [B, Lt, D]
    title_mask: torch.Tensor         # [B, Lt]
    title_mean: torch.Tensor         # [B, D]
    key_vectors: torch.Tensor        # [B, Na, D]
    value_vectors: torch.Tensor      # [B, Na, Lv, D]
    value_mask: torch.Tensor         # [B, Na, Lv]  уже занулена на паддинг-атрибутах
    attribute_mask: torch.Tensor     # [B, Na]


class KernelBank(nn.Module):
    """RBF-ядра KNRM: [...] -> [..., K]."""

    def __init__(self, mu: Sequence[float], sigma: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mu", torch.tensor(tuple(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(tuple(sigma), dtype=torch.float32))

    @property
    def num_kernels(self) -> int:
        return int(self.mu.numel())

    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        # similarities: [...]  ->  [..., K]; отклик неотрицателен по построению,
        # и это то свойство, на которое опирается взвешивание гейтом.
        mu = self.mu.to(similarities.dtype)
        sigma = self.sigma.to(similarities.dtype)
        difference = similarities.unsqueeze(-1) - mu
        return torch.exp(-(difference ** 2) / (2.0 * sigma ** 2))


def _masked_mean(vectors: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Среднее по реальным элементам. Пустой набор даёт нулевой вектор, не NaN."""
    weights = mask.unsqueeze(-1).to(vectors.dtype)
    total = (vectors * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


def build_embedding_matrix(
    word_vectors: dict[str, np.ndarray],
    corpus_tokens: Iterable[str],
    embedding_dim: int,
    seed: int = 20260815,
) -> tuple[np.ndarray, dict[str, int]]:
    """Таблица ``[vocab, D]`` и отображение токен -> id.

    Порядок словаря: ``<PAD>``, ``<UNK>``, затем слова Word2Vec, затем новые
    слова корпуса. Известные слова получают свой вектор Word2Vec; новые —
    случайный с тем же mean/std, что у Word2Vec, чтобы не выделяться по норме и
    не ломать масштаб косинусов; ``<PAD>`` — нулевой.

    Каждый встреченный в корпусе токен попадает в словарь: артикулы и модельные
    номера почти всегда редкие, и схлопывание их в общий ``<UNK>`` сделало бы все
    незнакомые коды одинаковыми — худший исход для точного совпадения.
    (Альтернатива, проверенная в ``knrm_model.vector_for_unknown``: выводить
    вектор детерминированно из строки токена, тогда один и тот же незнакомый код
    совпадает сам с собой и на инференсе. Здесь по спецификации используется
    ``<UNK>``.)
    """
    known = list(word_vectors)
    if known:
        stacked = np.stack([np.asarray(word_vectors[token], dtype=np.float32)
                            for token in known])
        if stacked.shape[1] != embedding_dim:
            raise ValueError(f"Word2Vec имеет размерность {stacked.shape[1]}, "
                             f"а запрошено {embedding_dim}")
        mean, std = float(stacked.mean()), float(stacked.std())
    else:
        stacked = np.zeros((0, embedding_dim), dtype=np.float32)
        mean, std = 0.0, 0.1

    vocabulary = [PAD_TOKEN, UNK_TOKEN] + known
    seen = set(vocabulary)
    fresh = [token for token in corpus_tokens if token not in seen and not seen.add(token)]
    vocabulary.extend(fresh)

    generator = np.random.default_rng(seed)
    table = generator.normal(mean, max(std, 1e-6),
                             size=(len(vocabulary), embedding_dim)).astype(np.float32)
    table[PAD_ID] = 0.0
    if known:
        table[2:2 + len(known)] = stacked
    token_id = {token: index for index, token in enumerate(vocabulary)}
    return table, token_id


class ProductMatcher(nn.Module):
    """Четыре канала взаимодействия над одной таблицей эмбеддингов."""

    def __init__(self, embedding_weight: torch.Tensor | np.ndarray,
                 config: KNRMConfig | None = None) -> None:
        super().__init__()
        self.config = config or KNRMConfig()
        weight = torch.as_tensor(embedding_weight, dtype=torch.float32)
        if weight.dim() != 2:
            raise ValueError("таблица эмбеддингов должна быть [vocab, D]")
        if weight.shape[1] != self.config.embedding_dim:
            raise ValueError(f"таблица имеет D={weight.shape[1]}, "
                             f"конфиг требует {self.config.embedding_dim}")
        self.embedding = nn.Embedding.from_pretrained(
            weight.clone(), freeze=self.config.freeze_embeddings,
            sparse=self.config.sparse_embedding, padding_idx=PAD_ID,
        )

        mu, sigma = self.config.kernels()
        self.kernels = KernelBank(mu, sigma)
        dim, num_kernels = self.config.embedding_dim, self.kernels.num_kernels

        # Гейт «название -> атрибут»: [k, t̄, k⊙t̄, |k−t̄|, cos] = 4D + 1.
        self.cross_gate = self._gate_mlp(4 * dim + 1)
        # Гейт «атрибут -> атрибут»: [k_A+k_B, |k_A−k_B|, k_A⊙k_B, cos] = 3D + 1.
        # Все четыре блока симметричны, поэтому g(i, j) == g(j, i) тождественно.
        self.attr_gate = self._gate_mlp(3 * dim + 1)

        self.final_mlp = nn.Sequential(
            nn.Linear(4 * num_kernels, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def _gate_mlp(self, in_features: int) -> nn.Sequential:
        hidden = nn.Linear(in_features, self.config.gate_hidden_dim)
        out = nn.Linear(self.config.gate_hidden_dim, 1)
        # Положительный bias на выходе: на старте sigmoid ≈ 0.88, то есть почти
        # все атрибуты открыты. Гейт должен научиться закрывать лишнее, а не
        # начинать с закрытых ворот, из-за которых градиент до ядер не дойдёт.
        nn.init.constant_(out.bias, self.config.gate_bias_init)
        return nn.Sequential(hidden, nn.GELU(), out)

    # ---------------------------------------------------------------- encoding

    def encode_item(self, item: ItemTensors) -> EncodedItem:
        """Lookup + L2-нормировка + согласование масок."""
        item.validate()
        eps = self.config.eps

        title_mask = item.title_mask.to(self.embedding.weight.dtype)
        title_vectors = F.normalize(self.embedding(item.title_ids), p=2, dim=-1, eps=eps)
        title_vectors = title_vectors * title_mask.unsqueeze(-1)   # [B, Lt, D]
        # Средний вектор названия для гейта. Нормируем: гейт сравнивает его с
        # ключом, и обе стороны должны жить на единичной сфере, иначе признак
        # cos(k, t̄) перестаёт быть косинусом, а |k − t̄| зависит от длины названия.
        title_mean = F.normalize(_masked_mean(title_vectors, title_mask, dim=1),
                                 p=2, dim=-1, eps=eps)             # [B, D]

        attribute_mask = item.attribute_mask.to(title_mask.dtype)  # [B, Na]
        # Токены паддинг-атрибутов гасим здесь один раз, чтобы дальше ни один
        # знаменатель не мог их посчитать.
        key_token_mask = item.key_token_mask.to(title_mask.dtype) * attribute_mask.unsqueeze(-1)
        value_mask = item.value_token_mask.to(title_mask.dtype) * attribute_mask.unsqueeze(-1)

        key_tokens = F.normalize(self.embedding(item.key_ids), p=2, dim=-1, eps=eps)
        key_tokens = key_tokens * key_token_mask.unsqueeze(-1)     # [B, Na, Lk, D]
        key_vectors = F.normalize(_masked_mean(key_tokens, key_token_mask, dim=2),
                                  p=2, dim=-1, eps=eps)            # [B, Na, D]

        value_vectors = F.normalize(self.embedding(item.value_ids), p=2, dim=-1, eps=eps)
        value_vectors = value_vectors * value_mask.unsqueeze(-1)   # [B, Na, Lv, D]

        return EncodedItem(title_vectors, title_mask, title_mean,
                           key_vectors, value_vectors, value_mask, attribute_mask)

    # ------------------------------------------------------------------- gates

    def build_cross_gate(self, title_mean: torch.Tensor, key_vectors: torch.Tensor
                         ) -> torch.Tensor:
        """[B, D] и [B, Na, D] -> [B, Na] в (0, 1)."""
        expanded = title_mean.unsqueeze(1).expand_as(key_vectors)          # [B, Na, D]
        cosine = (key_vectors * expanded).sum(dim=-1, keepdim=True)        # [B, Na, 1]
        features = torch.cat([key_vectors, expanded, key_vectors * expanded,
                              (key_vectors - expanded).abs(), cosine], dim=-1)
        return torch.sigmoid(self.cross_gate(features)).squeeze(-1)        # [B, Na]

    def build_attribute_gate(self, keys_a: torch.Tensor, keys_b: torch.Tensor
                             ) -> torch.Tensor:
        """[B, Na, D] и [B, Nb, D] -> [B, Na, Nb] в (0, 1), симметрично по (i, j)."""
        left = keys_a.unsqueeze(2)                                         # [B, Na, 1, D]
        right = keys_b.unsqueeze(1)                                        # [B, 1, Nb, D]
        cosine = (left * right).sum(dim=-1, keepdim=True)                  # [B, Na, Nb, 1]
        features = torch.cat([left + right, (left - right).abs(),
                              left * right, cosine], dim=-1)
        return torch.sigmoid(self.attr_gate(features)).squeeze(-1)         # [B, Na, Nb]

    # ---------------------------------------------------------------- channels

    def knrm_pool(self, left_vectors: torch.Tensor, left_mask: torch.Tensor,
                  right_vectors: torch.Tensor, right_mask: torch.Tensor) -> torch.Tensor:
        """Канал название-название в одну сторону: [B, Ll, D] × [B, Lr, D] -> [B, K]."""
        similarity = torch.einsum("bid,bjd->bij", left_vectors, right_vectors)  # [B, Ll, Lr]
        pair_mask = left_mask.unsqueeze(2) * right_mask.unsqueeze(1)            # [B, Ll, Lr]
        response = self.kernels(similarity) * pair_mask.unsqueeze(-1)           # [B, Ll, Lr, K]
        per_token = torch.log1p(response.sum(dim=2)) * left_mask.unsqueeze(-1)  # [B, Ll, K]
        denominator = left_mask.sum(dim=1, keepdim=True).clamp_min(1.0)         # [B, 1]
        return per_token.sum(dim=1) / denominator                               # [B, K]

    def cross_title_attributes(self, title_vectors: torch.Tensor, title_mask: torch.Tensor,
                               title_mean: torch.Tensor, key_vectors: torch.Tensor,
                               value_vectors: torch.Tensor, attribute_mask: torch.Tensor,
                               value_mask: torch.Tensor) -> torch.Tensor:
        """Токены названия одного товара против значений атрибутов другого -> [B, K]."""
        gate = self.build_cross_gate(title_mean, key_vectors) * attribute_mask   # [B, Nb]

        similarity = torch.einsum("bpd,bjqd->bpjq", title_vectors, value_vectors)
        pair_mask = title_mask[:, :, None, None] * value_mask[:, None, :, :]     # [B,Lt,Nb,Lv]
        response = self.kernels(similarity) * pair_mask.unsqueeze(-1)            # [...,Lv,K]
        per_attribute = response.sum(dim=3)                                      # [B,Lt,Nb,K]
        per_attribute = per_attribute * gate[:, None, :, None]

        # Делим на число атрибутов, иначе товар с длинным списком получал бы
        # больший score просто за длину списка.
        attribute_count = attribute_mask.sum(dim=1).clamp_min(1.0)               # [B]
        pooled = per_attribute.sum(dim=2) / attribute_count[:, None, None]       # [B, Lt, K]
        per_token = torch.log1p(pooled) * title_mask.unsqueeze(-1)
        denominator = title_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return per_token.sum(dim=1) / denominator                                # [B, K]

    def attribute_attribute(self, keys_a: torch.Tensor, values_a: torch.Tensor,
                            attribute_mask_a: torch.Tensor, value_mask_a: torch.Tensor,
                            keys_b: torch.Tensor, values_b: torch.Tensor,
                            attribute_mask_b: torch.Tensor, value_mask_b: torch.Tensor,
                            chunk_size: int | None = None) -> torch.Tensor:
        """Значения атрибутов A против значений атрибутов B, A -> B: [B, K].

        Чанкование идёт по атрибутам A и математику не меняет: слагаемые внешней
        суммы по ``i`` независимы, поэтому их можно накапливать порциями. Гейт
        считается внутри чанка, чтобы не держать ``[B, Na, Nb, 3D+1]`` целиком.
        """
        batch, num_a = attribute_mask_a.shape
        num_kernels = self.kernels.num_kernels
        step = chunk_size or self.config.attribute_chunk_size or num_a
        if step <= 0:
            raise ValueError("размер чанка должен быть положительным")

        attribute_count_b = attribute_mask_b.sum(dim=1).clamp_min(1.0)            # [B]
        total = torch.zeros(batch, num_kernels, dtype=values_a.dtype, device=values_a.device)

        for start in range(0, num_a, step):
            stop = min(start + step, num_a)
            keys_chunk = keys_a[:, start:stop]                                    # [B, C, D]
            values_chunk = values_a[:, start:stop]                                # [B, C, Lv, D]
            value_mask_chunk = value_mask_a[:, start:stop]                        # [B, C, Lv]
            attribute_chunk = attribute_mask_a[:, start:stop]                     # [B, C]

            gate = self.build_attribute_gate(keys_chunk, keys_b)                  # [B, C, Nb]
            gate = gate * attribute_chunk.unsqueeze(-1) * attribute_mask_b.unsqueeze(1)

            similarity = torch.einsum("bcpd,bjqd->bcpjq", values_chunk, values_b)
            pair_mask = (value_mask_chunk[:, :, :, None, None]
                         * value_mask_b[:, None, None, :, :])                     # [B,C,Lv,Nb,Lv]
            response = self.kernels(similarity) * pair_mask.unsqueeze(-1)
            per_attribute = response.sum(dim=4)                                   # [B,C,Lv,Nb,K]
            per_attribute = per_attribute * gate[:, :, None, :, None]

            pooled = per_attribute.sum(dim=3) / attribute_count_b[:, None, None, None]
            per_token = torch.log1p(pooled) * value_mask_chunk.unsqueeze(-1)      # [B,C,Lv,K]
            token_count = value_mask_chunk.sum(dim=2).clamp_min(1.0)              # [B, C]
            per_chunk_attribute = per_token.sum(dim=2) / token_count.unsqueeze(-1)
            total = total + (per_chunk_attribute * attribute_chunk.unsqueeze(-1)).sum(dim=1)

        return total / attribute_mask_a.sum(dim=1, keepdim=True).clamp_min(1.0)   # [B, K]

    # ----------------------------------------------------------------- forward

    def features(self, item_a: ItemTensors, item_b: ItemTensors) -> torch.Tensor:
        """[B, 4K]: z_tt, z_aa, z_cross_mean, z_cross_diff — все инвариантны к перестановке."""
        a = self.encode_item(item_a)
        b = self.encode_item(item_b)
        chunk = self.config.attribute_chunk_size

        z_tt = 0.5 * (self.knrm_pool(a.title_vectors, a.title_mask,
                                     b.title_vectors, b.title_mask)
                      + self.knrm_pool(b.title_vectors, b.title_mask,
                                       a.title_vectors, a.title_mask))

        z_aa = 0.5 * (self.attribute_attribute(a.key_vectors, a.value_vectors,
                                               a.attribute_mask, a.value_mask,
                                               b.key_vectors, b.value_vectors,
                                               b.attribute_mask, b.value_mask, chunk)
                      + self.attribute_attribute(b.key_vectors, b.value_vectors,
                                                 b.attribute_mask, b.value_mask,
                                                 a.key_vectors, a.value_vectors,
                                                 a.attribute_mask, a.value_mask, chunk))

        # Одни и те же веса на оба направления: это одно взаимодействие,
        # посмотренное с двух сторон, а не два разных.
        z_title_a_attr_b = self.cross_title_attributes(
            a.title_vectors, a.title_mask, a.title_mean,
            b.key_vectors, b.value_vectors, b.attribute_mask, b.value_mask)
        z_title_b_attr_a = self.cross_title_attributes(
            b.title_vectors, b.title_mask, b.title_mean,
            a.key_vectors, a.value_vectors, a.attribute_mask, a.value_mask)

        z_cross_mean = 0.5 * (z_title_a_attr_b + z_title_b_attr_a)
        z_cross_diff = (z_title_a_attr_b - z_title_b_attr_a).abs()

        return torch.cat([z_tt, z_aa, z_cross_mean, z_cross_diff], dim=-1)  # [B, 4K]

    def forward(self, item_a: ItemTensors, item_b: ItemTensors) -> torch.Tensor:
        return self.final_mlp(self.features(item_a, item_b)).squeeze(-1)    # [B]

    @torch.no_grad()
    def score(self, item_a: ItemTensors, item_b: ItemTensors) -> torch.Tensor:
        return torch.sigmoid(self.forward(item_a, item_b))

    def parameter_groups(self, learning_rate: float, embedding_scale: float = 0.1
                         ) -> list[dict]:
        """Таблице — меньший lr: она инициализирована Word2Vec, и полный шаг
        разрушал бы предобученную геометрию быстрее, чем голова успевает ею
        воспользоваться."""
        rest = [parameter for name, parameter in self.named_parameters()
                if not name.startswith("embedding.")]
        groups = [{"params": rest, "lr": learning_rate}]
        if not self.config.freeze_embeddings:
            groups.append({"params": list(self.embedding.parameters()),
                           "lr": learning_rate * embedding_scale})
        return groups

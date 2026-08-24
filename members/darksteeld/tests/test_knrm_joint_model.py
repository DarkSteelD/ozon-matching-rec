"""Тесты симметричной KNRM-сети: симметрия, маски, гейты, знаки, градиенты.

Запуск: .venv/bin/python -m pytest members/darksteeld/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knrm_joint_model import (  # noqa: E402
    PAD_ID, ItemTensors, KNRMConfig, KernelBank, ProductMatcher,
    build_embedding_matrix, default_kernels,
)

DIM = 16
VOCAB = ["<PAD>", "<UNK>", "шуруповерт", "bosch", "gsr", "12", "v",
         "бренд", "модель", "напряжение", "makita", "дрель", "цвет", "красный", "синий"]
INDEX = {token: position for position, token in enumerate(VOCAB)}


def make_model(seed: int = 0, **overrides) -> ProductMatcher:
    """Ортогональные-ish эмбеддинги: так совпадение токена даёт косинус 1,
    а несовпадение — около нуля, и тесты читают отклик ядер однозначно."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(len(VOCAB), DIM, generator=generator)
    weight[PAD_ID] = 0.0
    config = KNRMConfig(embedding_dim=DIM, hidden_dim=8, gate_hidden_dim=8,
                        dropout=0.0, **overrides)
    model = ProductMatcher(weight, config)
    model.eval()
    return model


def encode(titles: list[list[str]], attributes: list[list[tuple[list[str], list[str]]]],
           max_title: int | None = None, max_attributes: int | None = None,
           max_key: int | None = None, max_value: int | None = None) -> ItemTensors:
    """Списки токенов -> тензоры с паддингом. Пустые списки допустимы."""
    batch = len(titles)
    max_title = max_title or max(1, max((len(t) for t in titles), default=1))
    max_attributes = max_attributes or max(1, max((len(a) for a in attributes), default=1))
    max_key = max_key or max(1, max((len(k) for a in attributes for k, _ in a), default=1))
    max_value = max_value or max(1, max((len(v) for a in attributes for _, v in a), default=1))

    title_ids = torch.zeros(batch, max_title, dtype=torch.long)
    title_mask = torch.zeros(batch, max_title)
    key_ids = torch.zeros(batch, max_attributes, max_key, dtype=torch.long)
    key_mask = torch.zeros(batch, max_attributes, max_key)
    value_ids = torch.zeros(batch, max_attributes, max_value, dtype=torch.long)
    value_mask = torch.zeros(batch, max_attributes, max_value)
    attribute_mask = torch.zeros(batch, max_attributes)

    for row, tokens in enumerate(titles):
        for position, token in enumerate(tokens[:max_title]):
            title_ids[row, position] = INDEX[token]
            title_mask[row, position] = 1.0
    for row, attrs in enumerate(attributes):
        for slot, (keys, values) in enumerate(attrs[:max_attributes]):
            attribute_mask[row, slot] = 1.0
            for position, token in enumerate(keys[:max_key]):
                key_ids[row, slot, position] = INDEX[token]
                key_mask[row, slot, position] = 1.0
            for position, token in enumerate(values[:max_value]):
                value_ids[row, slot, position] = INDEX[token]
                value_mask[row, slot, position] = 1.0

    return ItemTensors(title_ids, title_mask, key_ids, key_mask,
                       value_ids, value_mask, attribute_mask)


SCREWDRIVER = (["шуруповерт", "bosch", "gsr", "12", "v"],
               [(["бренд"], ["bosch"]), (["модель"], ["gsr"]), (["напряжение"], ["12", "v"])])
DRILL = (["дрель", "makita"], [(["бренд"], ["makita"])])


# --------------------------------------------------------------- 1. симметрия

def test_forward_is_symmetric():
    model = make_model().double()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    forward = model(a, b)
    backward = model(b, a)
    assert torch.allclose(forward, backward, atol=1e-10), (forward, backward)


def test_forward_is_symmetric_float32():
    model = make_model()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    assert torch.allclose(model(a, b), model(b, a), atol=1e-5)


# ------------------------------------------------------- 2, 8. маски и паддинг

def test_extra_padding_does_not_change_score():
    model = make_model().double()
    tight_a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    tight_b = encode([DRILL[0]], [DRILL[1]], max_attributes=1)
    wide_a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_title=12, max_attributes=8,
                    max_key=5, max_value=7)
    wide_b = encode([DRILL[0]], [DRILL[1]], max_title=12, max_attributes=8,
                    max_key=5, max_value=7)
    assert torch.allclose(model(tight_a, tight_b), model(wide_a, wide_b), atol=1e-10)


def test_fully_masked_attribute_does_not_change_score():
    """Добавленный слот атрибута с нулевой маской обязан быть невидим — в том
    числе в знаменателях, которые делят на число атрибутов."""
    model = make_model().double()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=1)

    padded = encode([DRILL[0]], [DRILL[1]], max_attributes=4)
    # Слот 1 заполним мусорными id, но оставим attribute_mask = 0.
    padded.key_ids[0, 1, 0] = INDEX["цвет"]
    padded.value_ids[0, 1, 0] = INDEX["красный"]
    assert padded.attribute_mask[0, 1] == 0.0

    assert torch.allclose(model(a, b), model(a, padded), atol=1e-10)


def test_item_without_attributes_is_finite():
    model = make_model().double()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    empty = encode([["дрель"]], [[]], max_attributes=3, max_key=2, max_value=2)
    assert empty.attribute_mask.sum() == 0.0
    logit = model(a, empty)
    assert torch.isfinite(logit).all()
    both_empty = model(empty, empty)
    assert torch.isfinite(both_empty).all()


def test_empty_title_and_empty_everything_are_finite():
    model = make_model().double()
    nothing = encode([[]], [[]], max_title=3, max_attributes=2, max_key=2, max_value=2)
    assert torch.isfinite(model(nothing, nothing)).all()


# ------------------------------------------------- 3. кросс-канал ловит bosch

def test_cross_channel_fires_on_title_to_value_exact_match():
    """`bosch` в названии A и в значении атрибута `бренд` у B: канал titleA->attrB
    обязан дать заметный отклик ядра точного совпадения."""
    model = make_model()
    exact = 0  # ядро с mu = 1.0 идёт первым

    title = encode([["шуруповерт", "bosch"]], [[]], max_attributes=1, max_key=1, max_value=1)
    match = encode([["инструмент"] and ["дрель"]], [[(["бренд"], ["bosch"])]],
                   max_attributes=1, max_key=1, max_value=1)
    other = encode([["дрель"]], [[(["бренд"], ["makita"])]],
                   max_attributes=1, max_key=1, max_value=1)

    a = model.encode_item(title)
    hit = model.encode_item(match)
    miss = model.encode_item(other)

    z_hit = model.cross_title_attributes(a.title_vectors, a.title_mask, a.title_mean,
                                         hit.key_vectors, hit.value_vectors,
                                         hit.attribute_mask, hit.value_mask)
    z_miss = model.cross_title_attributes(a.title_vectors, a.title_mask, a.title_mean,
                                          miss.key_vectors, miss.value_vectors,
                                          miss.attribute_mask, miss.value_mask)
    assert z_hit[0, exact] > 10 * z_miss[0, exact] + 1e-6, (z_hit[0, exact], z_miss[0, exact])


# ------------------------------------------------------------- 4. гейт ключей

def test_key_gate_can_attenuate_matching_values():
    """Гейт умеет ослаблять вклад: при закрытом гейте отклик строго меньше."""
    model = make_model()
    values = encode([[("шуруповерт")] and ["шуруповерт"]],
                    [[(["цвет"], ["красный"])]], max_attributes=1, max_key=1, max_value=1)
    title = encode([["красный"]], [[]], max_attributes=1, max_key=1, max_value=1)

    a = model.encode_item(title)
    b = model.encode_item(values)
    open_gate = model.cross_title_attributes(a.title_vectors, a.title_mask, a.title_mean,
                                             b.key_vectors, b.value_vectors,
                                             b.attribute_mask, b.value_mask)

    # Закрываем гейт, сдвинув bias выходного слоя далеко в минус.
    with torch.no_grad():
        model.cross_gate[-1].bias.fill_(-20.0)
    closed_gate = model.cross_title_attributes(a.title_vectors, a.title_mask, a.title_mean,
                                               b.key_vectors, b.value_vectors,
                                               b.attribute_mask, b.value_mask)
    assert (closed_gate <= open_gate + 1e-6).all()
    assert closed_gate.abs().sum() < open_gate.abs().sum()


def test_gates_are_bounded_and_attribute_gate_is_symmetric():
    model = make_model().double()
    a = model.encode_item(encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3))
    b = model.encode_item(encode([DRILL[0]], [DRILL[1]], max_attributes=3))

    cross = model.build_cross_gate(a.title_mean, b.key_vectors)
    assert ((cross > 0.0) & (cross < 1.0)).all()

    forward = model.build_attribute_gate(a.key_vectors, b.key_vectors)
    backward = model.build_attribute_gate(b.key_vectors, a.key_vectors)
    assert torch.allclose(forward, backward.transpose(1, 2), atol=1e-12)
    assert ((forward > 0.0) & (forward < 1.0)).all()


# --------------------------------------------- 5. два минуса не дают ложный плюс

def test_two_negative_cosines_do_not_create_positive_evidence():
    """Регрессия на дефект прежней атрибутной модели.

    Там атрибут был ``normalize(k ⊙ v)``, и подстановка ``k_B = −k_A``,
    ``v_B = −v_A`` давала косинус ровно +1: два расхождения читались как точное
    совпадение. Здесь ключ входит только через sigmoid-гейт, умножающий
    неотрицательный отклик ядра, поэтому знак свидетельства поменять нечем.
    """
    model = make_model().double()
    kernel_response = model.kernels(torch.linspace(-1.0, 1.0, 41, dtype=torch.double))
    assert (kernel_response >= 0.0).all(), "отклик ядер обязан быть неотрицательным"

    keys_a = torch.nn.functional.normalize(torch.randn(2, 3, DIM, dtype=torch.double), dim=-1)
    gate = model.build_attribute_gate(keys_a, -keys_a)
    assert ((gate > 0.0) & (gate < 1.0)).all(), "гейт обязан жить в (0, 1)"

    # Старая конструкция на тех же данных: демонстрируем, что она вырождена,
    # а новая — нет.
    values_a = torch.nn.functional.normalize(torch.randn(2, 3, DIM, dtype=torch.double), dim=-1)
    old_slot_a = torch.nn.functional.normalize(keys_a * values_a, dim=-1)
    old_slot_b = torch.nn.functional.normalize((-keys_a) * (-values_a), dim=-1)
    old_cosine = (old_slot_a * old_slot_b).sum(-1)
    assert torch.allclose(old_cosine, torch.ones_like(old_cosine)), \
        "контроль: старая конструкция действительно читает антиподы как совпадение"

    # Новая: свидетельство берётся из косинуса ЗНАЧЕНИЙ, а он у антиподов равен −1,
    # и ядро точного совпадения на нём не срабатывает.
    value_cosine = (values_a * (-values_a)).sum(-1)
    exact = model.kernels(value_cosine)[..., 0]
    assert (exact < 1e-6).all(), "ядро точного совпадения не должно срабатывать на антиподах"


def test_gate_only_attenuates_never_flips_sign():
    model = make_model().double()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    ea, eb = model.encode_item(a), model.encode_item(b)
    z = model.attribute_attribute(ea.key_vectors, ea.value_vectors, ea.attribute_mask,
                                  ea.value_mask, eb.key_vectors, eb.value_vectors,
                                  eb.attribute_mask, eb.value_mask)
    assert (z >= 0.0).all(), "log1p от неотрицательной суммы не может быть отрицательным"


# ------------------------------------------------------------- 6. чанкование

def test_attribute_chunking_matches_full_computation():
    model = make_model().double()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    ea, eb = model.encode_item(a), model.encode_item(b)
    full = model.attribute_attribute(ea.key_vectors, ea.value_vectors, ea.attribute_mask,
                                     ea.value_mask, eb.key_vectors, eb.value_vectors,
                                     eb.attribute_mask, eb.value_mask, chunk_size=None)
    for chunk in (1, 2, 3):
        chunked = model.attribute_attribute(ea.key_vectors, ea.value_vectors,
                                            ea.attribute_mask, ea.value_mask,
                                            eb.key_vectors, eb.value_vectors,
                                            eb.attribute_mask, eb.value_mask,
                                            chunk_size=chunk)
        assert torch.allclose(full, chunked, atol=1e-12), chunk


# -------------------------------------------------------------- 7. градиенты

def test_gradients_reach_every_component():
    model = make_model()
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    target = torch.ones(1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(a, b), target)
    loss.backward()

    grad = model.embedding.weight.grad
    assert grad is not None and grad.abs().sum() > 0, "градиент не дошёл до таблицы"
    assert grad[PAD_ID].abs().sum() == 0, "padding_idx обязан остаться без градиента"
    used = {int(i) for i in a.title_ids.flatten().tolist() + b.title_ids.flatten().tolist()}
    used |= {int(i) for i in a.value_ids.flatten().tolist()} - {PAD_ID}
    assert any(grad[i].abs().sum() > 0 for i in used if i != PAD_ID)

    for name in ("cross_gate", "attr_gate", "final_mlp"):
        module = getattr(model, name)
        total = sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)
        assert total > 0, f"градиент не дошёл до {name}"


def test_frozen_embeddings_get_no_gradient():
    model = make_model(freeze_embeddings=True)
    a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    model(a, b).sum().backward()
    assert model.embedding.weight.grad is None
    assert not model.embedding.weight.requires_grad


def test_parameter_groups_lower_the_embedding_lr():
    model = make_model()
    groups = model.parameter_groups(1e-3, embedding_scale=0.1)
    assert len(groups) == 2
    assert pytest.approx(groups[1]["lr"]) == 1e-4
    frozen = make_model(freeze_embeddings=True).parameter_groups(1e-3)
    assert len(frozen) == 1


# ------------------------------------------------------- прочее: ядра, словарь

def test_default_kernels_have_exact_match_and_cover_the_range():
    mu, sigma = default_kernels(11)
    assert mu[0] == 1.0 and sigma[0] < 1e-2
    assert len(mu) == len(sigma) == 11
    assert min(mu[1:]) == pytest.approx(-0.9) and max(mu[1:]) == pytest.approx(0.9)
    bank = KernelBank(mu, sigma)
    response = bank(torch.tensor([1.0]))
    assert response[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_build_embedding_matrix_layout_and_statistics():
    rng = np.random.default_rng(0)
    word_vectors = {token: rng.normal(0.0, 0.5, DIM).astype(np.float32)
                    for token in ("bosch", "дрель")}
    table, token_id = build_embedding_matrix(word_vectors, ["gsr", "bosch", "новый"], DIM)

    assert token_id["<PAD>"] == PAD_ID and token_id["<UNK>"] == 1
    assert np.allclose(table[PAD_ID], 0.0)
    assert np.allclose(table[token_id["bosch"]], word_vectors["bosch"])
    assert "gsr" in token_id and "новый" in token_id, "новые токены обязаны попасть в словарь"
    assert len({token_id["gsr"], token_id["новый"], token_id["bosch"]}) == 3

    known = np.stack(list(word_vectors.values()))
    fresh = table[[token_id["gsr"], token_id["новый"]]]
    assert abs(fresh.std() - known.std()) < 0.4 * known.std() + 0.2


def test_batch_rows_are_independent():
    """LayerNorm вместо BatchNorm: строка батча не должна зависеть от соседей."""
    model = make_model().double()
    a = encode([SCREWDRIVER[0], DRILL[0]], [SCREWDRIVER[1], DRILL[1]], max_attributes=3)
    b = encode([DRILL[0], SCREWDRIVER[0]], [DRILL[1], SCREWDRIVER[1]], max_attributes=3)
    together = model(a, b)

    first_a = encode([SCREWDRIVER[0]], [SCREWDRIVER[1]], max_attributes=3)
    first_b = encode([DRILL[0]], [DRILL[1]], max_attributes=3)
    alone = model(first_a, first_b)
    assert torch.allclose(together[:1], alone, atol=1e-10)

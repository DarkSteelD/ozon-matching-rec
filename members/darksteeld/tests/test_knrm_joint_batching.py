"""Обрезка обязана быть тождественной: скор не должен от неё меняться."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knrm_joint_batching import (  # noqa: E402
    attribute_counts, bucketed_batches, make_batch,
)
from knrm_joint_model import KNRMConfig, ProductMatcher  # noqa: E402

VOCAB, DIM = 200, 16
MAX_TITLE, MAX_ATTRS, MAX_KEY, MAX_VALUE = 20, 24, 4, 6


def corpus(items: int = 64, seed: int = 0):
    """Корпус с РАЗНЫМИ длинами: у товара i занято i%… позиций, остальное паддинг."""
    generator = np.random.default_rng(seed)
    titles = np.zeros((items, MAX_TITLE), dtype=np.int32)
    keys = np.zeros((items, MAX_ATTRS, MAX_KEY), dtype=np.int32)
    values = np.zeros((items, MAX_ATTRS, MAX_VALUE), dtype=np.int32)
    for row in range(items):
        for column in range(1 + row % MAX_TITLE):
            titles[row, column] = generator.integers(1, VOCAB)
        for slot in range(1 + row % 7):                    # 1..7 атрибутов
            for column in range(1 + row % MAX_KEY):
                keys[row, slot, column] = generator.integers(1, VOCAB)
            for column in range(1 + row % 3):              # короткие значения
                values[row, slot, column] = generator.integers(1, VOCAB)
    return (torch.from_numpy(titles), torch.from_numpy(keys), torch.from_numpy(values))


def make_model(seed: int = 0) -> ProductMatcher:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(VOCAB, DIM, generator=generator)
    weight[0] = 0.0
    model = ProductMatcher(weight, KNRMConfig(embedding_dim=DIM, hidden_dim=8,
                                              gate_hidden_dim=8, dropout=0.0))
    return model.double().eval()


def test_trimming_does_not_change_the_score():
    """Главное свойство: обрезка режет только позиции, пустые у всех строк батча."""
    titles, keys, values = corpus()
    model = make_model()
    rows_a = np.arange(0, 32)
    rows_b = np.arange(32, 64)

    trimmed_a = make_batch(titles, keys, values, rows_a, trim=True)
    trimmed_b = make_batch(titles, keys, values, rows_b, trim=True)
    full_a = make_batch(titles, keys, values, rows_a, trim=False)
    full_b = make_batch(titles, keys, values, rows_b, trim=False)

    assert trimmed_a.key_ids.shape[1] < full_a.key_ids.shape[1], "обрезка обязана что-то срезать"
    assert torch.allclose(model(trimmed_a, trimmed_b), model(full_a, full_b), atol=1e-10)


def test_trimming_keeps_every_real_token():
    titles, keys, values = corpus()
    rows = np.arange(0, 16)
    trimmed = make_batch(titles, keys, values, rows, trim=True)
    full = make_batch(titles, keys, values, rows, trim=False)
    assert trimmed.title_mask.sum() == full.title_mask.sum()
    assert trimmed.value_token_mask.sum() == full.value_token_mask.sum()
    assert trimmed.attribute_mask.sum() == full.attribute_mask.sum()


def test_bucketed_batches_cover_every_row_once():
    counts = np.random.default_rng(0).integers(1, 24, 1000)
    batches = bucketed_batches(counts, batch_size=64,
                               generator=np.random.default_rng(1))
    seen = np.concatenate(batches)
    assert len(seen) == 1000
    assert set(seen.tolist()) == set(range(1000))


def test_bucketed_batches_are_homogeneous_and_shuffled():
    counts = np.random.default_rng(0).integers(1, 24, 4096)
    batches = bucketed_batches(counts, batch_size=64,
                               generator=np.random.default_rng(1))
    spread = np.mean([counts[batch].max() - counts[batch].min() for batch in batches])
    assert spread < 3.0, f"батчи должны быть однородными по длине, разброс {spread:.1f}"
    firsts = [int(batch[0]) for batch in batches]
    assert firsts != sorted(firsts), "порядок батчей обязан перемешиваться"


def test_attribute_counts_match_the_mask():
    titles, keys, values = corpus()
    counts = attribute_counts(keys, values)
    batch = make_batch(titles, keys, values, np.arange(len(counts)), trim=False)
    assert np.array_equal(counts, batch.attribute_mask.sum(dim=1).numpy().astype(int))

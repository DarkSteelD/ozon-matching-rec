"""Сборка батчей для совместной KNRM: обрезка по факту и бакетирование по длине.

**Почему это не микрооптимизация.** Стоимость канала «атрибуты против
атрибутов» пропорциональна ``(Na · Lv)²``: einsum считает все позиции, а маски
применяются уже после и умножают посчитанное на ноль. Паддинг здесь платный.

Замер на ручной вселенной: медиана 11 атрибутов при паддинге до 24 и медиана
**1** токен значения при паддинге до 6. Но обрезка по максимуму батча сама по
себе не даёт ничего: при случайном порядке в батче из 256 почти наверняка
найдётся товар с 24 атрибутами, и максимум остаётся 24. Экономия появляется
только вместе с сортировкой по длине — тогда средний максимум падает до 12, а
стоимость канала до 32.8% от полной, то есть ускорение 3.05×.

Порядок батчей при этом перемешивается (``bucketed_batches``): однородность
нужна внутри батча ради скорости, а последовательность шагов должна оставаться
случайной, иначе обучение сначала увидит все бедные атрибутами товары, потом все
богатые.
"""

from __future__ import annotations

import numpy as np
import torch

from knrm_joint_model import PAD_ID, ItemTensors


def _trim(ids: torch.Tensor, dim: int) -> int:
    """Сколько позиций по оси ``dim`` реально заняты хотя бы в одной строке."""
    occupied = (ids != PAD_ID)
    while occupied.dim() > dim + 1:
        occupied = occupied.any(dim=-1)
    while occupied.dim() > 1:
        occupied = occupied.any(dim=0)
    filled = int(occupied.sum())
    # Позиции заполняются слева направо, поэтому число занятых позиций и есть
    # граница обрезки; оставляем минимум одну, чтобы форма не выродилась.
    return max(filled, 1)


def make_batch(titles: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
               rows: np.ndarray, trim: bool = True,
               device: torch.device | str | None = None) -> ItemTensors:
    """Строки закодированного корпуса -> ``ItemTensors`` без паддинг-хвостов.

    Маски выводятся из id: ``PAD_ID = 0`` означает «нет токена». Обрезка режет
    только позиции, где маска нулевая у всех строк батча, поэтому на результат
    она не влияет — это проверено тестом на совпадение с необрезанным расчётом.
    """
    title_ids = titles[rows].long()
    key_ids = keys[rows].long()
    value_ids = values[rows].long()

    if trim:
        title_ids = title_ids[:, :_trim(title_ids, 1)]
        # Атрибут реален, только если есть и ключ, и значение: слот
        # "артикул": "" места в знаменателе занимать не должен.
        real = ((key_ids != PAD_ID).any(dim=2) & (value_ids != PAD_ID).any(dim=2))
        attribute_count = max(int(real.any(dim=0).sum()), 1)
        key_ids = key_ids[:, :attribute_count, :_trim(key_ids, 2)]
        value_ids = value_ids[:, :attribute_count, :_trim(value_ids, 2)]

    title_mask = (title_ids != PAD_ID).float()
    key_token_mask = (key_ids != PAD_ID).float()
    value_token_mask = (value_ids != PAD_ID).float()
    attribute_mask = ((key_ids != PAD_ID).any(dim=2)
                      & (value_ids != PAD_ID).any(dim=2)).float()
    batch = ItemTensors(title_ids, title_mask, key_ids, key_token_mask,
                        value_ids, value_token_mask, attribute_mask)
    # Переносим уже ОБРЕЗАННЫЙ батч: копировать на устройство полные формы
    # означало бы гнать через шину ровно тот паддинг, который мы только что
    # срезали.
    return batch.to(device) if device is not None else batch


def attribute_counts(keys: torch.Tensor, values: torch.Tensor) -> np.ndarray:
    """Число реальных атрибутов у каждого товара корпуса: ключ на сортировку."""
    real = (keys != PAD_ID).any(dim=2) & (values != PAD_ID).any(dim=2)
    return real.sum(dim=1).numpy()


def bucketed_batches(order_key: np.ndarray, batch_size: int,
                     generator: np.random.Generator | None = None) -> list[np.ndarray]:
    """Индексы батчей: однородные по длине внутри, перемешанные снаружи.

    ``order_key`` — величина, по которой сортируем (число атрибутов пары). Ties
    разрываются случайно, иначе внутри одной длины порядок был бы одним и тем же
    на каждой эпохе.
    """
    count = len(order_key)
    if generator is None:
        generator = np.random.default_rng()
    noise = generator.random(count)
    order = np.lexsort((noise, order_key))
    batches = [order[start:start + batch_size] for start in range(0, count, batch_size)]
    generator.shuffle(batches)
    return batches

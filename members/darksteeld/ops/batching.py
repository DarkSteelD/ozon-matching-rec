"""Группировка примеров в батчи по длине. Общий модуль для претрена и дообучения."""
from __future__ import annotations

import numpy as np
import torch


class LengthBucketBatches(torch.utils.data.Sampler):
    """Батчи из пар близкой длины: борьба с добивкой пустыми токенами.

    Батч добивается до самой длинной пары в нём, поэтому при случайном
    перемешивании один длинный экземпляр растягивает весь батч. Замер на 40
    тысячах пар: при батче 16 обрабатывается в 1.82 раза больше токенов, чем
    полезных, то есть 45% вычислений уходит в пустоту. Группировка по длине
    снижает этот множитель до 1.11.

    Полная сортировка по длине убила бы случайность и связала бы состав батча
    с длиной. Поэтому сортируем не весь корпус, а окно из window батчей: внутри
    окна длины сближаются, но само окно набирается случайно, и порядок готовых
    батчей потом ещё раз перемешивается.

    Длина считается в ТОКЕНАХ, а не в символах. Символьный прокси проще, но
    добивка идёт по токенам, и на нём множитель выходит 1.25 вместо 1.11 —
    треть выигрыша теряется. Разовая токенизация фолда стоит около полутора
    минут против полутора часов обучения, так что размен очевиден.
    """

    def __init__(self, lengths, batch_size, window=50, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.window = window
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lengths))
        span = self.batch_size * self.window
        batches = []
        for start in range(0, len(order), span):
            chunk = order[start:start + span]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for at in range(0, len(chunk), self.batch_size):
                batch = chunk[at:at + self.batch_size]
                if len(batch) == self.batch_size:      # drop_last
                    batches.append(batch.tolist())
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.lengths) // self.batch_size

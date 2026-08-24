"""Container entry point — совместная KNRM по названию и атрибутам.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Выход — CSV с колонками id1, id2, predict, по строке на каждую входную пару, в
порядке входа.

Одна сеть считает четыре взаимодействия: имя-имя, атрибуты-атрибуты и два
кросс-канала «имя одного товара против значений атрибутов другого». Модель
обучена на всех 365 654 ручных парах поверх предобучения на 4M пар matches_llm.

**Индексное пространство то же, что при обучении.** Артефакт везёт словарь
целиком, поэтому контейнер ничего не переиндексирует: токен теста ищется в
словаре напрямую, а незнакомый становится PAD и маскируется — ровно как при
обучении. Это отличается от ``knrm_llm_pretrain``/``knrm_attrs_llm``, которые
строят индекс под тестовые токены и восстанавливают незнакомые из строки;
здесь так делать нельзя, потому что таблица обучалась без таких векторов.

**Порядок пар сохраняется.** Внутри батчи сортируются по числу атрибутов —
стоимость канала атрибут-атрибут задаётся формами паддинга, и сортировка режет
её втрое, — но результат раскладывается обратно по исходным позициям.

**Время.** Инференс этой сети на боевом объёме НЕ измерен: замер без
бакетирования дал 13.4 минуты на 365 654 пары при лимите 20 минут на весь
прогон. С бакетированием должно быть заметно меньше, но это оценка. Лог печатает
скорость каждые 50 батчей, чтобы отставание было видно сразу.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 8))

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from knrm_joint_batching import attribute_counts, make_batch  # noqa: E402
from knrm_joint_tokens import parse_attributes, tokenize  # noqa: E402
from knrm_joint_model import PAD_ID, KNRMConfig, ProductMatcher  # noqa: E402


def log(message: str) -> None:
    print(f"[joint] {message}", flush=True)


def encode_items(names: list[str], attributes: list[str], token_id: dict[str, int],
                 shapes: dict[str, int], stemming: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Товары в те же формы, что при обучении. Незнакомый токен -> PAD."""
    count = len(names)
    titles = np.zeros((count, shapes["title"]), dtype=np.int32)
    keys = np.zeros((count, shapes["attrs"], shapes["key_tokens"]), dtype=np.int32)
    values = np.zeros((count, shapes["attrs"], shapes["value_tokens"]), dtype=np.int32)
    unknown, total = 0, 0
    for row, (name, raw) in enumerate(zip(names, attributes)):
        for column, token in enumerate(tokenize(name, stemming)[:shapes["title"]]):
            index = token_id.get(token, PAD_ID)
            titles[row, column] = index
            unknown += index == PAD_ID
            total += 1
        for slot, (key_tokens, value_tokens) in enumerate(
                parse_attributes(raw, stemming)[:shapes["attrs"]]):
            for column, token in enumerate(key_tokens[:shapes["key_tokens"]]):
                keys[row, slot, column] = token_id.get(token, PAD_ID)
            for column, token in enumerate(value_tokens[:shapes["value_tokens"]]):
                index = token_id.get(token, PAD_ID)
                values[row, slot, column] = index
                unknown += index == PAD_ID
                total += 1
    return titles, keys, values, (100 * unknown / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    blob = torch.load(HERE / "model.pt", map_location="cpu", weights_only=False)
    token_id, shapes = blob["token_id"], blob["max_shapes"]
    config = KNRMConfig(**blob["config"])
    model = ProductMatcher(blob["embedding_fp16"].to(torch.float32), config)
    model.load_state_dict(blob["head_state"], strict=False)
    model.eval()
    blob_stemming = blob.get("stemming", False)
    del blob
    log(f"токенизация: {'стеммы' if blob_stemming else 'сырые токены'}")
    log(f"модель: словарь {len(token_id):,}, ядер {model.kernels.num_kernels}, "
        f"загружена за {time.time() - started:.0f}s")

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    log(f"товаров {len(items):,}, пар {len(matches):,}")

    stemming = bool(blob_stemming)
    titles, keys, values, unknown_share = encode_items(
        items["name"].tolist(), items["attributes"].tolist(), token_id, shapes, stemming)
    log(f"закодировано за {time.time() - started:.0f}s; токенов вне словаря "
        f"{unknown_share:.1f}% (становятся PAD)")
    titles_t, keys_t, values_t = (torch.from_numpy(titles), torch.from_numpy(keys),
                                  torch.from_numpy(values))
    counts = attribute_counts(keys_t, values_t)

    row_of_id = {int(i): r for r, i in enumerate(items["id"].tolist())}
    del items
    left = matches["id1"].to_numpy()
    right = matches["id2"].to_numpy()
    known = np.array([int(a) in row_of_id and int(b) in row_of_id
                      for a, b in zip(left, right)])
    rows1 = np.array([row_of_id.get(int(a), 0) for a in left], dtype=np.int64)
    rows2 = np.array([row_of_id.get(int(b), 0) for b in right], dtype=np.int64)

    # Сортировка по числу атрибутов: стоимость канала атрибут-атрибут задаётся
    # формами паддинга, а не содержимым. Результат кладём обратно по позициям.
    order_key = np.maximum(counts[rows1], counts[rows2])
    order = np.argsort(order_key, kind="mergesort")
    predictions = np.full(len(matches), 0.5, dtype=np.float64)
    inference_started = time.time()
    with torch.no_grad():
        for index, start in enumerate(range(0, len(order), args.batch_size)):
            pick = order[start:start + args.batch_size]
            item_a = make_batch(titles_t, keys_t, values_t, rows1[pick])
            item_b = make_batch(titles_t, keys_t, values_t, rows2[pick])
            predictions[pick] = torch.sigmoid(model(item_a, item_b)).numpy()
            if (index + 1) % 50 == 0:
                done = start + len(pick)
                rate = done / max(time.time() - inference_started, 1e-9)
                log(f"  {done:,}/{len(order):,} ({rate:.0f} пар/с, "
                    f"осталось {(len(order) - done) / max(rate, 1e-9) / 60:.1f} мин)")

    if not known.all():
        predictions[~known] = 0.5
        log(f"{int((~known).sum()):,} пар получили 0.5 (товаров нет в items)")

    pd.DataFrame({"id1": left, "id2": right, "predict": predictions}).to_csv(
        args.output_path, index=False)
    log(f"записан {args.output_path} ({len(matches):,} строк), диапазон "
        f"{predictions.min():.6f}..{predictions.max():.6f} — всего "
        f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

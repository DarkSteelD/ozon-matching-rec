"""Точка входа контейнера — KNRM по словарю атрибутов товара.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Вывод: CSV с колонками id1, id2, predict — по строке на каждую входную пару, в
порядке входа.

Модель читает **только атрибуты**, название не используется вообще. Атрибут
кодируется двумя векторами — ключ из одной таблицы, значение из другой — и
представлением атрибута служит их поэлементное произведение. Дальше обычное
ядровое пулирование KNRM, но по матрице сходства атрибутов, а не токенов имени.

Артефакт везёт векторы, адресованные **строкой токена**, потому что на сабмите
товары другие и словарь у них другой. Этот скрипт строит индексное пространство
под те токены, которые есть в присланном файле: токен, знакомый по обучению,
получает обученный вектор, незнакомый — детерминированный вектор из своей же
строки. Для артикулов это принципиально: один и тот же код по обе стороны пары
даёт косинус ровно 1.0 и зажигает ядро точного совпадения, тогда как общая
строка ``<unk>`` сделала бы все незнакомые артикулы неотличимыми друг от друга.

torch, pandas, numpy и pyarrow есть в образе, поэтому ничего не вендорится и
ничего не скачивается.
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

from knrm_attrs_model import (  # noqa: E402
    MAX_ATTRS, MAX_KEY_TOKENS, MAX_VALUE_TOKENS, AttributeKNRM, encode_attributes,
    parse_attributes,
)
from knrm_model import PAD_ID, vector_for_unknown  # noqa: E402


def log(message: str) -> None:
    print(f"[run] {message}", flush=True)


def build_table(tokens_needed: dict[str, int], shipped: dict[str, int],
                vectors: np.ndarray, dim: int) -> tuple[torch.Tensor, int]:
    """Таблица под словарь теста: обученный вектор либо детерминированный из строки."""
    weight = np.zeros((len(tokens_needed) + 1, dim), dtype=np.float32)
    seen = 0
    for token, index in tokens_needed.items():
        row = shipped.get(token)
        if row is None:
            weight[index] = vector_for_unknown(token, dim)
        else:
            weight[index] = vectors[row]
            seen += 1
    return torch.from_numpy(weight), seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    started = time.time()
    torch.set_grad_enabled(False)
    artifact = json.loads((HERE / "artifact.json").read_text(encoding="utf-8"))
    blob = np.load(HERE / "model.npz", allow_pickle=True)
    dim = artifact["dim"]
    shipped_keys = {t: r for r, t in enumerate(blob["key_vocabulary"].tolist())}
    shipped_values = {t: r for r, t in enumerate(blob["value_vocabulary"].tolist())}
    key_vectors = blob["key_vectors"].astype(np.float32)
    value_vectors = blob["value_vectors"].astype(np.float32)
    log(f"артефакт: ключей {len(shipped_keys):,}, значений {len(shipped_values):,}, "
        f"dim {dim}; обучался на torch {artifact['torch_version']} (сейчас {torch.__version__})")

    items = pd.read_parquet(args.items_path, columns=["id", "attributes"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    attributes = items["attributes"].tolist()
    log(f"товаров {len(items):,} | пар {len(matches):,} | прочитано за {time.time() - started:.0f}s")

    # словарь теста
    test_keys: dict[str, int] = {}
    test_values: dict[str, int] = {}
    empty = 0
    for raw in attributes:
        pairs = parse_attributes(raw)
        if not pairs:
            empty += 1
        for key_tokens, value_tokens in pairs:
            for token in key_tokens:
                if token not in test_keys:
                    test_keys[token] = len(test_keys) + 1  # 0 остаётся под PAD
            for token in value_tokens:
                if token not in test_values:
                    test_values[token] = len(test_values) + 1
    key_weight, key_seen = build_table(test_keys, shipped_keys, key_vectors, dim)
    value_weight, value_seen = build_table(test_values, shipped_values, value_vectors, dim)
    log(f"словарь теста: ключей {len(test_keys):,} — обученных {key_seen:,} "
        f"({100 * key_seen / max(len(test_keys), 1):.1f}%); "
        f"значений {len(test_values):,} — обученных {value_seen:,} "
        f"({100 * value_seen / max(len(test_values), 1):.1f}%)")
    if empty:
        log(f"{empty:,} товаров без единого разобранного атрибута")

    model = AttributeKNRM(key_weight, value_weight, sparse=False)
    model.head.weight.copy_(torch.from_numpy(blob["head_weight"]))
    model.head.bias.copy_(torch.from_numpy(blob["head_bias"]))
    model.norm.weight.copy_(torch.from_numpy(blob["bn_weight"]))
    model.norm.bias.copy_(torch.from_numpy(blob["bn_bias"]))
    model.norm.running_mean.copy_(torch.from_numpy(blob["bn_mean"]))
    model.norm.running_var.copy_(torch.from_numpy(blob["bn_var"]))
    model.norm.eps = float(blob["bn_eps"])
    model.eval()

    keys_encoded, values_encoded = encode_attributes(attributes, test_keys, test_values)
    keys_t, values_t = torch.from_numpy(keys_encoded), torch.from_numpy(values_encoded)
    log(f"закодировано: ключи {keys_encoded.nbytes / 1e6:.0f} МБ, "
        f"значения {values_encoded.nbytes / 1e6:.0f} МБ")

    row_of_id = {int(item): row for row, item in enumerate(items["id"].to_numpy())}
    id1 = matches["id1"].to_numpy()
    id2 = matches["id2"].to_numpy()
    known = np.fromiter(((int(a) in row_of_id and int(b) in row_of_id) for a, b in zip(id1, id2)),
                        dtype=bool, count=len(id1))
    rows1 = np.array([row_of_id.get(int(a), 0) for a in id1], dtype=np.int64)
    rows2 = np.array([row_of_id.get(int(b), 0) for b in id2], dtype=np.int64)

    predictions = np.full(len(id1), 0.5, dtype=np.float64)
    order = np.flatnonzero(known)
    for start in range(0, len(order), args.batch_size):
        pick = order[start : start + args.batch_size]
        left = torch.from_numpy(rows1[pick])
        right = torch.from_numpy(rows2[pick])
        predictions[pick] = torch.sigmoid(model(
            keys_t[left].long(), values_t[left].long(),
            keys_t[right].long(), values_t[right].long(),
        )).numpy()
    if not known.all():
        log(f"{int((~known).sum()):,} пар получили 0.5 (товаров нет в файле товаров)")

    pd.DataFrame({"id1": id1, "id2": id2, "predict": predictions}).to_csv(
        args.output_path, index=False)
    log(f"записан {args.output_path} ({len(id1):,} строк) — всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

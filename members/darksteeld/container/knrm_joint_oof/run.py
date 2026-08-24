"""Container entry point — OOF-ансамбль совместной KNRM: четыре модели фолдов.

    python -u run.py --items_path <parquet> --matches_path <parquet> \
                     --output_path <csv>

Выход — CSV с колонками id1, id2, predict, по строке на каждую входную пару, в
порядке входа.

**Почему четыре модели, а не одна.** Обычный путь — доучить пятую модель на всех
ручных парах и отгрузить её. Здесь отгружаются те самые четыре, что дали
OOF-число на замороженных фолдах: модель фолда K обучена на остальных трёх и
фолда K не видела. Их выходы просто усредняются. Ансамбль из четырёх моделей,
обученных на пересекающихся, но разных подвыборках, устойчивее одиночной: то,
что каждая выучила из шума своей выборки, при усреднении гасится, а общее
остаётся. Плюс отгружается ровно то, что измерено, без ещё одного обучения,
результат которого пришлось бы принимать на веру.

Цена — четырёхкратный инференс. Модели считаются последовательно, потому что
держать четыре таблицы по 1.88 ГБ в памяти одновременно незачем: каждая
загружается, отрабатывает по всем парам и выгружается.

**Токенизация задаётся артефактом.** Флаг ``stemming`` лежит в каждом файле
весов; он же определял, как строился словарь. Кодирование делается один раз для
всех четырёх моделей — словарь у них общий, различаются только веса.

Незнакомый токен становится PAD и маскируется — ровно как при обучении.
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
sys.path.append(str(HERE / "vendor"))

from knrm_joint_batching import attribute_counts, make_batch  # noqa: E402
from knrm_joint_model import PAD_ID, KNRMConfig, ProductMatcher  # noqa: E402
from knrm_joint_tokens import parse_attributes, tokenize  # noqa: E402


def log(message: str) -> None:
    print(f"[oof] {message}", flush=True)


def build_lookup(token_id: dict[str, int]) -> tuple[dict[str, int], object]:
    """Поиск «сырой токен -> индекс строки таблицы».

    Модель обучена по стеммам, но стеммить в контейнере не обязательно: стем
    нужен лишь затем, чтобы найти строку, а токен без строки всё равно уходит в
    PAD. Поэтому основной путь — готовое отображение ``stem_map.npz``,
    посчитанное по всему каталогу. Ставить контейнер в зависимость от того,
    импортируется ли C-расширение PyStemmer в чужом образе, нельзя: не
    загрузится — токенизация разойдётся с обучением, и модель отдаст мусор
    вместо ошибки.

    Стеммер остаётся запасным путём для токенов, которых в каталоге не было, но
    чей стем в словаре есть. Нет стеммера — такие токены просто становятся PAD.
    """
    lookup: dict[str, int] = {}
    path = HERE / "stem_map.npz"
    if path.is_file():
        blob = np.load(path, allow_pickle=True)
        lookup = dict(zip(blob["tokens"].tolist(), blob["ids"].tolist()))
        log(f"отображение сырых токенов: {len(lookup):,} записей")
    else:
        lookup = dict(token_id)
        log("stem_map.npz отсутствует — поиск идёт напрямую по словарю модели")
    try:
        import Stemmer

        stemmer = Stemmer.Stemmer("russian")
        log("PyStemmer доступен: токены вне отображения будут стеммиться")
    except Exception as error:  # noqa: BLE001 — любая причина means «работаем без него»
        stemmer = None
        log(f"PyStemmer недоступен ({type(error).__name__}); токены вне отображения -> PAD")
    return lookup, stemmer


def encode_items(names: list[str], attributes: list[str], lookup: dict[str, int],
                 token_id: dict[str, int], stemmer, shapes: dict[str, int]
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Товары в те же формы, что при обучении. Незнакомый токен -> PAD."""
    count = len(names)
    titles = np.zeros((count, shapes["title"]), dtype=np.int32)
    keys = np.zeros((count, shapes["attrs"], shapes["key_tokens"]), dtype=np.int32)
    values = np.zeros((count, shapes["attrs"], shapes["value_tokens"]), dtype=np.int32)
    unknown, total = 0, 0
    miss_cache: dict[str, int] = {}

    def index_of(token: str) -> int:
        found = lookup.get(token)
        if found is not None:
            return found
        if stemmer is None:
            return PAD_ID
        found = miss_cache.get(token)
        if found is None:
            found = token_id.get(stemmer.stemWord(token), PAD_ID)
            miss_cache[token] = found
        return found

    for row, (name, raw) in enumerate(zip(names, attributes)):
        for column, token in enumerate(tokenize(name)[:shapes["title"]]):
            index = index_of(token)
            titles[row, column] = index
            unknown += index == PAD_ID
            total += 1
        for slot, (key_tokens, value_tokens) in enumerate(
                parse_attributes(raw)[:shapes["attrs"]]):
            for column, token in enumerate(key_tokens[:shapes["key_tokens"]]):
                keys[row, slot, column] = index_of(token)
            for column, token in enumerate(value_tokens[:shapes["value_tokens"]]):
                index = index_of(token)
                values[row, slot, column] = index
                unknown += index == PAD_ID
                total += 1
    return titles, keys, values, 100 * unknown / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, required=True)
    parser.add_argument("--matches_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    models = sorted((HERE / "models").glob("fold_*.pt"))
    if not models:
        raise SystemExit(f"нет файлов моделей в {HERE / 'models'}")
    log(f"моделей в ансамбле: {len(models)} ({', '.join(p.stem for p in models)})")

    # Словарь и формы общие у всех четырёх — читаем из первой и кодируем один раз.
    head = torch.load(models[0], map_location="cpu", weights_only=False)
    token_id, shapes = head["token_id"], head["max_shapes"]
    stemming = bool(head.get("stemming", False))
    config_dict = head["config"]
    del head
    log(f"словарь {len(token_id):,}, токенизация: "
        f"{'стеммы' if stemming else 'сырые токены'}")

    items = pd.read_parquet(args.items_path, columns=["id", "name", "attributes"])
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    log(f"товаров {len(items):,}, пар {len(matches):,}")

    lookup, stemmer = build_lookup(token_id)
    titles, keys, values, unknown_share = encode_items(
        items["name"].tolist(), items["attributes"].tolist(), lookup, token_id,
        stemmer, shapes)
    log(f"закодировано за {time.time() - started:.0f}s; токенов вне словаря "
        f"{unknown_share:.1f}%")
    titles_t, keys_t, values_t = (torch.from_numpy(titles), torch.from_numpy(keys),
                                  torch.from_numpy(values))
    counts = attribute_counts(keys_t, values_t)

    row_of_id = {int(i): r for r, i in enumerate(items["id"].tolist())}
    del items
    left, right = matches["id1"].to_numpy(), matches["id2"].to_numpy()
    known = np.array([int(a) in row_of_id and int(b) in row_of_id
                      for a, b in zip(left, right)])
    rows1 = np.array([row_of_id.get(int(a), 0) for a in left], dtype=np.int64)
    rows2 = np.array([row_of_id.get(int(b), 0) for b in right], dtype=np.int64)

    # Сортировка по числу атрибутов: стоимость канала атрибут-атрибут задаётся
    # формами паддинга. Порядок восстанавливается при записи.
    order = np.argsort(np.maximum(counts[rows1], counts[rows2]), kind="mergesort")

    total = np.zeros(len(matches), dtype=np.float64)
    for position, path in enumerate(models, start=1):
        model_started = time.time()
        blob = torch.load(path, map_location="cpu", weights_only=False)
        config = KNRMConfig(**config_dict)
        model = ProductMatcher(blob["embedding_fp16"].to(torch.float32)
                               if "embedding_fp16" in blob
                               else blob["state_dict"]["embedding.weight"], config)
        if "head_state" in blob:
            model.load_state_dict(blob["head_state"], strict=False)
        else:
            model.load_state_dict(blob["state_dict"])
        model.eval()
        held_out = blob.get("held_out", "?")
        del blob

        scores = np.empty(len(matches), dtype=np.float64)
        with torch.no_grad():
            for index, start in enumerate(range(0, len(order), args.batch_size)):
                pick = order[start:start + args.batch_size]
                item_a = make_batch(titles_t, keys_t, values_t, rows1[pick])
                item_b = make_batch(titles_t, keys_t, values_t, rows2[pick])
                scores[pick] = torch.sigmoid(model(item_a, item_b)).numpy()
                if (index + 1) % 100 == 0:
                    done = start + len(pick)
                    rate = done / max(time.time() - model_started, 1e-9)
                    log(f"  [{position}/{len(models)}] {done:,}/{len(order):,} "
                        f"({rate:.0f} пар/с, осталось "
                        f"{(len(order) - done) / max(rate, 1e-9) / 60:.1f} мин)")
        total += scores
        # Модель больше не нужна: держать четыре таблицы разом незачем.
        del model
        log(f"  модель {position}/{len(models)} ({held_out}) готова за "
            f"{time.time() - model_started:.0f}s, диапазон "
            f"{scores.min():.6f}..{scores.max():.6f}")

    predictions = total / len(models)
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

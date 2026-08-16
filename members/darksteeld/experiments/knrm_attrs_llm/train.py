"""KNRM по атрибутам, предобученный на matches_llm, затем дообученный по фолдам.

То же, что ``knrm_attrs``, но с первой стадией на 11.19M LLM-размеченных пар —
ровно тем приёмом, который поднял модель по названиям с 0.53078 до 0.56557.
Модель и разбор атрибутов импортируются из
``members/darksteeld/src/knrm_attrs_model.py``: архитектура не меняется, меняется
только то, с каких весов начинается дообучение.

**Что здесь принципиально иначе, чем у названий.** Название — это 20 токенов на
товар; словарь по всему каталогу уместился в 1.6M токенов, а кодирование 12.4M
товаров — в 990 МБ. Атрибуты дороже по обеим осям:

* *Словарь значений растёт быстрее.* Уже на 500 тысячах товаров каталога это
  701 тысяча токенов значений, и хвост почти весь одноразовый (39% встречаются
  дважды и чаще). Поэтому здесь та же отсечка, что и у названий: **все** токены
  ручной вселенной сохраняются независимо от частоты — на ней мы предсказываем,
  и артикулы в ней терять нельзя, — а токены каталога должны встретиться
  ``--min-count`` раз. Отброшенное почти не получает градиента и остаётся на
  инициализации, то есть ровно там, где его и восстановит детерминированный
  вектор из строки токена.
* *Кодирование дороже в 12 раз на товар.* Один товар — это [24,4] ключей и
  [24,6] значений, 960 байт против 80 у названия. Весь LLM-корпус занял бы
  ~11.9 ГБ, поэтому число пар предобучения ограничено ``--pretrain-pairs``, а
  требуемая память печатается и проверяется ДО выделения, а не после.

Дообучение и оценка не отличаются от ``knrm_attrs`` ни на строку: три фолда на
обучение, ранняя остановка patience 1 на срезе по компонентам связности,
предсказание отложенного фолда.

    .venv/bin/python members/darksteeld/experiments/knrm_attrs_llm/train.py \\
        --cache-dir <scratch>/attrcache
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_attrs_model import (  # noqa: E402
    DIM, MAX_ATTRS, MAX_KEY_TOKENS, MAX_VALUE_TOKENS, AttributeKNRM, encode_attributes,
    initial_weight, parse_attributes,
)
from knrm_model import PAD_ID  # noqa: E402
from lgbm_cheap import AUDIT_FILE, load_audit  # noqa: E402

DEFAULT_NAVEC = (REPOSITORY_ROOT / "members" / "darksteeld" / "models"
                 / "navec_hudlit_v1_12B_500K_300d_100q.tar")
VALIDATION_BUCKETS = 10
BYTES_PER_ITEM = (MAX_ATTRS * MAX_KEY_TOKENS + MAX_ATTRS * MAX_VALUE_TOKENS) * 4


def log(message: str) -> None:
    print(message, flush=True)


# ----------------------------------------------------------------- словари ---
def count_catalogue_tokens(parquet_path: Path, cache_path: Path,
                           batch_size: int = 200_000) -> tuple[Counter, Counter]:
    """Частоты токенов ключей и значений по всему каталогу; результат кэшируется."""
    if cache_path.is_file():
        blob = np.load(cache_path, allow_pickle=True)
        keys = Counter(dict(zip(blob["key_tokens"].tolist(), blob["key_counts"].tolist())))
        values = Counter(dict(zip(blob["value_tokens"].tolist(), blob["value_counts"].tolist())))
        log(f"частоты токенов каталога из кэша: ключей {len(keys):,}, значений {len(values):,}")
        return keys, values

    import pyarrow.parquet as pq

    keys, values = Counter(), Counter()
    scanned, started = 0, time.time()
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=batch_size,
                                                           columns=["attributes"]):
        for raw in batch.column("attributes").to_pylist():
            for key_tokens, value_tokens in parse_attributes(raw):
                keys.update(key_tokens)
                values.update(value_tokens)
        scanned += batch.num_rows
        if scanned % 2_000_000 == 0:
            log(f"    просмотрено {scanned:,}, значений {len(values):,}, "
                f"{time.time() - started:.0f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        key_tokens=np.array(list(keys), dtype=object),
        key_counts=np.array(list(keys.values()), dtype=np.int64),
        value_tokens=np.array(list(values), dtype=object),
        value_counts=np.array(list(values.values()), dtype=np.int64),
    )
    log(f"  посчитано за {time.time() - started:.0f}s: ключей {len(keys):,}, "
        f"значений {len(values):,} -> {cache_path.name}")
    return keys, values


def build_vocabulary(counts_big: Counter, tokens_hand: set[str], min_count: int) -> dict[str, int]:
    """Все токены ручной вселенной; токены каталога — от min_count вхождений."""
    keep = {token for token, count in counts_big.items() if count >= min_count}
    keep |= tokens_hand
    return {token: index for index, token in enumerate(sorted(keep), start=1)}  # 0 = PAD


def hand_attribute_tokens(attributes: list[str]) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    values: set[str] = set()
    for raw in attributes:
        for key_tokens, value_tokens in parse_attributes(raw):
            keys.update(key_tokens)
            values.update(value_tokens)
    return keys, values


# --------------------------------------------------------------- кодирование ---
def encode_stream(parquet_path: Path, wanted_ids: np.ndarray, key_ids: dict[str, int],
                  value_ids: dict[str, int], batch_size: int = 200_000
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Атрибуты нужных товаров, потоково из большого файла.

    ``wanted_ids`` отсортирован: принадлежность и номер строки — по одному
    searchsorted, чтобы миллионы товаров адресовались без словаря в питоне.
    """
    import pyarrow.parquet as pq

    keys = np.zeros((len(wanted_ids), MAX_ATTRS, MAX_KEY_TOKENS), dtype=np.int32)
    values = np.zeros((len(wanted_ids), MAX_ATTRS, MAX_VALUE_TOKENS), dtype=np.int32)
    seen, scanned, started = 0, 0, time.time()
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=batch_size,
                                                           columns=["id", "attributes"]):
        ids = np.asarray(batch.column("id"), dtype=np.int64)
        position = np.searchsorted(wanted_ids, ids)
        position[position >= len(wanted_ids)] = 0
        hit = wanted_ids[position] == ids
        if hit.any():
            raws = batch.column("attributes").to_pylist()
            for local in np.flatnonzero(hit).tolist():
                row = position[local]
                for slot, (key_tokens, value_tokens) in enumerate(parse_attributes(raws[local])):
                    for column, token in enumerate(key_tokens):
                        keys[row, slot, column] = key_ids.get(token, PAD_ID)
                    for column, token in enumerate(value_tokens):
                        values[row, slot, column] = value_ids.get(token, PAD_ID)
                seen += 1
        scanned += batch.num_rows
        if scanned % 2_000_000 == 0:
            log(f"    просмотрено {scanned:,}, найдено {seen:,}, {time.time() - started:.0f}s")
    log(f"  закодировано {seen:,}/{len(wanted_ids):,} товаров за {time.time() - started:.0f}s")
    return keys, values


# ------------------------------------------------------------------ обучение ---
def load_folds(targets_dir: Path, fold_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    folds: dict[str, dict[str, np.ndarray]] = {}
    for fold_id in fold_ids:
        path = targets_dir / f"{fold_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"нет {path}; собери цели: make validation-targets-v2")
        id1, id2, target = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                id1.append(int(row["id1"]))
                id2.append(int(row["id2"]))
                target.append(float(row["target"]))
        folds[fold_id] = {
            "id1": np.asarray(id1, dtype=np.int64),
            "id2": np.asarray(id2, dtype=np.int64),
            "target": np.asarray(target, dtype=np.float32),
        }
    return folds


def validation_mask(id1: np.ndarray, id2: np.ndarray, seed: str) -> np.ndarray:
    from validation.build_folds import connected_component_keys

    component_of_item = connected_component_keys(id1, id2)
    is_validation = np.zeros(len(id1), dtype=bool)
    bucket_of_key: dict[int, int] = {}
    for position, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        bucket = bucket_of_key.get(key)
        if bucket is None:
            digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
            bucket_of_key[key] = bucket
        is_validation[position] = bucket == 0
    return is_validation


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return 0.0
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@torch.no_grad()
def predict(model, keys, values, rows1, rows2, *, batch_size: int) -> np.ndarray:
    model.eval()
    rows1_tensor, rows2_tensor = torch.from_numpy(rows1), torch.from_numpy(rows2)
    scores = np.empty(len(rows1), dtype=np.float64)
    for start in range(0, len(rows1), batch_size):
        stop = min(start + batch_size, len(rows1))
        left, right = rows1_tensor[start:stop], rows2_tensor[start:stop]
        scores[start:stop] = torch.sigmoid(model(
            keys[left].long(), values[left].long(),
            keys[right].long(), values[right].long(),
        )).numpy()
    return scores


def run_epoch(model, optimizers, loss_function, keys, values, rows1, rows2, target,
              batch_size: int, generator: torch.Generator, label: str) -> float:
    model.train()
    permutation = torch.randperm(len(target), generator=generator)
    rows1_tensor, rows2_tensor = torch.from_numpy(rows1), torch.from_numpy(rows2)
    target_tensor = torch.from_numpy(target)
    running, batches, started, reported = 0.0, 0, time.time(), 0
    for start in range(0, len(target), batch_size):
        selection = permutation[start : start + batch_size]
        left, right = rows1_tensor[selection], rows2_tensor[selection]
        loss = loss_function(
            model(keys[left].long(), values[left].long(),
                  keys[right].long(), values[right].long()),
            target_tensor[selection],
        )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        running += float(loss.detach())
        batches += 1
        done = 100 * (start + batch_size) // max(len(target), 1)
        if done >= reported + 10:
            reported = done - done % 10
            log(f"    {label} {reported:>3}%  loss {running / batches:.5f}  "
                f"{time.time() - started:.0f}s")
    return running / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, required=True,
                        help="где кэшируются частоты токенов и кодирование LLM-товаров")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2"
                        / "darksteeld" / "knrm_attrs_llm")
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04",
                        help="каждый фолд списка по очереди held-out, обучение на ОСТАЛЬНЫХ "
                             "ИЗ СПИСКА; передав три фолда, получаем модели, не видевшие "
                             "четвёртый — это и есть вложенный OOF")
    parser.add_argument("--dump-all-folds", type=Path,
                        help="дополнительно писать in-sample предсказания на обучающих фолдах "
                             "в <путь>/trained_without_<held>/<fold>.csv")
    parser.add_argument("--navec", type=Path, default=DEFAULT_NAVEC)
    parser.add_argument("--min-count", type=int, default=5,
                        help="сколько раз токен должен встретиться в каталоге; "
                             "токены ручной вселенной сохраняются всегда")
    parser.add_argument("--pretrain-pairs", type=int, default=4_000_000,
                        help="сколько LLM-пар брать (0 = все 11.19M); ограничение по памяти")
    parser.add_argument("--max-encoding-gb", type=float, default=7.0,
                        help="потолок на кодирование LLM-товаров; проверяется ДО выделения")
    parser.add_argument("--cache-encoding", action="store_true",
                        help="сохранять закодированные атрибуты LLM-товаров на диск "
                             "(несколько ГБ; по умолчанию выключено)")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    import polars as pl

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = [f.strip() for f in args.folds.split(",") if f.strip()]

    # ---- словари по обеим вселенным ----------------------------------------
    hand = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "attributes"])
    hand_attributes = hand["attributes"].to_list()
    tokens_hand_keys, tokens_hand_values = hand_attribute_tokens(hand_attributes)
    log(f"ручная вселенная: {len(hand_attributes):,} товаров, токенов в ключах "
        f"{len(tokens_hand_keys):,}, в значениях {len(tokens_hand_values):,}")

    counts_keys, counts_values = count_catalogue_tokens(
        args.data_dir / "items.parquet", args.cache_dir / "attr_token_counts.npz")
    key_ids = build_vocabulary(counts_keys, tokens_hand_keys, args.min_count)
    value_ids = build_vocabulary(counts_values, tokens_hand_values, args.min_count)
    table_gb = (len(key_ids) + len(value_ids) + 2) * DIM * 4 / 1e9
    log(f"словарь: ключей {len(key_ids):,}, значений {len(value_ids):,} "
        f"(min_count={args.min_count}) | таблицы {table_gb:.2f} ГБ, x3 со SparseAdam")

    # ---- стадия 1: предобучение на matches_llm ------------------------------
    llm = pl.read_parquet(args.data_dir / "matches_llm.parquet")
    if args.pretrain_pairs and args.pretrain_pairs < llm.height:
        pick = np.random.default_rng(args.seed).choice(llm.height, args.pretrain_pairs,
                                                       replace=False)
        pick.sort()
        llm = llm[pick]
    llm_ids = np.unique(np.concatenate([llm["id1"].to_numpy(), llm["id2"].to_numpy()]))
    needed_gb = len(llm_ids) * BYTES_PER_ITEM / 1e9
    log(f"LLM-пар взято {llm.height:,} из 11,187,780; товаров {len(llm_ids):,}; "
        f"кодирование потребует {needed_gb:.2f} ГБ")
    if needed_gb > args.max_encoding_gb:
        raise SystemExit(
            f"кодирование {needed_gb:.2f} ГБ превышает потолок {args.max_encoding_gb} ГБ — "
            f"уменьши --pretrain-pairs (примерно до "
            f"{int(args.pretrain_pairs * args.max_encoding_gb / needed_gb):,}) "
            f"или подними --max-encoding-gb")
    if np.intersect1d(llm_ids, hand["id"].to_numpy()).size:
        raise AssertionError("вселенные LLM и ручной разметки пересекаются — предобучение потечёт")
    log("пересечение с ручной вселенной: 0 (проверено, а не предположено)")

    # Состояние после предобучения кладём на диск: предобучение стоит десятки
    # минут и зависит только от (эпохи, min_count, число пар), а не от того,
    # какие ручные фолды считать обучающими. Имя считаем здесь, до кодирования,
    # потому что при готовом чекпоинте кодировать LLM-товары незачем — это
    # несколько ГБ и минуты работы ради массива, который никто не прочитает.
    checkpoint = (args.cache_dir /
                  f"attrs_pretrained_e{args.pretrain_epochs}_mc{args.min_count}"
                  f"_p{llm.height}.pt")
    pretrained_ready = checkpoint.is_file()

    # Кэш кодирования по умолчанию выключен: он весит столько же, сколько сам
    # массив (здесь несколько ГБ), а диск на этой машине уже один раз кончился
    # посреди прогона. Пересчёт стоит минуты, потеря места — весь прогон.
    cache = args.cache_dir / f"llm_attrs_mc{args.min_count}_p{llm.height}.npz"
    llm_keys = llm_values = None
    if pretrained_ready:
        log(f"веса после предобучения уже есть ({checkpoint.name}) — "
            f"кодирование {len(llm_ids):,} LLM-товаров пропущено")
    elif args.cache_encoding and cache.is_file():
        blob = np.load(cache)
        llm_keys, llm_values = blob["keys"], blob["values"]
        log(f"кодирование LLM-товаров из кэша {cache.name} {llm_keys.shape}")
    else:
        llm_keys, llm_values = encode_stream(
            args.data_dir / "items.parquet", llm_ids, key_ids, value_ids)
        if args.cache_encoding:
            np.savez(cache, keys=llm_keys, values=llm_values)

    base_key_weight, key_covered = initial_weight(key_ids, DIM, args.navec)
    base_value_weight, value_covered = initial_weight(value_ids, DIM, args.navec)
    log(f"navec покрыл ключей {key_covered:,}/{len(key_ids):,}, "
        f"значений {value_covered:,}/{len(value_ids):,}")

    model = AttributeKNRM(base_key_weight.clone(), base_value_weight.clone(), sparse=True)

    if pretrained_ready:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        log(f"веса после предобучения из кэша: {checkpoint.name}")
    else:
        optimizers = [
            torch.optim.SparseAdam(
                list(model.key_embedding.parameters()) + list(model.value_embedding.parameters()),
                lr=args.learning_rate),
            torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                             lr=args.learning_rate),
        ]
        llm_rows1 = np.searchsorted(llm_ids, llm["id1"].to_numpy()).astype(np.int64)
        llm_rows2 = np.searchsorted(llm_ids, llm["id2"].to_numpy()).astype(np.int64)
        llm_target = llm["target"].to_numpy().astype(np.float32)
        llm_keys_t, llm_values_t = torch.from_numpy(llm_keys), torch.from_numpy(llm_values)
        generator = torch.Generator().manual_seed(args.seed)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = run_epoch(model, optimizers, nn.BCEWithLogitsLoss(), llm_keys_t, llm_values_t,
                             llm_rows1, llm_rows2, llm_target, args.batch_size, generator,
                             f"предобучение e{epoch}")
            log(f"  предобучение, эпоха {epoch}: loss {loss:.5f}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        log(f"веса после предобучения сохранены: {checkpoint.name}")
        del llm_keys_t, llm_values_t, llm_rows1, llm_rows2, llm_target
    pretrained_state = copy.deepcopy(model.state_dict())
    del llm_keys, llm_values

    # ---- стадия 2: дообучение по фолдам ------------------------------------
    hand_keys, hand_values = encode_attributes(hand_attributes, key_ids, value_ids)
    hand_keys_t, hand_values_t = torch.from_numpy(hand_keys), torch.from_numpy(hand_values)
    row_of_id = {int(i): r for r, i in enumerate(hand["id"].to_list())}
    folds = load_folds(args.targets_dir, fold_ids)
    corrections = load_audit() if args.audit else {}
    log(f"доразметка: {len(corrections)} исправлений" if corrections
        else "доразметка не применяется (--audit чтобы включить)")

    zero_shot, fine_tuned = {}, {}
    for held_out in fold_ids:
        log(f"\n=== {held_out}")
        train_ids = [f for f in fold_ids if f != held_out]
        id1 = np.concatenate([folds[f]["id1"] for f in train_ids])
        id2 = np.concatenate([folds[f]["id2"] for f in train_ids])
        target = np.concatenate([folds[f]["target"] for f in train_ids])
        if corrections:
            for position, pair in enumerate(zip(id1.tolist(), id2.tolist())):
                if pair in corrections:
                    target[position] = corrections[pair]

        fold = folds[held_out]
        held_rows1 = np.array([row_of_id[int(i)] for i in fold["id1"]], dtype=np.int64)
        held_rows2 = np.array([row_of_id[int(i)] for i in fold["id2"]], dtype=np.int64)

        model.load_state_dict(pretrained_state)
        scores = predict(model, hand_keys_t, hand_values_t, held_rows1, held_rows2,
                         batch_size=args.batch_size * 4)
        zero_shot[held_out] = average_precision(fold["target"], scores)
        log(f"    zero-shot (только LLM, ни одной ручной метки): "
            f"PR-AUC {zero_shot[held_out]:.6f}")

        is_validation = validation_mask(id1, id2, f"{args.seed}:{held_out}")
        rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int64)
        rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int64)
        fold_optimizers = [
            torch.optim.SparseAdam(
                list(model.key_embedding.parameters())
                + list(model.value_embedding.parameters()), lr=args.learning_rate),
            torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                             lr=args.learning_rate),
        ]
        loss_function = nn.BCEWithLogitsLoss()
        fold_generator = torch.Generator().manual_seed(args.seed)
        best, best_epoch, best_state, waited = -1.0, 0, None, 0
        for epoch in range(1, args.max_epochs + 1):
            epoch_started = time.time()
            loss = run_epoch(model, fold_optimizers, loss_function, hand_keys_t, hand_values_t,
                             rows1[~is_validation], rows2[~is_validation],
                             target[~is_validation], args.batch_size, fold_generator,
                             f"дообучение e{epoch}")
            validation_score = average_precision(
                target[is_validation],
                predict(model, hand_keys_t, hand_values_t, rows1[is_validation],
                        rows2[is_validation], batch_size=args.batch_size * 4))
            improved = validation_score > best
            log(f"      эпоха {epoch:>2}  loss {loss:.5f}  val PR-AUC {validation_score:.6f}"
                f"{'  *' if improved else ''}  {time.time() - epoch_started:.0f}s")
            if improved:
                best, best_epoch, waited = validation_score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                waited += 1
                if waited >= args.patience:
                    log(f"      ранняя остановка, возвращаю эпоху {best_epoch}")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

        scores = predict(model, hand_keys_t, hand_values_t, held_rows1, held_rows2,
                         batch_size=args.batch_size * 4)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / f"{held_out}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for a, b, s in zip(fold["id1"].tolist(), fold["id2"].tolist(), scores.tolist(),
                               strict=True):
                writer.writerow([a, b, f"{s:.8f}"])
        fine_tuned[held_out] = average_precision(fold["target"], scores)
        log(f"    {held_out}: лучшая эпоха {best_epoch} (val {best:.6f}) "
            f"-> PR-AUC на фолде {fine_tuned[held_out]:.6f}")

        if args.dump_all_folds:
            # Та же модель на ОБУЧАЮЩИХ фолдах. Числа in-sample, метрикой быть
            # не могут; нужны как признак модели второго уровня, которой иначе
            # нечего положить в свои обучающие строки. Кладём отдельно от
            # out-of-fold файла, чтобы ни один скоринг их не подобрал.
            dump_dir = args.dump_all_folds / f"trained_without_{held_out}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            for other in train_ids:
                other_fold = folds[other]
                other_rows1 = np.array([row_of_id[int(i)] for i in other_fold["id1"]],
                                       dtype=np.int64)
                other_rows2 = np.array([row_of_id[int(i)] for i in other_fold["id2"]],
                                       dtype=np.int64)
                in_sample = predict(model, hand_keys_t, hand_values_t, other_rows1, other_rows2,
                                    batch_size=args.batch_size * 4)
                with (dump_dir / f"{other}.csv").open("w", newline="", encoding="utf-8") as sink:
                    writer = csv.writer(sink, lineterminator="\n")
                    writer.writerow(["id1", "id2", "predict"])
                    for a, b, s in zip(other_fold["id1"].tolist(), other_fold["id2"].tolist(),
                                       in_sample.tolist(), strict=True):
                        writer.writerow([a, b, f"{s:.8f}"])
            log(f"      in-sample предсказания на обучающих фолдах -> {dump_dir}")

    log("\n" + "=" * 62)
    zs = ", ".join(f"{f}={zero_shot[f]:.4f}" for f in fold_ids)
    ft = ", ".join(f"{f}={fine_tuned[f]:.4f}" for f in fold_ids)
    log(f"zero-shot  mean {np.mean(list(zero_shot.values())):.6f}  ({zs})")
    log(f"дообучено  mean {np.mean(list(fine_tuned.values())):.6f}  ({ft})")
    log("контроли на тех же фолдах spec-v2:")
    log("  knrm_attrs        0.551883   (та же модель без предобучения)")
    log("  knrm_llm_pretrain 0.565568   (по названию, предобучен на matches_llm)")
    log(f"предсказания -> {args.out_dir}   всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

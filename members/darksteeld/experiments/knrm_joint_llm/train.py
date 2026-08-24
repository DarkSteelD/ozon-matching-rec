"""Предобучение совместной KNRM-сети на matches_llm: название и атрибуты сразу.

Обе прежние сети предобучались на этом же корпусе по отдельности — одна по
названиям, другая по атрибутам. Здесь один проход обучает одну сеть, которая
видит четыре взаимодействия: имя-имя, атрибуты-атрибуты и два кросс-канала
«имя одного против значений атрибутов другого».

**Утечки нет и это проверяется на старте, а не предполагается:** вселенная
товаров LLM-разметки не пересекается с ручной (0 из 12,384,610), поэтому
предобучение на любых LLM-парах не касается ни одного фолда ручной валидации.

**Словарь один на все три поля.** Токены названий, ключей и значений
индексируются одной таблицей — иначе косинус между названием и значением
атрибута нечего было бы считать. Берутся все токены ручной вселенной (именно на
них модель в итоге предсказывает) плюс токены каталога от ``--min-count``
вхождений. Частоты каталога переиспользуются из кэшей, оставленных
экспериментами ``knrm_llm_pretrain`` и ``knrm_attrs_llm``, поэтому повторного
прохода по items.parquet не требуется.

    .venv/bin/python members/darksteeld/experiments/knrm_joint_llm/train.py \\
        --cache-dir <scratch>/knrm_cache --out <scratch>/joint_pretrained.pt \\
        --pretrain-pairs 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_joint_batching import attribute_counts, bucketed_batches, make_batch  # noqa: E402
from knrm_joint_tokens import parse_attributes, tokenize  # noqa: E402
from knrm_joint_model import (  # noqa: E402
    PAD_ID, ItemTensors, KNRMConfig, ProductMatcher, build_embedding_matrix,
)

DEFAULT_NAVEC = (REPOSITORY_ROOT / "members" / "darksteeld" / "models"
                 / "navec_hudlit_v1_12B_500K_300d_100q.tar")
MAX_TITLE = 20
MAX_ATTRS = 24
MAX_KEY_TOKENS = 4
MAX_VALUE_TOKENS = 6
BYTES_PER_ITEM = 4 * (MAX_TITLE + MAX_ATTRS * (MAX_KEY_TOKENS + MAX_VALUE_TOKENS))


def log(message: str) -> None:
    print(message, flush=True)


def hand_universe_tokens(items: pl.DataFrame, stemming: bool) -> set[str]:
    """Все токены ручной вселенной: названия, ключи и значения в одном множестве."""
    tokens: set[str] = set()
    for name in items["name"].to_list():
        tokens.update(tokenize(name, stemming))
    for raw in items["attributes"].to_list():
        for key_tokens, value_tokens in parse_attributes(raw, stemming):
            tokens.update(key_tokens)
            tokens.update(value_tokens)
    return tokens


def vocabulary_counts(path: Path, stemming: bool) -> Counter:
    """Частоты из build_vocabulary.py: один файл, обе версии внутри.

    Считаны по объединению items.parquet и items_human.parquet (вложенность
    второго в первый проверена при подсчёте), поэтому отдельно досчитывать
    ручную вселенную не нужно.
    """
    blob = np.load(path, allow_pickle=True)
    prefix = "stem" if stemming else "raw"
    counts: Counter = Counter()
    for field in ("name", "key", "value"):
        tokens = blob[f"{prefix}_{field}_tokens"].tolist()
        values = blob[f"{prefix}_{field}_counts"].tolist()
        for token, count in zip(tokens, values):
            counts[token] = max(counts.get(token, 0), count)
    log(f"  частоты из {path.name} ({prefix}): {len(counts):,} токенов")
    return counts


def catalogue_counts(cache_dir: Path) -> Counter:
    """Частоты токенов каталога из кэшей прежних экспериментов, объединённые.

    ``big_token_counts.npz`` — токены названий, ``attr_token_counts.npz`` — ключи
    и значения. Оба лежат рядом после прогонов ``knrm_llm_pretrain`` и
    ``knrm_attrs_llm``; пересчитывать каталог заново незачем.
    """
    counts: Counter = Counter()
    names_cache = cache_dir / "big_token_counts.npz"
    if names_cache.is_file():
        blob = np.load(names_cache, allow_pickle=True)
        field = "tokens" if "tokens" in blob.files else blob.files[0]
        values = "counts" if "counts" in blob.files else blob.files[1]
        counts.update(dict(zip(blob[field].tolist(), blob[values].tolist())))
        log(f"  токены названий из кэша: {len(counts):,}")
    attrs_cache = cache_dir / "attr_token_counts.npz"
    if attrs_cache.is_file():
        blob = np.load(attrs_cache, allow_pickle=True)
        for token_field, count_field in (("key_tokens", "key_counts"),
                                         ("value_tokens", "value_counts")):
            if token_field in blob.files:
                counts.update(dict(zip(blob[token_field].tolist(),
                                       blob[count_field].tolist())))
        log(f"  плюс ключи и значения: всего {len(counts):,}")
    if not counts:
        raise SystemExit(
            f"в {cache_dir} нет ни big_token_counts.npz, ни attr_token_counts.npz — "
            "сначала прогоните knrm_llm_pretrain или knrm_attrs_llm, они их оставляют")
    return counts


def encode_stream(parquet_path: Path, wanted_ids: np.ndarray, token_id: dict[str, int],
                  stemming: bool = False, batch_size: int = 200_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Названия и атрибуты нужных товаров, потоково из большого файла.

    ``wanted_ids`` отсортирован, поэтому принадлежность и номер строки берутся
    одним searchsorted — миллионы товаров адресуются без питоновского словаря.
    """
    import pyarrow.parquet as pq

    titles = np.zeros((len(wanted_ids), MAX_TITLE), dtype=np.int32)
    keys = np.zeros((len(wanted_ids), MAX_ATTRS, MAX_KEY_TOKENS), dtype=np.int32)
    values = np.zeros((len(wanted_ids), MAX_ATTRS, MAX_VALUE_TOKENS), dtype=np.int32)
    seen, scanned, started = 0, 0, time.time()
    for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=batch_size, columns=["id", "name", "attributes"]):
        ids = np.asarray(batch.column("id"), dtype=np.int64)
        position = np.searchsorted(wanted_ids, ids)
        position[position >= len(wanted_ids)] = 0
        hit = wanted_ids[position] == ids
        if hit.any():
            names = batch.column("name").to_pylist()
            raws = batch.column("attributes").to_pylist()
            for local in np.flatnonzero(hit).tolist():
                row = position[local]
                for column, token in enumerate(tokenize(names[local], stemming)[:MAX_TITLE]):
                    titles[row, column] = token_id.get(token, PAD_ID)
                for slot, (key_tokens, value_tokens) in enumerate(
                        parse_attributes(raws[local], stemming)[:MAX_ATTRS]):
                    for column, token in enumerate(key_tokens[:MAX_KEY_TOKENS]):
                        keys[row, slot, column] = token_id.get(token, PAD_ID)
                    for column, token in enumerate(value_tokens[:MAX_VALUE_TOKENS]):
                        values[row, slot, column] = token_id.get(token, PAD_ID)
                seen += 1
        scanned += batch.num_rows
        if scanned % 2_000_000 == 0:
            log(f"    просмотрено {scanned:,}, найдено {seen:,}, "
                f"{time.time() - started:.0f}s")
    log(f"  закодировано {seen:,}/{len(wanted_ids):,} товаров за {time.time() - started:.0f}s")
    return titles, keys, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="куда писать чекпоинт")
    parser.add_argument("--navec", type=Path, default=DEFAULT_NAVEC)
    parser.add_argument("--vocabulary", type=Path,
                        help="npz от build_vocabulary.py; без него — старые кэши")
    parser.add_argument("--stem", action="store_true",
                        help="стемминг: словарь и кодирование по стеммам")
    parser.add_argument("--min-count", type=int, default=5,
                        help="порог частоты токена каталога для попадания в словарь")
    parser.add_argument("--pretrain-pairs", type=int, default=2_000_000, help="0 = все 11.19M")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-scale", type=float, default=0.1,
                        help="во сколько раз lr таблицы меньше остального")
    parser.add_argument("--chunk", type=int, default=8, help="чанк по атрибутам A в канале A-A")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-kernels", type=int, default=11)
    parser.add_argument("--max-encoding-gb", type=float, default=14.0)
    parser.add_argument("--device", default="cpu",
                        help="cpu или cuda; на cuda таблица идёт плотным AdamW")
    parser.add_argument("--sparse", action="store_true",
                        help="разреженный градиент таблицы + SparseAdam")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = без ограничения; для замера")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="сохранять промежуточное состояние каждые N батчей; 0 = не сохранять")
    parser.add_argument("--resume", action="store_true",
                        help="продолжить с промежуточного состояния, если оно есть")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # ---- словарь -----------------------------------------------------------
    hand = pl.read_parquet(args.data_dir / "items_human.parquet",
                           columns=["id", "name", "attributes"])
    tokens_hand = hand_universe_tokens(hand, args.stem)
    log(f"ручная вселенная: {hand.height:,} товаров, {len(tokens_hand):,} уникальных токенов"
        f"{' (стеммы)' if args.stem else ''}")
    counts = (vocabulary_counts(args.vocabulary, args.stem) if args.vocabulary
              else catalogue_counts(args.cache_dir))
    frequent = {token for token, count in counts.items() if count >= args.min_count}
    corpus_tokens = sorted(tokens_hand | frequent)
    log(f"словарь: {len(tokens_hand):,} ручных + каталог от {args.min_count} вхождений "
        f"-> {len(corpus_tokens):,} токенов")

    word_vectors: dict[str, np.ndarray] = {}
    if args.navec.is_file():
        from navec import Navec

        navec = Navec.load(str(args.navec))
        known = set(navec.vocab.words)
        word_vectors = {token: np.asarray(navec[token], dtype=np.float32)
                        for token in corpus_tokens if token in known}
        log(f"navec покрыл {len(word_vectors):,} токенов "
            f"({100 * len(word_vectors) / len(corpus_tokens):.1f}%)")
    table, token_id = build_embedding_matrix(word_vectors, corpus_tokens,
                                             embedding_dim=300, seed=args.seed)
    log(f"таблица {table.shape}, {table.nbytes / 1e9:.2f} ГБ | setup {time.time() - started:.0f}s")

    # ---- пары LLM ----------------------------------------------------------
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
            f"кодирование {needed_gb:.2f} ГБ выше потолка {args.max_encoding_gb} ГБ — "
            f"уменьшите --pretrain-pairs примерно до "
            f"{int(args.pretrain_pairs * args.max_encoding_gb / needed_gb):,}")
    overlap = np.intersect1d(llm_ids, hand["id"].to_numpy()).size
    if overlap:
        raise AssertionError(f"вселенные пересекаются в {overlap} товарах — предобучение потечёт")
    log("пересечение с ручной вселенной: 0 (проверено, а не предположено)")
    del hand

    titles, keys, values = encode_stream(args.data_dir / "items.parquet", llm_ids,
                                     token_id, args.stem)
    rows1 = np.searchsorted(llm_ids, llm["id1"].to_numpy())
    rows2 = np.searchsorted(llm_ids, llm["id2"].to_numpy())
    target = llm["target"].to_numpy().astype(np.float32)
    del llm
    titles_t = torch.from_numpy(titles)
    keys_t = torch.from_numpy(keys)
    values_t = torch.from_numpy(values)

    # ---- модель ------------------------------------------------------------
    config = KNRMConfig(embedding_dim=300, num_kernels=args.num_kernels,
                        hidden_dim=args.hidden_dim, dropout=args.dropout,
                        attribute_chunk_size=args.chunk,
                        sparse_embedding=args.sparse)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Разреженный градиент нужен там, где плотный шаг по таблице стоит
        # столько же, сколько forward. На GPU это уже не так.
        config = replace(config, sparse_embedding=False)
    model = ProductMatcher(table, config)
    del table
    model.to(device)
    log(f"устройство: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    # Плотный шаг оптимизатора по таблице 943k x 300 стоит столько же, сколько
    # весь forward, хотя за батч трогается доля процента строк. Со sparse=True
    # градиент по таблице разрежен, и её ведёт SparseAdam, а остальную сеть —
    # обычный AdamW; ни один оптимизатор не умеет оба режима сразу.
    head = [parameter for name, parameter in model.named_parameters()
            if not name.startswith("embedding.")]
    optimizers = [torch.optim.AdamW(head, lr=args.learning_rate)]
    if not config.freeze_embeddings:
        embedding_lr = args.learning_rate * args.embedding_scale
        optimizers.append(
            torch.optim.SparseAdam(list(model.embedding.parameters()), lr=embedding_lr)
            if config.sparse_embedding else
            torch.optim.AdamW(list(model.embedding.parameters()), lr=embedding_lr))
    loss_function = nn.BCEWithLogitsLoss()
    parameters = sum(p.numel() for p in model.parameters())
    log(f"модель: {parameters / 1e6:.1f}M параметров, ядер {model.kernels.num_kernels}, "
        f"чанк {args.chunk}")

    # Ключ сортировки — максимум числа атрибутов у двух товаров пары: именно он
    # задаёт форму тензора в канале A-A, а значит и стоимость шага.
    counts = attribute_counts(keys_t, values_t)
    order_key = np.maximum(counts[rows1], counts[rows2])
    log(f"атрибутов в паре (max из двух): медиана {np.median(order_key):.0f}, "
        f"p90 {np.quantile(order_key, 0.9):.0f}, max {order_key.max()}")

    # Промежуточное состояние: прогон идёт часами, и внешняя остановка не должна
    # стоить всего. Порядок батчей эпохи выводится из (seed, epoch), поэтому
    # после загрузки достаточно пропустить уже пройденные батчи — список тот же.
    partial_path = args.out.with_suffix(".partial.pt")
    start_epoch, skip_batches = 1, 0
    if args.resume and partial_path.is_file():
        saved = torch.load(partial_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"])
        start_epoch, skip_batches = saved["epoch"], saved["completed_batches"]
        log(f"продолжаю с {partial_path.name}: эпоха {start_epoch}, "
            f"пройдено батчей {skip_batches:,}")

    def save_partial(epoch: int, completed: int) -> None:
        # Пишем во временный файл и переименовываем: прерывание на середине
        # записи не должно оставить битый чекпоинт вместо целого.
        temporary = partial_path.with_suffix(".tmp")
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "completed_batches": completed}, temporary)
        temporary.replace(partial_path)

    total_steps = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # rng выводим из эпохи, а не тянем состояние: список батчей обязан быть
        # воспроизводим, иначе пропуск пройденных батчей пропустит не те.
        batch_index = bucketed_batches(order_key, args.batch_size,
                                       np.random.default_rng([args.seed, epoch]))
        if skip_batches:
            log(f"  пропускаю {skip_batches:,} уже пройденных батчей эпохи {epoch}")
            batch_index = batch_index[skip_batches:]
            skip_batches = 0
        running, batches, epoch_started, done = 0.0, 0, time.time(), 0
        for pick in batch_index:
            item_a = make_batch(titles_t, keys_t, values_t, rows1[pick], device=device)
            item_b = make_batch(titles_t, keys_t, values_t, rows2[pick], device=device)
            logits = model(item_a, item_b)
            loss = loss_function(logits, torch.from_numpy(target[pick]).to(device))

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Клип только по плотной части: у разреженного градиента таблицы
            # clip_grad_norm_ материализовал бы её целиком и съел бы весь выигрыш.
            torch.nn.utils.clip_grad_norm_(head, 5.0)
            for optimizer in optimizers:
                optimizer.step()

            running += float(loss.detach())
            batches += 1
            total_steps += 1
            done += len(pick)
            if batches % args.log_every == 0:
                rate = done / max(time.time() - epoch_started, 1e-9)
                left = (len(target) - done) / max(rate, 1e-9)
                log(f"    эпоха {epoch}  {100 * done / len(target):4.1f}%  "
                    f"loss {running / batches:.5f}  {rate:.0f} пар/с  "
                    f"осталось {left / 60:.0f} мин")
            if args.checkpoint_every and batches % args.checkpoint_every == 0:
                save_partial(epoch, batches)
                log(f"      промежуточное состояние сохранено (батч {batches:,})")
            if args.max_steps and total_steps >= args.max_steps:
                break
        log(f"  эпоха {epoch}: loss {running / max(batches, 1):.5f}, "
            f"{time.time() - epoch_started:.0f}s")
        if args.max_steps and total_steps >= args.max_steps:
            log(f"остановлено по --max-steps {args.max_steps}")
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "token_id": token_id,
        "config": config.__dict__,
        "pretrain_pairs": int(len(target)),
        "epochs": args.epochs,
        "seed": args.seed,
        "stemming": bool(args.stem),
        "min_count": args.min_count,
        "max_shapes": {"title": MAX_TITLE, "attrs": MAX_ATTRS,
                       "key_tokens": MAX_KEY_TOKENS, "value_tokens": MAX_VALUE_TOKENS},
    }, args.out)
    partial_path.unlink(missing_ok=True)
    log(f"\nчекпоинт -> {args.out} ({args.out.stat().st_size / 1e9:.2f} ГБ), "
        f"всего {time.time() - started:.0f}s")
    (args.out.with_suffix(".json")).write_text(json.dumps({
        "vocabulary": len(token_id), "navec_covered": len(word_vectors),
        "pairs": int(len(target)), "parameters": parameters,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

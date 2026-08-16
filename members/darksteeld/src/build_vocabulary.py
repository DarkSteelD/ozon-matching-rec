"""Словарь токенов по всему каталогу: сырой и со стеммингом, в один проход.

Считает частоты по объединению ``items.parquet`` и ``items_human.parquet``
(товары второго содержатся в первом — это проверяется, а не предполагается, и
дубли по id отбрасываются). Токенизация та же, что у моделей:
``knrm_model.tokenize`` для названий и ``knrm_attrs_model.parse_attributes`` для
ключей и значений атрибутов.

**Зачем стемминг.** Словарь строится с порогом по частоте, и всё, что ниже
порога, модель на инференсе теряет. Русские словоформы дробят частоту одного
смысла на десяток вариантов: ``кроссовки``, ``кроссовок``, ``кроссовками``
считаются по отдельности, и каждая по отдельности может не дотянуть до порога,
хотя вместе они частотны. Стемминг их склеивает, поэтому при том же пороге
словарь покрывает больше текста.

Артикулы и модельные номера стемминг не трогает: Snowball сводит только
русские суффиксы, а ``gsr``, ``12v``, ``ts830p`` проходят как есть — что и
нужно, потому что именно они несут точное совпадение.

    .venv/bin/python members/darksteeld/src/build_vocabulary.py --out <путь>.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knrm_attrs_model import parse_attributes  # noqa: E402
from knrm_model import tokenize  # noqa: E402

FIELDS = ("name", "key", "value")


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0, help="0 = весь каталог; для отладки")
    args = parser.parse_args()

    import Stemmer

    stemmer = Stemmer.Stemmer("russian")
    stem_cache: dict[str, str] = {}

    def stem(token: str) -> str:
        # Кэш обязателен: токенов-вхождений сотни миллионов, а типов — миллионы,
        # и без него стемминг стал бы главной статьёй расхода.
        value = stem_cache.get(token)
        if value is None:
            value = stemmer.stemWord(token)
            stem_cache[token] = value
        return value

    raw: dict[str, Counter] = {field: Counter() for field in FIELDS}
    stemmed: dict[str, Counter] = {field: Counter() for field in FIELDS}

    # items_human должен целиком лежать внутри items — README это утверждает,
    # проверяем. Если так, второй файл читать незачем: его товары уже посчитаны.
    human_ids = set()
    for batch in pq.ParquetFile(args.data_dir / "items_human.parquet").iter_batches(
            batch_size=args.batch_size, columns=["id"]):
        human_ids.update(np.asarray(batch.column("id"), dtype=np.int64).tolist())
    log(f"items_human: {len(human_ids):,} товаров")

    seen_ids: set[int] = set()
    scanned, started = 0, time.time()
    for batch in pq.ParquetFile(args.data_dir / "items.parquet").iter_batches(
            batch_size=args.batch_size, columns=["id", "name", "attributes"]):
        ids = np.asarray(batch.column("id"), dtype=np.int64).tolist()
        names = batch.column("name").to_pylist()
        raws = batch.column("attributes").to_pylist()
        for item_id, name, attributes in zip(ids, names, raws):
            seen_ids.add(item_id)
            for token in tokenize(name):
                raw["name"][token] += 1
                stemmed["name"][stem(token)] += 1
            for key_tokens, value_tokens in parse_attributes(attributes):
                for token in key_tokens:
                    raw["key"][token] += 1
                    stemmed["key"][stem(token)] += 1
                for token in value_tokens:
                    raw["value"][token] += 1
                    stemmed["value"][stem(token)] += 1
        scanned += len(ids)
        if scanned % 2_000_000 < args.batch_size:
            log(f"  просмотрено {scanned:,}, типов имён {len(raw['name']):,}, "
                f"{time.time() - started:.0f}s")
        if args.limit and scanned >= args.limit:
            break

    log(f"\nпросмотрено {scanned:,} товаров каталога за {time.time() - started:.0f}s")
    if args.limit:
        log("вложенность items_human не проверялась: скан оборван --limit")
    else:
        missing = len(human_ids - seen_ids)
        log(f"товаров items_human, отсутствующих в items: {missing:,} "
            f"({'вложенность подтверждена' if missing == 0 else 'ВЛОЖЕННОСТЬ НАРУШЕНА'})")
        if missing:
            # Досчитываем недостающие: словарь обязан покрывать обе вселенные.
            log(f"  досчитываю {missing:,} товаров из items_human...")
            for batch in pq.ParquetFile(args.data_dir / "items_human.parquet").iter_batches(
                    batch_size=args.batch_size, columns=["id", "name", "attributes"]):
                ids = np.asarray(batch.column("id"), dtype=np.int64).tolist()
                names = batch.column("name").to_pylist()
                raws = batch.column("attributes").to_pylist()
                for item_id, name, attributes in zip(ids, names, raws):
                    if item_id in seen_ids:
                        continue
                    for token in tokenize(name):
                        raw["name"][token] += 1
                        stemmed["name"][stem(token)] += 1
                    for key_tokens, value_tokens in parse_attributes(attributes):
                        for token in key_tokens:
                            raw["key"][token] += 1
                            stemmed["key"][stem(token)] += 1
                        for token in value_tokens:
                            raw["value"][token] += 1
                            stemmed["value"][stem(token)] += 1

    log(f"\n{'поле':<10} {'типов сырых':>14} {'типов со стеммингом':>21} {'сжатие':>8}")
    for field in FIELDS:
        before, after = len(raw[field]), len(stemmed[field])
        log(f"{field:<10} {before:>14,} {after:>21,} {before / max(after, 1):>7.2f}x")

    log(f"\n{'поле':<10} {'порог':>6} {'словарь сырой':>15} {'покрытие':>9} "
        f"{'словарь со стем.':>17} {'покрытие':>9}")
    for field in FIELDS:
        r = np.array(sorted(raw[field].values()), dtype=np.int64)
        s = np.array(sorted(stemmed[field].values()), dtype=np.int64)
        for threshold in (1, 2, 5, 10):
            rv, sv = int((r >= threshold).sum()), int((s >= threshold).sum())
            rc = r[r >= threshold].sum() / r.sum()
            sc = s[s >= threshold].sum() / s.sum()
            log(f"{field:<10} {threshold:>6} {rv:>15,} {100 * rc:>8.2f}% "
                f"{sv:>17,} {100 * sc:>8.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for field in FIELDS:
        for label, table in (("raw", raw), ("stem", stemmed)):
            tokens, counts = zip(*table[field].items()) if table[field] else ((), ())
            payload[f"{label}_{field}_tokens"] = np.array(tokens, dtype=object)
            payload[f"{label}_{field}_counts"] = np.array(counts, dtype=np.int64)
    np.savez_compressed(args.out, **payload)
    log(f"\nсловари -> {args.out} ({args.out.stat().st_size / 1e6:.0f} МБ), "
        f"уникальных стемов в кэше {len(stem_cache):,}")


if __name__ == "__main__":
    main()

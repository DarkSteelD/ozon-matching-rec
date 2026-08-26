"""Компактный паркет текстов для LLM-претрена: только то, что поедет на GPU-бокс.

``items.parquet`` весит 4.1 ГБ, но претрену нужны не поля товара, а готовая
строка «имя | категория | атрибуты» ровно того вида, что строит инференс. Здесь
она собирается один раз, для тех 12.38M товаров, которые реально встречаются в
``matches_llm.parquet``, и пишется с сильным сжатием. Всё, что не участвует в
парах, отбрасывается.

    .venv/bin/python members/darksteeld/ops/build_llm_texts.py --out <файл>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW = REPOSITORY_ROOT / "data" / "raw"
ATTRS_LIMIT = 800


def compact_attrs(raw) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(d, dict):
        return ""
    parts = []
    for k in sorted(d, key=str.lower):
        v = d[k]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v[:6])
        parts.append(f"{k}:{v}")
    return "; ".join(parts)[:ATTRS_LIMIT]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    pairs = pl.read_parquet(RAW / "matches_llm.parquet", columns=["id1", "id2"])
    needed = pl.concat([pairs["id1"], pairs["id2"]]).unique()
    print(f"пар {pairs.height:,}; уникальных товаров {needed.len():,}")

    # 13.4M товаров с атрибутами в память не влезают (проверено: OOM на 36 ГБ),
    # поэтому идём по row group'ам исходного файла и пишем шардами.
    import pyarrow.parquet as pq

    keep = set(needed.to_list())
    del needed
    source = pq.ParquetFile(RAW / "items.parquet")
    shards = args.out.parent / (args.out.stem + "_shards")
    shards.mkdir(parents=True, exist_ok=True)
    for old in shards.glob("*.parquet"):
        old.unlink()

    written = 0
    for index in range(source.num_row_groups):
        table = source.read_row_group(index, columns=["id", "name", "category", "attributes"])
        ids = table.column("id").to_pylist()
        pick = [i for i, v in enumerate(ids) if v in keep]
        if not pick:
            continue
        names = table.column("name").to_pylist()
        cats = table.column("category").to_pylist()
        attrs = table.column("attributes").to_pylist()
        chunk_ids, chunk_text = [], []
        for i in pick:
            parts = (str(names[i] or ""), str(cats[i] or ""), compact_attrs(attrs[i]))
            chunk_ids.append(ids[i])
            chunk_text.append(" | ".join(x for x in parts if x))
        pl.DataFrame({"id": chunk_ids, "text": chunk_text}).write_parquet(
            shards / f"part_{index:03d}.parquet", compression="zstd", compression_level=9)
        written += len(chunk_ids)
        print(f"  группа {index+1}/{source.num_row_groups}: отобрано {len(chunk_ids):,} "
              f"(всего {written:,})", flush=True)
        del table, ids, names, cats, attrs, chunk_ids, chunk_text

    print(f"\nсобрано {written:,} товаров в {len(list(shards.glob('*.parquet')))} шардах")
    if written != len(keep):
        print(f"  ВНИМАНИЕ: {len(keep) - written:,} товаров пары ссылаются, а в каталоге их нет")
    out = pl.read_parquet(shards / "*.parquet")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out, compression="zstd", compression_level=9)
    for old in shards.glob("*.parquet"):
        old.unlink()
    shards.rmdir()
    print(f"записан {args.out} — {out.height:,} строк, {args.out.stat().st_size/1e6:.0f} МБ")
    print(f"длина текста: медиана {int(out['text'].str.len_chars().median())}, "
          f"95-й перцентиль {int(out['text'].str.len_chars().quantile(0.95))}")


if __name__ == "__main__":
    main()

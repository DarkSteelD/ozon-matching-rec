"""Отображение «сырой токен -> индекс словаря» для контейнера со стеммингом.

**Зачем.** Модель обучена по стеммам, значит контейнеру нужно стеммить тестовые
токены. Ставить его в зависимость от того, импортируется ли C-расширение
PyStemmer в чужом образе, опасно: если оно не заведётся, токенизация разойдётся
с обучением и модель молча выдаст мусор вместо ошибки.

Обхода нет только на первый взгляд. На инференсе стем нужен ровно затем, чтобы
найти строку таблицы; токен, чьего стема в словаре нет, всё равно становится
PAD. Значит достаточно заранее посчитать, какому индексу соответствует каждый
**сырой** токен каталога, и отгрузить это отображение. Стеммер в контейнере
тогда не нужен вовсе.

Теряется единственный случай: тестовый токен, которого нет в каталоге, но чей
стем в словаре есть (новая словоформа знакомого слова). Для него контейнер
попробует стеммер, если тот доступен, и иначе отправит токен в PAD.

    .venv/bin/python members/darksteeld/src/build_stem_map.py \\
        --token-id <path>/token_id.npz --vocabulary <path>/vocabulary_full.npz \\
        --out <container>/stem_map.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token-id", type=Path, required=True,
                        help="npz со стеммами словаря модели: tokens, ids")
    parser.add_argument("--vocabulary", type=Path, required=True,
                        help="npz от build_vocabulary.py: сырые токены каталога")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import Stemmer

    stemmer = Stemmer.Stemmer("russian")

    blob = np.load(args.token_id, allow_pickle=True)
    stem_to_id = dict(zip(blob["tokens"].tolist(), blob["ids"].tolist()))
    print(f"словарь модели: {len(stem_to_id):,} стеммов")

    vocabulary = np.load(args.vocabulary, allow_pickle=True)
    raw_tokens: set[str] = set()
    for field in ("name", "key", "value"):
        raw_tokens.update(vocabulary[f"raw_{field}_tokens"].tolist())
    print(f"сырых токенов каталога: {len(raw_tokens):,}")

    tokens, ids = [], []
    cache: dict[str, str] = {}
    for token in raw_tokens:
        stem = cache.get(token)
        if stem is None:
            stem = stemmer.stemWord(token)
            cache[token] = stem
        index = stem_to_id.get(stem)
        if index is not None:
            tokens.append(token)
            ids.append(index)
    print(f"из них ведут в словарь: {len(tokens):,} "
          f"({100 * len(tokens) / len(raw_tokens):.1f}%)")

    # Стеммы тоже кладём: токен может совпасть со своим стемом (латиница, цифры,
    # артикулы), и тогда прямой поиск сработает без всякого отображения.
    for stem, index in stem_to_id.items():
        tokens.append(stem)
        ids.append(index)

    unique: dict[str, int] = {}
    for token, index in zip(tokens, ids):
        unique[token] = index
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out,
                        tokens=np.array(list(unique.keys()), dtype=object),
                        ids=np.array(list(unique.values()), dtype=np.int32))
    print(f"отображение -> {args.out} ({args.out.stat().st_size / 1e6:.0f} МБ), "
          f"{len(unique):,} записей")


if __name__ == "__main__":
    main()

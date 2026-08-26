"""Обучающий паркет для сильнейшей одиночной модели команды (ce_priodistill, 0.8517).

Рецепт принадлежит dzkhomidov; здесь собран только вход к нему, потому что двух
шагов подготовки в ``members/dzkhomidov/src/`` нет — есть тренер, но нет того,
что ему скармливают. Восстанавливаются оба:

**Мягкие цели дистилляции.** Студент учится не на жёсткой метке, а на
``0.7 * final_stack_OOF + 0.3 * target``. Переобучать ради этого ансамбль не
нужно: его OOF лежит колонкой ``final_stack_all`` в
``members/dzkhomidov/preds/all_model_predictions_oof.parquet``.

**Prio-тексты.** Атрибуты переставлены бренд -> модель/артикул -> цвет вместо
алфавита, у фэшн-категорий выброшены размерные атрибуты, и в текст пары дописан
символьный блок ``@@ сравнение: цвет=...; артикул=...``. Логика взята дословно
из ``members/dzkhomidov/container/run_v4.py`` — там она на стороне инференса, а
обучение обязано видеть ровно те же строки, иначе модель учится на одном
распределении, а работает на другом.

**Про утечку в дистилляции.** Мягкая цель строки из фолда J — это предсказание
ансамбля, обученного на фолдах кроме J, то есть в том числе на фолде K. Модель
фолда K учится на таких целях и проверяется на K, так что часть измеренного
прироста дистилляции (+0.0081 по их логу) может быть переносом меток фолда K
через цель. На отправку это не влияет: финальная модель учится на всех ручных
парах, а тест ей не виден ни прямо, ни через цель. Но локальное 0.8517 стоит
читать как верхнюю оценку.

    .venv/bin/python members/darksteeld/src/build_distill_pairs.py --out <файл>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW = REPOSITORY_ROOT / "data" / "raw"
OOF = REPOSITORY_ROOT / "members" / "dzkhomidov" / "preds" / "all_model_predictions_oof.parquet"

ATTRS_LIMIT = 800
PRIO = ['бренд', 'модель', 'артикул', 'код товара', 'партномер', 'цвет',
        'название цвета', 'тип', 'материал']
DROP_FASHION = ['размер', 'российский размер', 'длина стельки']
FASHION = {'Обувь', 'Одежда', 'Галантерея и аксессуары', 'Ювелирные изделия'}


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


def parse_kv(attr: str) -> list[tuple[str, str]]:
    out = []
    if attr:
        for part in attr.split('; '):
            k, _, v = part.partition(':')
            if v:
                out.append((k.strip().lower(), v.strip()))
    return out


def prio_attrs(attr: str, fashion: bool) -> str:
    kv = parse_kv(attr)
    if fashion:
        kv = [(k, v) for k, v in kv if not any(d in k for d in DROP_FASHION)]

    def idx(k: str) -> int:
        for i, p in enumerate(PRIO):
            if p in k:
                return i
        return len(PRIO)

    kv.sort(key=lambda x: (idx(x[0]), x[0]))
    return '; '.join(f'{k}:{v}' for k, v in kv)[:700]


def getv(kv, keys):
    for k, v in kv:
        if any(kk in k for kk in keys):
            return v.lower()
    return None


def cmp_tok(a, b) -> str:
    if a is None or b is None:
        return 'неизвестно'
    if a == b or (len(a) > 4 and (a in b or b in a)):
        return 'совпал'
    return 'различен'


def closure_rows(oof, prepared, cache):
    """Выведенные транзитивностью пары с ручными вердиктами.

    Учителя у них нет — их нет в OOF-паркете, потому что ансамбль их никогда не
    предсказывал. Мягкой целью служит сам вердикт: 1 для подтверждённых, 0 для
    отвергнутых. Пары с вердиктом «не определить» не берутся вовсе.

    Фолд считается по тому же разбиению, что у паркета: обе стороны выведенной
    пары лежат в одной компоненте по всем размеченным парам, значит и в одном
    фолде, и build_closure это проверяет ассертом.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from closure_pairs import build_closure

    verdicts_file = (REPOSITORY_ROOT / "members" / "darksteeld" / "data"
                     / "closure_verdicts.jsonl")
    verdict = {}
    if verdicts_file.is_file():
        for line in verdicts_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                verdict[(r["id1"], r["id2"])] = r["audited_label"]
    if not verdict:
        raise SystemExit(f"нет вердиктов в {verdicts_file}")

    fold_index = {f: i for i, f in enumerate(sorted(set(oof["fold"].to_list())))}
    pairs = list(zip(oof["id1"].to_list(), oof["id2"].to_list()))
    labels = oof["target"].to_list()
    folds = [fold_index[f] for f in oof["fold"].to_list()]
    rejected = {k for k, v in verdict.items() if v == 0}
    produced, produced_y, produced_fold, _ = build_closure(pairs, labels, folds)

    back = {i: f for f, i in fold_index.items()}
    rows = []
    for key, lab, fold in zip(produced, produced_y, produced_fold):
        decided = verdict.get(key)
        if lab == 1.0 and decided is None:
            continue                      # выведенный позитив без вердикта не берём
        if decided == -1:
            continue
        target = float(decided) if decided is not None else lab
        n1, c1, kv1, a1 = prepared(key[0])
        n2, c2, kv2, a2 = prepared(key[1])
        diff = (" @@ сравнение: цвет=" + cmp_tok(getv(kv1, ["цвет"]), getv(kv2, ["цвет"]))
                + "; артикул=" + cmp_tok(
                    getv(kv1, ["артикул", "модель", "код товара", "партномер"]),
                    getv(kv2, ["артикул", "модель", "код товара", "партномер"])))
        rows.append({"fold": back[fold], "id1": key[0], "id2": key[1],
                     "target": int(target), "soft_target": target,
                     "text1": f"{n1} | {c1} | {a1}{diff}",
                     "text2": f"{n2} | {c2} | {a2}{diff}", "source": "closure"})
    from collections import Counter
    print(f"замыкание: {len(rows):,} пар "
          f"({dict(Counter(r['target'] for r in rows))}), "
          f"отвергнутых вручную исключено {len(rejected):,}")
    return pl.DataFrame(rows, schema=["fold", "id1", "id2", "target",
                                      "soft_target", "text1", "text2", "source"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", default="final_stack_all",
                        help="колонка OOF, с которой снимаются мягкие цели")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="вес мягкой цели против жёсткой метки")
    parser.add_argument("--with-closure", action="store_true",
                        help="добавить выведенные транзитивностью пары, отсуженные вручную")
    args = parser.parse_args()

    oof = pl.read_parquet(OOF, columns=["fold", "id1", "id2", "target", args.teacher])
    items = pl.read_parquet(RAW / "items_human.parquet",
                            columns=["id", "name", "category", "attributes"])
    name = dict(zip(items["id"].to_list(), items["name"].to_list()))
    category = dict(zip(items["id"].to_list(), items["category"].to_list()))
    attributes = dict(zip(items["id"].to_list(), items["attributes"].to_list()))
    print(f"пар {oof.height:,}; учитель {args.teacher}, alpha {args.alpha}")

    cache: dict[int, tuple[str, str, list, str]] = {}

    def prepared(item: int):
        if item not in cache:
            cat = category.get(item) or ""
            compact = compact_attrs(attributes.get(item))
            # Их run_v4 восстанавливает имя как text.split(" | ")[0] из склейки
            # "имя | категория | атрибуты", поэтому у 0.4% товаров, где ' | '
            # встречается внутри самого имени, оно обрезается. Это дефект, но он
            # происходит НА ИНФЕРЕНСЕ, и обучение обязано видеть ровно то же —
            # иначе train и test разойдутся по распределению. Воспроизводим.
            plain = str(name.get(item) or "").split(" | ")[0]
            cache[item] = (plain, cat, parse_kv(compact),
                           prio_attrs(compact, cat in FASHION))
        return cache[item]

    text1, text2 = [], []
    for left, right in zip(oof["id1"].to_list(), oof["id2"].to_list()):
        n1, c1, kv1, a1 = prepared(left)
        n2, c2, kv2, a2 = prepared(right)
        diff = (" @@ сравнение: цвет=" + cmp_tok(getv(kv1, ["цвет"]), getv(kv2, ["цвет"]))
                + "; артикул=" + cmp_tok(
                    getv(kv1, ["артикул", "модель", "код товара", "партномер"]),
                    getv(kv2, ["артикул", "модель", "код товара", "партномер"])))
        text1.append(f"{n1} | {c1} | {a1}{diff}")
        text2.append(f"{n2} | {c2} | {a2}{diff}")

    soft = (args.alpha * oof[args.teacher].to_numpy()
            + (1 - args.alpha) * oof["target"].to_numpy().astype(float))
    out = oof.select("fold", "id1", "id2", "target").with_columns(
        pl.Series("soft_target", soft),
        pl.Series("text1", text1),
        pl.Series("text2", text2),
        pl.lit("hand").alias("source"),
    )

    if args.with_closure:
        out = pl.concat([out, closure_rows(oof, prepared, cache)], how="vertical")
        print(f"с замыканием: {out.height:,} строк "
              f"({int((out['source'] == 'closure').sum()):,} выведенных)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print(f"записан {args.out} ({out.height:,} строк, "
          f"{args.out.stat().st_size / 1e6:.0f} МБ)")
    print(f"мягкая цель: медиана {float(out['soft_target'].median()):.4f}, "
          f"диапазон {float(out['soft_target'].min()):.4f}..{float(out['soft_target'].max()):.4f}")
    print(f"длина текста: медиана {int(out['text1'].str.len_chars().median())} символов, "
          f"95-й перцентиль {int(out['text1'].str.len_chars().quantile(0.95))}")


if __name__ == "__main__":
    main()

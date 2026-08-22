"""Очередь доразметки по выведенным парам, с цепочкой, которая их породила.

Выведенная пара — это утверждение транзитивности: a≡b и b≡c, значит a≡c. Судить
её в отрыве нельзя, потому что вердикт «не дубль» на самом деле относится не к
ней, а к цепочке: если a и c не дубли, то неверно одно из **размеченных** рёбер
a—b или b—c. Поэтому очередь везёт путь целиком.

Ценность здесь именно в этом. Сами выведенные пары в обучении измерены и стоят
−0.0002 (см. leaderboard_v2), так что чистить их незачем. А вот рёбра цепочки
лежат и в обучающей выборке, и в целевых файлах фолдов — их исправление имеет
вес. И, в отличие от прежних очередей, сигнал здесь структурный: пара попадает в
очередь потому, что метки образовали треугольник, а не потому, что модель
поспорила. Прежние очереди отбирались ошибками модели и потому смещены к тому,
что модель и так не умеет.

    .venv/bin/python members/darksteeld/src/build_closure_queue.py \\
        --judge-dir <каталог с кросс-энкодером>   # ранжирование, необязательно

Пишет ``members/darksteeld/data/closure_queue.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closure_pairs import build_closure, positive_components  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPOSITORY_ROOT / "data" / "raw"
TARGETS_DIR = REPOSITORY_ROOT / "validation" / "targets_v2"
AUDIT_FILE = REPOSITORY_ROOT / "members" / "darksteeld" / "data" / "label_audit.jsonl"
OUT_FILE = REPOSITORY_ROOT / "members" / "darksteeld" / "data" / "closure_queue.csv"


def load_audit() -> dict[tuple[int, int], int]:
    if not AUDIT_FILE.is_file():
        return {}
    latest: dict[tuple[int, int], dict] = {}
    for line in AUDIT_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["id1"], r["id2"])] = r
    return {k: v["audited_label"] for k, v in latest.items() if v["audited_label"] >= 0}


def load_pairs() -> tuple[list[tuple[int, int]], list[int], list[int], dict]:
    pairs, labels, folds, category = [], [], [], {}
    for index, path in enumerate(sorted(TARGETS_DIR.glob("fold_*.csv"))):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            key = (int(row["id1"]), int(row["id2"]))
            pairs.append(key)
            labels.append(int(row["target"]))
            folds.append(index)
            category[key] = row["category"]
    return pairs, labels, folds, category


def shortest_chain(adjacency: dict[int, set[int]], source: int, target: int) -> list[int]:
    """Путь по РАЗМЕЧЕННЫМ положительным рёбрам — то, из чего выведена пара."""
    previous = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbour in adjacency[node]:
            if neighbour not in previous:
                previous[neighbour] = node
                queue.append(neighbour)
    if target not in previous:
        return []
    path, node = [], target
    while node is not None:
        path.append(node)
        node = previous[node]
    return path[::-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=None,
                        help="каталог с кросс-энкодером для ранжирования очереди")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    import polars as pl

    pairs, labels, folds, category = load_pairs()
    corrections = load_audit()
    applied = 0
    for position, key in enumerate(pairs):
        if key in corrections and corrections[key] != labels[position]:
            labels[position] = corrections[key]
            applied += 1
    print(f"пар {len(pairs):,}; исправлений журнала применено {applied}")

    produced, produced_labels, produced_folds, contradictions = build_closure(
        pairs, labels, folds)
    if contradictions:
        print(f"противоречий после исправлений: {len(contradictions)} "
              f"(их компоненты исключены)")
    implied = [(p, f) for p, y, f in zip(produced, produced_labels, produced_folds) if y == 1.0]
    print(f"выведенных положительных пар: {len(implied):,}")

    # уже отсуженные выведенные пары в очередь не попадают
    judged = {k for k in corrections}
    implied = [(p, f) for p, f in implied if p not in judged]
    print(f"из них ещё не отсужено: {len(implied):,}")

    component = positive_components(pairs, labels)
    members = defaultdict(list)
    for node, root in component.items():
        members[root].append(node)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (left, right), label in zip(pairs, labels):
        if label == 1:
            adjacency[left].add(right)
            adjacency[right].add(left)

    items = pl.read_parquet(RAW_DIR / "items_human.parquet",
                            columns=["id", "name", "category", "attributes"])
    name = dict(zip(items["id"].to_list(), items["name"].to_list()))
    attributes = dict(zip(items["id"].to_list(), items["attributes"].to_list()))
    item_category = dict(zip(items["id"].to_list(), items["category"].to_list()))

    rows = []
    for (left, right), fold in implied:
        chain = shortest_chain(adjacency, left, right)
        rows.append({
            "id1": left, "id2": right, "fold": f"fold_{fold + 1:02d}",
            "category": item_category.get(left, ""),
            "component_size": len(members[component[left]]),
            "chain_len": max(len(chain) - 1, 0),
            "chain": " -> ".join(str(x) for x in chain),
        })
    print(f"длина цепочки: { {k: sum(1 for r in rows if r['chain_len'] == k) for k in sorted({r['chain_len'] for r in rows})} }")

    scores = None
    if args.judge_dir:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        def text(item: int) -> str:
            return f"{name.get(item, '')} | {item_category.get(item, '')} | {attributes.get(item, '')}"

        tokenizer = AutoTokenizer.from_pretrained(str(args.judge_dir))
        device = torch.device(args.device)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(args.judge_dir), dtype=torch.float16 if device.type != "cpu" else torch.float32)
        model = model.to(device).eval()
        use_type_ids = getattr(model.config, "type_vocab_size", 0) > 1
        left_text = [text(r["id1"]) for r in rows]
        right_text = [text(r["id2"]) for r in rows]
        scores = np.zeros(len(rows))
        with torch.inference_mode():
            for start in range(0, len(rows), 64):
                batch = tokenizer(left_text[start:start + 64], right_text[start:start + 64],
                                  truncation=True, max_length=args.max_len,
                                  padding=True, return_tensors="pt")
                if not use_type_ids:
                    batch.pop("token_type_ids", None)
                batch = {k: v.to(device) for k, v in batch.items()}
                logit = model(**batch).logits.float()
                probability = (torch.softmax(logit, -1)[:, 1] if logit.shape[-1] > 1
                               else torch.sigmoid(logit[:, 0]))
                scores[start:start + 64] = probability.cpu().numpy()
        print(f"арбитр: медиана {np.median(scores):.4f}, доля <0.5 {100 * (scores < 0.5).mean():.1f}%")

    for position, row in enumerate(rows):
        row["judge"] = f"{scores[position]:.6f}" if scores is not None else ""
    rows.sort(key=lambda r: (float(r["judge"]) if r["judge"] else 1.0, -r["component_size"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"записано {len(rows):,} строк -> {args.out.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()

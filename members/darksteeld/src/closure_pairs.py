"""Пары, выводимые из ручной разметки транзитивностью, с проверкой на утечку.

Разметка организаторов согласована с разбиением товаров на классы: отрицательных
рёбер внутри положительной компоненты нет ни одного из 271 764, а у размеченных
пар вообще нет общего соседа — треугольник ни разу не замкнут. Значит компонента
связности по положительным рёбрам — это класс эквивалентности, и из него
выводятся метки, которых в файле нет:

* **положительные** — пара товаров внутри одной компоненты (a≡b, b≡c ⟹ a≡c);
* **отрицательные** — пара из двух разных компонент, между которыми есть хотя бы
  одно размеченное отрицательное ребро (a≡b, b⊥c ⟹ a⊥c).

Утечки это не создаёт, но полагаться на рассуждение здесь нельзя, поэтому
``build_closure`` проверяет каждую выведенную пару: оба её товара обязаны лежать
в одном фолде, иначе исключение. Аргумент таков — фолды бьются по компонентам
**всех** пар, а обе конструкции выше связывают свои концы цепочкой размеченных
рёбер, то есть держат их в одной такой компоненте. Проверка ловит случай, когда
это перестанет быть правдой (например, если разбиение поменяют).

Выведенные пары идут только в обучение. Целевые файлы фолдов остаются те же, и
метрика считается на них, иначе скор перестанет сравниваться с лидербордом.
"""

from __future__ import annotations

from collections import defaultdict


def positive_components(pairs: list[tuple[int, int]],
                        labels) -> dict[int, int]:
    """Товар -> корень его компоненты по рёбрам с меткой 1."""
    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for (left, right), label in zip(pairs, labels, strict=True):
        find(left), find(right)
        if label == 1:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
    return {node: find(node) for node in parent}


def build_closure(pairs: list[tuple[int, int]], labels, fold_of_row,
                  rejected: set[tuple[int, int]] | None = None,
                  ) -> tuple[list[tuple[int, int]], list[float], list[int], list[tuple[int, int]]]:
    """Выведенные пары, их метки, фолды и список противоречий.

    Противоречие — отрицательное ребро внутри положительной компоненты. В
    исходных данных таких нет, но исправления журнала сливают компоненты и могут
    их создать. Такая компонента целиком исключается из замыкания: её класс
    эквивалентности перестал быть классом, и выводить из него нечего.

    ``rejected`` — выведенные пары, отсуженные вручную как «не дубль». Это то же
    самое противоречие, только доказанное человеком, а не найденное в файле:
    если a и c не дубли, то цепочка a≡b≡c держится на неверном размеченном
    ребре. Компонента такой пары исключается наравне с остальными.
    """
    rejected = rejected or set()
    component = positive_components(pairs, labels)
    members: dict[int, list[int]] = defaultdict(list)
    for node, root in component.items():
        members[root].append(node)
    for group in members.values():
        group.sort()

    fold_of_item: dict[int, int] = {}
    for (left, right), fold in zip(pairs, fold_of_row, strict=True):
        for node in (left, right):
            previous = fold_of_item.setdefault(node, int(fold))
            if previous != int(fold):
                raise AssertionError(f"товар {node} встречается в фолдах {previous} и {int(fold)}")

    contradictions = [(left, right)
                      for (left, right), label in zip(pairs, labels, strict=True)
                      if label == 0 and component[left] == component[right]]
    poisoned = {component[left] for left, _ in contradictions}
    for left, right in rejected:
        if left in component and right in component and component[left] == component[right]:
            poisoned.add(component[left])
            contradictions.append((left, right))

    known = {(min(a, b), max(a, b)) for a, b in pairs}
    produced: dict[tuple[int, int], float] = {}

    for root, group in members.items():
        if len(group) < 3 or root in poisoned:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                key = (group[i], group[j])
                if key not in known:
                    produced[key] = 1.0

    for (left, right), label in zip(pairs, labels, strict=True):
        if label != 0 or component[left] in poisoned or component[right] in poisoned:
            continue
        for node_left in members[component[left]]:
            for node_right in members[component[right]]:
                if node_left == node_right:
                    continue
                key = (min(node_left, node_right), max(node_left, node_right))
                if key not in known and key not in produced:
                    produced[key] = 0.0

    out_pairs, out_labels, out_folds = [], [], []
    for key in sorted(produced):
        left_fold, right_fold = fold_of_item[key[0]], fold_of_item[key[1]]
        if left_fold != right_fold:
            raise AssertionError(
                f"выведенная пара {key} разрывает фолды {left_fold} и {right_fold} — "
                "это утечка, разбиение больше не совместимо с замыканием")
        out_pairs.append(key)
        out_labels.append(produced[key])
        out_folds.append(left_fold)
    return out_pairs, out_labels, out_folds, contradictions

"""E-CUP 2026 matching: стек RuModernBERT (prio, 224) + rubert-base (простые, 384).

Инференс повторяет ``members/dzkhomidov/container/run_v4.py`` в части сборки
текстов — дословно, включая обрезку имени по первому ``' | '``. Это дефект их
реализации, но обучающий паркет собран ``build_distill_pairs.py`` с ровно тем же
дефектом, и расхождение здесь означало бы расхождение train с test.

Два отличия от run_v4, оба существенные:

*   **Голова на два логита.** Их модели отдают один логит, и они берут
    ``sigmoid(logits.squeeze(-1))``. Наша обучена с ``num_labels=2`` и скор — это
    ``sigmoid(logit1 - logit0)``. Прогон нашей модели через их код не упал бы с
    ошибкой формы только случайно: ``squeeze(-1)`` на форме (N, 2) ничего не
    делает, и присваивание (N, 2) в срез (N,) уже валится. Здесь форма
    определяется на месте, так что контейнер работает с любой из двух голов.
*   **Одна модель вместо ансамбля.** Отгружается усреднение весов четырёх
    фолдовых моделей, а не их предсказания: одна модель на инференсе вместо
    четырёх, то есть вчетверо меньше времени при том же числе обученных сетей.

Штраф за расхождение размеров в фэшн-категориях сохранён: локально он метрику
роняет на 0.0218, но на лидерборде дал +0.00233 (v3 0.49460 -> v4 0.49693).
Это прямое свидетельство, что соглашения о размерах в train и test разные, и
верить тут надо лидерборду, а не локальной валидации.
"""
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pair_budget import fit_pair

T0 = time.time()
TIME_BUDGET = 15 * 60  # запас 5 минут внутри 20-минутного лимита проверки.
# Пять, а не три: посылка с бюджетом 17 минут скор получила, но потом падала
# по таймауту, то есть трёх минут на загрузку моделей и сборку текстов не
# хватало. Работа теперь вдвое меньше, так что запас берём с избытком.
ATTRS_LIMIT = 800

PRIO = ['бренд', 'модель', 'артикул', 'код товара', 'партномер', 'цвет',
        'название цвета', 'тип', 'материал']
DROP_FASHION = ['размер', 'российский размер', 'длина стельки']
FASHION = {'Обувь', 'Одежда', 'Галантерея и аксессуары', 'Ювелирные изделия'}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def elapsed() -> float:
    return time.time() - T0


def compact_attrs(raw) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    parts = []
    for key in sorted(parsed, key=str.lower):
        value = parsed[key]
        if isinstance(value, list):
            value = ",".join(str(x) for x in value[:6])
        parts.append(f"{key}:{value}")
    return "; ".join(parts)[:ATTRS_LIMIT]


def parse_kv(attr):
    out = []
    if attr:
        for part in attr.split('; '):
            key, _, value = part.partition(':')
            if value:
                out.append((key.strip().lower(), value.strip()))
    return out


def prio_attrs(attr, fashion):
    kv = parse_kv(attr)
    if fashion:
        kv = [(k, v) for k, v in kv if not any(d in k for d in DROP_FASHION)]

    def index(key):
        for i, name in enumerate(PRIO):
            if name in key:
                return i
        return len(PRIO)
    kv.sort(key=lambda pair: (index(pair[0]), pair[0]))
    return '; '.join(f'{k}:{v}' for k, v in kv)[:700]


def getv(kv, keys):
    for key, value in kv:
        if any(wanted in key for wanted in keys):
            return value.lower()
    return None


def cmp_tok(left, right):
    if left is None or right is None:
        return 'неизвестно'
    if left == right or (len(left) > 4 and (left in right or right in left)):
        return 'совпал'
    return 'различен'


def size_sets(kv):
    out = {}
    for key, value in kv:
        if 'размер' not in key or 'упаков' in key:
            continue
        lowered = value.lower()
        if 'росс' in key or ' ru' in lowered or key.endswith('ru'):
            system = 'ru'
        elif 'производител' in key or 'eu' in lowered or 'us' in lowered:
            system = 'mk'
        else:
            system = 'plain'
        numbers = set(x.replace(',', '.') for x in _NUM.findall(value))
        if numbers:
            out.setdefault(system, set()).update(numbers)
    return out


def size_mismatch(first, second):
    for system in ('ru', 'plain', 'mk'):
        if system in first and system in second:
            return not (first[system] & second[system])
    return False


def build_prio_texts(matches, id2name, id2cat, id2attr):
    """prio-тексты и флаги штрафа за размер. Кэш по товару: в матчах один и тот
    же товар встречается многократно, а разбор атрибутов недёшев."""
    left_texts, right_texts, penalty = [], [], []
    cache = {}
    for first, second in zip(matches["id1"], matches["id2"]):
        for item in (first, second):
            if item not in cache:
                category = id2cat.get(item) or ""
                compact = compact_attrs(id2attr.get(item))
                # Обрезка имени по первому ' | ' — воспроизведение дефекта run_v4,
                # который видело обучение. Затрагивает ~0.4% товаров.
                name = str(id2name.get(item) or "").split(" | ")[0]
                cache[item] = (name, category, parse_kv(compact),
                               prio_attrs(compact, category in FASHION))
        name1, cat1, kv1, attrs1 = cache[first]
        name2, cat2, kv2, attrs2 = cache[second]
        diff = (" @@ сравнение: цвет=" + cmp_tok(getv(kv1, ["цвет"]), getv(kv2, ["цвет"]))
                + "; артикул=" + cmp_tok(
                    getv(kv1, ["артикул", "модель", "код товара", "партномер"]),
                    getv(kv2, ["артикул", "модель", "код товара", "партномер"])))
        left_texts.append(f"{name1} | {cat1} | {attrs1}{diff}")
        right_texts.append(f"{name2} | {cat2} | {attrs2}{diff}")
        penalty.append(cat1 in FASHION
                       and size_mismatch(size_sets(kv1), size_sets(kv2)))
    return left_texts, right_texts, penalty


def build_plain_texts(matches, id2name, id2cat, id2attr):
    """Простые тексты «имя | категория | атрибуты» — то, что читает rubase384.
    Имя здесь НЕ обрезается: обрезка живёт только в prio-ветке, и обучающий
    паркет для этой модели собран ровно так же, с полным именем."""
    cache = {}
    for item in set(matches["id1"]) | set(matches["id2"]):
        text = str(id2name.get(item) or "")
        category = id2cat.get(item)
        if category:
            text = f"{text} | {category}"
        attrs = compact_attrs(id2attr.get(item))
        if attrs:
            text = f"{text} | {attrs}"
        cache[item] = text
    return ([cache[i] for i in matches["id1"]], [cache[i] for i in matches["id2"]])


def score(spec, left, right, device, deadline=None):
    tokenizer = AutoTokenizer.from_pretrained(spec["path"])
    if spec.get("budget_attrs"):
        # Модель обучалась с обрезкой хвостов атрибутов вместо конца пары.
        # Инференс обязан повторять это дословно, иначе вход разойдётся с train.
        fitted = [fit_pair(x, y, tokenizer, spec.get("max_len", 224))
                  for x, y in zip(left, right)]
        left = [f[0] for f in fitted]
        right = [f[1] for f in fitted]
        print(f"  {spec['path']}: атрибуты подрезаны под {spec.get('max_len')}",
              flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(spec["path"])
    model = model.half().to(device) if device.type == "cuda" else model.to(device)
    model.eval()
    # ModernBERT не использует token_type_ids; у моделей на BERT они нужны.
    use_type_ids = getattr(model.config, "type_vocab_size", 0) > 1
    max_len = spec.get("max_len", 224)

    total = len(left)
    # Сортировка по грубой длине: внутри батча padding до самой длинной пары,
    # поэтому близкие по длине пары дают меньше пустой работы.
    rough = np.array([len(a) + len(b) for a, b in zip(left, right)])
    order = np.argsort(rough, kind="stable")
    out = np.zeros(total, dtype=np.float64)
    batch = spec.get("batch", 256) if device.type == "cuda" else 32
    done, started = 0, time.time()
    with torch.inference_mode():
        for start in range(0, total, batch):
            picked = order[start:start + batch]
            # Если по текущей скорости не укладываемся, режем длину. Батчи
            # отсортированы, так что урезается только дорогой хвост.
            if done > 5000 and max_len > 144:
                rate = done / (time.time() - started)
                if elapsed() + (total - done) / rate > (deadline or TIME_BUDGET * 0.95):
                    max_len = max(144, int(max_len * 0.75))
                    print(f"режу max_len -> {max_len} (прошло {elapsed():.0f}с)",
                          flush=True)
            encoded = tokenizer([left[i] for i in picked], [right[i] for i in picked],
                                truncation=True, max_length=max_len, padding=True,
                                return_tensors="pt").to(device)
            if not use_type_ids:
                encoded.pop("token_type_ids", None)
            logits = model(**encoded).logits
            # Голова на два логита: скор — разность, как в обучении. На одном
            # логите берём его сам.
            margin = (logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2
                      else logits[:, 0])
            out[picked] = torch.sigmoid(margin.float()).cpu().numpy()
            done += len(picked)
            if done % (batch * 50) < batch:
                print(f"{done}/{total} прошло {elapsed():.0f}с", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", required=True)
    parser.add_argument("--matches_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство {device}, прошло {elapsed():.0f}с", flush=True)

    items = pd.read_parquet(args.items_path,
                            columns=["id", "name", "category", "attributes"])
    matches = pd.read_parquet(args.matches_path)[["id1", "id2"]]
    id2name = dict(zip(items["id"], items["name"]))
    id2cat = dict(zip(items["id"], items["category"]))
    id2attr = dict(zip(items["id"], items["attributes"]))
    del items

    # prio-тексты нужны всегда: из них же берутся флаги штрафа за размер.
    prio_left, prio_right, penalty = build_prio_texts(matches, id2name, id2cat, id2attr)
    specs = json.load(open("models.json"))
    # models.json бывает списком моделей (старая форма) или объектом с полем
    # routes: категория -> [[номер модели, вес], ...]. Вторая нужна затем, что в
    # четырёх категориях впереди не одиночная модель, а смесь двух.
    if isinstance(specs, dict):
        ROUTES = {c: [(int(i), float(w)) for i, w in v]
                  for c, v in specs.get("routes", {}).items()}
        specs = specs["models"]
    else:
        ROUTES = {c: [(i, 1.0)] for i, spec in enumerate(specs)
                  for c in spec.get("categories", [])}
    plain_left = plain_right = None
    if any(spec.get("texts", "prio") == "plain" for spec in specs):
        plain_left, plain_right = build_plain_texts(matches, id2name, id2cat, id2attr)
    print(f"{len(prio_left)} пар, тексты собраны, прошло {elapsed():.0f}с", flush=True)

    routes = ROUTES
    pair_category = np.array([id2cat.get(i) or "" for i in matches["id1"]])
    total_pairs = len(pair_category)
    routed_any = (np.isin(pair_category, list(routes)) if routes
                  else np.zeros(total_pairs, dtype=bool))
    blend_mask = ~routed_any

    # Каждая модель считает только те пары, которые дойдут до ответа: общий
    # бленд плюс её собственные маршруты.
    todo = {}
    for number, spec in enumerate(specs):
        want = (blend_mask.copy() if spec.get("weight", 0.0) > 0
                else np.zeros(total_pairs, dtype=bool))
        own = [c for c, members in routes.items() if any(i == number for i, _ in members)]
        if own:
            want |= np.isin(pair_category, own)
        if want.any():
            todo[number] = np.flatnonzero(want)
    planned = sum(len(v) for v in todo.values())
    print(f"к счёту {planned} пар вместо {total_pairs * len(specs)} "
          f"({planned / max(1, total_pairs * len(specs)):.0%} работы)", flush=True)

    ranks, weights, raw_scores = [], [], {}
    remaining = planned
    for number, spec in enumerate(specs):
        if number not in todo:
            print(f"{spec['path']}: ни одной пары, пропускаю", flush=True)
            continue
        index = todo[number]
        plain = spec.get("texts", "prio") == "plain"
        source_left = plain_left if plain else prio_left
        source_right = plain_right if plain else prio_right
        deadline = elapsed() + (TIME_BUDGET * 0.95 - elapsed()) * len(index) / max(1, remaining)
        part = score(spec, [source_left[i] for i in index],
                     [source_right[i] for i in index], device, deadline)
        remaining -= len(index)
        predicted = np.full(total_pairs, np.nan)
        predicted[index] = part
        raw_scores[number] = predicted
        if spec.get("weight", 0.0) > 0:
            rank = np.zeros(total_pairs)
            rank[index] = pd.Series(part).rank().to_numpy() / len(part)
            ranks.append(rank * spec["weight"])
            weights.append(spec["weight"])
        print(f"{spec['path']} готов на {len(index)} парах, "
              f"прошло {elapsed():.0f}с", flush=True)

    # Незакрытые пары получают 0.5: до этого доходит только если модель совсем
    # не отработала, и лучше ровная середина, чем NaN в ответе.
    final = np.full(total_pairs, 0.5, dtype=np.float64)
    if ranks:
        blended = np.sum(ranks, axis=0) / np.sum(weights)
        final[blend_mask] = blended[blend_mask]
    for category, members in routes.items():
        picked = pair_category == category
        if not picked.any():
            continue
        # Внутри категории берём ранги каждой назначенной модели и смешиваем по
        # весам. Метрика читает только порядок внутри категории, поэтому шкалы
        # с другими категориями согласовывать не нужно.
        mixed, total_weight, missing = np.zeros(int(picked.sum())), 0.0, []
        for number, weight in members:
            scored = raw_scores.get(number)
            if scored is None or np.isnan(scored[picked]).any():
                missing.append(number)
                continue
            inside = scored[picked]
            mixed += weight * (np.argsort(np.argsort(inside)) + 1) / len(inside)
            total_weight += weight
        if total_weight <= 0:
            print(f"  {category}: ни одна модель не отработала, остаётся бленд",
                  flush=True)
            continue
        if missing:
            print(f"  {category}: пропущены модели {missing}, смесь из оставшихся",
                  flush=True)
        final[picked] = mixed / total_weight
        names = "+".join(specs[i]["path"] for i, _ in members if i not in missing)
        print(f"  {category}: {int(picked.sum())} пар отданы {names}", flush=True)

    mask = np.array(penalty)
    if mask.any():
        final[mask] *= 0.25
        print(f"штраф за размер применён к {int(mask.sum())} фэшн-парам", flush=True)

    pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"],
                  "predict": final}).to_csv(args.output_path, index=False)
    print(f"записан {args.output_path}, всего {elapsed():.0f}с", flush=True)


if __name__ == "__main__":
    main()

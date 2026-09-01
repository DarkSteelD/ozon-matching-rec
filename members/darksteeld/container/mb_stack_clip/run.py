"""E-CUP 2026 matching: стек RuModernBERT + student, с обрезкой уверенности.

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

T0 = time.time()
TIME_BUDGET = 17 * 60  # запас 3 минуты внутри 20-минутного лимита проверки

# Обрезка уверенности по явному требованию. Замер на честном OOF: она СТОИТ
# -0.000968 макро. Причина в том, что PR-AUC читает только порядок, а обрезка
# схлопывает 17.4 тыс. пар в связку на нуле и 17.3 тыс. на единице — внутри
# связки порядка не остаётся, а верхушка ранжирования весит в метрике больше
# всего. Порога, при котором обрезка помогает, не существует; 0.01/0.99 даёт
# +0.00002, то есть ноль в пределах шума. Оставлено осознанно.
CLIP_LOW, CLIP_HIGH = 0.05, 0.95
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


def score(spec, left, right, device):
    tokenizer = AutoTokenizer.from_pretrained(spec["path"])
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
            if done > 20000 and max_len > 144:
                rate = done / (time.time() - started)
                if elapsed() + (total - done) / rate > TIME_BUDGET * 0.95:
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
    plain_left = plain_right = None
    if any(spec.get("texts", "prio") == "plain" for spec in specs):
        plain_left, plain_right = build_plain_texts(matches, id2name, id2cat, id2attr)
    print(f"{len(prio_left)} пар, тексты собраны, прошло {elapsed():.0f}с", flush=True)

    ranks, weights = [], []
    model_cost = None
    for number, spec in enumerate(specs):
        # Защита по времени: если по стоимости предыдущей модели видно, что
        # следующая выведет за бюджет, она пропускается. Лучше отдать бленд из
        # трёх моделей, чем не отдать ничего. Первая считается всегда.
        if number > 0 and model_cost is not None \
                and elapsed() + model_cost * spec.get("cost", 1.0) * 1.1 > TIME_BUDGET:
            print(f"защита по времени: пропускаю {spec['path']} "
                  f"(прошло {elapsed():.0f}с)", flush=True)
            continue
        started_model = time.time()
        plain = spec.get("texts", "prio") == "plain"
        predicted = score(spec, plain_left if plain else prio_left,
                          plain_right if plain else prio_right, device)
        # Стоимость нормируем на заявленную относительную: следующая модель может
        # быть дороже или дешевле, и сравнивать надо приведённые величины.
        model_cost = (time.time() - started_model) / spec.get("cost", 1.0)
        # Ранги, а не сырые вероятности: штраф ниже умножает на 0.25, и на
        # лидерборде +0.00233 измерен именно на этой комбинации.
        rank = pd.Series(predicted).rank().to_numpy() / len(predicted)
        ranks.append(rank * spec.get("weight", 1.0))
        weights.append(spec.get("weight", 1.0))
        print(f"{spec['path']} готов, прошло {elapsed():.0f}с", flush=True)

    final = np.asarray(np.sum(ranks, axis=0) / np.sum(weights), dtype=np.float64)
    mask = np.array(penalty)
    if mask.any():
        final[mask] *= 0.25
        print(f"штраф за размер применён к {int(mask.sum())} фэшн-парам", flush=True)

    low, high = final < CLIP_LOW, final > CLIP_HIGH
    final[low] = 0.0
    final[high] = 1.0
    print(f"обрезка уверенности: {int(low.sum())} пар -> 0, "
          f"{int(high.sum())} пар -> 1", flush=True)

    pd.DataFrame({"id1": matches["id1"], "id2": matches["id2"],
                  "predict": final}).to_csv(args.output_path, index=False)
    print(f"записан {args.output_path}, всего {elapsed():.0f}с", flush=True)


if __name__ == "__main__":
    main()

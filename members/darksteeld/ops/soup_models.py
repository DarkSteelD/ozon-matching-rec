"""Усреднение весов моделей, обученных по фолдам («model soup»).

Почему это вообще законно. Усреднять веса двух независимо обученных сетей
нельзя: у них перепутаны нейроны, и среднее двух разных решений — не решение.
Здесь случай другой: все фолдовые модели дообучались из ОДНОГО чекпоинта
претрена, включая голову классификатора, и ушли от него недалеко. В такой
постановке они лежат в одной чаше функции потерь, и среднее весов остаётся
осмысленной моделью. Скрипт печатает, насколько далеко модели разошлись, —
если расхождение окажется большим, к результату надо относиться с недоверием.

Чего этим измерить НЕЛЬЗЯ. Каждая фолдовая модель не видела своего фолда, но
суп видел все четыре: любая оценка супа на наших размеченных парах будет
внутривыборочной и несравнимой с OOF-числом 0.863304. Ровно на этом уже
обжигались: у выложенной rubase внутривыборочные 0.8858 против честных 0.851713.

    python soup_models.py --models runs/mb_adan_1e-4/model_fold_0{1,2,3,4} \
        --out runs/mb_soup
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="сохранять в fp16 (на инференсе разницы нет)")
    args = parser.parse_args()

    states = []
    for path in args.models:
        weights = path / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} не найден")
        states.append(load_file(str(weights)))
    print(f"моделей: {len(states)}")

    keys = set(states[0])
    for i, state in enumerate(states[1:], 1):
        if set(state) != keys:
            raise ValueError(f"модель {args.models[i]} имеет другой набор тензоров")

    souped: dict[str, torch.Tensor] = {}
    drift = []
    for key in sorted(keys):
        tensors = [state[key] for state in states]
        if not tensors[0].is_floating_point():
            # Целочисленные буферы усреднять бессмысленно, берём первый.
            souped[key] = tensors[0].clone()
            continue
        stacked = torch.stack([t.float() for t in tensors])
        mean = stacked.mean(0)
        # Расхождение: насколько модели разошлись относительно масштаба весов.
        spread = float(stacked.std(0).mean())
        scale = float(mean.abs().mean()) + 1e-12
        drift.append((spread / scale, key, spread))
        souped[key] = mean.half() if args.fp16 else mean

    drift.sort(reverse=True)
    print("\nрасхождение между моделями (отклонение / масштаб веса):")
    for ratio, key, spread in drift[:6]:
        print(f"  {ratio:7.3f}  {key}  (разброс {spread:.2e})")
    median = drift[len(drift) // 2][0]
    print(f"\nмедианное относительное расхождение: {median:.4f}")
    if median > 0.5:
        print("ВНИМАНИЕ: модели разошлись сильно, суп может быть хуже отдельных моделей")
    else:
        print("модели близки друг к другу — усреднение весов обосновано")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(souped, str(args.out / "model.safetensors"),
              metadata={"format": "pt"})
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.txt"):
        source = args.models[0] / name
        if source.is_file():
            shutil.copy2(source, args.out / name)
    print(f"\nсуп сохранён в {args.out}")


if __name__ == "__main__":
    main()

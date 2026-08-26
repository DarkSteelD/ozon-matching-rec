"""Дообуч кросс-энкодера на готовых prio-текстах с мягкими целями.

Отличается от ``members/dzkhomidov/src/train_hand_fast.py`` только входом: тексты
и цели уже собраны ``build_distill_pairs.py``, поэтому здесь нет ни склейки
атрибутов, ни обращения к каталогу — только парквет с колонками
``fold, id1, id2, target, soft_target, text1, text2, source``.

Два режима, как у Далера:

    --stage folds : для каждого фолда обучиться на трёх остальных и предсказать
                    его, получив честный OOF. Так меряется качество.
    --stage final : обучиться на всех парах и сохранить модель для контейнера.
                    Так делается посылка; локально её измерить нельзя.

Потери — BCE по мягкой цели. Жёсткая метка используется только для отчёта.

    python train_distill.py --data hand_pairs_distill_aug.parquet \
        --init <чекпоинт после LLM-претрена> --stage folds --max-len 224
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from pair_budget import fit_pair
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return float("nan")
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


class Pairs(Dataset):
    def __init__(self, frame: pl.DataFrame):
        self.t1 = frame["text1"].to_list()
        self.t2 = frame["text2"].to_list()
        self.y = frame["soft_target"].to_numpy().astype("float32")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.t1[i], self.t2[i], self.y[i]


class LengthBucketBatches(torch.utils.data.Sampler):
    """Батчи из пар близкой длины: борьба с добивкой пустыми токенами.

    Батч добивается до самой длинной пары в нём, поэтому при случайном
    перемешивании один длинный экземпляр растягивает весь батч. Замер на 40
    тысячах пар: при батче 16 обрабатывается в 1.82 раза больше токенов, чем
    полезных, то есть 45% вычислений уходит в пустоту. Группировка по длине
    снижает этот множитель до 1.11.

    Полная сортировка по длине убила бы случайность и связала бы состав батча
    с длиной. Поэтому сортируем не весь корпус, а окно из window батчей: внутри
    окна длины сближаются, но само окно набирается случайно, и порядок готовых
    батчей потом ещё раз перемешивается.

    Длина считается в ТОКЕНАХ, а не в символах. Символьный прокси проще, но
    добивка идёт по токенам, и на нём множитель выходит 1.25 вместо 1.11 —
    треть выигрыша теряется. Разовая токенизация фолда стоит около полутора
    минут против полутора часов обучения, так что размен очевиден.
    """

    def __init__(self, lengths, batch_size, window=50, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.window = window
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.lengths))
        span = self.batch_size * self.window
        batches = []
        for start in range(0, len(order), span):
            chunk = order[start:start + span]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for at in range(0, len(chunk), self.batch_size):
                batch = chunk[at:at + self.batch_size]
                if len(batch) == self.batch_size:      # drop_last
                    batches.append(batch.tolist())
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.lengths) // self.batch_size


def collate(batch, tokenizer, max_len, pad_multiple=0, budget_attrs=False,
            swap=False):
    t1, t2, y = zip(*batch)
    if swap:
        # «А дубль Б» и «Б дубль А» — одно утверждение, но модель видит порядок
        # фиксированным: у ModernBERT нет сегментных эмбеддингов, и половины
        # различаются только положением относительно [SEP]. Случайная
        # перестановка учит инвариантности вместо запоминания порядка.
        # torch.rand берётся из RNG воркера, который DataLoader сеет по-разному.
        flip = torch.rand(len(t1)) < 0.5
        t1, t2 = (tuple(b if f else a for a, b, f in zip(t1, t2, flip)),
                  tuple(a if f else b for a, b, f in zip(t1, t2, flip)))
    if budget_attrs:
        # Режем хвосты атрибутов, а не конец пары: иначе первым гибнет блок
        # сравнения, стоящий в конце prio-текста. Подробности в pair_budget.
        fitted = [fit_pair(a, b, tokenizer, max_len) for a, b in zip(t1, t2)]
        t1 = [f[0] for f in fitted]
        t2 = [f[1] for f in fitted]
    encoded = tokenizer(list(t1), list(t2), truncation=True, max_length=max_len,
                        padding=True, pad_to_multiple_of=pad_multiple or None,
                        return_tensors="pt")
    return encoded, torch.tensor(y, dtype=torch.float32)


def make_optimizer(model, args):
    """AdamW по рецепту Далера либо Adan.

    Скорость обучения намеренно одна и та же: в сравнении меняется ровно один
    фактор — сам оптимизатор. Оговорка в том, что Adan на дообуче нередко любит
    lr в полтора-два раза выше, поэтому если эта рука проиграет, первый
    подозреваемый — не оптимизатор, а неподобранный шаг.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "nadam":
        # Adam с поправкой Нестерова: шаг делается по уже экстраполированному
        # моменту. decoupled_weight_decay=True приводит регуляризацию к тому же
        # виду, что у AdamW, иначе распад веса подмешивался бы в градиент и
        # руки различались бы не только правилом обновления.
        return torch.optim.NAdam(trainable, lr=args.lr, weight_decay=0.01,
                                 decoupled_weight_decay=True)
    if args.optimizer == "adan":
        try:
            from adan import Adan
            # max_grad_norm=0: обрезка градиента делается снаружи, как и для AdamW,
            # иначе руки различались бы ещё и способом клиппинга.
            return Adan(trainable, lr=args.lr, betas=(0.98, 0.92, 0.99),
                        weight_decay=0.02, max_grad_norm=0.0, foreach=True)
        except ImportError:
            from adan_pytorch import Adan
            # ВНИМАНИЕ: у этой реализации betas заданы как (1 - beta) из статьи.
            # Её умолчание (0.02, 0.08, 0.01) — это те же самые (0.98, 0.92, 0.99),
            # что у sail-sg выше. Числа выглядят непохоже, но означают одно и то
            # же; менять их «на правильные» нельзя, оптимизатор станет другим.
            return Adan(trainable, lr=args.lr, betas=(0.02, 0.08, 0.01),
                        weight_decay=0.02)
    return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)


def fit(model, tokenizer, frame, args, device):
    dataset = Pairs(frame)
    common = dict(num_workers=args.workers, pin_memory=True,
                  collate_fn=lambda b: collate(b, tokenizer, args.max_len,
                                               args.pad_multiple, args.budget_attrs,
                                               args.swap_augment))
    if args.length_buckets:
        started_lengths = time.time()
        lengths = []
        for at in range(0, len(dataset.t1), 4000):
            lengths.extend(len(x) for x in tokenizer(
                dataset.t1[at:at + 4000], dataset.t2[at:at + 4000],
                truncation=True, max_length=args.max_len)["input_ids"])
        print(f"  длины посчитаны за {time.time() - started_lengths:.0f} с; "
              f"медиана {int(np.median(lengths))} токенов", flush=True)
        loader = DataLoader(dataset, batch_sampler=LengthBucketBatches(
            lengths, args.bs, seed=args.seed), **common)
    else:
        loader = DataLoader(dataset, batch_size=args.bs, shuffle=True,
                            drop_last=True, **common)
    steps = len(loader) * args.epochs
    optimizer = make_optimizer(model, args)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear")
    # Масштабирование потерь придумано против исчезновения градиентов в fp16, у
    # которого узкий диапазон экспоненты. Мы считаем в bf16, а у него диапазон
    # как у fp32 — исчезать нечему, и скейлер становится чистыми накладными
    # расходами: лишний проход по всем градиентам в unscale_ (150M параметров,
    # это около 1.2 ГБ трафика на шаг) плюс синхронизация GPU с CPU на каждом
    # шаге ради проверки на inf. Отключён. Объект оставлен: при enabled=False
    # все его методы — сквозные, и структура цикла не меняется.
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # Компилируем обёртку, а не подменяем модель: параметры у них общие, поэтому
    # оптимизатор, обрезка градиента, предсказание и сохранение продолжают
    # работать с исходным модулем и ничего не знают про inductor. Предсказание
    # намеренно идёт по НЕскомпилированной модели — там формы батчей гуляют, и
    # статичная компиляция пересобиралась бы на каждой новой длине.
    trained = (torch.compile(model, dynamic=not args.pad_multiple)
               if args.compile else model)
    if args.compile:
        print(f"  модель скомпилирована (dynamic={not args.pad_multiple})", flush=True)
    model.train()
    seen, started = 0, time.time()
    for epoch in range(args.epochs):
        for encoded, y in loader:
            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = trained(**encoded).logits
                logits = logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2 else logits[:, 0]
                loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); schedule.step()
            seen += len(y)
            if seen % (args.bs * 200) < args.bs:
                rate = seen / (time.time() - started)
                print(f"  эпоха {epoch+1}/{args.epochs} {seen:,}/{steps*args.bs:,} "
                      f"loss {loss.item():.4f} {rate:.0f} пар/с", flush=True)
    return model


@torch.inference_mode()
def predict(model, tokenizer, frame, args, device) -> np.ndarray:
    model.eval()
    t1, t2 = frame["text1"].to_list(), frame["text2"].to_list()
    order = np.argsort([len(a) + len(b) for a, b in zip(t1, t2)], kind="stable")
    out = np.zeros(len(t1), dtype=np.float64)
    for start in range(0, len(order), args.eval_bs):
        pick = order[start:start + args.eval_bs]
        encoded = tokenizer([t1[i] for i in pick], [t2[i] for i in pick],
                            truncation=True, max_length=args.max_len,
                            padding=True, return_tensors="pt").to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(**encoded).logits
        logits = logits[:, 1] - logits[:, 0] if logits.shape[-1] == 2 else logits[:, 0]
        out[pick] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def build(args, device, tokenizer):
    """Модель под задачу. Отдельно обслуживаются декодеры вроде Qwen3.5.

    У них три отличия от bert-подобных: нужен pad-токен (иначе класс-голова не
    найдёт последний токен последовательности), веса лучше держать в bfloat16, а
    у мультимодальных вариантов есть башня зрения, которая для пар текстов
    мёртвый груз — её замораживаем, чтобы не тратить на неё состояния AdamW.
    """
    kwargs = {"num_labels": args.num_labels}
    if args.bf16_weights:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        args.init or args.model, **kwargs)
    # У мультимодальных конфигов (Qwen3.5) верхний уровень — обёртка, и
    # pad_token_id живёт в text_config; на верхнем уровне атрибута нет вовсе,
    # поэтому обращение к нему падает с AttributeError, а не возвращает None.
    # Класс-голова ищет последний непустой токен именно по pad_token_id, так что
    # проставляем его везде, где он мог бы читаться.
    for cfg in {id(model.config): model.config,
                id(getattr(model.config, "text_config", model.config)):
                    getattr(model.config, "text_config", model.config)}.values():
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tokenizer.pad_token_id
    frozen = 0
    if args.freeze_vision:
        for name, param in model.named_parameters():
            if "vis" in name.lower() or "image" in name.lower():
                param.requires_grad_(False)
                frozen += param.numel()
        if frozen:
            print(f"  заморожена башня зрения: {frozen/1e6:.0f}M параметров", flush=True)
    model = model.to(device)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--init", default=None, help="чекпоинт после LLM-претрена")
    parser.add_argument("--stage", choices=["folds", "final"], default="folds")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--eval-bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=224)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--optimizer", choices=["adamw", "adan", "nadam"], default="adamw")
    parser.add_argument("--swap-augment", action="store_true",
                        help="случайно менять товары местами: задача симметрична, "
                             "а модель видит порядок фиксированным")
    parser.add_argument("--length-buckets", action="store_true",
                        help="собирать батчи из пар близкой длины: убирает "
                             "45%% вычислений, уходящих в добивку пустыми токенами")
    parser.add_argument("--budget-attrs", action="store_true",
                        help="обрезать хвосты атрибутов вместо конца пары: "
                             "блок сравнения выживает в 100%% случаев против 7.5%%")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile обучающего прохода: на RuModernBERT +59%%")
    parser.add_argument("--pad-multiple", type=int, default=0,
                        help="дополнять длину батча до кратной N. Делает формы "
                             "статичными, из-за чего компиляция даёт лучшие ядра")
    parser.add_argument("--save-models", action="store_true",
                        help="сохранять веса каждого фолда, а не только предсказания")
    parser.add_argument("--folds", default="", help="список фолдов через запятую; "
                        "пусто — все. Для свипа гиперпараметров хватает одного")
    parser.add_argument("--bf16-weights", action="store_true",
                        help="держать веса в bfloat16 (нужно крупным моделям)")
    parser.add_argument("--freeze-vision", action="store_true",
                        help="заморозить башню зрения у мультимодальных моделей")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--drop-closure", action="store_true",
                        help="выбросить выведенные пары из обучения (базовая рука)")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pl.read_parquet(args.data)
    if args.drop_closure and "source" in frame.columns:
        before = frame.height
        frame = frame.filter(pl.col("source") == "hand")
        print(f"выведенные пары исключены: {before - frame.height:,}")
    tokenizer = AutoTokenizer.from_pretrained(args.init or args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"устройство {device}; строк {frame.height:,}; оптимизатор {args.optimizer}; "
          f"старт из {args.init or args.model}", flush=True)

    if args.stage == "final":
        model = fit(build(args, device, tokenizer), tokenizer, frame, args, device)
        model.half().save_pretrained(args.out / "model")
        tokenizer.save_pretrained(args.out / "model")
        print(f"сохранено в {args.out / 'model'}")
        return

    # Проверяем ТОЛЬКО на ручных парах: выведенные в оценку не идут, иначе скор
    # перестанет сравниваться с 0.851713 у ce_priodistill.
    wanted = [f.strip() for f in args.folds.split(",") if f.strip()]
    scores = []
    for fold in sorted(set(frame["fold"].to_list())):
        if wanted and fold not in wanted:
            continue
        train = frame.filter(pl.col("fold") != fold)
        held = frame.filter((pl.col("fold") == fold) & (pl.col("source") == "hand"))
        print(f"\n{fold}: обучение {train.height:,}, проверка {held.height:,}", flush=True)
        model = fit(build(args, device, tokenizer), tokenizer, train, args, device)
        predicted = predict(model, tokenizer, held, args, device)
        score = average_precision(held["target"].to_numpy().astype(float), predicted)
        scores.append(score)
        print(f"{fold}: PR-AUC {score:.6f}", flush=True)
        pl.DataFrame({"id1": held["id1"], "id2": held["id2"],
                      "predict": predicted}).write_csv(args.out / f"{fold}.csv")
        if args.save_models:
            # fp16: веса фолдовых моделей нужны для ансамбля и для проверки, а
            # держать их в fp32 незачем — на инференсе разница не видна.
            where = args.out / f"model_{fold}"
            model.half().save_pretrained(where)
            tokenizer.save_pretrained(where)
            print(f"{fold}: модель сохранена в {where}", flush=True)
        del model; torch.cuda.empty_cache()
    print(f"\nmean PR-AUC {np.mean(scores):.6f} по {len(scores)} фолдам")
    print("контроль ce_priodistill у dzkhomidov: 0.851713")


if __name__ == "__main__":
    main()

"""LLM-претрен кросс-энкодера на 11.19M мягких пар — шаг, дающий +0.038 у Далера.

Их ``train_ce.py --stage llm`` делает то же самое, но читает подготовленный
``llm_pairs_<N>.parquet`` и заранее пре-токенизированный memmap. Здесь вход
компактнее: справочник ``llm_texts.parquet`` (id -> готовая строка) плюс сами
пары. Тексты держатся стрелочным массивом и токенизируются на лету воркерами
загрузчика — на GPU-боксе с 16+ ядрами токенизация не является узким местом, а
7-гигабайтного кэша идентификаторов на диске не возникает.

Цель мягкая: у ``matches_llm.parquet`` колонка target — это вероятность от LLM,
а не 0/1, и BCE берётся по ней напрямую. В этом весь смысл шага: модель учится
у разметчика распределению, а не жёстким меткам.

Чекпоинт пишется каждые --save-every шагов и при старте подхватывается, чтобы
падение или вытеснение инстанса не стоили всего прогона.

    python pretrain_llm.py --texts llm_texts.parquet --pairs matches_llm.parquet \
        --out ckpt/rubase_llm --epochs 2 --max-len 160 --bs 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LlmPairs(Dataset):
    """Пары из matches_llm; текст берётся по индексу из общего стрелочного массива."""

    def __init__(self, texts, sorted_ids, order, id1, id2, target):
        # Поиск по отсортированному numpy-массиву, а не по словарю: словарь на
        # 12.38M записей это ~1.2 ГБ, и при форке воркеров copy-on-write на нём
        # ломается о пересчёт ссылок — каждый воркер получил бы свою копию.
        # Массивы numpy и буферы arrow при форке действительно разделяются.
        self.texts = texts
        self.sorted_ids, self.order = sorted_ids, order
        self.id1, self.id2, self.target = id1, id2, target

    def _row(self, item: int) -> int:
        return int(self.order[np.searchsorted(self.sorted_ids, item)])

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i):
        a = self.texts[self._row(self.id1[i])].as_py() or ""
        b = self.texts[self._row(self.id2[i])].as_py() or ""
        return a, b, self.target[i]


def collate(batch, tokenizer, max_len, pad_multiple=0):
    t1, t2, y = zip(*batch)
    encoded = tokenizer(list(t1), list(t2), truncation=True, max_length=max_len,
                        padding=True,
                        pad_to_multiple_of=pad_multiple or None,
                        return_tensors="pt")
    return encoded, torch.tensor(y, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=2000, help="шагов между чекпоинтами")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--limit", type=int, default=0,
                        help="взять только N пар. Для крупных моделей это не отладка, "
                             "а способ уложить претрен в бюджет времени: полный корпус "
                             "им не по силам на одной карте")
    parser.add_argument("--bf16-weights", action="store_true")
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--optimizer", choices=["adamw", "nadam"], default="adamw")
    parser.add_argument("--compile", action="store_true", help="torch.compile модели")
    parser.add_argument("--pad-multiple", type=int, default=0,
                        help="дополнять длину батча до кратной N (тензорные ядра "
                             "любят кратность 8/64; заодно уменьшает число форм)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("нужен CUDA: на CPU этот шаг занимает недели")

    table = pq.read_table(args.texts, columns=["id", "text"])
    ids = table.column("id").to_numpy()
    texts = table.column("text").combine_chunks()
    order = np.argsort(ids, kind="stable").astype("int32")
    sorted_ids = ids[order]
    del ids
    print(f"справочник текстов: {len(sorted_ids):,} товаров", flush=True)

    pairs = pq.read_table(args.pairs, columns=["id1", "id2", "target"])
    id1 = pairs.column("id1").to_numpy()
    id2 = pairs.column("id2").to_numpy()
    target = pairs.column("target").to_numpy().astype("float32")
    if args.limit:
        id1, id2, target = id1[:args.limit], id2[:args.limit], target[:args.limit]
    print(f"пар: {len(target):,}; мягкая цель: медиана {np.median(target):.3f}, "
          f"доля >0.5 {float((target > 0.5).mean()):.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    state_file = args.out / "state.json"
    start_epoch, start_step = 0, 0
    if state_file.is_file() and (args.out / "config.json").is_file():
        state = json.loads(state_file.read_text())
        start_epoch, start_step = state["epoch"], state["step"]
        source = str(args.out)
        print(f"подхватываю чекпоинт: эпоха {start_epoch}, шаг {start_step}", flush=True)
    else:
        source = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # sdpa вместо eager-внимания: та же математика, но матрица внимания не
    # материализуется целиком — на bs 256 x len 160 это разница между 19 ГБ и ~10 ГБ
    kwargs = {"num_labels": 2, "attn_implementation": "sdpa"}
    if args.bf16_weights:
        kwargs["dtype"] = torch.bfloat16
    try:
        model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
    except (ValueError, KeyError):
        kwargs.pop("attn_implementation")
        model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        print("sdpa недоступно, работаем на eager", flush=True)
    # У мультимодальных конфигов (Qwen3.5) pad_token_id живёт в text_config, а на
    # верхнем уровне атрибута нет вовсе — обращение к нему падает, а не даёт None.
    for cfg in {id(model.config): model.config,
                id(getattr(model.config, "text_config", model.config)):
                    getattr(model.config, "text_config", model.config)}.values():
        if getattr(cfg, "pad_token_id", None) is None:
            cfg.pad_token_id = tokenizer.pad_token_id
    if args.freeze_vision:
        frozen = 0
        for name, param in model.named_parameters():
            if "vis" in name.lower() or "image" in name.lower():
                param.requires_grad_(False)
                frozen += param.numel()
        if frozen:
            print(f"заморожена башня зрения: {frozen/1e6:.0f}M параметров", flush=True)
    model = model.to(device)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if args.compile:
        # Динамические формы дешевле компилировать, но inductor выдаёт под них
        # заведомо более медленные ядра. Если длина батча кратна --pad-multiple,
        # различных форм остаётся max_len/pad_multiple штук — их дешевле
        # скомпилировать по отдельности и получить статичные ядра.
        dynamic = not args.pad_multiple
        model = torch.compile(model, dynamic=dynamic)
        print(f"модель скомпилирована (dynamic={dynamic}); первые шаги медленные",
              flush=True)
    loader = DataLoader(LlmPairs(texts, sorted_ids, order, id1, id2, target),
                        batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=True, persistent_workers=True,
                        prefetch_factor=4,
                        collate_fn=lambda b: collate(b, tokenizer, args.max_len,
                                                     args.pad_multiple))
    total = len(loader) * args.epochs
    trainable = [p for p in model.parameters() if p.requires_grad]
    # fused=True собирает обновление всех тензоров в одно ядро CUDA. У модели
    # больше двухсот отдельных параметров, и на каждый обычный AdamW запускает
    # свои ядра — при коротком шаге этот запуск стоит заметной доли времени.
    optimizer = (torch.optim.NAdam(trainable, lr=args.lr, weight_decay=0.01,
                                   decoupled_weight_decay=True, foreach=True)
                 if args.optimizer == "nadam"
                 else torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01,
                                        fused=(device.type == "cuda")))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total, pct_start=0.06,
        anneal_strategy="linear")
    for _ in range(start_epoch * len(loader) + start_step):
        schedule.step()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    step = start_step
    started, seen = time.time(), 0
    for epoch in range(start_epoch, args.epochs):
        for batch, (encoded, y) in enumerate(loader):
            if epoch == start_epoch and batch < start_step:
                continue
            encoded = {k: v.to(device, non_blocking=True) for k, v in encoded.items()}
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**encoded).logits
                loss = loss_fn((logits[:, 1] - logits[:, 0]).float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); schedule.step()
            step += 1; seen += len(y)
            if step % 200 == 0:
                rate = seen / (time.time() - started)
                left = (total - (epoch * len(loader) + batch)) * args.bs / max(rate, 1) / 3600
                print(f"эпоха {epoch+1}/{args.epochs} шаг {batch:,}/{len(loader):,} "
                      f"loss {loss.item():.4f} {rate:.0f} пар/с осталось {left:.1f} ч",
                      flush=True)
            if step % args.save_every == 0:
                model.save_pretrained(args.out); tokenizer.save_pretrained(args.out)
                state_file.write_text(json.dumps({"epoch": epoch, "step": batch + 1}))
        start_step = 0
    model.save_pretrained(args.out); tokenizer.save_pretrained(args.out)
    state_file.write_text(json.dumps({"epoch": args.epochs, "step": 0, "done": True}))
    print(f"готово, чекпоинт в {args.out}")


if __name__ == "__main__":
    main()

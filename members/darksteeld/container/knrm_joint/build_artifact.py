"""Обучить отгружаемую совместную KNRM и выгрузить её для контейнера.

Две стадии, как у остальных сетей репозитория:

1. предобучение на ``matches_llm`` — берётся готовым из чекпоинта
   ``experiments/knrm_joint_llm/train.py`` (3 часа, переобучать незачем);
2. дообучение на **всех** 365 654 ручных парах с ранней остановкой по срезу,
   вырезанному из них по компонентам связности, и выгрузка.

OOF-модели фолдов для отгрузки не годятся: каждая видела три четверти ручных
пар. Контейнеру нужна одна модель, видевшая всё.

**Что выгружается.** Таблица эмбеддингов в float16 (943 580 x 300 — 566 МБ
вместо 1.13 ГБ; косинус от половинной точности не страдает, вход всё равно
нормируется) плюс словарь токен -> строка таблицы и веса гейтов и головы.
Индексное пространство сохраняется тем же, что при обучении, поэтому контейнер
ничего не переиндексирует: токен теста ищется в словаре напрямую.

**Токен вне словаря становится PAD**, то есть маскируется и в суммы не входит.
Это ровно то, что происходило при обучении (``encode_stream`` в train.py
отображает незнакомые токены в PAD, а не в ``<UNK>``), поэтому инференс и
обучение видят одну и ту же величину. Строка ``<UNK>`` в таблице есть, но она
осталась необученной и намеренно не используется.

    .venv/bin/python members/darksteeld/container/knrm_joint/build_artifact.py \\
        --checkpoint <scratch>/joint_pretrained_4m.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "experiments"
                       / "knrm_joint_llm"))

from finetune import average_precision, encode_hand, load_folds, predict  # noqa: E402
from knrm_joint_batching import attribute_counts, bucketed_batches, make_batch  # noqa: E402
from knrm_joint_model import KNRMConfig, ProductMatcher  # noqa: E402

VALIDATION_BUCKETS = 10
FOLDS = [f"fold_{k:02d}" for k in range(1, 5)]


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--log-every", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    started = time.time()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    token_id, shapes = blob["token_id"], blob["max_shapes"]
    config = KNRMConfig(**blob["config"])
    log(f"предобучение: {blob['pretrain_pairs']:,} пар, словарь {len(token_id):,}")

    items = pl.read_parquet(args.data_dir / "items_human.parquet",
                            columns=["id", "name", "attributes"])
    titles, keys, values, unknown = encode_hand(items, token_id, shapes)
    log(f"ручная вселенная: {items.height:,} товаров, токенов вне словаря {unknown:,}")
    titles_t, keys_t, values_t = (torch.from_numpy(titles), torch.from_numpy(keys),
                                  torch.from_numpy(values))
    counts = attribute_counts(keys_t, values_t)
    row_of_id = {int(i): r for r, i in enumerate(items["id"].to_list())}
    del items

    # Все ручные пары: контейнеру нужна модель, видевшая всё.
    folds = load_folds(args.targets_dir, FOLDS)
    id1 = np.concatenate([folds[f]["id1"] for f in FOLDS])
    id2 = np.concatenate([folds[f]["id2"] for f in FOLDS])
    target = np.concatenate([folds[f]["target"] for f in FOLDS])
    rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int64)
    rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int64)
    order_key = np.maximum(counts[rows1], counts[rows2])
    log(f"обучающих пар {len(target):,}, позитивов {int(target.sum()):,}")

    from validation.build_folds import connected_component_keys

    # Срез для ранней остановки — по компонентам связности: товар не может
    # оказаться и в обучении, и в срезе, иначе остановка запаздывает.
    component_of_item = connected_component_keys(id1, id2)
    bucket: dict[int, int] = {}
    is_validation = np.zeros(len(id1), dtype=bool)
    for index, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        if key not in bucket:
            digest = hashlib.sha256(f"{args.seed}:ship:{key}".encode()).digest()
            bucket[key] = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
        is_validation[index] = bucket[key] == 0
    train_mask = ~is_validation
    log(f"обучение {int(train_mask.sum()):,}, срез {int(is_validation.sum()):,}")

    model = ProductMatcher(torch.zeros(len(token_id), config.embedding_dim), config)
    model.load_state_dict(blob["state_dict"])
    del blob

    head = [parameter for name, parameter in model.named_parameters()
            if not name.startswith("embedding.")]
    optimizers = [torch.optim.AdamW(head, lr=args.learning_rate)]
    embedding_lr = args.learning_rate * args.embedding_scale
    optimizers.append(
        torch.optim.SparseAdam(list(model.embedding.parameters()), lr=embedding_lr)
        if config.sparse_embedding else
        torch.optim.AdamW(list(model.embedding.parameters()), lr=embedding_lr))
    loss_function = nn.BCEWithLogitsLoss()

    train_rows1, train_rows2 = rows1[train_mask], rows2[train_mask]
    train_target, train_key = target[train_mask], order_key[train_mask]
    rng = np.random.default_rng(args.seed)
    best, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_started, running, batches, done = time.time(), 0.0, 0, 0
        for pick in bucketed_batches(train_key, args.batch_size, rng):
            item_a = make_batch(titles_t, keys_t, values_t, train_rows1[pick])
            item_b = make_batch(titles_t, keys_t, values_t, train_rows2[pick])
            loss = loss_function(model(item_a, item_b),
                                 torch.from_numpy(train_target[pick]))
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head, 5.0)
            for optimizer in optimizers:
                optimizer.step()
            running += float(loss.detach())
            batches += 1
            done += len(pick)
            if batches % args.log_every == 0:
                rate = done / max(time.time() - epoch_started, 1e-9)
                log(f"      эпоха {epoch}  {100 * done / len(train_target):4.1f}%  "
                    f"loss {running / batches:.5f}  {rate:.0f} пар/с")

        validation_score = average_precision(
            target[is_validation],
            predict(model, titles_t, keys_t, values_t, rows1[is_validation],
                    rows2[is_validation], args.batch_size, order_key[is_validation]))
        improved = validation_score > best
        log(f"    эпоха {epoch}: loss {running / max(batches, 1):.5f}, "
            f"срез PR-AUC {validation_score:.6f}{'  *' if improved else ''}, "
            f"{time.time() - epoch_started:.0f}s")
        if improved:
            best, best_epoch, waited = validation_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= args.patience:
                log(f"    ранняя остановка, возвращаю эпоху {best_epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- выгрузка ----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    embedding = state.pop("embedding.weight").to(torch.float16)
    torch.save({
        "embedding_fp16": embedding,
        "head_state": {k: v for k, v in state.items()},
        "token_id": token_id,
        "config": config.__dict__,
        "max_shapes": shapes,
    }, args.out_dir / "model.pt")

    size_gb = (args.out_dir / "model.pt").stat().st_size / 1e9
    artifact = {
        "experiment": "knrm_joint",
        "reads": "название и атрибуты; четыре канала взаимодействия",
        "vocabulary": len(token_id),
        "embedding_dtype": "float16",
        "unknown_token_policy": "PAD (маскируется) — как при обучении",
        "pairs_finetune": int(train_mask.sum()),
        "validation_slice": int(is_validation.sum()),
        "best_epoch": best_epoch,
        "validation_prauc": float(best),
        "max_shapes": shapes,
        "config": config.__dict__,
        "torch_version": torch.__version__,
        "local_cv": {
            "spec_v2_mean_prauc": 0.61915363,
            "spec_v2_zero_shot": 0.517825,
            "control_blend_name_attrs_50_50": 0.617125,
            "note": ("против оптимального бленда двух прежних KNRM выигрыш +0.0020 при "
                     "парной выборочной ошибке 0.0013 — в пределах шума; см. "
                     "reports/KNRM_JOINT.md"),
        },
        "timing_warning": ("инференс НЕ измерен на боевом объёме. Замер без "
                           "бакетирования дал 13.4 мин на 365,654 пары при лимите 20 мин "
                           "на весь прогон; с бакетированием ожидается меньше, но это "
                           "оценка, а не факт"),
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"\nартефакт -> {args.out_dir}/model.pt ({size_gb:.2f} ГБ), "
        f"лучшая эпоха {best_epoch} (срез {best:.6f}), всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

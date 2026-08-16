"""Дообучение совместной KNRM по замороженным фолдам и OOF-предсказания.

Схема та же, что у ``knrm_llm_pretrain`` и ``knrm_attrs_llm``, чтобы число было
сравнимо с ними по построению, а не «примерно»: для фолда K модель стартует с
предобученного состояния, дообучается на остальных фолдах ИЗ СПИСКА ``--folds``,
ранняя остановка patience 1 по срезу, вырезанному из обучающих пар **по
компонентам связности** (товар не может оказаться и в обучении, и в срезе), и
предсказывает фолд K.

Передав три фолда вместо четырёх, получаем модели, не видевшие четвёртый, —
это вложенный OOF для моделей второго уровня.

    .venv/bin/python members/darksteeld/experiments/knrm_joint_llm/finetune.py \\
        --checkpoint <scratch>/joint_pretrained_4m.pt \\
        --out-dir validation/predictions_v2/darksteeld/knrm_joint
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_joint_batching import attribute_counts, bucketed_batches, make_batch  # noqa: E402
from knrm_joint_tokens import parse_attributes, tokenize  # noqa: E402
from knrm_joint_model import PAD_ID, KNRMConfig, ProductMatcher  # noqa: E402

VALIDATION_BUCKETS = 10


def log(message: str) -> None:
    print(message, flush=True)


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


def load_folds(targets_dir: Path, fold_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    folds = {}
    for fold_id in fold_ids:
        path = targets_dir / f"{fold_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"нет {path}")
        id1, id2, target = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                id1.append(int(row["id1"]))
                id2.append(int(row["id2"]))
                target.append(float(row["target"]))
        folds[fold_id] = {"id1": np.asarray(id1, dtype=np.int64),
                          "id2": np.asarray(id2, dtype=np.int64),
                          "target": np.asarray(target, dtype=np.float32)}
    return folds


def encode_hand(items: pl.DataFrame, token_id: dict[str, int], shapes: dict[str, int],
                stemming: bool = False):
    """Ручная вселенная в те же формы, что использовались при предобучении.

    ``stemming`` берётся из чекпоинта, а не из аргументов: токенизация обязана
    совпадать с той, на которой строился словарь."""
    count = items.height
    titles = np.zeros((count, shapes["title"]), dtype=np.int32)
    keys = np.zeros((count, shapes["attrs"], shapes["key_tokens"]), dtype=np.int32)
    values = np.zeros((count, shapes["attrs"], shapes["value_tokens"]), dtype=np.int32)
    unknown = 0
    for row, (name, raw) in enumerate(zip(items["name"].to_list(),
                                          items["attributes"].to_list())):
        for column, token in enumerate(tokenize(name, stemming)[:shapes["title"]]):
            index = token_id.get(token, PAD_ID)
            titles[row, column] = index
            unknown += index == PAD_ID
        for slot, (key_tokens, value_tokens) in enumerate(
                parse_attributes(raw, stemming)[:shapes["attrs"]]):
            for column, token in enumerate(key_tokens[:shapes["key_tokens"]]):
                keys[row, slot, column] = token_id.get(token, PAD_ID)
            for column, token in enumerate(value_tokens[:shapes["value_tokens"]]):
                values[row, slot, column] = token_id.get(token, PAD_ID)
    return titles, keys, values, unknown


@torch.no_grad()
def predict(model: ProductMatcher, titles, keys, values, rows1, rows2,
            batch_size: int, order_key: np.ndarray, device=None) -> np.ndarray:
    """Предсказания в исходном порядке пар. Батчи бакетируем ради скорости и
    возвращаем на место — порядок строк файла предсказаний контрактный.

    Устройство берётся у самой модели, а параметр ``device`` оставлен только для
    совместимости вызовов и игнорируется. Передавать его руками оказалось
    ошибкоопасно: из четырёх вызовов один забыли, и прогон падал на
    несовпадении устройств уже после первой эпохи, то есть спустя минуты работы.
    """
    model.eval()
    device = next(model.parameters()).device
    scores = np.empty(len(rows1), dtype=np.float64)
    order = np.argsort(order_key, kind="mergesort")
    for start in range(0, len(order), batch_size):
        pick = order[start:start + batch_size]
        item_a = make_batch(titles, keys, values, rows1[pick], device=device)
        item_b = make_batch(titles, keys, values, rows2[pick], device=device)
        scores[pick] = torch.sigmoid(model(item_a, item_b)).cpu().numpy()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dump-all-folds", type=Path,
                        help="in-sample предсказания на обучающих фолдах для стекинга")
    parser.add_argument("--save-weights", type=Path,
                        help="каталог для весов каждого фолда; БЕЗ него часы обучения\n                             оставляют только CSV, а модели теряются")
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04",
                        help="каждый по очереди held-out, обучение на ОСТАЛЬНЫХ ИЗ СПИСКА")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-scale", type=float, default=0.1)
    parser.add_argument("--device", default="cpu",
                        help="cpu или cuda; на cuda таблица обновляется плотным\n                             AdamW — SparseAdam там медленнее и капризнее")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = без ограничения; отладочный лимит батчей на эпоху")
    args = parser.parse_args()

    fold_ids = [name.strip() for name in args.folds.split(",") if name.strip()]
    if len(fold_ids) < 2:
        raise SystemExit("нужно хотя бы два фолда: один held-out, один обучающий")
    torch.manual_seed(args.seed)
    started = time.time()

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    token_id = blob["token_id"]
    shapes = blob["max_shapes"]
    config = KNRMConfig(**blob["config"])
    log(f"чекпоинт {args.checkpoint.name}: словарь {len(token_id):,}, "
        f"пар предобучения {blob['pretrain_pairs']:,}, формы {shapes}")

    items = pl.read_parquet(args.data_dir / "items_human.parquet",
                            columns=["id", "name", "attributes"])
    stemming = bool(blob.get("stemming", False))
    log(f"токенизация: {'стеммы' if stemming else 'сырые токены'}")
    titles, keys, values, unknown = encode_hand(items, token_id, shapes, stemming)
    log(f"ручная вселенная: {items.height:,} товаров, токенов названия вне словаря "
        f"{unknown:,}")
    titles_t, keys_t, values_t = (torch.from_numpy(titles), torch.from_numpy(keys),
                                  torch.from_numpy(values))
    counts = attribute_counts(keys_t, values_t)
    row_of_id = {int(i): r for r, i in enumerate(items["id"].to_list())}
    del items

    folds = load_folds(args.targets_dir, fold_ids)
    rows1 = {f: np.array([row_of_id[int(i)] for i in folds[f]["id1"]], dtype=np.int64)
             for f in fold_ids}
    rows2 = {f: np.array([row_of_id[int(i)] for i in folds[f]["id2"]], dtype=np.int64)
             for f in fold_ids}

    from validation.build_folds import connected_component_keys

    # Таблица большая: держим ОДИН экземпляр модели и возвращаем её к
    # предобученному состоянию перед каждым фолдом, а не строим заново.
    # Ровно len(token_id), без +1: build_embedding_matrix кладёт <PAD> и <UNK>
    # ВНУТРЬ словаря, в отличие от старых скриптов, где таблица была на строку
    # длиннее самого словаря.
    device = torch.device(args.device)
    if device.type == "cuda":
        # Разреженный градиент таблицы нужен был там, где плотный шаг по 283M
        # параметров стоил столько же, сколько forward. На GPU это уже не так.
        config = replace(config, sparse_embedding=False)
    model = ProductMatcher(torch.zeros(len(token_id), config.embedding_dim), config)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    pretrained_state = copy.deepcopy(model.state_dict())
    del blob
    loss_function = nn.BCEWithLogitsLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zero_shot, fine_tuned = {}, {}
    for position, held_out in enumerate(fold_ids):
        model.load_state_dict(pretrained_state)
        held_key = np.maximum(counts[rows1[held_out]], counts[rows2[held_out]])
        scores = predict(model, titles_t, keys_t, values_t, rows1[held_out],
                         rows2[held_out], args.batch_size, held_key, device)
        zero_shot[held_out] = average_precision(folds[held_out]["target"], scores)
        log(f"\n  {held_out} zero-shot (только LLM, ручных меток не видел): "
            f"PR-AUC {zero_shot[held_out]:.6f}")

        train_ids = [f for f in fold_ids if f != held_out]
        pool_id1 = np.concatenate([folds[f]["id1"] for f in train_ids])
        pool_rows1 = np.concatenate([rows1[f] for f in train_ids])
        pool_rows2 = np.concatenate([rows2[f] for f in train_ids])
        pool_target = np.concatenate([folds[f]["target"] for f in train_ids])
        pool_key = np.maximum(counts[pool_rows1], counts[pool_rows2])

        # Срез для ранней остановки режется по компонентам связности: иначе
        # товар попал бы и в обучение, и в срез, и остановка бы запаздывала.
        component_of_item = connected_component_keys(
            pool_id1, np.concatenate([folds[f]["id2"] for f in train_ids]))
        bucket: dict[int, int] = {}
        is_validation = np.zeros(len(pool_id1), dtype=bool)
        for index, item in enumerate(pool_id1.tolist()):
            key = component_of_item[item]
            if key not in bucket:
                digest = hashlib.sha256(f"{args.seed}:{held_out}:{key}".encode()).digest()
                bucket[key] = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
            is_validation[index] = bucket[key] == 0

        train_mask = ~is_validation
        head = [parameter for name, parameter in model.named_parameters()
                if not name.startswith("embedding.")]
        optimizers = [torch.optim.AdamW(head, lr=args.learning_rate)]
        if not config.freeze_embeddings:
            embedding_lr = args.learning_rate * args.embedding_scale
            optimizers.append(
                torch.optim.SparseAdam(list(model.embedding.parameters()), lr=embedding_lr)
                if config.sparse_embedding else
                torch.optim.AdamW(list(model.embedding.parameters()), lr=embedding_lr))

        rng = np.random.default_rng(args.seed + position)
        best, best_epoch, best_state, waited = -1.0, 0, None, 0
        train_rows1, train_rows2 = pool_rows1[train_mask], pool_rows2[train_mask]
        train_target, train_key = pool_target[train_mask], pool_key[train_mask]
        log(f"    обучение на {int(train_mask.sum()):,} парах "
            f"({', '.join(train_ids)}), срез {int(is_validation.sum()):,}")

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            epoch_started, running, batches, done = time.time(), 0.0, 0, 0
            for index, pick in enumerate(bucketed_batches(train_key, args.batch_size, rng)):
                if args.max_batches and index >= args.max_batches:
                    break
                item_a = make_batch(titles_t, keys_t, values_t, train_rows1[pick],
                                    device=device)
                item_b = make_batch(titles_t, keys_t, values_t, train_rows2[pick],
                                    device=device)
                loss = loss_function(model(item_a, item_b),
                                     torch.from_numpy(train_target[pick]).to(device))
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
                pool_target[is_validation],
                predict(model, titles_t, keys_t, values_t, pool_rows1[is_validation],
                        pool_rows2[is_validation], args.batch_size, pool_key[is_validation]))
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

        if args.save_weights:
            # Веса сохраняем ДО записи предсказаний: дообучение фолда стоит
            # десятки минут, и терять его из-за сбоя на последнем шаге нельзя.
            args.save_weights.mkdir(parents=True, exist_ok=True)
            weight_path = args.save_weights / f"{held_out}.pt"
            torch.save({"state_dict": model.state_dict(), "token_id": token_id,
                        "config": config.__dict__, "max_shapes": shapes,
                        "stemming": stemming,
                        "held_out": held_out, "trained_on": train_ids,
                        "best_epoch": best_epoch}, weight_path)
            log(f"    веса -> {weight_path} "
                f"({weight_path.stat().st_size / 1e9:.2f} ГБ)")

        scores = predict(model, titles_t, keys_t, values_t, rows1[held_out],
                         rows2[held_out], args.batch_size, held_key, device)
        fine_tuned[held_out] = average_precision(folds[held_out]["target"], scores)
        with (args.out_dir / f"{held_out}.csv").open("w", newline="", encoding="utf-8") as sink:
            writer = csv.writer(sink, lineterminator="\n")
            writer.writerow(["id1", "id2", "predict"])
            for a, b, s in zip(folds[held_out]["id1"].tolist(),
                               folds[held_out]["id2"].tolist(), scores.tolist(), strict=True):
                writer.writerow([a, b, f"{s:.8f}"])
        log(f"    {held_out}: лучшая эпоха {best_epoch} -> PR-AUC {fine_tuned[held_out]:.6f}")

        if args.dump_all_folds:
            # In-sample предсказания на обучающих фолдах: метрикой быть не могут,
            # нужны как признак модели второго уровня. Кладём отдельно от OOF.
            dump_dir = args.dump_all_folds / f"trained_without_{held_out}"
            dump_dir.mkdir(parents=True, exist_ok=True)
            for other in train_ids:
                other_key = np.maximum(counts[rows1[other]], counts[rows2[other]])
                in_sample = predict(model, titles_t, keys_t, values_t, rows1[other],
                                    rows2[other], args.batch_size, other_key, device)
                with (dump_dir / f"{other}.csv").open("w", newline="",
                                                      encoding="utf-8") as sink:
                    writer = csv.writer(sink, lineterminator="\n")
                    writer.writerow(["id1", "id2", "predict"])
                    for a, b, s in zip(folds[other]["id1"].tolist(),
                                       folds[other]["id2"].tolist(),
                                       in_sample.tolist(), strict=True):
                        writer.writerow([a, b, f"{s:.8f}"])
            log(f"      in-sample -> {dump_dir}")

    log("\n" + "=" * 62)
    log(f"zero-shot  среднее {np.mean(list(zero_shot.values())):.6f}  "
        f"({', '.join(f'{k}={v:.4f}' for k, v in zero_shot.items())})")
    log(f"дообучено  среднее {np.mean(list(fine_tuned.values())):.6f}  "
        f"({', '.join(f'{k}={v:.4f}' for k, v in fine_tuned.items())})")
    log("контроли на тех же фолдах spec-v2: knrm_name_noaudit 0.568294, "
        "knrm_attrs_llm 0.570753, lgbm_cheap_v1 0.638171")
    log(f"предсказания -> {args.out_dir} | всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

"""KNRM over the attribute dictionary — keys and values as two separate layers.

The two models we already have both read the product **name**: ``lgbm_cheap``
through hand-built similarity features, ``knrm_llm_pretrain`` through token
kernels. Their 0.5/0.5 blend gained +0.056 on the public board over the better
of them, far more than any single-model change in this project — and the reason
is that they fail in different places (prediction correlation 0.62). This
experiment goes after that same lever deliberately: a model that reads a
*different field*, so its errors have no reason to line up with theirs.

Attributes are not free text, they are key/value pairs, and the pairing carries
the meaning. ``цвет = красный`` versus ``цвет = синий`` shares a token and means
the opposite; ``цвет = красный`` versus ``оттенок = красный`` shares the other
token and means nearly the same. Flattening the dictionary into a token bag —
which is what the existing ``kv_jaccard`` feature effectively does — throws that
structure away.

So each attribute is embedded twice, from two independent tables, and the
attribute's vector is the **element-wise product** of the two. The product is
unchanged only when both halves match; agreeing on one side alone lands strictly
between match and mismatch. From there it is ordinary KNRM with attributes in
place of tokens: cosine matrix between the two products' rows, RBF kernel
pooling, BatchNorm, linear ranking layer. Model code lives in
``members/darksteeld/src/knrm_attrs_model.py`` so a container can import the
same file.

Names are not used at all. That is the point, not an omission.

Protocol is identical to ``knrm_name_v2`` so the numbers are comparable: train
on three folds, early stopping (patience 1) on a component-grouped slice carved
out of those three, predict the held-out fold, never touch it during training.

    .venv/bin/python members/darksteeld/experiments/knrm_attrs/train.py \\
        --targets-dir validation/targets_v2 \\
        --out-dir validation/predictions_v2/darksteeld/knrm_attrs
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

from knrm_attrs_model import (  # noqa: E402
    DIM, MAX_ATTRS, AttributeKNRM, build_attribute_vocabularies, encode_attributes,
    initial_weight,
)
from lgbm_cheap import AUDIT_FILE, load_audit  # noqa: E402

DEFAULT_NAVEC = (REPOSITORY_ROOT / "members" / "darksteeld" / "models"
                 / "navec_hudlit_v1_12B_500K_300d_100q.tar")
VALIDATION_BUCKETS = 10


def log(message: str) -> None:
    print(message, flush=True)


def load_folds(targets_dir: Path, fold_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    folds: dict[str, dict[str, np.ndarray]] = {}
    for fold_id in fold_ids:
        path = targets_dir / f"{fold_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"нет {path}; собери цели: make validation-targets-v2")
        id1, id2, target = [], [], []
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                id1.append(int(row["id1"]))
                id2.append(int(row["id2"]))
                target.append(float(row["target"]))
        folds[fold_id] = {
            "id1": np.asarray(id1, dtype=np.int64),
            "id2": np.asarray(id2, dtype=np.int64),
            "target": np.asarray(target, dtype=np.float32),
        }
    return folds


def validation_mask(id1: np.ndarray, id2: np.ndarray, seed: str) -> np.ndarray:
    """Держим в валидации целые компоненты связности — ни один товар не окажется
    одновременно в обучении и в валидации."""
    from validation.build_folds import connected_component_keys

    component_of_item = connected_component_keys(id1, id2)
    is_validation = np.zeros(len(id1), dtype=bool)
    bucket_of_key: dict[int, int] = {}
    for position, item in enumerate(id1.tolist()):
        key = component_of_item[item]
        bucket = bucket_of_key.get(key)
        if bucket is None:
            digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
            bucket = int.from_bytes(digest[:8], "big") % VALIDATION_BUCKETS
            bucket_of_key[key] = bucket
        is_validation[position] = bucket == 0
    return is_validation


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    labels, ranked = target[order], score[order]
    cumulative = np.cumsum(labels)
    if cumulative[-1] == 0:
        return 0.0
    last = np.r_[ranked[1:] != ranked[:-1], True]
    precision = cumulative[last] / (np.arange(len(labels))[last] + 1)
    recall = cumulative[last] / cumulative[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


@torch.no_grad()
def predict(model: AttributeKNRM, keys: torch.Tensor, values: torch.Tensor,
            rows1: np.ndarray, rows2: np.ndarray, *, batch_size: int) -> np.ndarray:
    model.eval()
    rows1_tensor, rows2_tensor = torch.from_numpy(rows1), torch.from_numpy(rows2)
    scores = np.empty(len(rows1), dtype=np.float64)
    for start in range(0, len(rows1), batch_size):
        stop = min(start + batch_size, len(rows1))
        left, right = rows1_tensor[start:stop], rows2_tensor[start:stop]
        scores[start:stop] = torch.sigmoid(model(
            keys[left].long(), values[left].long(),
            keys[right].long(), values[right].long(),
        )).numpy()
    return scores


def train_fold(model: AttributeKNRM, keys: torch.Tensor, values: torch.Tensor,
               train_rows1, train_rows2, train_target,
               validation_rows1, validation_rows2, validation_target,
               *, max_epochs: int, patience: int, batch_size: int,
               learning_rate: float, generator: torch.Generator) -> tuple[int, float]:
    embedding_optimizer = torch.optim.SparseAdam(
        list(model.key_embedding.parameters()) + list(model.value_embedding.parameters()),
        lr=learning_rate,
    )
    dense_optimizer = torch.optim.Adam(
        list(model.norm.parameters()) + list(model.head.parameters()), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    rows1_tensor = torch.from_numpy(train_rows1)
    rows2_tensor = torch.from_numpy(train_rows2)
    target_tensor = torch.from_numpy(train_target)
    pair_count = len(train_target)

    best_score, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(pair_count, generator=generator)
        running, batches, started = 0.0, 0, time.time()
        for start in range(0, pair_count, batch_size):
            selection = permutation[start : start + batch_size]
            left, right = rows1_tensor[selection], rows2_tensor[selection]
            loss = loss_function(
                model(keys[left].long(), values[left].long(),
                      keys[right].long(), values[right].long()),
                target_tensor[selection],
            )
            embedding_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            embedding_optimizer.step()
            dense_optimizer.step()
            running += float(loss.detach())
            batches += 1

        scores = predict(model, keys, values, validation_rows1, validation_rows2,
                         batch_size=batch_size * 4)
        validation_score = average_precision(validation_target, scores)
        improved = validation_score > best_score
        log(f"      epoch {epoch:>2}  loss {running / batches:.5f}  "
            f"val PR-AUC {validation_score:.6f}{'  *' if improved else ''}  "
            f"{time.time() - started:.0f}s")
        if improved:
            best_score, best_epoch, waited = validation_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= patience:
                log(f"      ранняя остановка: нет прироста {waited} эпох(и), "
                    f"возвращаю эпоху {best_epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best_score


def write_predictions(out_dir: Path, fold_id: str, fold: dict[str, np.ndarray],
                      scores: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{fold_id}.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(["id1", "id2", "predict"])
        for id1, id2, score in zip(fold["id1"].tolist(), fold["id2"].tolist(),
                                   scores.tolist(), strict=True):
            writer.writerow([id1, id2, f"{score:.8f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--targets-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "targets_v2")
    parser.add_argument("--out-dir", type=Path,
                        default=REPOSITORY_ROOT / "validation" / "predictions_v2"
                        / "darksteeld" / "knrm_attrs")
    parser.add_argument("--folds", default="fold_01,fold_02,fold_03,fold_04")
    parser.add_argument("--navec", type=Path, default=DEFAULT_NAVEC)
    parser.add_argument("--audit", action="store_true",
                        help="обучать на исправленных метках; по умолчанию исходные, "
                             "чтобы сравнение с knrm_name_v2 и knrm_llm_pretrain было честным")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    import polars as pl

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.time()
    fold_ids = [f.strip() for f in args.folds.split(",") if f.strip()]

    items = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "attributes"])
    attributes = items["attributes"].to_list()
    row_of_id = {int(i): r for r, i in enumerate(items["id"].to_list())}
    log(f"товаров {len(attributes):,}")

    key_ids, value_ids = build_attribute_vocabularies(attributes)
    log(f"словари: токенов в ключах {len(key_ids):,}, в значениях {len(value_ids):,}")
    keys_encoded, values_encoded = encode_attributes(attributes, key_ids, value_ids)
    filled = (keys_encoded[:, :, 0] != 0).sum(axis=1)
    log(f"атрибутов на товар после разбора: медиана {np.median(filled):.0f}, "
        f"среднее {filled.mean():.1f}, обрезано до {MAX_ATTRS}; "
        f"без единого атрибута {(filled == 0).sum():,} товаров")
    log(f"кодирование: ключи {keys_encoded.nbytes / 1e6:.0f} МБ, "
        f"значения {values_encoded.nbytes / 1e6:.0f} МБ, {time.time() - started:.0f}s")
    keys_tensor = torch.from_numpy(keys_encoded)
    values_tensor = torch.from_numpy(values_encoded)

    folds = load_folds(args.targets_dir, fold_ids)
    corrections = load_audit() if args.audit else {}
    if args.audit:
        digest = hashlib.sha256(AUDIT_FILE.read_bytes()).hexdigest()
        log(f"доразметка: {len(corrections)} исправлений, журнал sha256 {digest[:12]}")
    else:
        log("доразметка не применяется (--audit чтобы включить)")

    # Инициализация считается один раз и клонируется под каждый фолд.
    # nn.Embedding.from_pretrained НЕ копирует тензор: без clone() второй фолд
    # стартовал бы с обученных весов первого, то есть с утечкой через модель.
    base_key_weight, key_covered = initial_weight(key_ids, DIM, args.navec)
    base_value_weight, value_covered = initial_weight(value_ids, DIM, args.navec)
    log(f"navec покрыл ключей {key_covered:,}/{len(key_ids):,}, "
        f"значений {value_covered:,}/{len(value_ids):,}; "
        f"таблицы {(base_key_weight.numel() + base_value_weight.numel()) * 4 / 1e6:.0f} МБ")

    scores_by_fold = {}
    for held_out in fold_ids:
        log(f"\n=== {held_out}")
        train_ids = [f for f in fold_ids if f != held_out]
        id1 = np.concatenate([folds[f]["id1"] for f in train_ids])
        id2 = np.concatenate([folds[f]["id2"] for f in train_ids])
        target = np.concatenate([folds[f]["target"] for f in train_ids])
        if corrections:
            applied = 0
            for position, pair in enumerate(zip(id1.tolist(), id2.tolist())):
                if pair in corrections:
                    target[position] = corrections[pair]
                    applied += 1
            log(f"    исправлений в обучающей части: {applied}")

        is_validation = validation_mask(id1, id2, f"{args.seed}:{held_out}")
        rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int64)
        rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int64)
        log(f"    обучение {int((~is_validation).sum()):,} пар / валидация "
            f"{int(is_validation.sum()):,}")

        model = AttributeKNRM(base_key_weight.clone(), base_value_weight.clone(), sparse=True)

        best_epoch, best_score = train_fold(
            model, keys_tensor, values_tensor,
            rows1[~is_validation], rows2[~is_validation], target[~is_validation],
            rows1[is_validation], rows2[is_validation], target[is_validation],
            max_epochs=args.max_epochs, patience=args.patience,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            generator=torch.Generator().manual_seed(args.seed),
        )

        fold = folds[held_out]
        held_rows1 = np.array([row_of_id[int(i)] for i in fold["id1"]], dtype=np.int64)
        held_rows2 = np.array([row_of_id[int(i)] for i in fold["id2"]], dtype=np.int64)
        scores = predict(model, keys_tensor, values_tensor, held_rows1, held_rows2,
                         batch_size=args.batch_size * 4)
        write_predictions(args.out_dir, held_out, fold, scores)
        held_score = average_precision(fold["target"], scores)
        scores_by_fold[held_out] = held_score
        log(f"    {held_out}: лучшая эпоха {best_epoch} (val {best_score:.6f}) "
            f"-> PR-AUC на фолде {held_score:.6f}")
        del model

    log("\n" + "=" * 62)
    mean = float(np.mean(list(scores_by_fold.values())))
    detail = ", ".join(f"{f}={scores_by_fold[f]:.4f}" for f in fold_ids)
    log(f"knrm_attrs mean {mean:.6f}  ({detail})")
    log("контроли на тех же фолдах spec-v2:")
    log("  knrm_name_v2      0.530775   (KNRM по названию, без предобучения)")
    log("  knrm_llm_pretrain 0.565568   (KNRM по названию, предобучен на matches_llm)")
    log("  lgbm_cheap_v1     0.638171   (бустинг по 21 признаку, включая атрибутные)")
    log(f"предсказания -> {args.out_dir}   всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

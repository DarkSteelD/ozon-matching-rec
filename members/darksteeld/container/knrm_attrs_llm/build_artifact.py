"""Обучить отгружаемый KNRM по атрибутам и выгрузить его с адресацией по токенам.

Две стадии, как в проверенном эксперименте
(``members/darksteeld/experiments/knrm_attrs_llm``, mean PR-AUC 0.570335 на
фолдах spec-v2 против 0.551883 без предобучения):

1. предобучение одну эпоху на всех 11,187,780 парах ``matches_llm`` с мягкими
   целями, по словарю, охватывающему обе вселенные;
2. дообучение на **всех** 365,654 ручных парах с ранней остановкой (patience 1)
   по срезу, вырезанному из них по компонентам связности, и выгрузка.

Состояние после предобучения кэшируется, поэтому повторная выгрузка не повторяет
пятидесятиминутную первую стадию.

**Почему выгружается словарь, а не индексы.** Таблица эмбеддингов адресуется
номерами обучающего словаря, а на сабмите товары другие и словарь у них другой.
Поэтому артефакт везёт отображение **токен -> вектор**, а контейнер строит
индексное пространство под те токены, которые встретились в тестовом файле.
Токен, которого в обучении не было, получает детерминированный вектор из своей
же строки — тот самый, с которого он стартовал бы при обучении. Для артикулов
это принципиально: один и тот же код по обе стороны пары даёт косинус ровно 1.0
и зажигает ядро точного совпадения, а общая строка ``<unk>`` сделала бы все
незнакомые артикулы одинаковыми, то есть худшим из возможных вариантов.

    .venv/bin/python members/darksteeld/container/knrm_attrs_llm/build_artifact.py \\
        --cache-dir <scratch>/attrcache
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "members" / "darksteeld" / "src"))

EXPERIMENT = REPOSITORY_ROOT / "members" / "darksteeld" / "experiments" / "knrm_attrs_llm"
sys.path.insert(0, str(EXPERIMENT))

from knrm_attrs_model import (  # noqa: E402
    DIM, MAX_ATTRS, MAX_KEY_TOKENS, MAX_VALUE_TOKENS, AttributeKNRM, encode_attributes,
    initial_weight,
)
from knrm_model import PAD_ID  # noqa: E402
from train import (  # noqa: E402  — тот же код, что у проверенного эксперимента
    average_precision, build_vocabulary, count_catalogue_tokens, encode_stream,
    hand_attribute_tokens, predict, run_epoch, validation_mask,
)

SEED = 20260815
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
NAVEC = (REPOSITORY_ROOT / "members" / "darksteeld" / "models"
         / "navec_hudlit_v1_12B_500K_300d_100q.tar")


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--navec", type=Path, default=NAVEC)
    parser.add_argument("--min-count", type=int, default=5,
                        help="отсечка по каталогу для ОБУЧАЮЩЕГО словаря")
    parser.add_argument("--pretrain-pairs", type=int, default=0, help="0 = все 11.19M")
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--ship-all", action="store_true",
                        help="отгружать все векторы, включая восстановимые из строки")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    import polars as pl

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    started = time.time()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- словари по обеим вселенным ----------------------------------------
    hand = pl.read_parquet(args.data_dir / "items_human.parquet", columns=["id", "attributes"])
    hand_attributes = hand["attributes"].to_list()
    tokens_hand_keys, tokens_hand_values = hand_attribute_tokens(hand_attributes)
    counts_keys, counts_values = count_catalogue_tokens(
        args.data_dir / "items.parquet", args.cache_dir / "attr_token_counts.npz")
    key_ids = build_vocabulary(counts_keys, tokens_hand_keys, args.min_count)
    value_ids = build_vocabulary(counts_values, tokens_hand_values, args.min_count)
    log(f"словарь: ключей {len(key_ids):,}, значений {len(value_ids):,}")

    # ---- стадия 1: предобучение (кэшируется) --------------------------------
    checkpoint = args.cache_dir / f"attrs_pretrained_e{args.pretrain_epochs}_mc{args.min_count}.pt"
    base_key_weight, key_covered = initial_weight(key_ids, DIM, args.navec)
    base_value_weight, value_covered = initial_weight(value_ids, DIM, args.navec)
    log(f"navec покрыл ключей {key_covered:,}/{len(key_ids):,}, "
        f"значений {value_covered:,}/{len(value_ids):,}")
    init_key = base_key_weight.clone()
    init_value = base_value_weight.clone()
    model = AttributeKNRM(base_key_weight, base_value_weight, sparse=True)

    if checkpoint.is_file():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        log(f"веса после предобучения из кэша: {checkpoint.name}")
    else:
        llm = pl.read_parquet(args.data_dir / "matches_llm.parquet")
        if args.pretrain_pairs and args.pretrain_pairs < llm.height:
            pick = np.random.default_rng(SEED).choice(llm.height, args.pretrain_pairs,
                                                      replace=False)
            pick.sort()
            llm = llm[pick]
        llm_ids = np.unique(np.concatenate([llm["id1"].to_numpy(), llm["id2"].to_numpy()]))
        if np.intersect1d(llm_ids, hand["id"].to_numpy()).size:
            raise AssertionError("вселенные пересекаются — предобучение потечёт")
        log(f"LLM-пар {llm.height:,}, товаров {len(llm_ids):,}; кодирование "
            f"{len(llm_ids) * MAX_ATTRS * (MAX_KEY_TOKENS + MAX_VALUE_TOKENS) * 4 / 1e9:.2f} ГБ")
        llm_keys, llm_values = encode_stream(
            args.data_dir / "items.parquet", llm_ids, key_ids, value_ids)
        optimizers = [
            torch.optim.SparseAdam(
                list(model.key_embedding.parameters())
                + list(model.value_embedding.parameters()), lr=LEARNING_RATE),
            torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                             lr=LEARNING_RATE),
        ]
        rows1 = np.searchsorted(llm_ids, llm["id1"].to_numpy()).astype(np.int64)
        rows2 = np.searchsorted(llm_ids, llm["id2"].to_numpy()).astype(np.int64)
        target = llm["target"].to_numpy().astype(np.float32)
        keys_t, values_t = torch.from_numpy(llm_keys), torch.from_numpy(llm_values)
        generator = torch.Generator().manual_seed(SEED)
        for epoch in range(1, args.pretrain_epochs + 1):
            loss = run_epoch(model, optimizers, nn.BCEWithLogitsLoss(), keys_t, values_t,
                             rows1, rows2, target, BATCH_SIZE, generator,
                             f"предобучение e{epoch}")
            log(f"  предобучение, эпоха {epoch}: loss {loss:.5f}")
        torch.save(model.state_dict(), checkpoint)
        del keys_t, values_t, llm_keys, llm_values, rows1, rows2, target, llm

    # ---- стадия 2: дообучение на всех ручных парах --------------------------
    matches = pl.read_parquet(args.data_dir / "matches.parquet")
    hand_keys, hand_values = encode_attributes(hand_attributes, key_ids, value_ids)
    hand_keys_t, hand_values_t = torch.from_numpy(hand_keys), torch.from_numpy(hand_values)
    row_of_id = {int(i): r for r, i in enumerate(hand["id"].to_list())}
    id1, id2 = matches["id1"].to_numpy(), matches["id2"].to_numpy()
    labels = matches["target"].to_numpy().astype(np.float32)

    corrections_applied, audit_digest = 0, None
    if args.audit:
        from lgbm_cheap import AUDIT_FILE, load_audit

        corrections = load_audit()
        for position, pair in enumerate(zip(id1.tolist(), id2.tolist())):
            if pair in corrections:
                labels[position] = corrections[pair]
                corrections_applied += 1
        audit_digest = hashlib.sha256(AUDIT_FILE.read_bytes()).hexdigest()
        log(f"доразметка: применено {corrections_applied} из {len(corrections)}")
    else:
        log("доразметка: не применяется")

    rows1 = np.array([row_of_id[int(i)] for i in id1], dtype=np.int64)
    rows2 = np.array([row_of_id[int(i)] for i in id2], dtype=np.int64)
    is_validation = validation_mask(id1, id2, f"{SEED}:submit")
    log(f"дообучение: обучение {int((~is_validation).sum()):,} / валидация "
        f"{int(is_validation.sum()):,}")

    optimizers = [
        torch.optim.SparseAdam(
            list(model.key_embedding.parameters())
            + list(model.value_embedding.parameters()), lr=LEARNING_RATE),
        torch.optim.Adam(list(model.norm.parameters()) + list(model.head.parameters()),
                         lr=LEARNING_RATE),
    ]
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(SEED)
    best, best_epoch, best_state, waited = -1.0, 0, None, 0
    for epoch in range(1, args.finetune_epochs + 1):
        run_epoch(model, optimizers, loss_function, hand_keys_t, hand_values_t,
                  rows1[~is_validation], rows2[~is_validation], labels[~is_validation],
                  BATCH_SIZE, generator, f"дообучение e{epoch}")
        score = average_precision(
            labels[is_validation],
            predict(model, hand_keys_t, hand_values_t, rows1[is_validation],
                    rows2[is_validation], batch_size=BATCH_SIZE * 4))
        improved = score > best
        log(f"  дообучение, эпоха {epoch}: val PR-AUC {score:.6f}{'  *' if improved else ''}")
        if improved:
            best, best_epoch, waited = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= args.patience:
                log(f"  ранняя остановка, возвращаю эпоху {best_epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- выгрузка -----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key_tokens = sorted(key_ids, key=lambda t: key_ids[t])
    value_tokens = sorted(value_ids, key=lambda t: value_ids[t])
    key_trained = model.key_embedding.weight.detach().numpy()
    value_trained = model.value_embedding.weight.detach().numpy()

    # Отгружаем только те векторы, которые контейнер НЕ восстановит сам.
    #
    # Для незнакомого токена контейнер берёт vector_for_unknown(токен) — чистую
    # функцию от строки. Значит токен, чей обученный вектор так и остался равен
    # этому же хеш-вектору, отгружать незачем: контейнер получит ровно его. Это
    # не приближение и не отсечка по частоте, а точная проверка «совпадает ли
    # обученный вектор с тем, что и так будет восстановлен».
    #
    # Критерий именно по хеш-вектору, а не по инициализации: токен, начавшийся с
    # navec и не сдвинувшийся, восстановлен НЕ будет, поэтому он отгружается.
    def shippable(trained: np.ndarray, vocabulary: dict[str, int]) -> np.ndarray:
        hashed, _ = initial_weight(vocabulary, DIM, None)   # navec выключен -> чистый хеш
        hashed = hashed.numpy()
        deviation = np.abs(trained[1:] - hashed[1:]).max(axis=1)
        return deviation > 1e-6

    keep_key = np.ones(len(key_tokens), dtype=bool) if args.ship_all \
        else shippable(key_trained, key_ids)
    keep_value = np.ones(len(value_tokens), dtype=bool) if args.ship_all \
        else shippable(value_trained, value_ids)
    log(f"отгружается ключей {int(keep_key.sum()):,}/{len(key_tokens):,} "
        f"({100*keep_key.mean():.1f}%), значений {int(keep_value.sum()):,}/"
        f"{len(value_tokens):,} ({100*keep_value.mean():.1f}%); "
        f"остальные контейнер восстановит из строки токена")
    key_tokens = [t for t, keep in zip(key_tokens, keep_key.tolist()) if keep]
    value_tokens = [t for t, keep in zip(value_tokens, keep_value.tolist()) if keep]
    key_trained_ship = key_trained[1:][keep_key]
    value_trained_ship = value_trained[1:][keep_value]

    np.savez_compressed(
        args.out_dir / "model.npz",
        key_vocabulary=np.array(key_tokens, dtype=object),
        key_vectors=key_trained_ship.astype(np.float16),
        value_vocabulary=np.array(value_tokens, dtype=object),
        value_vectors=value_trained_ship.astype(np.float16),
        head_weight=model.head.weight.detach().numpy().astype(np.float32),
        head_bias=model.head.bias.detach().numpy().astype(np.float32),
        bn_weight=model.norm.weight.detach().numpy().astype(np.float32),
        bn_bias=model.norm.bias.detach().numpy().astype(np.float32),
        bn_mean=model.norm.running_mean.numpy().astype(np.float32),
        bn_var=model.norm.running_var.numpy().astype(np.float32),
        bn_eps=np.float32(model.norm.eps),
    )
    # Имя эксперимента выводим из фактического корпуса предобучения, а не
    # прибиваем: сборка с --pretrain-pairs 4000000 — это knrm_attrs_llm, и
    # называть её _full значило бы приписать ей чужой OOF на лидерборде.
    if args.audit:
        experiment_name = "knrm_attrs_llm_audit"
    elif args.pretrain_pairs and args.pretrain_pairs < 11_187_780:
        experiment_name = "knrm_attrs_llm"
    else:
        experiment_name = "knrm_attrs_llm_full"
    artifact = {
        "experiment": experiment_name,
        "reads": "attributes only; product name is not used",
        "corrections_applied": corrections_applied,
        "audit_journal_sha256": audit_digest,
        "pairs_pretrain": 11_187_780 if not args.pretrain_pairs else args.pretrain_pairs,
        "pairs_finetune": int(len(labels)),
        "key_vocabulary_trained": len(key_ids),
        "value_vocabulary_trained": len(value_ids),
        "key_vocabulary_shipped": len(key_tokens),
        "value_vocabulary_shipped": len(value_tokens),
        "min_count": args.min_count,
        "dim": DIM,
        "max_attrs": MAX_ATTRS,
        "max_key_tokens": MAX_KEY_TOKENS,
        "max_value_tokens": MAX_VALUE_TOKENS,
        "seed": SEED,
        "pretrain_epochs": args.pretrain_epochs,
        "best_finetune_epoch": best_epoch,
        "validation_prauc": best,
        "navec_key_rows": key_covered,
        "navec_value_rows": value_covered,
        "torch_version": torch.__version__,
        "repo_commit": git_commit(REPOSITORY_ROOT),
        "local_cv": {
            "spec_v2_mean_prauc": 0.57033477,
            "spec_v2_zero_shot_mean_prauc": 0.455377,
            "control_knrm_attrs_no_pretrain": 0.551883,
            "control_knrm_llm_pretrain_on_names": 0.565568,
            "note": "out-of-fold на замороженных фолдах; отгружаемая модель обучена на всех парах",
        },
    }
    (args.out_dir / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    size = (args.out_dir / "model.npz").stat().st_size / 1e6
    log(f"model.npz {size:.0f} МБ | ключей {len(key_tokens):,}, значений {len(value_tokens):,} "
        f"| лучшая эпоха {best_epoch} (val {best:.6f}) | всего {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()

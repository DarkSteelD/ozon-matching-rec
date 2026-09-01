# E-CUP 2026 Ozon Matching — финальные решения команды

Этот документ фиксирует **две посылки, отмеченные `final` на платформе ODS**.
Это не выбор двух наибольших публичных скоров: поздняя `mb_qwen_v2.zip`
показывала `0.542073`, но final-флаг стоял у решений ниже.

| Решение | ODS submission ID | Время, UTC | Mean PR-AUC |
| --- | --- | --- | ---: |
| `mb_route_mega.zip` | `5016818c-565d-48ab-bad9-0dbe6c36a6f1` | 2026-08-30 19:52:17 | **0.5414130948985605** |
| `mb_ce1_soup.zip` | `009f5b47-a4be-493d-84bd-77d1ea755dcc` | 2026-08-27 15:23:58 | **0.5135322712877685** |

Архивы, их точный состав и SHA-256 закреплены в
`final_solutions/manifest.json`. Скрипт `final_solutions/manage.py` скачивает,
проверяет, безопасно распаковывает и детерминированно перепаковывает решения.
Все точные веса уже находятся внутри двух ODS-архивов по закреплённым прямым
ссылкам; в основной Git они повторно не загружаются.

## 1. Быстрый точный повтор отправленных контейнеров

### 1.1. Подготовить клон

```bash
python final_solutions/manage.py fetch
python final_solutions/manage.py verify
python final_solutions/manage.py extract
```

По умолчанию архивы и распакованные контейнеры попадают в
`artifacts/final_solutions/`; эта директория игнорируется Git. `verify`
проверяет одновременно:

1. размер и SHA-256 всего ZIP;
2. отсутствие дубликатов, абсолютных путей, `..` и симлинков;
3. точный набор файлов;
4. размер и SHA-256 каждого файла внутри ZIP.

Для одного решения используйте, например:

```bash
python final_solutions/manage.py fetch --solution mb_ce1_soup
python final_solutions/manage.py extract --solution mb_ce1_soup
```

### 1.2. Контракт запуска

Обе посылки используют образ `odsai/ecup26-matching-baseline:1.0` и команду
`python -u run.py`. Из корня распакованного решения эквивалентный запуск:

```bash
python -u run.py \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output_path /output/submission.csv
```

Входные схемы:

- `items.parquet`: `id`, `name`, `category`, `attributes`;
- `matches.parquet`: `id1`, `id2`, без `target`.

Выходная схема: CSV с колонками `id1,id2,predict` и одной строкой на входную
пару. Все модели загружаются только с локального диска; сеть на инференсе не
нужна.

### 1.3. Пересобрать ZIP из проверенного каталога

```bash
python final_solutions/manage.py pack \
  --solution mb_ce1_soup \
  --source artifacts/final_solutions/mb_ce1_soup \
  --archive artifacts/final_solutions/mb_ce1_soup.repacked.zip
```

Сборщик сортирует пути, ставит фиксированное время `1980-01-01`, фиксирует
права `0644` и использует Deflate level 9. ZIP может отличаться от ODS-архива
на уровне служебных полей, но распакованное содержимое обязано побайтово
совпасть с manifest. Для намеренно переобученных весов есть явный флаг
`--allow-content-drift`; случайно подменить final-вес без него нельзя.

## 2. Данные и общий обучающий конвейер

Исходные данные соревнования не хранятся в Git. Нужны:

```text
data/raw/items.parquet
data/raw/items_human.parquet
data/raw/matches.parquet
data/raw/matches_llm.parquet
```

Ручной набор содержит 365 654 пары. LLM-набор содержит 11.19 млн мягко
размеченных пар и не пересекается с ручным по товарам, поэтому претрен можно
честно оценивать на всей ручной выборке. Командный validation contract —
grouped 4-fold по компонентам связности; товар не встречается в двух фолдах.

Установите как минимум Python 3.12 и библиотеки `torch`, `transformers`,
`safetensors`, `numpy`, `pandas`, `polars`, `pyarrow`. Зафиксированные версии
для финального ModernBERT-конвейера: `torch 2.11.0+cu128`,
`transformers 5.15.1`. Обучение проводилось на RTX 5070 Ti; инференс был
рассчитан на T4 и лимит 20 минут.

Не понижайте Transformers до 4.x: точные final-tokenizer configs используют
класс `TokenizersBackend`, которого в 4.57 ещё нет. Контейнерный образ ODS уже
содержал совместимое окружение.

### 2.1. Подготовить простые тексты для LLM-претрена

```bash
mkdir -p work/final
python members/darksteeld/ops/build_llm_texts.py \
  --out work/final/llm_texts.parquet
```

На выходе остаются только товары из LLM-пар. Текст имеет вид
`имя | категория | атрибуты`; атрибуты сортируются по имени ключа и
ограничиваются 800 символами.

### 2.2. Подготовить ручные пары и prio-тексты

```bash
python members/darksteeld/src/build_distill_pairs.py \
  --teacher final_stack_all \
  --alpha 0.7 \
  --out work/final/hand_pairs_distill_aug.parquet
```

Команда читает OOF-учителя из
`members/dzkhomidov/preds/all_model_predictions_oof.parquet`. Мягкая цель:
`0.7 * final_stack_all + 0.3 * target`. Для обучения строятся prio-тексты:

- порядок атрибутов: бренд → модель/артикул → цвет → остальные;
- в конец добавляется `@@ сравнение: цвет=...; артикул=...`;
- у четырёх fashion-категорий убираются размерные атрибуты;
- имя намеренно обрезается по первому `" | "`, повторяя поведение inference.

Не добавляйте `--with-closure`: 6 648 пар, полученных транзитивным замыканием,
ухудшили результат на `0.001594` на всех четырёх фолдах.

### 2.3. Претрен RuModernBERT на LLM-парах

```bash
python members/darksteeld/ops/pretrain_llm.py \
  --texts work/final/llm_texts.parquet \
  --pairs data/raw/matches_llm.parquet \
  --model deepvk/RuModernBERT-base \
  --num-labels 1 \
  --out work/final/rumodernbert_llm_pretrain_512 \
  --epochs 1 \
  --max-len 512 \
  --bs 32 \
  --workers 16 \
  --lr 5e-5 \
  --compile \
  --pad-multiple 64 \
  --length-buckets \
  --swap-augment \
  --save-every 2000
```

Один проход занимал около 16 часов на RTX 5070 Ti. Модель имеет один логит,
учится BCE на мягкой цели, случайно меняет стороны пары и группирует примеры по
длине. Точный checkpoint уже доступен как
`members/darksteeld/models/rumodernbert_llm_pretrain_512/`.

Критическая деталь: используйте претрен длины 512. Старый checkpoint длины 160
становился хуже при инференсе на 512 (`0.714784` против `0.733139`), тогда как
настоящий 512-претрен давал `0.748740` на том же контроле.

## 3. Final №1: `mb_ce1_soup`

### 3.1. Идея

Это один RuModernBERT cross-encoder длины 512. Он получен усреднением весов
четырёх моделей, каждая из которых стартовала из одного LLM-претрена и училась
на трёх ручных фолдах. На инференсе работает одна сеть, а не ансамбль из
четырёх проходов.

Честные OOF-оценки одиночных фолдовых моделей:

| Fold | PR-AUC |
| --- | ---: |
| 01 | 0.862819 |
| 02 | 0.870183 |
| 03 | 0.866205 |
| 04 | 0.866304 |
| Mean | **0.866378** |
| Macro over 20 categories | **0.819570** |

Это оценка **одиночной held-out модели**, не супа. Суп вместе видел все четыре
фолда, поэтому локальная оценка супа на ручных данных внутривыборочна.

### 3.2. Обучить четыре fold-модели

```bash
python members/darksteeld/ops/train_distill.py \
  --data work/final/hand_pairs_distill_aug.parquet \
  --model deepvk/RuModernBERT-base \
  --init work/final/rumodernbert_llm_pretrain_512 \
  --num-labels 1 \
  --out work/final/ce1 \
  --stage folds \
  --epochs 1 \
  --bs 24 \
  --eval-bs 24 \
  --max-len 512 \
  --workers 16 \
  --optimizer adan \
  --lr 1e-4 \
  --compile \
  --pad-multiple 64 \
  --drop-closure \
  --length-buckets \
  --duplicate-swapped \
  --save-models
```

`--duplicate-swapped` добавляет оба порядка пары отдельными примерами. Для
обучения веса должны быть fp32: наследование `float16` из inference-checkpoint
раньше давало NaN с первого шага. Скрипт проверяет dtype и падает рано.

### 3.3. Усреднить веса

```bash
python members/darksteeld/ops/soup_models.py \
  --models work/final/ce1/model_fold_01 \
           work/final/ce1/model_fold_02 \
           work/final/ce1/model_fold_03 \
           work/final/ce1/model_fold_04 \
  --out work/final/rumodernbert_ce1_soup
```

Финальный median relative drift был `0.0595`, то есть fold-модели остались в
одной области параметров и weight averaging осмыслен. Точный отправленный суп:
`members/darksteeld/models/rumodernbert_ce1_soup_final/`.

### 3.4. Инференс

Источник контейнера: `members/darksteeld/container/mb_ce1_soup/`.

1. Строятся prio-тексты и флаг несовпадения размера.
2. Пары сортируются по грубой длине, чтобы снизить padding.
3. Cross-encoder считает `sigmoid(logit)`.
4. Вероятности заменяются глобальными рангами.
5. Для конфликтующих размеров в fashion score умножается на `0.25`.
6. Если прогноз времени выходит за 17 минут, `max_len` дорогого хвоста
   последовательно сокращается, но не ниже 144.

Штраф размера локально вредил (`-0.0218` macro), но на leaderboard дал
`+0.00233`; это зафиксированный distribution shift, поэтому финал повторяет
leaderboard-проверенное поведение.

## 4. Final №2: `mb_route_mega`

### 4.1. Идея

Метрика — среднее PR-AUC по 20 категориям. Внутри каждой категории важен только
порядок пар, поэтому разные категории можно обслуживать разными моделями.
`mb_route_mega` содержит пять моделей, но каждая считает только принадлежащие ей
пары: суммарная работа около `1.1` полного прохода, а не пять проходов.

Индексы в `models.json`:

| № | Каталог | Модель | Вход / длина |
| ---: | --- | --- | --- |
| 0 | `model_ce1` | точный `ce1_soup` из final №1 | prio / 512 |
| 1 | `model_pretrain` | RuModernBERT LLM-pretrain без hand fine-tune | plain / 512 |
| 2 | `model_mb224` | старый RuModernBERT model soup | prio / 224 |
| 3 | `model_student384` | rubert-base student e3 от dzkhomidov | prio / 384 |
| 4 | `model_fold01` | ce1 fold-01, обучен на folds 02–04 | prio / 512 |

Точные веса и токенизаторы всех пяти каталогов входят в final archive и
закреплены в manifest. Модель №0 общая с `mb_ce1_soup`; модель №1 — fp16
inference-копия `rumodernbert_llm_pretrain_512`.

### 4.2. Маршруты

| Категории | Выход |
| --- | --- |
| Аптека; Бытовая техника; Мебель; Строительство и ремонт | model 0 (`ce1`) |
| Обувь; Одежда; Галантерея и аксессуары; Красота и гигиена; Детские товары; Хобби и творчество; Товары для животных; Дом и сад; Канцелярские товары | model 1 (`pretrain`) |
| Автотовары; Электроника | model 2 (`mb224`) |
| Бытовая химия; Музыкальные инструменты; Продукты питания; Спорт и отдых | rank blend `0.75 * model 2 + 0.25 * model 3` |
| Ювелирные изделия | model 4 (`fold01`) |

Для смеси ранги вычисляются отдельно внутри категории. После маршрутизации
применяется тот же fashion size penalty `0.25`.

### 4.3. Почему маршрутизация сработала

Fine-tuned-модели сходились к похожему ранжированию `ce1` и почти не выигрывали
новых категорий. LLM-pretrain, не видевший ручных меток, имел rank agreement
около `0.78–0.79` и выигрывал девять категорий. В fashion это объяснимо:
prio-конвейер удалял размеры, а plain-тексты претрена их сохраняли.

Последовательность публичных проверок:

- `ce1_soup`: `0.513532`;
- консервативный route четырёх категорий: около `0.53175`;
- полный measured routing ceiling: `0.541413`.

Финальная конфигурация выбирает победителя по опубликованным покатегорийным
результатам. Это сильнее подогнано к public split, чем `mb_ce1_soup`; оба
решения и были оставлены final как разные точки по риску.

### 4.4. Защита по времени

Контейнер сначала вычисляет маску пар для каждого model route и не считает
предсказания, которые затем были бы перезаписаны. Бюджет inference — 15 минут
из платформенных 20; остаётся пять минут на загрузку весов, тексты и запись
CSV. Дедлайн распределяется пропорционально числу оставшихся пар. При отставании
уменьшается `max_len` только у ещё не обработанного длинного хвоста.

## 5. Проверки перед повторной отправкой

Минимальный обязательный набор:

```bash
python -m py_compile \
  final_solutions/manage.py \
  members/darksteeld/container/mb_ce1_soup/run.py \
  members/darksteeld/container/mb_route_mega/run.py

python final_solutions/manage.py verify
git lfs fsck
git status --short
```

Затем выполните smoke-run на небольшом parquet с той же схемой и проверьте:

- число выходных строк равно числу входных пар;
- `id1,id2` и порядок строк не изменились;
- `predict` конечен, без `NaN`, в диапазоне `[0, 1]`;
- в логе нет пропущенной модели или срабатывания fallback;
- полный прогон укладывается в 20 минут на целевой GPU.

## 6. Важные ограничения воспроизводимости

- Competition data нельзя восстановить из Git: их нужно положить в
  `data/raw/` вручную.
- Exact inference воспроизводится через сохранённые final-веса и manifest.
  Повторное обучение может дать другой бинарный SHA из-за CUDA kernels и
  недетерминированности, даже при seed `20260814`.
- Public score воспроизводится только на скрытых данных ODS; локальные OOF
  числа проверяют pipeline, но не равны leaderboard score.
- Не «исправляйте» обрезку имени, fashion size penalty или форматы plain/prio:
  это часть обучающего распределения и отправленного решения.
- Не оценивайте model soup на тех же четырёх ручных фолдах как честный OOF:
  вместе его компоненты видели всю ручную выборку.

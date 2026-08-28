# Ops: внутренний бенч, TOP-5, MLflow, реальный ЛБ, сабмит

Портируемый тулкит для этого трека. Поведение задаётся `ops_config.json`
(метрика трека, направление, публичная колонка). Никаких путей vast — запускается
на любой машине, где лежат предсказания (сейчас ноут участника, позже общий
инстанс).

## Два лидерборда — не путать

- **Наша валидация (CV)** — `validation/leaderboard.{csv,md}` и `validation/TOP5.md`:
  наши эксперименты на замороженных фолдах, наша собственная метрика. Публичный
  скор наших посылок стоит там лишь как диагностическая колонка (`public_*`).
- **Реальный ЛБ соревнования** — `PUBLIC_LEADERBOARD.md` в корне репо: места
  **всех команд** на ODS, тянется живьём из API (`pull_public_lb.py`). Это то,
  что видят организаторы; для отбора фолдов/калибровки/весов НЕ используется
  (CAMPAIGN_RULES #2).

## Ночной бенч

```bash
python validation/ops/bench.py            # пересобрать лидерборд + TOP5 + MLflow, без пуша
python validation/ops/bench.py --commit   # то же + git add/commit/pull --rebase/push
python validation/ops/bench.py --no-mlflow --py /path/to/python
```

Шаги: (1) пересобрать `validation/leaderboard.csv/.md` из закоммиченных
result-JSON (`make leaderboard`); (2) сгенерировать `validation/TOP5.md` —
топ-5 по первичной метрике; (3) залогировать все result-JSON в MLflow;
(4) при `--commit` — закоммитить и запушить лидерборды, TOP5 и results.

Деплой на 23:00 (хост определяется позже — vast или чей-то бокс; здесь НЕ
установлен):

```cron
0 23 * * * cd <repo> && <py> validation/ops/bench.py --commit >> ~/bench.log 2>&1
```

`make score` (обучение → предсказания → result-JSON) остаётся за участником;
бенч консолидирует и коммитит лучшее.

## Реальный ЛБ ODS

```bash
python validation/ops/pull_public_lb.py [--commit]   # -> PUBLIC_LEADERBOARD.md
python validation/ops/refresh_ods_detail.py           # полный срез + commit при изменении
```

Тянет публичный лидерборд всех команд (нужна кука `~/.config/ecup-agent/ods_cookie`),
находит нашу команду по `user_id`, пишет место / отрыв от лидеров / топ-15.
`run_all.py` в хабе прогоняет и бенч (CV), и этот пул по всем трём трекам.

## MLflow

Бэкенд — sqlite (file-store в MLflow 3.x закрыт), по умолчанию
`~/.config/ecup-agent/mlflow.db`, по одному эксперименту на трек, ран на
`<member>/<experiment>` с пофолдовыми и агрегатными метриками + публичным скором.
Посмотреть:

```bash
mlflow ui --backend-store-uri sqlite:///$HOME/.config/ecup-agent/mlflow.db
# http://127.0.0.1:5000
```

Общий бэкенд позже (когда поднимем vast-хранилище) — задать `ECUP_MLFLOW_URI`
(напр. `postgresql://…` или `http://<host>:5000`); код не меняется.

## Сабмит всех решений

`TOP5.md` говорит, что достойно отправки. Единый сабмиттер для всех трёх треков —
`members/darksteeld/scripts/ods_autosubmit.py` в репозитории `ozon-ltv` (трек 3):
presign → S3 → `submissions/add`, dry-run по умолчанию, живой контроль дневного
бюджета (5/день), md5-дедуп. Очередь — `auto_candidates.json` (по треку список
файлов); наполняется из TOP5 теми экспериментами, у которых есть собранный
артефакт сабмита (контейнер `.zip` для треков 1/2, CSV для трека 3).

```bash
python members/darksteeld/scripts/ods_autosubmit.py --auto            # dry-run
python members/darksteeld/scripts/ods_autosubmit.py --auto --yes      # боевой
```

## Кросс-трековый запуск

`ozon-ltv/members/darksteeld/ops/ecup_repos.json` — конфиг всех трёх репо;
`run_all.py` там прогоняет бенч по всем трекам одной командой.

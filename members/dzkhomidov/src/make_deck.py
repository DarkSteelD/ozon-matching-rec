# -*- coding: utf-8 -*-
"""Презентация трека матчинга: слайды 16:9 в PDF, мягкая палитра."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

BG, INK, SUB = '#faf7f2', '#44403c', '#8a837b'
BLUE, ORANGE, GREEN, PURPLE, RED, GRID = '#8fb4dd', '#eab48f', '#9ccfb0', '#bfa8d9', '#dfa0a0', '#e7e2da'
DARKBLUE = '#5b87ba'
plt.rcParams.update({'font.size': 13, 'text.color': INK, 'axes.edgecolor': GRID,
                     'axes.labelcolor': SUB, 'xtick.color': SUB, 'ytick.color': INK,
                     'figure.facecolor': BG, 'axes.facecolor': BG, 'font.family': 'DejaVu Sans'})

PDF = os.path.expanduser('~/matching-work/zs/matching_track_report.pdf')
pp = PdfPages(PDF)
NUM = [0]

def slide(title, body=''):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    fig.patch.set_facecolor(BG)
    fig.text(0.06, 0.90, title, fontsize=24, fontweight='bold', color=INK)
    if body:
        assert len(body) <= 200, f'body>200: {len(body)} {title}'
        fig.text(0.06, 0.83, body, fontsize=14.5, color=SUB, wrap=True)
    NUM[0] += 1
    fig.text(0.97, 0.03, str(NUM[0]), fontsize=11, color=SUB, ha='right')
    return fig

def ax_of(fig, rect=(0.08, 0.10, 0.86, 0.64)):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return ax

def done(fig):
    pp.savefig(fig)
    plt.close(fig)

def hbars(ax, labels, vals, colors, fmt='{:.3f}', xlim=None):
    y = range(len(labels))
    ax.barh(y, vals, height=0.6, color=colors, zorder=3)
    for i, v in enumerate(vals):
        ax.text(v + (xlim or max(vals)) * 0.012, i, fmt.format(v), va='center', fontsize=12)
    ax.set_yticks(list(y), labels)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    if xlim:
        ax.set_xlim(0, xlim)

def card(fig, x, y, w, h, title, lines, color):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                         fc='white', ec=color, lw=2, transform=fig.transFigure)
    fig.patches.append(box)
    fig.text(x + 0.015, y + h - 0.055, title, fontsize=12.5, fontweight='bold', color=INK)
    for i, ln in enumerate(lines):
        fig.text(x + 0.015, y + h - 0.10 - i * 0.042, ln, fontsize=10.5, color=SUB)

# ---------- 1. Титул ----------
f = slide('Матчинг товаров · Ozon E-CUP 2026',
          'Команда Roma Bazuka · трек 1 · итог на 18.08: локально 0.856, публичный лб 0.4881 — 10 место из 81')
ax = ax_of(f, (0.15, 0.12, 0.7, 0.55))
xs = list(range(8))
ys = [0.257, 0.329, 0.638, 0.789, 0.827, 0.845, 0.8555, 0.8564]
ax.plot(xs, ys, '-o', color=DARKBLUE, lw=3, ms=8, zorder=3)
ax.fill_between(xs, ys, 0.2, color=BLUE, alpha=0.25)
ax.set_xticks(xs, ['prior', 'tfidf', 'lgbm', 'CE', '+LLM', '+len', 'анс', 'стек'], fontsize=11)
ax.set_ylim(0.2, 0.92)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 2. Задача ----------
f = slide('Задача', 'Кандидатные пары ДАНЫ: только классификация «та же карточка или нет». Retrieval не нужен. Вход — текст: имя, категория, атрибуты. Картинок нет.')
card(f, 0.08, 0.35, 0.28, 0.3, 'Карточка A', ['name, category', 'attributes (JSON)'], BLUE)
card(f, 0.64, 0.35, 0.28, 0.3, 'Карточка B', ['name, category', 'attributes (JSON)'], ORANGE)
card(f, 0.40, 0.38, 0.20, 0.24, 'Модель', ['score ∈ [0,1]', 'PR-AUC'], GREEN)
f.text(0.37, 0.49, '→', fontsize=30, color=SUB)
f.text(0.61, 0.49, '←', fontsize=30, color=SUB)
done(f)

# ---------- 3. Данные ----------
f = slide('Данные', 'Ручная разметка 366К пар (метки 0/1) + LLM-разметка 11.2М пар (мягкие 0…1). Вселенные товаров не пересекаются — LLM-пары леик-фри для претрена.')
ax = ax_of(f)
hbars(ax, ['matches (ручные пары)', 'matches_llm (LLM-пары)', 'items_human (товары)', 'items (все товары)'],
      [0.366, 11.2, 0.355, 4.6], [BLUE, PURPLE, BLUE, GRID], fmt='{:.2f}M', xlim=13)
ax.set_xlabel('млн записей')
done(f)

# ---------- 4. Карточка ----------
f = slide('Как выглядит карточка', 'Название свободной формы + свёрнутый JSON атрибутов. Ключи и стиль у продавцов разные — прямое сравнение полей почти невозможно.')
card(f, 0.10, 0.18, 0.80, 0.5, 'шлепанцы adissage f35579 adidas синий 46 eu · [Обувь]',
     ['артикул производителя: 0000200858654-46;  бренд: adidas;',
      'код товара: 100052528706;  материал верха: резина, пластик;',
      'модель: adissage f35579;  пол: унисекс;  размер: 45 46 eu;',
      'размер ru: 45;  размер производителя: 46 eu;  серия: 888 …'], BLUE)
done(f)

# ---------- 5. Пара-позитив ----------
f = slide('Пара-позитив: размер НЕ важен', 'Vans: размер 44,5 RU против EU 38.5 — всё равно матч. Конвенция: «тот же товар» = модель + цвет, размерная сетка игнорируется.')
card(f, 0.07, 0.22, 0.40, 0.42, 'A · vans кеды, 44,5 ru, белый/чёрный',
     ['артикул: vd3hy28 11', 'бренд: vans', 'размер: 44,5 ru', 'цвет: черный, белый'], GREEN)
card(f, 0.53, 0.22, 0.40, 0.42, 'B · кроссовки vans',
     ['бренд: vans', 'размер производителя: eu 38.5', 'российский размер: 37,5', 'цвет: черно-белый'], GREEN)
f.text(0.5, 0.13, 'target = 1', fontsize=18, fontweight='bold', color=GREEN, ha='center')
done(f)

# ---------- 6. Пара-негатив ----------
f = slide('Пара-негатив: цвет решает', 'Fre gamo, обе — сандалии той же марки. Чёрные против жёлтых → разные товары. Негативы в фэшне — «почти тот же» товар.')
card(f, 0.07, 0.22, 0.40, 0.42, 'A · сандалии fre gamo',
     ['бренд: fre gamo', 'название цвета: черный', 'размер производителя: 43', 'пол: мужской'], RED)
card(f, 0.53, 0.22, 0.40, 0.42, 'B · fre gamo / сандалии',
     ['российский размер: 41', 'цвет: желтый', 'комплектация: 1 пара', 'страна: китай'], RED)
f.text(0.5, 0.13, 'target = 0', fontsize=18, fontweight='bold', color=RED, ha='center')
done(f)

# ---------- 7. Позитив-рейты ----------
f = slide('Категории неравны', 'Позитив-рейт от 11.8% (Одежда) до 56% (Детские товары): фэшн намеренно набит трудными негативами «почти тот же товар».')
ax = ax_of(f)
cats = ['Одежда', 'Ювелирные изделия', 'Мебель', 'Электроника', 'Автотовары', 'Дом и сад', 'Канц. товары', 'Красота', 'Бытовая химия', 'Детские товары']
rates = [0.118, 0.124, 0.150, 0.173, 0.175, 0.262, 0.339, 0.363, 0.469, 0.562]
cols = [ORANGE if c in ('Одежда', 'Ювелирные изделия') else BLUE for c in cats]
hbars(ax, cats[::-1], rates[::-1], cols[::-1], fmt='{:.0%}', xlim=0.68)
ax.set_xlabel('доля позитивов')
done(f)

# ---------- 8. Валидация ----------
f = slide('Замороженная валидация', '4 фолда по компонентам связности графа пар: товар никогда не в двух фолдах. Таргеты запинены SHA256 — фолды не двигаются под модель.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.55))
folds = ['fold_01', 'fold_02', 'fold_03', 'fold_04']
ax.bar(folds, [91157, 91474, 91615, 91408], color=[BLUE, GREEN, ORANGE, PURPLE], zorder=3, width=0.6)
for i, v in enumerate([91157, 91474, 91615, 91408]):
    ax.text(i, v + 1200, f'{v:,}', ha='center', fontsize=12)
ax.set_ylim(0, 105000)
ax.yaxis.grid(True, color=GRID)
ax.set_ylabel('пар')
done(f)

# ---------- 9. Метрика ----------
f = slide('Метрика: расшифровали total_prauc', 'API вернул per-category скоры: среднее 20 категорий сходится с total до 4 знака. Метрика — МАКРО по категориям, каждая весит 1/20.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.55))
ax.bar(['среднее 20\nкатегорий', 'total_prauc\n(лб)'], [0.4881, 0.4881], color=[GREEN, DARKBLUE], width=0.45, zorder=3)
for i, v in enumerate([0.48811, 0.48811]):
    ax.text(i, v + 0.01, f'{v:.5f}', ha='center', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.6)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 10. Прогресс ----------
f = slide('Путь скора: 0.638 → 0.856', '30 скоримых итераций на замороженных фолдах. Каждый шаг ниже — отдельный слайд с решением и заплаченной ценой.')
ax = ax_of(f)
steps = ['lgbm\nбейслайн', 'CE tiny', 'CE base', '+LLM\nпретрен', '+len\n224', 'e2+len\n384', 'анс 7\nмоделей', '+LGBM\nстек', '+zs LLM\nстек']
vals = [0.638, 0.580, 0.789, 0.827, 0.838, 0.845, 0.854, 0.8555, 0.8564]
ax.plot(range(len(vals)), vals, '-o', color=DARKBLUE, lw=3, ms=9, zorder=3)
for i, v in enumerate(vals):
    ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=11)
ax.set_xticks(range(len(vals)), steps, fontsize=10.5)
ax.set_ylim(0.55, 0.9)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 11. Бейзлайны ----------
f = slide('Бейзлайны и классика', 'Имена почти не разделяют классы: tfidf 0.33. Дерево на 21 дешёвой фиче — 0.638. KNRM с navec-инициализацией — 0.53.')
ax = ax_of(f)
hbars(ax, ['const_prior', 'attr_jaccard', 'name_tfidf', 'KNRM v2', 'lgbm_cheap'][::-1],
      [0.257, 0.255, 0.329, 0.530, 0.638][::-1],
      [GREEN, GREEN, GREEN, GREEN, DARKBLUE][::-1], xlim=0.8)
done(f)

# ---------- 12. Масштаб ----------
f = slide('Решение 1: масштаб энкодера', 'Кросс-энкодер. rubert-tiny2 хуже lgbm (0.565). rubert-base сразу +0.15 к лучшему бейзлайну. Масштаб — самый большой одиночный рычаг: +0.21.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.55))
ax.bar(['tiny2\n29M', 'base\n180M'], [0.565, 0.789], color=[GRID, BLUE], width=0.5, zorder=3)
for i, v in enumerate([0.565, 0.789]):
    ax.text(i, v + 0.012, f'{v:.3f}', ha='center', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.9)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 13. LLM-претрен ----------
f = slide('Решение 2: претрен на 11.2М LLM-пар', 'BCE на мягких метках чужой вселенной товаров (пересечение с ручной = 0), потом FT на ручных. +0.038. Второй проход претрена — ещё +0.002.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.55))
ax.bar(['hand-only', '+LLM претрен', '+2-я эпоха'], [0.789, 0.827, 0.829],
       color=[BLUE, PURPLE, PURPLE], width=0.55, zorder=3)
for i, v in enumerate([0.789, 0.827, 0.829]):
    ax.text(i, v + 0.004, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')
ax.set_ylim(0.75, 0.86)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 14. Длина ----------
f = slide('Решение 3: длина контекста', 'Самый надёжный рычаг: каждый шаг 128→160→224→288→384 платил +0.002…0.006. Атрибуты до 800 символов, обрезаются только 9%.')
ax = ax_of(f)
lens = [128, 160, 224, 288, 384]
sc = [0.8266, 0.8329, 0.8383, 0.8439, 0.8453]
ax.plot(lens, sc, '-o', color=DARKBLUE, lw=3, ms=9, zorder=3)
for x, v in zip(lens, sc):
    ax.annotate(f'{v:.4f}', (x, v), textcoords='offset points', xytext=(0, 12), ha='center', fontsize=11)
ax.set_xticks(lens)
ax.set_xlabel('max_len (токены)')
ax.set_ylim(0.82, 0.855)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 15. Архитектуры ----------
f = slide('Решение 4: разные архитектуры', 'e5-base и mdeberta-v3 слабее rubert-base поодиночке, но кросс-архитектурный ансамбль даёт +0.006–0.008 — а same-arch бэггинг лишь +0.0005.')
ax = ax_of(f)
hbars(ax, ['mdeberta-v3', 'e5-base', 'rubert-base e2'], [0.8311, 0.8344, 0.8453],
      [PURPLE, ORANGE, BLUE], xlim=1.0)
done(f)

# ---------- 16. Ансамбль ----------
f = slide('Решение 5: ансамбль + стек', '7 членов (3 архитектуры × длины) → ранк-среднее 0.8541. LGBM-стек с 21 фичей — 0.8543. Их ранк-микс — 0.8555.')
ax = ax_of(f)
hbars(ax, ['лучшая одиночная', 'ранк-анс 7', 'LGBM стек', 'final_combo'],
      [0.8453, 0.8541, 0.8543, 0.8555], [BLUE, GREEN, GREEN, DARKBLUE], xlim=1.0)
done(f)

# ---------- 17. zero-shot LLM ----------
f = slide('Решение 6: zero-shot LLM свип', 'Гипотеза Романа: LLM-ризонинг для пар. P(«1») первого токена, промпт с атрибутами. Лучшие — gemma-4-E4B и Qwen3.5-4B ≈ 0.67.')
ax = ax_of(f)
hbars(ax, ['Qwen3-VL-2B', 'gemma-4-E2B', 'gemma-4-E4B', 'Qwen3.5-4B', 'бленд двух'],
      [0.494, 0.611, 0.672, 0.676, 0.692], [ORANGE] * 4 + [DARKBLUE], xlim=0.85)
done(f)

# ---------- 18. Стек с LLM ----------
f = slide('Решение 7: LLM в стек', 'rank(CE)·0.94 + rank(LLM)·0.06 = 0.85638. Дельта +0.0009 — маленькая, но плюс на всех 4 фолдах. Новый топ-1 команды.')
ax = ax_of(f)
folds = ['fold_01', 'fold_02', 'fold_03', 'fold_04']
d = [0.0011, 0.0007, 0.0010, 0.0008]
ax.bar(folds, d, color=GREEN, width=0.55, zorder=3)
for i, v in enumerate(d):
    ax.text(i, v + 0.00004, f'+{v:.4f}', ha='center', fontsize=12)
ax.set_ylim(0, 0.0014)
ax.yaxis.grid(True, color=GRID)
ax.set_ylabel('дельта к final_combo')
done(f)

# ---------- 19. Гипотезы: обзор ----------
f = slide('Проверенные гипотезы', 'Каждая гипотеза — измерение на замороженных фолдах. Критерий значимости: одинаковый знак дельты на всех четырёх фолдах.')
ax = ax_of(f, (0.22, 0.10, 0.72, 0.62))
hyp = ['длина контекста', 'LLM-претрен', 'кросс-арх анс', 'zs-LLM в стек', 'LGBM >> ранк-анс', 'same-arch бэггинг', 'KNRM в стек', 'авто-флип меток']
delt = [0.017, 0.038, 0.007, 0.0009, 0.0002, 0.0005, 0.00006, -0.0014]
cols = [GREEN if d > 0.0006 else (RED if d < 0 else GRID) for d in delt]
y = range(len(hyp))
ax.barh(y, delt, color=cols, height=0.6, zorder=3)
ax.set_yticks(list(y), hyp)
ax.axvline(0, color=SUB, lw=1)
ax.set_xscale('symlog', linthresh=0.001)
ax.set_xlabel('дельта mean_prauc (symlog)')
ax.xaxis.grid(True, color=GRID)
done(f)

# ---------- 20. H: KNRM ----------
f = slide('Гипотеза: KNRM докинет стеку — НЕТ', 'Воспроизвели knrm_name_v2 (OOF 0.531 = 0.530 у Романа). Лучший вес 0.01: +0.00006, знак гуляет по фолдам. Кернелы имён CE уже выучил.')
ax = ax_of(f)
ws = ['0.01', '0.02', '0.03', '0.05', '0.08']
dd = [0.00006, 0.00002, -0.00009, -0.00054, -0.00166]
ax.bar(ws, dd, color=[GREEN if x > 0 else RED for x in dd], width=0.55, zorder=3)
ax.axhline(0, color=SUB, lw=1)
ax.set_xlabel('вес KNRM в стеке')
ax.set_ylabel('дельта')
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 21. H: авто-флип ----------
f = slide('Гипотеза: учиться на исправленных метках — НЕТ', 'Флипнули 1336 пар, где CE и LLM уверенно против метки. −0.0014 на всех фолдах; на парах без флагов ≈ 0. Нужна ручная доразметка.')
ax = ax_of(f)
x = np.arange(4)
ax.bar(x - 0.2, [-0.00188, -0.00126, -0.00126, -0.00129], 0.38, color=RED, zorder=3, label='полный eval')
ax.bar(x + 0.2, [-0.00001, 0.00027, 0.00006, 0.00133], 0.38, color=GRID, zorder=3, label='чистые пары')
ax.set_xticks(x, ['fold_01', 'fold_02', 'fold_03', 'fold_04'])
ax.axhline(0, color=SUB, lw=1)
ax.legend(frameon=False)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 22. Контейнер ----------
f = slide('Посылка ОДС = код-контейнер', 'Лимит 20 минут на весь тест (~366К пар). Наши посылки: fp16, батчи по длине, авто-деградация max_len и тайм-гард на вторую модель.')
card(f, 0.06, 0.30, 0.20, 0.3, 'zip ~0.9 ГБ', ['run.py', 'models/ (fp16)', 'metadata.json'], BLUE)
card(f, 0.31, 0.30, 0.20, 0.3, 'модель 1', ['rubase len192', '~8 мин'], GREEN)
card(f, 0.56, 0.30, 0.20, 0.3, 'тайм-гард', ['осталось время?', 'да → модель 2', 'нет → пишем'], ORANGE)
card(f, 0.79, 0.30, 0.16, 0.3, 'вывод', ['id1,id2,predict', '< 20 мин'], PURPLE)
done(f)

# ---------- 23. Посылки ----------
f = slide('Четыре посылки первого дня', 'Фулл-бленд 3 архитектур показал 0.4902, но не влез в лимит (error). Засчитан пара rubase+mdeb: 0.4881 — 10 место из 81.')
ax = ax_of(f)
labels = ['#1 стендалон rubase192', '#2 бленд rub+e5', '#3 фулл-бленд (error)', '#4 rub+mdeb (зачтена)']
vals = [0.4719, 0.4778, 0.4902, 0.4881]
cols = [BLUE, BLUE, RED, DARKBLUE]
hbars(ax, labels[::-1], vals[::-1], cols[::-1], fmt='{:.4f}', xlim=0.62)
ax.axvline(0.5251, color=SUB, ls='--', lw=1.5)
ax.text(0.527, 3.3, 'топ-1: 0.525', fontsize=11, color=SUB)
done(f)

# ---------- 24. Локал vs лб ----------
f = slide('Локальной валидации можно верить', 'Шкалы разные (0.85 ↔ 0.48), но порядок решений сохраняется: бленд лучше стендалона на +0.006 и локально, и на лб.')
ax = ax_of(f)
x = np.arange(2)
ax.bar(x - 0.2, [0.8407, 0.8457], 0.38, color=BLUE, zorder=3, label='локально (mean_prauc)')
ax.bar(x + 0.2, [0.4719, 0.4778], 0.38, color=ORANGE, zorder=3, label='лб ОДС (total_prauc)')
for xi, v in zip(x - 0.2, [0.8407, 0.8457]): ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)
for xi, v in zip(x + 0.2, [0.4719, 0.4778]): ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=11)
ax.set_xticks(x, ['стендалон', 'бленд'])
ax.set_ylim(0, 1.0)
ax.legend(frameon=False)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 25. Фэшн-провал ----------
f = slide('Главная находка: фэшн лежит на тесте', 'Одежда 0.095 и Обувь 0.099 на тесте против 0.60/0.58 локально. Именно фэшн топит макро-метрику — вес каждой категории 1/20.')
ax = ax_of(f)
cats = ['Одежда', 'Обувь', 'Галантерея', 'Ювелирка', 'Мебель', 'Электроника', 'Автотовары']
loc = [0.596, 0.576, 0.740, 0.615, 0.776, 0.792, 0.728]
tst = [0.095, 0.099, 0.237, 0.260, 0.377, 0.586, 0.827]
x = np.arange(len(cats))
ax.bar(x - 0.2, loc, 0.38, color=BLUE, zorder=3, label='локально')
ax.bar(x + 0.2, tst, 0.38, color=ORANGE, zorder=3, label='тест ОДС')
ax.set_xticks(x, cats, fontsize=11)
ax.legend(frameon=False)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 26. Имена анти-сигнал ----------
f = slide('Почему фэшн трудный: имена — анти-сигнал', 'В Обуви у негативов перекрытие имён ВЫШЕ, чем у позитивов (0.337 против 0.308). В Электронике наоборот: имя решает (+0.107).')
ax = ax_of(f)
cats = ['Обувь', 'Одежда', 'Бытовая химия', 'Электроника']
gaps = [-0.029, 0.024, 0.051, 0.107]
ax.bar(cats, gaps, color=[RED if g < 0 else GREEN for g in gaps], width=0.55, zorder=3)
ax.axhline(0, color=SUB, lw=1)
ax.set_ylabel('jaccard(pos) − jaccard(neg)')
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 27. Цвет/артикул ----------
f = slide('Что реально разделяет пары в фэшне', 'Совпадение цвета: 64% у позитивов против 40% у негативов. Артикул почти безошибочен (37% против 3%), но есть лишь у 5–37% пар.')
ax = ax_of(f)
x = np.arange(3)
ax.bar(x - 0.2, [0.638, 0.611, 0.368], 0.38, color=GREEN, zorder=3, label='P(совпал | матч)')
ax.bar(x + 0.2, [0.397, 0.322, 0.032], 0.38, color=RED, zorder=3, label='P(совпал | не матч)')
ax.set_xticks(x, ['цвет · Обувь', 'цвет · Одежда', 'артикул · Ювелирка'])
ax.legend(frameon=False)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 28. Шум меток ----------
f = slide('И разметка местами битая', 'Одинаковый артикул lbjx010013, те же габариты и цвет — метка 0. Такие дубли-с-меткой-0 мы ловим согласием двух независимых моделей.')
card(f, 0.07, 0.22, 0.40, 0.42, 'A · сумка baseus easyjourney dark grey',
     ['артикул: lbjx010013', 'высота: 11.5 см · ширина: 17.5 см', 'цвет товара: серый'], ORANGE)
card(f, 0.53, 0.22, 0.40, 0.42, 'B · сумка baseus easyjourney серая',
     ['артикул: lbjx010013', 'высота: 11.5 см · ширина: 17.5 см', 'цвет товара: серый'], ORANGE)
f.text(0.5, 0.13, 'target = 0 ?!', fontsize=18, fontweight='bold', color=RED, ha='center')
done(f)

# ---------- 29. Доразметка ----------
f = slide('1336 пар на доразметку', 'CE-ансамбль и zero-shot LLM уверенно согласны ПРОТИВ метки: 993 «на деле матч» + 343 «на деле разные». CSV и паркет предиктов — в репе.')
ax = ax_of(f)
cats = ['Бытовая химия', 'Детские товары', 'Хобби', 'Бытовая техника', 'Животные', 'Муз. инструменты']
fp = [123, 120, 113, 91, 73, 65]
fn = [10, 9, 18, 6, 18, 34]
x = np.arange(len(cats))
ax.bar(x - 0.2, fp, 0.38, color=BLUE, zorder=3, label='метка 0, модели: матч')
ax.bar(x + 0.2, fn, 0.38, color=ORANGE, zorder=3, label='метка 1, модели: разные')
ax.set_xticks(x, cats, fontsize=10.5)
ax.legend(frameon=False)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 30. Направления ошибок ----------
f = slide('Где теряем total_prauc', 'Разрыв (локальный − тест) на вес 1/20: фэшн-категории дают половину всей потери метрики. Обувь и Одежда — по 0.025 total каждая.')
ax = ax_of(f)
cats = ['Одежда', 'Обувь', 'Галантерея', 'Ювелирка', 'Мебель', 'Спорт', 'Дом и сад']
loss = [(0.596 - 0.095) / 20, (0.576 - 0.099) / 20, (0.740 - 0.237) / 20, (0.615 - 0.260) / 20,
        (0.776 - 0.377) / 20, (0.750 - 0.419) / 20, (0.851 - 0.420) / 20]
hbars(ax, cats[::-1], loss[::-1], [ORANGE if c in ('Одежда', 'Обувь', 'Галантерея', 'Ювелирка') else BLUE for c in cats][::-1], fmt='{:.4f}', xlim=0.032)
ax.set_xlabel('потеря total_prauc')
done(f)

# ---------- 31. Ошибки: конвенция ----------
f = slide('Направление 1: конвенция матча', 'Модель должна выучить «модель+цвет, размер игнорируй». В работе: приоритет атрибутов + символьные сравнения цвета/артикула прямо в тексте пары.')
card(f, 0.10, 0.25, 0.80, 0.38, 'Новый текст пары (эксперимент prio)',
     ['бренд → модель/артикул → цвет → материал → …  (вместо алфавита)',
      'фэшн: размерные атрибуты выброшены',
      '@@ сравнение: цвет=различен; артикул=неизвестно'], GREEN)
done(f)

# ---------- 32. Ошибки: дистилляция ----------
f = slide('Направление 2: дистилляция ансамбля', 'Ансамбль 0.856 не влезает в 20 минут. Студент rubert-base учится на 0.7·стек + 0.3·метка — переносим ансамбль в одну модель контейнера.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.52))
ax.bar(['одиночная\n(сейчас в лб)', 'ансамбль\n(не влезает)', 'студент\n(цель)'],
       [0.841, 0.856, 0.850], color=[BLUE, GRID, GREEN], width=0.55, zorder=3)
for i, v in enumerate([0.841, 0.856, 0.850]):
    ax.text(i, v + 0.001, f'{v:.3f}' + ('?' if i == 2 else ''), ha='center', fontsize=13, fontweight='bold')
ax.set_ylim(0.83, 0.865)
ax.yaxis.grid(True, color=GRID)
done(f)

# ---------- 33. Ошибки: метки ----------
f = slide('Направление 3: чистые метки', 'Авто-флип не сработал (−0.0014) — исправления должен подтвердить человек. 1336 пар ждут ручной доразметки, приоритет — фэшн и Галантерея.')
ax = ax_of(f, (0.15, 0.14, 0.7, 0.52))
ax.bar(['авто-флип', 'ручная доразметка'], [-0.0014, 0.0], color=[RED, GRID], width=0.5, zorder=3)
ax.text(0, -0.0013, '−0.0014\nизмерено', ha='center', fontsize=12, color=INK)
ax.text(1, 0.0002, '? — следующий шаг', ha='center', fontsize=12, color=SUB)
ax.axhline(0, color=SUB, lw=1)
ax.set_ylim(-0.002, 0.001)
done(f)

# ---------- 34. План ----------
f = slide('План до дедлайна 30.08', 'Сегодня: спец-фэшн посылка + дистилляция. Дальше: ручная доразметка, len384 в контейнер через деградацию, финальный отбор 2 посылок.')
ax = ax_of(f, (0.08, 0.12, 0.84, 0.58))
tasks = ['спец-фэшн тексты', 'дистилляция студента', 'ручная доразметка', 'len384 контейнер', 'отбор 2 финальных']
start = [0, 0, 1, 2, 9]
dur = [1, 1, 4, 2, 1]
cols = [GREEN, GREEN, ORANGE, BLUE, PURPLE]
for i, (t, s, d, c) in enumerate(zip(tasks, start, dur, cols)):
    ax.barh(len(tasks) - 1 - i, d, left=s, height=0.55, color=c, zorder=3)
ax.set_yticks(range(len(tasks)), tasks[::-1])
ax.set_xticks(range(0, 11, 2), [f'{20 + x}.08' for x in range(0, 11, 2)])
ax.xaxis.grid(True, color=GRID)
ax.set_xlim(0, 10.5)
done(f)

pp.close()
print('slides:', NUM[0], '->', PDF, round(os.path.getsize(PDF) / 1e6, 1), 'MB')

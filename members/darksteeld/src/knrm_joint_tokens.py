"""Токенизация совместной сети: общая для обучения, дообучения и контейнера.

Стемминг здесь — **свойство словаря**, а не опция вызова. Флаг едет в чекпоинте
рядом с ``token_id``, и все три потребителя (предобучение, дообучение,
контейнер) берут его оттуда. Иначе достаточно забыть один флаг в одном месте,
чтобы инференс токенизировал не так, как обучение, и модель молча деградировала
до случайной — ровно тот класс ошибок, который уже стоил нам одного прогона на
несовпадении устройств.

Snowball сводит только русские суффиксы. Артикулы, модели и размеры (``gsr``,
``12v``, ``ts830p``) проходят нетронутыми — что и требуется: на них держится
ядро точного совпадения, и склеивать их с чем-либо нельзя.

Замер по всему каталогу (13 397 761 товар): стемминг сжимает словарь названий в
1.08x, значений в 1.04x, ключей в 1.75x — то есть работает почти только на
нормальной русской речи в ключах, а товарные тексты состоят в основном из
латиницы и цифр.
"""

from __future__ import annotations

from knrm_attrs_model import parse_attributes as _parse_attributes
from knrm_model import tokenize as _tokenize

_STEM_CACHE: dict[str, str] = {}
_STEMMER = None


def _stemmer():
    global _STEMMER
    if _STEMMER is None:
        import Stemmer

        _STEMMER = Stemmer.Stemmer("russian")
    return _STEMMER


def stem(token: str) -> str:
    """Стем токена с кэшем: вхождений сотни миллионов, типов — миллионы."""
    value = _STEM_CACHE.get(token)
    if value is None:
        value = _stemmer().stemWord(token)
        _STEM_CACHE[token] = value
    return value


def tokenize(name: str, stemming: bool = False) -> list[str]:
    tokens = _tokenize(name)
    return [stem(t) for t in tokens] if stemming else tokens


def parse_attributes(raw: str, stemming: bool = False) -> list[tuple[list[str], list[str]]]:
    parsed = _parse_attributes(raw)
    if not stemming:
        return parsed
    return [([stem(k) for k in keys], [stem(v) for v in values]) for keys, values in parsed]


def item_tokens(name: str, attributes: str, stemming: bool = False) -> set[str]:
    """Все токены товара одним множеством — для построения словаря."""
    tokens = set(tokenize(name, stemming))
    for keys, values in parse_attributes(attributes, stemming):
        tokens.update(keys)
        tokens.update(values)
    return tokens

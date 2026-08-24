from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

NUM = r"\d+(?:[.,]\d+)?"
DIM_RE = re.compile(rf"(?<!\w)({NUM})\s*[xх×]\s*({NUM})\s*(мм|mm|см|cm|м|m)(?!\w)", re.I)
QTY_RE = re.compile(
    rf"(?<!\w)({NUM})\s*(килограмм(?:а|ов)?|кг|kg|грамм(?:а|ов)?|гр\.?|г|g|"
    rf"миллилитр(?:а|ов)?|мл|ml|литр(?:а|ов)?|л|l|штук(?:а|и)?|шт\.?|pcs?|pieces?)(?!\w)", re.I)
SIZE_WORD_RE = re.compile(r"(?<!\w)(размер|size)\s*[:=]?\s*(xxxl|xxl|xl|xs|s|m|l|\d{1,3})(?!\w)", re.I)
SIZE_SCALE_RE = re.compile(r"(?<!\w)(\d{1,3})\s*(ru|eu|росс(?:ия)?|eur)(?!\w)", re.I)


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    family: str
    source: str
    normalized: str
    corrupted: str


def compact_number(value: Decimal) -> str:
    plain = format(value.normalize(), "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    candidates = [plain]
    if value != 0 and value == value.to_integral():
        s = str(abs(int(value)))
        zeros = len(s) - len(s.rstrip("0"))
        sign = "-" if value < 0 else ""
        if zeros:
            candidates.append(f"{sign}{s[:-zeros]}e{zeros}")
    return min(candidates, key=lambda x: (len(x), x))


def pad(value: str, width: int) -> str | None:
    return value + " " * (width - len(value)) if len(value) <= width else None


def corrupt(value: str) -> str:
    for i, char in enumerate(value):
        if char.isdigit():
            return value[:i] + str((int(char) + 5) % 10) + value[i + 1:]
    swaps = {"s": "l", "m": "s", "l": "m", "x": "s"}
    for i, char in enumerate(value.lower()):
        if char in swaps:
            return value[:i] + swaps[char] + value[i + 1:]
    return ("z" + value[1:]) if value else value


def quantity(match: re.Match[str]) -> tuple[str, str]:
    number = Decimal(match.group(1).replace(",", "."))
    unit = match.group(2).lower().rstrip(".")
    if unit in {"килограмм", "килограмма", "килограммов", "кг", "kg"}:
        return compact_number(number * 1000) + "g", "mass"
    if unit in {"грамм", "грамма", "граммов", "гр", "г", "g"}:
        return compact_number(number) + "g", "mass"
    if unit in {"литр", "литра", "литров", "л", "l"}:
        return compact_number(number * 1000) + "ml", "volume"
    if unit in {"миллилитр", "миллилитра", "миллилитров", "мл", "ml"}:
        return compact_number(number) + "ml", "volume"
    return compact_number(number) + "pc", "count"


def candidates(text: str):
    for match in DIM_RE.finditer(text):
        yield match, f"{match.group(1).replace(',', '.')}x{match.group(2).replace(',', '.')}{match.group(3).lower().translate(str.maketrans({'х':'x','×':'x','м':'m','с':'c'}))}", "dimension"
    for match in SIZE_WORD_RE.finditer(text):
        yield match, "sz=" + match.group(2).lower(), "fashion_size"
    for match in SIZE_SCALE_RE.finditer(text):
        scale = "eu" if match.group(2).lower() in {"eu", "eur"} else "ru"
        yield match, f"{scale}={match.group(1)}", "fashion_size"
    for match in QTY_RE.finditer(text):
        try:
            value, family = quantity(match)
        except InvalidOperation:
            continue
        yield match, value, family


def normalize(text: str | None) -> tuple[str, str, list[Edit]]:
    original = text or ""
    accepted = []
    occupied: list[tuple[int, int]] = []
    for match, value, family in sorted(candidates(original), key=lambda x: (x[0].start(), -(x[0].end()-x[0].start()))):
        start, end = match.span()
        if any(start < b and end > a for a, b in occupied):
            continue
        norm = pad(value, end - start)
        if norm is None or norm == original[start:end]:
            continue
        bad = corrupt(norm)
        assert len(norm) == len(bad) == end - start and bad != norm
        accepted.append(Edit(start, end, family, original[start:end], norm, bad))
        occupied.append((start, end))
    norm_chars, bad_chars = list(original), list(original)
    for edit in accepted:
        norm_chars[edit.start:edit.end] = edit.normalized
        bad_chars[edit.start:edit.end] = edit.corrupted
    normalized, corrupted = "".join(norm_chars), "".join(bad_chars)
    assert len(original) == len(normalized) == len(corrupted)
    return normalized, corrupted, accepted


def self_check() -> None:
    examples = [
        "масса 1 кг и 1000 г", "объем 0,5 л / 500 мл", "размер 10х20 см",
        "упаковка 10 штук", "Размер: XL", "42 RU",
    ]
    for source in examples:
        good, bad, edits = normalize(source)
        assert edits and len(source) == len(good) == len(bad) and good != bad
    a, _, _ = normalize("1 кг")
    b, _, _ = normalize("1000 г")
    assert a.strip() == b.strip() == "1e3g"


if __name__ == "__main__":
    self_check()
    print("self-check passed")

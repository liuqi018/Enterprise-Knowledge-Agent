import re


CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

CHINESE_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}


def extract_amount(text: str) -> float:
    """Extract a RMB amount from common Chinese business expressions."""
    if not text:
        return 0.0

    normalized = text.replace(",", "").replace("，", "")
    numeric = _extract_numeric_amount(normalized)
    if numeric > 0:
        return numeric
    return _extract_chinese_amount(normalized)


def normalize_amount(value, fallback_text: str = "") -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        parsed = extract_amount(value)
        if parsed > 0:
            return parsed
    return extract_amount(fallback_text)


def _extract_numeric_amount(text: str) -> float:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(万元|万块|万人民币|万)",
        r"(\d+(?:\.\d+)?)\s*(千元|千块|千)",
        r"(\d+(?:\.\d+)?)\s*(元|块|人民币)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2) or "元"
        if unit.startswith("万"):
            return value * 10000
        if unit.startswith("千"):
            return value * 1000
        return value
    return 0.0


def _extract_chinese_amount(text: str) -> float:
    match = re.search(r"([零一二两三四五六七八九十百千万]+)\s*(元|块|人民币)?", text)
    if not match:
        return 0.0
    return float(chinese_number_to_int(match.group(1)))


def chinese_number_to_int(text: str) -> int:
    if not text:
        return 0

    total = 0
    section = 0
    number = 0
    for char in text:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            continue
        unit = CHINESE_UNITS.get(char)
        if not unit:
            continue
        if unit == 10000:
            section = (section + number) or 1
            total += section * unit
            section = 0
            number = 0
            continue
        section += (number or 1) * unit
        number = 0
    return total + section + number

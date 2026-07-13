from __future__ import annotations

import json
import re
from typing import Any


class JSONGuardError(ValueError):
    pass


def _strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_balanced(text: str, open_char: str, close_char: str) -> str:
    start = text.find(open_char)
    if start < 0:
        raise JSONGuardError(f"no JSON payload starting with {open_char!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise JSONGuardError(f"unclosed JSON payload starting with {open_char!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    payload = _extract_balanced(_strip_code_fence(text), "{", "}")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise JSONGuardError("JSON payload is not an object")
    return data


def parse_json_array(text: str) -> list[Any]:
    cleaned = _strip_code_fence(text)
    try:
        payload = _extract_balanced(cleaned, "[", "]")
        data = json.loads(payload)
    except JSONGuardError:
        obj = parse_json_object(cleaned)
        data = obj.get("plan") or obj.get("tools") or obj.get("steps") or obj.get("actions")
    if not isinstance(data, list):
        raise JSONGuardError("JSON payload is not an array")
    return data


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def coerce_float(value: Any, default: float = 0.0, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))

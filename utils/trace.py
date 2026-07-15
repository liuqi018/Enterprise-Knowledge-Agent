import json
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator


_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def current_trace_id() -> str:
    return _trace_id.get() or "-"


@contextmanager
def trace_scope(trace_id: str = None) -> Iterator[str]:
    active_trace_id = trace_id or new_trace_id()
    token = _trace_id.set(active_trace_id)
    try:
        yield active_trace_id
    finally:
        try:
            _trace_id.reset(token)
        except ValueError:
            # StreamingResponse may resume a sync generator in a different worker context.
            # Dropping the reset is safer than failing the response after work completed.
            pass


def now_ms() -> float:
    return time.perf_counter() * 1000


def elapsed_ms(start_ms: float) -> int:
    return int(now_ms() - start_ms)


def short_text(text: Any, limit: int = 120) -> str:
    value = "" if text is None else str(text)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def log_trace(logger, event: str, trace_id: str = None, **fields: Any) -> None:
    safe_fields = _safe_fields(fields)
    payload = {"trace_id": trace_id or current_trace_id(), "event": event, **safe_fields}
    logger.info("[Trace] %s", json.dumps(payload, ensure_ascii=False))


def _safe_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            result[key] = round(value, 4)
        elif isinstance(value, (int, bool)):
            result[key] = value
        else:
            result[key] = short_text(value)
    return result

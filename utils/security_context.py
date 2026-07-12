from contextvars import ContextVar
from typing import Optional

current_tenant_id: ContextVar[Optional[str]] = ContextVar("current_tenant_id", default=None)


def get_current_tenant_id() -> Optional[str]:
    return current_tenant_id.get()

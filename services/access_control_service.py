from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from AIRAGAgent.config.settings import settings
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path


DEFAULT_ACCESS_CONFIG: Dict[str, Any] = {
    "restricted_message": "抱歉，你暂时没有这个权限哦~",
    "sensitive_detail_keywords": [
        "金额",
        "标准",
        "明细",
        "名单",
        "具体数额",
        "工资条",
        "密码",
        "账号",
        "权限开通",
        "数据库",
        "密级",
    ],
    "domains": {
        "finance": {
            "name": "财务制度",
            "keywords": ["财务", "付款", "收款", "资金", "预算", "发票", "借款", "报销"],
        },
        "salary_performance": {
            "name": "薪酬绩效制度",
            "keywords": ["薪资", "工资", "绩效", "奖金", "提成", "调薪", "考核"],
        },
        "security": {
            "name": "信息安全制度",
            "keywords": ["信息安全", "权限", "账号", "访问控制", "保密", "数据", "系统权限"],
        },
    },
    "roles": {
        "admin": {"default": "allow"},
        "user": {
            "default": "allow",
            "domains": {
                "finance": "general_only",
                "salary_performance": "general_only",
                "security": "general_only",
            },
        },
    },
}

VALID_ACTIONS = {"allow", "general_only", "deny"}


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str = ""
    domain: str = ""
    message: str = ""
    action: str = ""
    role: str = ""


class AccessControlService:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or settings.ACCESS_CONTROL_CONFIG_PATH
        self.config = self._load_config()

    def reload(self) -> None:
        self.config = self._load_config()

    def can_access_query(self, query: str, role: str = "user") -> AccessDecision:
        normalized_role = (role or "user").strip().lower()
        domain = self._detect_sensitive_domain(query)
        if not domain:
            return AccessDecision(True, reason="non_sensitive", role=normalized_role)

        action = self._role_action(normalized_role, domain)
        if action == "allow":
            return AccessDecision(
                True,
                reason="domain_allowed",
                domain=domain,
                action=action,
                role=normalized_role,
            )

        if action == "deny":
            return self._deny(
                reason="sensitive_domain_restricted",
                domain=domain,
                action=action,
                role=normalized_role,
            )

        if action == "general_only" and self._asks_sensitive_detail(query):
            return self._deny(
                reason="sensitive_detail_restricted",
                domain=domain,
                action=action,
                role=normalized_role,
            )

        return AccessDecision(
            True,
            reason="sensitive_domain_general_allowed",
            domain=domain,
            action=action,
            role=normalized_role,
        )

    def _deny(self, reason: str, domain: str, action: str, role: str) -> AccessDecision:
        return AccessDecision(
            allowed=False,
            reason=reason,
            domain=domain,
            action=action,
            role=role,
            message=self.config.get("restricted_message") or DEFAULT_ACCESS_CONFIG["restricted_message"],
        )

    def _detect_sensitive_domain(self, query: str) -> str:
        for domain, domain_config in self._domains().items():
            keywords = self._string_list(domain_config.get("keywords"))
            if any(keyword and keyword in query for keyword in keywords):
                return domain
        return ""

    def _asks_sensitive_detail(self, query: str) -> bool:
        return any(keyword and keyword in query for keyword in self._detail_keywords())

    def _role_action(self, role: str, domain: str) -> str:
        roles = self.config.get("roles") or {}
        role_config = roles.get(role) or roles.get("user") or {}
        domain_actions = role_config.get("domains") or {}
        action = domain_actions.get(domain) or role_config.get("default") or "allow"
        if action not in VALID_ACTIONS:
            logger.warning("[access control] invalid action=%s role=%s domain=%s, fallback to general_only", action, role, domain)
            return "general_only"
        return action

    def _load_config(self) -> Dict[str, Any]:
        path = Path(self.config_path)
        if not path.is_absolute():
            path = Path(get_abs_path(str(path)))
        try:
            with path.open("r", encoding="utf-8") as f:
                loaded = yaml.load(f, Loader=yaml.FullLoader) or {}
            return self._merge_config(DEFAULT_ACCESS_CONFIG, loaded)
        except Exception as exc:
            logger.warning("[access control] failed to load %s, using defaults: %s", path, exc)
            return DEFAULT_ACCESS_CONFIG

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _domains(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get("domains") or {}

    def _detail_keywords(self) -> List[str]:
        return self._string_list(self.config.get("sensitive_detail_keywords"))

    def _string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


access_control_service = AccessControlService()

from __future__ import annotations

from dataclasses import dataclass


SENSITIVE_DOMAIN_KEYWORDS = {
    "finance": ["财务", "付款", "收款", "资金", "预算", "发票", "借款"],
    "salary_performance": ["薪资", "工资", "绩效", "奖金", "提成", "调薪", "考核"],
    "security": ["信息安全", "权限", "账号", "访问控制", "保密", "数据", "系统权限"],
}

SENSITIVE_DETAIL_KEYWORDS = [
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
]


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str = ""
    domain: str = ""
    message: str = ""


class AccessControlService:
    def can_access_query(self, query: str, role: str = "user") -> AccessDecision:
        if role == "admin":
            return AccessDecision(True)

        domain = self._detect_sensitive_domain(query)
        if not domain:
            return AccessDecision(True)

        if self._asks_sensitive_detail(query):
            return AccessDecision(
                allowed=False,
                reason="sensitive_detail_restricted",
                domain=domain,
                message=(
                    "该问题涉及财务、薪资绩效或信息安全等敏感制度细节。"
                    "当前账号无权直接查询具体明细，请联系管理员或对应归口部门确认。"
                ),
            )

        return AccessDecision(
            allowed=True,
            reason="sensitive_domain_general_allowed",
            domain=domain,
        )

    def _detect_sensitive_domain(self, query: str) -> str:
        for domain, keywords in SENSITIVE_DOMAIN_KEYWORDS.items():
            if any(keyword in query for keyword in keywords):
                return domain
        return ""

    def _asks_sensitive_detail(self, query: str) -> bool:
        return any(keyword in query for keyword in SENSITIVE_DETAIL_KEYWORDS)


access_control_service = AccessControlService()

import argparse
import sys
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from AIRAGAgent.services.access_control_service import AccessControlService  # noqa: E402


DEFAULT_QUERIES = [
    "财务报销流程怎么走？",
    "财务报销金额标准是多少？",
    "员工工资明细怎么查看？",
    "账号权限开通需要谁审批？",
    "新员工入职需要准备哪些材料？",
]


def main():
    parser = argparse.ArgumentParser(description="Check configurable access-control decisions.")
    parser.add_argument("--role", default="user", help="Role name, for example: user/admin/hr/finance/security")
    parser.add_argument("--query", action="append", help="Query to check. Can be passed multiple times.")
    args = parser.parse_args()

    service = AccessControlService()
    queries = args.query or DEFAULT_QUERIES
    for query in queries:
        decision = service.can_access_query(query, args.role)
        print(
            f"role={args.role} allowed={decision.allowed} "
            f"domain={decision.domain or '-'} action={decision.action or '-'} "
            f"reason={decision.reason or '-'} query={query}"
        )


if __name__ == "__main__":
    main()

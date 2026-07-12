# -*- coding: utf-8 -*-
"""Classify enterprise knowledge documents into semantic folders.

The script only moves files under data/enterprise. It is safe to rerun:
already-classified files are skipped, destination folders are created
automatically, and duplicate names are preserved with a numeric suffix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DOCUMENT_SUFFIXES = {".doc", ".docx", ".pdf", ".txt"}

TARGETS = {
    "procurement": "policies/procurement",
    "leave_attendance": "policies/leave_attendance",
    "onboarding": "policies/onboarding",
    "reimbursement": "policies/reimbursement",
    "security": "policies/security",
    "administration": "policies/administration",
    "finance": "policies/finance",
    "salary_performance": "policies/salary_performance",
    "warehouse": "policies/warehouse",
    "sales": "policies/sales",
    "hr": "policies/hr",
    "general_policy": "policies/general",
    "contract_template": "templates/contracts",
    "form_template": "templates/forms",
    "job_description": "references/job_descriptions",
    "training_material": "references/training_materials",
    "org_structure": "references/org_structure",
    "culture_team": "references/culture_team",
    "meeting_report": "references/meeting_reports",
    "reference": "references/general",
}


DOMAIN_RULES = [
    ("procurement", ["采购", "请购", "供应商", "询价", "报价", "招标", "物资"]),
    ("leave_attendance", ["请假", "休假", "考勤", "病假", "年假", "调休", "加班"]),
    ("onboarding", ["入职", "转正", "试用期", "招聘", "录用", "离职", "辞退"]),
    ("reimbursement", ["报销", "差旅", "出差", "费用", "发票", "借款", "补贴"]),
    ("security", ["保密", "信息安全", "权限", "账号", "数据安全", "安全生产", "消防", "环保"]),
    ("administration", ["办公室", "办公用品", "行政", "档案", "印章", "车辆", "会议", "接待", "6s", "7s"]),
    ("finance", ["财务", "资金", "现金", "预算", "固定资产", "资产", "股权", "股份", "分红", "入股", "股东"]),
    ("salary_performance", ["薪酬", "薪资", "绩效", "提成", "奖金", "年终奖", "激励", "奖惩"]),
    ("warehouse", ["仓库", "库存", "入库", "出库"]),
    ("sales", ["销售", "客户", "门店", "直播", "售后", "工单"]),
    ("hr", ["人力资源", "人事", "员工", "人才", "晋升", "培养", "守则", "劳动"]),
]


def normalize_name(path: Path) -> str:
    return path.stem.lower().replace("（", "(").replace("）", ")")


def is_policy_name(name: str) -> bool:
    return any(term in name for term in ["制度", "办法", "规定", "规章", "细则", "流程", "机制", "规范", "管理"])


def classify_file(path: Path) -> str:
    name = normalize_name(path)
    policy_name = is_policy_name(name)

    if policy_name:
        for target, keywords in DOMAIN_RULES:
            if any(keyword in name for keyword in keywords):
                return target
        return "general_policy"

    if any(term in name for term in ["组织架构", "部门职责"]):
        return "org_structure"
    if any(term in name for term in ["岗位职责", "职责说明", "岗位责任"]):
        return "job_description"
    if any(term in name for term in ["培训", "教材", "员工手册"]):
        return "training_material"
    if any(term in name for term in ["团建", "年会", "企业文化"]):
        return "culture_team"
    if any(term in name for term in ["工作汇报", "总结", "早会"]):
        return "meeting_report"

    if not policy_name:
        if any(term in name for term in ["合同", "协议", "承诺书", "授权委托书"]):
            return "contract_template"
        if any(term in name for term in ["申请", "申请表", "登记表", "清单", "模板", "确认书", "通知书", "流程图", "方案"]):
            return "form_template"

    for target, keywords in DOMAIN_RULES:
        if any(keyword in name for keyword in keywords):
            return target

    return "reference"


def iter_root_files(data_dir: Path):
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and not path.name.startswith("~$") and path.suffix.lower() in DOCUMENT_SUFFIXES:
            yield path


def unique_destination(source: Path, path: Path) -> Path:
    if not path.exists():
        return path
    if path.stat().st_size == source.stat().st_size:
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def build_plan(data_dir: Path):
    moves = []
    for source in iter_root_files(data_dir):
        category = classify_file(source)
        target_dir = data_dir / TARGETS[category]
        destination = unique_destination(source, target_dir / source.name)
        moves.append(
            {
                "source": source,
                "destination": destination,
                "category": category,
                "target_folder": TARGETS[category],
            }
        )
    return moves


def write_report(data_dir: Path, moves, applied: bool) -> Path:
    report_path = data_dir.parent / "enterprise_classification_report.json"
    serializable = [
        {
            "source": str(item["source"]),
            "destination": str(item["destination"]),
            "category": item["category"],
            "target_folder": item["target_folder"],
        }
        for item in moves
    ]
    report_path.write_text(
        json.dumps({"applied": applied, "moves": serializable}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify documents under data/enterprise.")
    parser.add_argument("--data-dir", default="data/enterprise", help="Enterprise document root.")
    parser.add_argument("--apply", action="store_true", help="Move files. Without this flag, only preview.")
    parser.add_argument("--report", action="store_true", help="Write data/enterprise_classification_report.json.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"data dir not found: {data_dir}")

    moves = build_plan(data_dir)
    counts = Counter(item["target_folder"] for item in moves)
    samples = defaultdict(list)
    for item in moves:
        if len(samples[item["target_folder"]]) < 5:
            samples[item["target_folder"]].append(item["source"].name)

    print(f"待分类文件数: {len(moves)}")
    for folder, count in sorted(counts.items()):
        print(f"- {folder}: {count}")
        for sample in samples[folder]:
            print(f"  · {sample}")

    if args.apply:
        for item in moves:
            item["destination"].parent.mkdir(parents=True, exist_ok=True)
            if item["destination"].exists() and item["destination"].stat().st_size == item["source"].stat().st_size:
                item["source"].unlink()
            else:
                shutil.move(str(item["source"]), str(item["destination"]))
        print(f"已移动文件数: {len(moves)}")
    else:
        print("当前为预览模式；确认后加 --apply 执行移动。")

    if args.report:
        report_path = write_report(data_dir, moves, args.apply)
        print(f"分类报告: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

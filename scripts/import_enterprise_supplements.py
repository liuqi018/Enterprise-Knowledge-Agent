# -*- coding: utf-8 -*-
"""Import high-value supplemental policy documents into data/enterprise.

This script copies, never moves, files from the larger 01-10 corpus into the
curated enterprise knowledge base. It intentionally skips industry cases,
forms, records, templates, temporary files, and obvious duplicates.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


DOCUMENT_SUFFIXES = {".doc", ".docx", ".pdf", ".txt"}

SOURCE_RULES = [
    {
        "source": "03-财务管理制度",
        "target": "policies/finance",
        "limit": 80,
        "include": [
            "财务", "资金", "现金", "预算", "付款", "收款", "应收", "应付",
            "发票", "审批", "资产", "盘点",
        ],
        "exclude": ["报销", "差旅", "借款", "费用开支"],
    },
    {
        "source": "03-财务管理制度",
        "target": "policies/reimbursement",
        "limit": 35,
        "include": ["报销", "差旅", "借款", "费用", "开支", "发票"],
    },
    {
        "source": "09-生产管理制度",
        "target": "policies/production",
        "limit": 80,
        "include": ["生产", "车间", "调度", "物料", "设备", "工艺", "异常", "订单", "计划", "开停"],
    },
    {
        "source": "09-生产管理制度",
        "target": "policies/quality",
        "limit": 45,
        "include": ["质量", "检验", "事故", "控制", "检查", "不合格", "纠正", "预防"],
    },
    {
        "source": "06质量管理体系",
        "target": "policies/quality",
        "limit": 25,
        "include": ["质量体系", "质量控制", "质量管理", "供方", "评审", "控制程序"],
    },
    {
        "source": "08-研发管理制度",
        "target": "policies/research",
        "limit": 60,
        "include": ["研发", "项目", "立项", "结项", "专利", "配置管理", "开发规范", "测试", "产品"],
    },
    {
        "source": "05信息技术服务管理",
        "target": "policies/it_service",
        "limit": 30,
        "include": ["事件", "问题", "变更", "配置", "服务", "容量", "连续性", "供应商", "业务关系"],
    },
    {
        "source": "10-信息安全管理",
        "target": "policies/security",
        "limit": 45,
        "include": ["信息安全", "访问控制", "风险", "备份", "恶意软件", "物理访问", "第三方", "业务持续", "法律法规"],
    },
    {
        "source": "02-人事管理制度",
        "target": "policies/salary_performance",
        "limit": 35,
        "include": ["薪酬", "薪资", "绩效", "考核", "奖金", "提成"],
    },
    {
        "source": "02-人事管理制度",
        "target": "policies/hr",
        "limit": 19,
        "include": ["招聘", "培训", "晋升", "员工关系", "奖惩", "人力资源"],
        "exclude": ["doc ", "DOC ", "模版"],
    },
]

EXCLUDE_TERMS = [
    ".ds_store", "~$", ".~", "行业案例", "案例", "表格", "模板", "大全",
    "汇总", "报告", "总结", "台帐", "台账", "记录", "申请", "登记",
    "通知单", "计划表", "签到", "讲话", "发言", "考试", "题库", "简表",
    "合作社", "农村信用", "医院", "医疗", "酒店", "餐饮", "房地产",
    "物业", "家政", "旅行社", "美容", "医药", "药业", "学校", "教育",
    "建筑", "项目部", "装饰", "电力", "电商", "商业银行", "新能源",
    "制造业", "机械", "加工厂", "广告", "金融", "财务部", "〜", "～",
]

POLICY_TERMS = ["制度", "办法", "规定", "流程", "规范", "细则", "手册", "标准", "程序"]

PREFERRED_TERMS = [
    "标准制度", "管理制度", "管理办法", "审批", "流程", "规范", "控制程序",
    "管理手册", "管理细则",
]


def normalize_name(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"\(doc\s*\d+\)|（doc\s*\d+）", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"v\d+(\.\d+)*|版本|修订版|完整版|最新版|打印版|副本", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[\s_\-—（）()【】\[\]《》]+", "", stem)
    stem = re.sub(r"^\d+", "", stem)
    return stem


def should_exclude(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in DOCUMENT_SUFFIXES:
        return True
    if name.startswith(".") or name.startswith("~"):
        return True
    return any(term.lower() in name for term in EXCLUDE_TERMS)


def is_policy_like(path: Path) -> bool:
    return any(term in path.name for term in POLICY_TERMS)


def score_file(path: Path, include_terms: list[str]) -> int:
    name = path.name
    score = 0
    score += sum(6 for term in PREFERRED_TERMS if term in name)
    score += sum(4 for term in include_terms if term in name)
    score += sum(2 for term in POLICY_TERMS if term in name)
    if "流程图" in name:
        score -= 3
    if len(name) > 70:
        score -= 1
    return score


def existing_names(target_root: Path) -> set[str]:
    names = set()
    if not target_root.exists():
        return names
    for path in target_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES:
            names.add(normalize_name(path.name))
    return names


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def candidates_for_rule(
    data_root: Path,
    enterprise_root: Path,
    rule: dict,
    seen_names: set[str],
    seen_sources: set[Path],
):
    source_dir = data_root / rule["source"]
    if not source_dir.exists():
        return []

    scored = []
    scored_names = set()
    for path in source_dir.rglob("*"):
        if not path.is_file() or should_exclude(path) or not is_policy_like(path):
            continue
        normalized_name = normalize_name(path.name)
        resolved_source = path.resolve()
        if normalized_name in scored_names or resolved_source in seen_sources:
            continue
        if any(term in path.name for term in rule.get("exclude", [])):
            continue
        if not any(term in path.name for term in rule["include"]):
            continue
        score = score_file(path, rule["include"])
        if score <= 0:
            continue
        scored.append((score, len(path.name), path))
        scored_names.add(normalized_name)

    scored.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    selected = []
    for _, _, source in scored[: rule["limit"]]:
        normalized_name = normalize_name(source.name)
        resolved_source = source.resolve()
        if normalized_name in seen_names or resolved_source in seen_sources:
            seen_sources.add(resolved_source)
            continue
        target_dir = enterprise_root / rule["target"]
        destination = unique_destination(target_dir / source.name)
        selected.append(
            {
                "source": source,
                "destination": destination,
                "target": rule["target"],
                "score": score_file(source, rule["include"]),
            }
        )
        seen_names.add(normalized_name)
        seen_sources.add(resolved_source)
    return selected


def build_plan(data_root: Path, enterprise_root: Path):
    seen_names = existing_names(enterprise_root)
    seen_sources = set()
    imports = []
    for rule in SOURCE_RULES:
        imports.extend(candidates_for_rule(data_root, enterprise_root, rule, seen_names, seen_sources))
    return imports


def write_report(data_root: Path, imports, applied: bool) -> Path:
    report_path = data_root / "enterprise_supplement_import_report.json"
    payload = [
        {
            "source": str(item["source"]),
            "destination": str(item["destination"]),
            "target": item["target"],
            "score": item["score"],
        }
        for item in imports
    ]
    report_path.write_text(
        json.dumps({"applied": applied, "imports": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def completed_report_exists(data_root: Path) -> bool:
    report_path = data_root / "enterprise_supplement_import_report.json"
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("applied"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Import supplemental enterprise policy documents.")
    parser.add_argument("--data-root", default="data", help="Root containing 01-10 folders and enterprise.")
    parser.add_argument("--enterprise-root", default="data/enterprise", help="Curated enterprise knowledge root.")
    parser.add_argument("--apply", action="store_true", help="Copy files. Without this flag, only preview.")
    parser.add_argument("--report", action="store_true", help="Write import report JSON under data/.")
    parser.add_argument("--ignore-report", action="store_true", help="Recalculate even if an applied report exists.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    enterprise_root = Path(args.enterprise_root).resolve()
    if completed_report_exists(data_root) and not args.ignore_report:
        print("已存在已执行的补充导入报告；默认不继续追加低优先级候选。需要重算时加 --ignore-report。")
        return 0

    imports = build_plan(data_root, enterprise_root)

    counts = Counter(item["target"] for item in imports)
    samples = defaultdict(list)
    for item in imports:
        if len(samples[item["target"]]) < 8:
            samples[item["target"]].append(item["source"].name)

    print(f"待补充文件数: {len(imports)}")
    for target, count in sorted(counts.items()):
        print(f"- {target}: {count}")
        for sample in samples[target]:
            print(f"  · {sample}")

    if args.apply:
        for item in imports:
            item["destination"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source"], item["destination"])
        print(f"已复制文件数: {len(imports)}")
    else:
        print("当前为预览模式；确认后加 --apply 执行复制。")

    if args.report:
        report_path = write_report(data_root, imports, args.apply)
        print(f"补充报告: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

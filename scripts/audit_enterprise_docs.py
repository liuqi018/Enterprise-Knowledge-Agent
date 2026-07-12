# -*- coding: utf-8 -*-
"""Audit and optionally quarantine enterprise knowledge documents.

The audit is intentionally conservative for the manufacturing-enterprise
scenario: it keeps broad corporate policies and manufacturing-related content,
flags questionable files for review, and only quarantines clearly irrelevant or
broken files when --apply is provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DOCUMENT_SUFFIXES = {".doc", ".docx", ".pdf", ".txt"}

KEEP_TERMS = [
    "制造", "生产", "车间", "设备", "设施", "物料", "原辅料", "包材", "产成品",
    "仓库", "库存", "出入库", "采购", "供应商", "质量", "检验", "不合格",
    "研发", "项目立项", "项目结项", "配置管理", "开发规范", "测试",
    "财务", "付款", "收款", "应收", "应付", "预算", "资金", "现金", "资产",
    "报销", "差旅", "借款", "发票", "人力资源", "人事", "招聘", "入职",
    "转正", "离职", "培训", "绩效", "薪酬", "考勤", "请假", "行政",
    "会议", "档案", "印章", "ITSM", "信息安全", "访问控制", "备份",
    "风险评估", "业务持续性", "安全生产", "消防", "EHS", "环保",
]

GENERAL_CORPORATE_TERMS = [
    "公司", "企业", "员工", "部门", "岗位", "组织架构", "管理制度", "管理办法",
    "管理规定", "流程", "规范", "细则", "手册", "制度",
]

MISMATCH_TERMS = [
    "医院", "医疗", "门诊", "院所", "药业", "制药", "医药",
    "学校", "教育", "培训学校", "教师", "学生",
    "银行", "金融", "证券", "基金", "保险公司",
    "房地产", "物业", "酒店", "餐饮", "餐厅", "旅行社", "美容",
    "超市", "便利店", "连锁店", "门店", "店铺", "服装", "酒业",
    "猪场", "养殖", "农场", "合作社", "农村信用",
    "施工企业", "建筑工程", "工程项目", "施工噪声", "项目部",
    "烟草", "电商", "主播", "直播", "MCN", "广告公司",
    "工会", "律师拟定", "讲话", "发言",
]

LOW_VALUE_TERMS = [
    "模板", "表格", "清单", "台账", "台帐", "记录表", "登记表", "申请表",
    "签到", "证明", "收入证明", "任职证明", "通知书", "确认书", "讲话",
    "发言", "总结", "计划表", "万能模板", "面试题", "公式", "思维导图",
]

TEMP_TERMS = [".ds_store", "thumbs.db", "~$", ".~", "〜", "～"]

REVIEW_TERMS = [
    "股份有限公司", "上市公司", "集团股份", "集团公司", "适用于股份",
    "适用于工程公司", "中英文对照", "最新版", "完整版", "范本", "大全",
]


@dataclass
class AuditItem:
    path: Path
    decision: str
    reason: str
    target_bucket: str


def normalize_name(path: Path) -> str:
    return path.name.lower()


def relpath(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def has_any(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term.lower() in text.lower()]


def classify(path: Path, root: Path) -> AuditItem:
    name = path.name
    normalized = normalize_name(path)
    relative = relpath(path, root)
    parent = relative.lower()

    if path.suffix.lower() not in DOCUMENT_SUFFIXES:
        return AuditItem(path, "quarantine", "unsupported_suffix", "unsupported")
    if has_any(normalized, TEMP_TERMS):
        return AuditItem(path, "quarantine", "temporary_file", "temporary")
    if path.stat().st_size == 0:
        return AuditItem(path, "quarantine", "zero_size", "zero_size")

    keep_hits = has_any(name, KEEP_TERMS)
    general_hits = has_any(name, GENERAL_CORPORATE_TERMS)
    mismatch_hits = has_any(name, MISMATCH_TERMS)
    low_value_hits = has_any(name, LOW_VALUE_TERMS)
    review_hits = has_any(name, REVIEW_TERMS)

    # Core manufacturing domains are intentionally protected unless the file is
    # clearly temporary/broken. This avoids losing useful production policies
    # just because a filename contains a broad term like "工程".
    if any(bucket in parent for bucket in [
        "policies/production",
        "policies/quality",
        "policies/warehouse",
        "policies/procurement",
        "policies/research",
        "policies/it_service",
        "policies/security",
    ]):
        if low_value_hits and not keep_hits:
            return AuditItem(path, "review", f"low_value_candidate:{','.join(low_value_hits[:3])}", "review")
        return AuditItem(path, "keep", "protected_core_domain", "")

    if any(bucket in parent for bucket in ["policies/onboarding", "policies/hr", "templates/forms"]):
        if low_value_hits:
            return AuditItem(path, "review", f"hr_form_or_proof_review:{','.join(low_value_hits[:3])}", "review")

    if mismatch_hits and not keep_hits:
        return AuditItem(path, "quarantine", f"business_mismatch:{','.join(mismatch_hits[:3])}", "business_mismatch")

    if mismatch_hits and keep_hits:
        return AuditItem(
            path,
            "review",
            f"mixed_business_context:keep={','.join(keep_hits[:3])};mismatch={','.join(mismatch_hits[:3])}",
            "review",
        )

    if low_value_hits and not keep_hits and not general_hits:
        return AuditItem(path, "quarantine", f"low_value:{','.join(low_value_hits[:3])}", "low_value")

    if low_value_hits and (keep_hits or general_hits):
        return AuditItem(path, "review", f"low_value_candidate:{','.join(low_value_hits[:3])}", "review")

    if review_hits:
        return AuditItem(path, "review", f"version_or_scope_review:{','.join(review_hits[:3])}", "review")

    return AuditItem(path, "keep", "scenario_relevant_or_general_corporate", "")


def file_digest(path: Path, size_limit_mb: int = 20) -> str:
    if path.stat().st_size > size_limit_mb * 1024 * 1024:
        return ""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_reviews(files: list[Path]) -> list[AuditItem]:
    by_digest = defaultdict(list)
    for path in files:
        if path.stat().st_size == 0:
            continue
        digest = file_digest(path)
        if digest:
            by_digest[digest].append(path)

    duplicates = []
    for paths in by_digest.values():
        if len(paths) <= 1:
            continue
        for path in sorted(paths)[1:]:
            duplicates.append(AuditItem(path, "review", "duplicate_content_candidate", "review"))
    return duplicates


def iter_documents(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


def unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    index = 2
    while True:
        candidate = destination.with_name(f"{destination.stem}_{index}{destination.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def quarantine_item(item: AuditItem, root: Path, quarantine_root: Path) -> Path:
    relative = item.path.relative_to(root)
    destination = unique_destination(quarantine_root / item.target_bucket / relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(item.path), str(destination))
    return destination


def write_reports(root: Path, items: list[AuditItem], duplicate_items: list[AuditItem], applied: bool) -> tuple[Path, Path]:
    report_dir = root.parent / "audit_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "enterprise_audit_report.json"
    csv_path = report_dir / "enterprise_audit_report.csv"

    payload = {
        "applied": applied,
        "items": [
            {
                "path": str(item.path),
                "decision": item.decision,
                "reason": item.reason,
                "target_bucket": item.target_bucket,
            }
            for item in items
        ],
        "duplicate_review_items": [
            {
                "path": str(item.path),
                "decision": item.decision,
                "reason": item.reason,
                "target_bucket": item.target_bucket,
            }
            for item in duplicate_items
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["decision", "reason", "target_bucket", "path"])
        for item in items + duplicate_items:
            writer.writerow([item.decision, item.reason, item.target_bucket, str(item.path)])

    return json_path, csv_path


def print_summary(items: list[AuditItem], duplicate_items: list[AuditItem]) -> None:
    counts = Counter(item.decision for item in items)
    bucket_counts = Counter(item.target_bucket for item in items if item.target_bucket)
    reason_counts = Counter(item.reason for item in items)

    print("审计结果:")
    for decision in ["keep", "review", "quarantine"]:
        print(f"- {decision}: {counts.get(decision, 0)}")
    if duplicate_items:
        print(f"- duplicate review: {len(duplicate_items)}")

    print("\n隔离/复核类别:")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"- {bucket}: {count}")

    print("\n主要原因:")
    for reason, count in reason_counts.most_common(12):
        print(f"- {reason}: {count}")

    for decision in ["quarantine", "review"]:
        samples = [item for item in items if item.decision == decision][:20]
        if samples:
            print(f"\n{decision} 样例:")
            for item in samples:
                print(f"- [{item.reason}] {item.path}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Audit enterprise docs for manufacturing scenario relevance.")
    parser.add_argument("--data-dir", default="data/enterprise", help="Curated enterprise document root.")
    parser.add_argument("--quarantine-dir", default="data/quarantine", help="Quarantine output root.")
    parser.add_argument("--apply", action="store_true", help="Move quarantine items. Review items are never moved.")
    parser.add_argument("--include-duplicates", action="store_true", help="Also compute duplicate-content review list.")
    args = parser.parse_args()

    root = Path(args.data_dir).resolve()
    quarantine_root = Path(args.quarantine_dir).resolve()
    if not root.exists():
        raise SystemExit(f"data dir not found: {root}")

    files = iter_documents(root)
    items = [classify(path, root) for path in files]
    duplicate_items = duplicate_reviews(files) if args.include_duplicates else []

    print(f"扫描文档数: {len(files)}")
    print_summary(items, duplicate_items)

    if args.apply:
        moved = []
        for item in items:
            if item.decision == "quarantine":
                moved.append((item.path, quarantine_item(item, root, quarantine_root)))
        print(f"\n已隔离文件数: {len(moved)}")
    else:
        print("\n当前为预览模式；确认后加 --apply 执行隔离。review 文件不会自动移动。")

    json_path, csv_path = write_reports(root, items, duplicate_items, args.apply)
    print(f"JSON 报告: {json_path}")
    print(f"CSV 报告: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

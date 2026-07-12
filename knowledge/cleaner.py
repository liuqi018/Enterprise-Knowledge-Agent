import os
import re
from collections import Counter
from typing import Iterable, List


LOW_VALUE_LINE_PATTERNS = [
    r"^第\s*\d+\s*页\s*/\s*共\s*\d+\s*页$",
    r"^page\s*\d+\s*(of\s*\d+)?$",
    r"^公司内部资料$",
    r"^内部资料$",
    r"^confidential$",
    r"^[-_=]{3,}$",
]

SIGNATURE_TERMS = ["签字", "签名", "盖章", "审批意见", "审核意见", "负责人", "经办人", "日期", "年   月   日"]

FORM_FIELD_TERMS = ["序号", "名称", "规格", "型号", "数量", "单价", "金额", "备注", "部门", "申请人"]


def clean_text(text: str) -> str:
    text = normalize_whitespace(text)
    lines = [clean_line(line) for line in text.splitlines()]
    lines = remove_low_value_lines(lines)
    lines = collapse_repeated_lines(lines)
    return "\n".join(line for line in lines if line).strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s*\|\s*", " | ", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line


def remove_low_value_lines(lines: Iterable[str]) -> List[str]:
    result = []
    for line in lines:
        if not line:
            continue
        normalized = line.strip().lower()
        if any(re.match(pattern, normalized, flags=re.IGNORECASE) for pattern in LOW_VALUE_LINE_PATTERNS):
            continue
        if is_repeated_punctuation(line):
            continue
        result.append(line)
    return result


def collapse_repeated_lines(lines: Iterable[str], max_repeats: int = 2) -> List[str]:
    seen = Counter()
    result = []
    for line in lines:
        key = normalize_repeat_key(line)
        seen[key] += 1
        if seen[key] <= max_repeats:
            result.append(line)
    return result


def normalize_repeat_key(line: str) -> str:
    return re.sub(r"\s+", "", line.lower())


def is_repeated_punctuation(line: str) -> bool:
    compact = re.sub(r"\s+", "", line)
    return bool(compact) and len(set(compact)) <= 2 and all(char in "-_=|·.。" for char in set(compact))


def infer_document_type(path: str, content: str = "") -> str:
    name = os.path.basename(path)
    text = f"{name} {content[:800]}"
    if any(term in name for term in ["合同", "协议", "承诺书"]) and "制度" not in name:
        return "contract_template"
    if any(term in name for term in ["岗位职责", "职责说明", "组织架构", "岗位责任"]):
        return "job_description"
    if any(term in name for term in ["申请表", "登记表", "清单", "模板", "确认书", "流程图"]):
        return "form_template"
    if any(term in name for term in ["培训", "教材", "手册"]):
        return "training_material"
    if any(term in text for term in ["制度", "办法", "规定", "细则", "流程", "规范", "管理"]):
        return "policy"
    return "unknown"


def infer_policy_domain(path: str, content: str = "") -> str:
    name = os.path.basename(path)
    document_type = infer_document_type(path, content)
    text = f"{name} {content[:1000]}"
    if document_type in {"contract_template", "job_description"} and "采购" not in name:
        text = content[:1000]

    domain_keywords = {
        "reimbursement": ["报销", "差旅", "费用", "发票", "借款"],
        "leave_attendance": ["请假", "事假", "考勤", "病假", "年假", "调休", "休假"],
        "procurement": ["采购", "请购", "供应商", "询价", "报价", "验收", "仓库", "办公用品", "物资"],
        "security": ["信息安全", "权限", "账号", "数据安全", "保密"],
        "onboarding": ["入职", "转正", "试用期", "录用", "离职"],
        "ticket_sop": ["工单", "SOP", "客户", "售后"],
        "administration": ["印章", "档案", "会议", "车辆", "行政"],
        "salary_performance": ["薪资", "绩效", "奖金", "提成"],
    }
    for domain, keywords in domain_keywords.items():
        if any(keyword in text for keyword in keywords):
            return domain
    return "general"


def is_low_value_chunk(text: str, metadata: dict = None) -> bool:
    metadata = metadata or {}
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 18:
        return True

    document_type = metadata.get("document_type")
    if document_type == "form_template" and len(compact) < 80:
        return True

    signature_hits = sum(1 for term in SIGNATURE_TERMS if term in text)
    form_hits = sum(1 for term in FORM_FIELD_TERMS if term in text)
    policy_signal_hits = sum(1 for term in ["制度", "流程", "规定", "审批", "申请", "材料", "要求", "标准"] if term in text)

    if signature_hits >= 4 and policy_signal_hits == 0:
        return True
    if form_hits >= 6 and len(compact) < 160 and policy_signal_hits <= 1:
        return True
    if table_noise_ratio(text) > 0.55 and policy_signal_hits <= 1:
        return True
    return False


def table_noise_ratio(text: str) -> float:
    if not text:
        return 0.0
    noisy_chars = sum(1 for char in text if char in "|:_-—")
    return noisy_chars / max(len(text), 1)

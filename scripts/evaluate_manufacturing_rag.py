# -*- coding: utf-8 -*-
"""Evaluate manufacturing-enterprise RAG retrieval and answer quality."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = PROJECT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.rag.rag_service import RagSummarizeService
from AIRAGAgent.utils.json_guard import JSONGuardError, coerce_bool, coerce_float, parse_json_object


KEYWORD_SYNONYMS = {
    "流程": ["步骤", "程序", "环节", "路径", "办理", "流转"],
    "处理": ["处置", "整改", "跟进", "办理", "解决", "管控"],
    "上报": ["报告", "汇报", "提交", "反馈", "报送"],
    "检查": ["检验", "核查", "审核", "复核", "确认", "验收"],
    "责任": ["负责人", "责任人", "责任部门", "职责", "分工"],
    "材料": ["资料", "凭证", "单据", "附件", "证明", "记录"],
    "审批": ["审核", "批准", "确认", "签批", "审批人"],
    "风险": ["合规", "禁止", "不得", "注意", "隐患", "控制"],
    "权限": ["账号", "授权", "访问控制", "访问权限"],
    "生产": ["车间", "作业", "生产现场", "生产过程"],
    "质量": ["品质", "检验", "不合格", "质量控制"],
    "财务": ["资金", "付款", "收款", "会计", "预算"],
    "报销": ["费用", "差旅", "借款", "发票"],
    "采购": ["请购", "供应商", "采购申请", "采购合同"],
    "仓库": ["库存", "入库", "出库", "物资"],
    "研发": ["项目", "立项", "结项", "开发"],
    "服务": ["ITSM", "事件", "问题", "服务请求"],
    "安全": ["信息安全", "保密", "风险", "控制"],
    "员工": ["人员", "职工", "新员工"],
    "指标": ["KPI", "考核项", "权重", "评价"],
}


STRICT_ANSWER_JUDGE_PROMPT = """
You are a strict evaluator for an enterprise policy RAG system.
The question, answer, expected keywords, and source snippets may be Chinese.

Judge the answer only against:
1. the retrieved source snippets,
2. the expected answer keywords,
3. the user's question.

Do not use external knowledge. Be strict:
- Penalize unsupported details such as made-up amounts, deadlines, roles, departments, forms, or approval levels.
- Penalize answers that are vague when the retrieved sources contain concrete procedure or responsibility information.
- Penalize answers that refuse or say there is no basis when the retrieved sources support an answer.
- Penalize answers that answer a different topic.
- Passing requires the answer to be useful for an employee and grounded in the retrieved sources.
- If expected_no_answer is true, pass only when the answer clearly says the evidence is insufficient
  or no clear policy basis is available, and does not invent a concrete policy.

Return one JSON object only. No markdown. No extra text.
Schema:
{
  "accuracy_score": 1-5,
  "faithfulness_score": 1-5,
  "completeness_score": 1-5,
  "groundedness_score": 1-5,
  "usability_score": 1-5,
  "passed": true/false,
  "missing_points": ["short item"],
  "unsupported_claims": ["short item"],
  "reason": "short reason"
}

Scoring:
5 = fully correct and specific, 4 = mostly correct with minor omissions,
3 = partially correct but incomplete, 2 = weak or likely misleading,
1 = wrong, unsupported, or off-topic.

Evaluation payload:
{payload}
"""


NO_ANSWER_MARKERS = [
    "\u6ca1\u6709\u660e\u786e\u4f9d\u636e",
    "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u660e\u786e\u4f9d\u636e",
    "\u8d44\u6599\u672a\u660e\u786e",
    "\u672a\u660e\u786e",
    "\u65e0\u6cd5\u786e\u8ba4",
    "\u4e0d\u80fd\u7f16\u9020",
    "\u4e0d\u5e94\u7f16\u9020",
    "\u672a\u68c0\u7d22\u5230",
    "\u672a\u67e5\u8be2\u5230",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            required = [
                "id",
                "query",
                "expected_domain",
                "expected_source_keywords",
                "expected_answer_keywords",
                "question_type",
            ]
            for field in required:
                if field not in case:
                    raise ValueError(f"missing {field} at {path}:{line_no}")
            rows.append(case)
    return rows


def text_of_doc(doc) -> str:
    metadata = doc.metadata or {}
    parts = [
        metadata.get("file_name"),
        metadata.get("source"),
        metadata.get("section_title"),
        metadata.get("policy_domain"),
        doc.page_content[:300],
    ]
    return " ".join(str(part) for part in parts if part)


def metadata_domain(doc) -> str:
    return str((doc.metadata or {}).get("policy_domain") or "")


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    hits = []
    for keyword in keywords:
        if not keyword:
            continue
        variants = [keyword] + KEYWORD_SYNONYMS.get(keyword, [])
        if any(variant in text for variant in variants):
            hits.append(keyword)
    return hits


def as_string_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        text = compact_text(str(item), 140)
        if text:
            result.append(text)
    return result


def judge_payload(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "question": case["query"],
        "question_type": case["question_type"],
        "hard_type": case.get("hard_type", ""),
        "expected_no_answer": bool(case.get("expected_no_answer", False)),
        "expected_domain": case["expected_domain"],
        "expected_source_keywords": case["expected_source_keywords"],
        "expected_answer_keywords": case["expected_answer_keywords"],
        "answer": row.get("answer", ""),
        "answer_keyword_hits": row.get("answer_keyword_hits", []),
        "missing_answer_keywords": row.get("missing_answer_keywords", []),
        "retrieved_sources": [
            {
                "rank": source.get("rank"),
                "file_name": source.get("file_name"),
                "policy_domain": source.get("policy_domain"),
                "section_title": source.get("section_title"),
                "domain_match": source.get("domain_match"),
                "source_match": source.get("source_match"),
                "source_keyword_hits": source.get("source_keyword_hits", []),
                "preview": compact_text(source.get("preview", ""), 700),
            }
            for source in row.get("retrieved_sources", [])[:5]
        ],
    }


def normalize_judge_result(data: dict[str, Any], min_score: float) -> dict[str, Any]:
    scores = {
        "accuracy_score": round(coerce_float(data.get("accuracy_score"), 0.0, 0.0, 5.0), 2),
        "faithfulness_score": round(coerce_float(data.get("faithfulness_score"), 0.0, 0.0, 5.0), 2),
        "completeness_score": round(coerce_float(data.get("completeness_score"), 0.0, 0.0, 5.0), 2),
        "groundedness_score": round(coerce_float(data.get("groundedness_score"), 0.0, 0.0, 5.0), 2),
        "usability_score": round(coerce_float(data.get("usability_score"), 0.0, 0.0, 5.0), 2),
    }
    score_passed = all(
        scores[key] >= min_score
        for key in [
            "accuracy_score",
            "faithfulness_score",
            "completeness_score",
            "groundedness_score",
        ]
    )
    explicit_passed = coerce_bool(data.get("passed"), score_passed)
    return {
        **scores,
        "passed": bool(explicit_passed and score_passed),
        "missing_points": as_string_list(data.get("missing_points")),
        "unsupported_claims": as_string_list(data.get("unsupported_claims")),
        "reason": compact_text(data.get("reason", ""), 300),
        "parse_failed": False,
    }


def judge_answer(case: dict[str, Any], row: dict[str, Any], min_score: float) -> dict[str, Any]:
    payload = json.dumps(judge_payload(case, row), ensure_ascii=False, indent=2)
    prompt = STRICT_ANSWER_JUDGE_PROMPT.replace("{payload}", payload)
    try:
        response = chat_model.invoke(prompt)
        raw = getattr(response, "content", response)
        data = parse_json_object(str(raw))
        return normalize_judge_result(data, min_score)
    except (JSONGuardError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "accuracy_score": 0.0,
            "faithfulness_score": 0.0,
            "completeness_score": 0.0,
            "groundedness_score": 0.0,
            "usability_score": 0.0,
            "passed": False,
            "missing_points": [],
            "unsupported_claims": [],
            "reason": f"judge JSON parse failed: {type(exc).__name__}: {compact_text(str(exc), 160)}",
            "parse_failed": True,
        }
    except Exception as exc:
        return {
            "accuracy_score": 0.0,
            "faithfulness_score": 0.0,
            "completeness_score": 0.0,
            "groundedness_score": 0.0,
            "usability_score": 0.0,
            "passed": False,
            "missing_points": [],
            "unsupported_claims": [],
            "reason": f"judge failed: {type(exc).__name__}: {compact_text(str(exc), 160)}",
            "parse_failed": False,
        }


def detects_no_answer(answer: str) -> bool:
    return any(marker in str(answer or "") for marker in NO_ANSWER_MARKERS)


def dcg(relevance: list[int]) -> float:
    total = 0.0
    for index, rel in enumerate(relevance, start=1):
        if rel:
            total += rel / math.log2(index + 1)
    return total


def evaluate_case(
    service: RagSummarizeService,
    case: dict[str, Any],
    max_k: int,
    skip_answer: bool,
    judge_answer_enabled: bool,
    judge_min_score: float,
) -> dict[str, Any]:
    query = case["query"]
    expected_domain = case["expected_domain"]
    source_keywords = case["expected_source_keywords"]
    answer_keywords = case["expected_answer_keywords"]
    hard_type = str(case.get("hard_type") or case["question_type"])
    expected_no_answer = bool(case.get("expected_no_answer", False))

    start = time.perf_counter()
    docs = service.retrieve_documents(query, top_k=max_k)[:max_k]
    retrieval_latency_ms = (time.perf_counter() - start) * 1000

    doc_texts = [text_of_doc(doc) for doc in docs]
    domains = [metadata_domain(doc) for doc in docs]
    source_keyword_hits_by_doc = [keyword_hits(text, source_keywords) for text in doc_texts]
    source_relevance = [1 if hits else 0 for hits in source_keyword_hits_by_doc]
    domain_relevance = [1 if domain == expected_domain else 0 for domain in domains]

    combined_relevance = [
        1 if source_hit or domain_hit else 0
        for source_hit, domain_hit in zip(source_relevance, domain_relevance)
    ]
    first_hit_rank = next((idx for idx, rel in enumerate(combined_relevance, start=1) if rel), 0)
    hit_at_1 = 1 if any(combined_relevance[:1]) else 0
    hit_at_3 = 1 if any(combined_relevance[:3]) else 0
    hit_at_5 = 1 if any(combined_relevance[:5]) else 0
    domain_hit_at_3 = 1 if any(domain_relevance[:3]) else 0
    source_hit_at_3 = 1 if any(source_relevance[:3]) else 0
    mrr = 1 / first_hit_rank if first_hit_rank else 0.0
    ideal = sorted(combined_relevance, reverse=True)
    ndcg_at_5 = dcg(combined_relevance[:5]) / dcg(ideal[:5]) if dcg(ideal[:5]) else 0.0

    answer = ""
    answer_latency_ms = 0.0
    answer_keyword_hit_list: list[str] = []
    if not skip_answer:
        answer_start = time.perf_counter()
        answer = service.generate_answer(query, docs) if docs else ""
        answer_latency_ms = (time.perf_counter() - answer_start) * 1000
        answer_keyword_hit_list = keyword_hits(answer, answer_keywords)
    no_answer_detected = detects_no_answer(answer) if not skip_answer else False
    missing_answer_keywords = [
        keyword for keyword in answer_keywords if keyword not in answer_keyword_hit_list
    ]

    answer_keyword_coverage = (
        len(answer_keyword_hit_list) / len(answer_keywords)
        if answer_keywords
        else 0.0
    )

    result = {
        "id": case["id"],
        "query": query,
        "question_type": case["question_type"],
        "hard_type": hard_type,
        "expected_no_answer": expected_no_answer,
        "expected_domain": expected_domain,
        "expected_source_keywords": source_keywords,
        "expected_answer_keywords": answer_keywords,
        "retrieved_domains": domains,
        "retrieved_sources": [
            {
                "rank": index + 1,
                "file_name": (doc.metadata or {}).get("file_name"),
                "policy_domain": metadata_domain(doc),
                "section_title": (doc.metadata or {}).get("section_title"),
                "domain_match": domain_relevance[index] == 1,
                "source_match": source_relevance[index] == 1,
                "source_keyword_hits": source_keyword_hits_by_doc[index],
                "preview": doc.page_content[:260],
            }
            for index, doc in enumerate(docs)
        ],
        "first_hit_rank": first_hit_rank,
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "hit@5": hit_at_5,
        "recall@1": hit_at_1,
        "recall@3": hit_at_3,
        "recall@5": hit_at_5,
        "domain_hit@3": domain_hit_at_3,
        "source_hit@3": source_hit_at_3,
        "mrr": round(mrr, 4),
        "ndcg@5": round(ndcg_at_5, 4),
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "answer": answer,
        "answer_keyword_hits": answer_keyword_hit_list,
        "missing_answer_keywords": missing_answer_keywords,
        "answer_keyword_coverage": round(answer_keyword_coverage, 4),
        "answer_latency_ms": round(answer_latency_ms, 2),
        "no_answer_detected": 1 if no_answer_detected else 0,
        "no_answer_pass": 1 if expected_no_answer and no_answer_detected else 0,
    }
    if judge_answer_enabled and not skip_answer:
        judge_start = time.perf_counter()
        judge = judge_answer(case, result, judge_min_score)
        judge_latency_ms = (time.perf_counter() - judge_start) * 1000
        judge["latency_ms"] = round(judge_latency_ms, 2)
        result["judge"] = judge
        result["judge_passed"] = 1 if judge.get("passed") else 0
        result["judge_accuracy_score"] = judge.get("accuracy_score", 0.0)
        result["judge_faithfulness_score"] = judge.get("faithfulness_score", 0.0)
        result["judge_completeness_score"] = judge.get("completeness_score", 0.0)
        result["judge_groundedness_score"] = judge.get("groundedness_score", 0.0)
        result["judge_usability_score"] = judge.get("usability_score", 0.0)
        result["judge_latency_ms"] = round(judge_latency_ms, 2)
    return result


def average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 4)


def summarize(rows: list[dict[str, Any]], skip_answer: bool) -> dict[str, Any]:
    retrieval_rows = [row for row in rows if not row.get("expected_no_answer")]
    no_answer_rows = [row for row in rows if row.get("expected_no_answer")]
    summary = {
        "cases": len(rows),
        "retrieval_cases": len(retrieval_rows),
        "hit@1": average(retrieval_rows, "hit@1"),
        "hit@3": average(retrieval_rows, "hit@3"),
        "hit@5": average(retrieval_rows, "hit@5"),
        "recall@1": average(retrieval_rows, "recall@1"),
        "recall@3": average(retrieval_rows, "recall@3"),
        "recall@5": average(retrieval_rows, "recall@5"),
        "domain_hit@3": average(retrieval_rows, "domain_hit@3"),
        "source_hit@3": average(retrieval_rows, "source_hit@3"),
        "mrr": average(retrieval_rows, "mrr"),
        "ndcg@5": average(retrieval_rows, "ndcg@5"),
        "avg_retrieval_latency_ms": round(
            sum(float(row.get("retrieval_latency_ms", 0.0)) for row in rows) / max(len(rows), 1),
            2,
        ),
    }
    if no_answer_rows:
        summary["no_answer_cases"] = len(no_answer_rows)
        summary["no_answer_pass_rate"] = average(no_answer_rows, "no_answer_pass")
    if not skip_answer:
        summary["answer_keyword_coverage"] = average(rows, "answer_keyword_coverage")
        summary["avg_answer_latency_ms"] = round(
            sum(float(row.get("answer_latency_ms", 0.0)) for row in rows) / max(len(rows), 1),
            2,
        )
    if rows and any("judge" in row for row in rows):
        summary["judge_pass_rate"] = average(rows, "judge_passed")
        summary["judge_avg_accuracy"] = average(rows, "judge_accuracy_score")
        summary["judge_avg_faithfulness"] = average(rows, "judge_faithfulness_score")
        summary["judge_avg_completeness"] = average(rows, "judge_completeness_score")
        summary["judge_avg_groundedness"] = average(rows, "judge_groundedness_score")
        summary["judge_avg_usability"] = average(rows, "judge_usability_score")
        summary["avg_judge_latency_ms"] = round(
            sum(float(row.get("judge_latency_ms", 0.0)) for row in rows) / max(len(rows), 1),
            2,
        )
    return summary


def summarize_by_group(rows: list[dict[str, Any]], group_key: str, skip_answer: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "unknown"))].append(row)
    return {name: summarize(group_rows, skip_answer) for name, group_rows in sorted(groups.items())}


def coverage_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = {
        "by_domain": dict(sorted(Counter(str(case.get("expected_domain", "unknown")) for case in cases).items())),
        "by_question_type": dict(sorted(Counter(str(case.get("question_type", "unknown")) for case in cases).items())),
    }
    hard_type_counts = Counter(str(case.get("hard_type")) for case in cases if case.get("hard_type"))
    if hard_type_counts:
        summary["by_hard_type"] = dict(sorted(hard_type_counts.items()))
    no_answer_count = sum(1 for case in cases if case.get("expected_no_answer"))
    if no_answer_count:
        summary["by_expected_no_answer"] = {
            "false": len(cases) - no_answer_count,
            "true": no_answer_count,
        }
    return summary


def diagnose_failure(row: dict[str, Any], skip_answer: bool, min_answer_coverage: float) -> str:
    if row.get("expected_no_answer"):
        if row.get("no_answer_pass"):
            return "expected_no_answer_passed"
        return "no_answer_policy_failed"
    if row["hit@3"] == 0:
        return "retrieval_miss"
    if row["source_hit@3"] == 0:
        return "source_keyword_mismatch"
    if row["domain_hit@3"] == 0:
        return "domain_mismatch"
    if row.get("judge") and not row.get("judge_passed"):
        if (row.get("judge") or {}).get("parse_failed"):
            return "judge_parse_failed"
        return "answer_judge_failed"
    if not skip_answer and row["answer_keyword_coverage"] < min_answer_coverage:
        return "answer_keyword_gap"
    return "unknown"


def failure_category(row: dict[str, Any], skip_answer: bool, min_answer_coverage: float) -> str:
    reason = row.get("failure_reason") or diagnose_failure(row, skip_answer, min_answer_coverage)
    if reason == "no_answer_policy_failed":
        return "no_answer_failure"
    if reason == "retrieval_miss":
        return "retrieval_failure"
    if reason in {"domain_mismatch", "source_keyword_mismatch"}:
        return "retrieval_label_or_metadata_issue"
    if reason == "answer_keyword_gap":
        if row.get("judge") and row.get("judge_passed"):
            return "keyword_false_negative"
        return "answer_missing_expected_keywords"
    if reason == "judge_parse_failed":
        return "judge_output_failure"
    if reason == "answer_judge_failed":
        judge = row.get("judge") or {}
        unsupported = judge.get("unsupported_claims") or []
        completeness = float(row.get("judge_completeness_score") or 0.0)
        faithfulness = float(row.get("judge_faithfulness_score") or 0.0)
        if unsupported or faithfulness < 4.0:
            return "answer_faithfulness_or_hallucination_risk"
        if completeness < 4.0:
            return "answer_incomplete"
        return "judge_failed_other"
    return "unknown"


def failure_analysis_summary(failed: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(row.get("failure_category", "unknown") for row in failed)
    reasons = Counter(row.get("failure_reason", "unknown") for row in failed)
    category_descriptions = {
        "retrieval_failure": "No relevant evidence was retrieved in top results.",
        "retrieval_label_or_metadata_issue": "Evidence was found, but source keywords or domain metadata did not match the expected labels.",
        "keyword_false_negative": "The LLM judge passed the answer, but keyword coverage rules marked it as failed.",
        "answer_missing_expected_keywords": "The answer missed expected keywords and did not pass the keyword threshold.",
        "judge_output_failure": "The LLM judge did not return a valid structured result.",
        "answer_faithfulness_or_hallucination_risk": "The judge found unsupported or weakly grounded answer content.",
        "answer_incomplete": "The answer was grounded but missed required points.",
        "no_answer_failure": "The case expected no direct policy basis, but the answer did not clearly refuse or state insufficient evidence.",
        "judge_failed_other": "The judge failed the answer for mixed or non-specific reasons.",
        "unknown": "The failure did not match a known category.",
    }
    return {
        "total_failed": len(failed),
        "by_category": [
            {
                "category": category,
                "count": count,
                "description": category_descriptions.get(category, ""),
            }
            for category, count in categories.most_common()
        ],
        "by_reason": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common()
        ],
    }


def compact_text(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def markdown_cell(text: Any, limit: int = 180) -> str:
    return compact_text(str(text or ""), limit).replace("|", "\\|")


def failed_rows(rows: list[dict[str, Any]], skip_answer: bool, min_answer_coverage: float) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row.get("expected_no_answer"):
            failed = not bool(row.get("no_answer_pass"))
        else:
            failed = row["hit@3"] == 0
        judge_passed = bool(row.get("judge") and row.get("judge_passed"))
        if not skip_answer and row["answer_keyword_coverage"] < min_answer_coverage:
            # Keyword coverage is a cheap diagnostic, not a stricter signal than
            # the answer judge. If the judge passes, keep the row out of the
            # failed list and report the low keyword coverage as a residual metric.
            failed = failed or (not row.get("expected_no_answer") and not judge_passed)
        if row.get("judge") and not row.get("judge_passed") and not row.get("expected_no_answer"):
            failed = True
        if failed:
            failed_row = dict(row)
            failed_row["failure_reason"] = diagnose_failure(row, skip_answer, min_answer_coverage)
            failed_row["failure_category"] = failure_category(failed_row, skip_answer, min_answer_coverage)
            result.append(failed_row)
    return result


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    lines = [
        "# Manufacturing RAG Evaluation Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: `{metrics['cases']}`",
        f"- Vector backend: `{report['vector_backend']}`",
        f"- Max K: `{report['max_k']}`",
        f"- Skip answer: `{report['skip_answer']}`",
        f"- Strict answer judge: `{report.get('judge_answer', False)}`",
        f"- Judge min score: `{report.get('judge_min_score', '')}`",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Dataset Coverage", ""])
    lines.extend(["### By Domain", "", "| Domain | Cases |", "| --- | ---: |"])
    for domain, count in report.get("coverage", {}).get("by_domain", {}).items():
        lines.append(f"| {domain} | {count} |")
    lines.extend(["", "### By Question Type", "", "| Question Type | Cases |", "| --- | ---: |"])
    for question_type, count in report.get("coverage", {}).get("by_question_type", {}).items():
        lines.append(f"| {question_type} | {count} |")
    if report.get("coverage", {}).get("by_hard_type"):
        lines.extend(["", "### By Hard Type", "", "| Hard Type | Cases |", "| --- | ---: |"])
        for hard_type, count in report.get("coverage", {}).get("by_hard_type", {}).items():
            lines.append(f"| {hard_type} | {count} |")
    if report.get("coverage", {}).get("by_expected_no_answer"):
        lines.extend(["", "### By Expected No Answer", "", "| Expected No Answer | Cases |", "| --- | ---: |"])
        for flag, count in report.get("coverage", {}).get("by_expected_no_answer", {}).items():
            lines.append(f"| {flag} | {count} |")

    lines.extend(["", "## Metrics By Domain", "", "| Domain | Cases | Recall@3 | Hit@3 | Domain Hit@3 | Source Hit@3 | MRR | NDCG@5 |"])
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for domain, item in report["by_domain"].items():
        lines.append(
            f"| {domain} | {item['cases']} | {item.get('recall@3', item['hit@3'])} | {item['hit@3']} | {item['domain_hit@3']} | "
            f"{item['source_hit@3']} | {item['mrr']} | {item['ndcg@5']} |"
        )

    if report.get("by_hard_type"):
        lines.extend(["", "## Metrics By Hard Type", "", "| Hard Type | Cases | Retrieval Cases | Recall@3 | MRR | NDCG@5 | No-Answer Pass Rate | Judge Pass Rate |"])
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for hard_type, item in report["by_hard_type"].items():
            lines.append(
                f"| {hard_type} | {item['cases']} | {item.get('retrieval_cases', item['cases'])} | "
                f"{item.get('recall@3', item.get('hit@3', 0))} | {item.get('mrr', 0)} | {item.get('ndcg@5', 0)} | "
                f"{item.get('no_answer_pass_rate', 'N/A')} | {item.get('judge_pass_rate', 'N/A')} |"
            )

    if report["failed"]:
        reason_counts = Counter(row.get("failure_reason", "unknown") for row in report["failed"])
        lines.extend(["", "## Failure Reason Counts", "", "| Reason | Count |", "| --- | ---: |"])
        for reason, count in reason_counts.most_common():
            lines.append(f"| {reason} | {count} |")

    failure_analysis = report.get("failure_analysis") or {}
    if failure_analysis.get("by_category"):
        lines.extend(["", "## Failure Analysis", "", "| Category | Count | Description |", "| --- | ---: | --- |"])
        for item in failure_analysis.get("by_category", []):
            lines.append(
                f"| {markdown_cell(item.get('category', ''), 100)} | {item.get('count', 0)} | "
                f"{markdown_cell(item.get('description', ''), 260)} |"
            )

    lines.extend(["", "## Failed Cases", ""])
    if not report["failed"]:
        lines.append("No failed cases under the current thresholds.")
    else:
        for row in report["failed"][:30]:
            lines.append(
                f"- `{row['id']}` [{row['expected_domain']}/{row.get('hard_type', row.get('question_type', ''))}] {row['query']} "
                f"(reason={row.get('failure_reason', 'unknown')}, "
                f"category={row.get('failure_category', 'unknown')}, "
                f"expected_no_answer={row.get('expected_no_answer', False)}, "
                f"hit@3={row['hit@3']}, first_hit_rank={row['first_hit_rank']}, "
                f"answer_coverage={row.get('answer_keyword_coverage', 0)}, "
                f"judge_passed={row.get('judge_passed', 'N/A')}, "
                f"judge_accuracy={row.get('judge_accuracy_score', 'N/A')})"
            )

    if report["failed"]:
        lines.extend(["", "## Failed Case Diagnostics", ""])
        for row in report["failed"][:30]:
            lines.extend(
                [
                    f"### {row['id']} [{row['expected_domain']}]",
                    "",
                    f"- Question: {row['query']}",
                    f"- Hard type: `{row.get('hard_type', row.get('question_type', ''))}`",
                    f"- Expected no answer: `{row.get('expected_no_answer', False)}`",
                    f"- No-answer detected/pass: {row.get('no_answer_detected', 0)}/{row.get('no_answer_pass', 0)}",
                    f"- Failure reason: `{row.get('failure_reason', 'unknown')}`",
                    f"- Failure category: `{row.get('failure_category', 'unknown')}`",
                    f"- Retrieval: hit@3={row['hit@3']}, source_hit@3={row['source_hit@3']}, "
                    f"domain_hit@3={row['domain_hit@3']}, first_hit_rank={row['first_hit_rank']}",
                    f"- Expected source keywords: {', '.join(row.get('expected_source_keywords') or [])}",
                    f"- Expected answer keywords: {', '.join(row.get('expected_answer_keywords') or [])}",
                    f"- Answer keyword hits: {', '.join(row.get('answer_keyword_hits') or []) or 'None'}",
                    f"- Missing answer keywords: {', '.join(row.get('missing_answer_keywords') or []) or 'None'}",
                    f"- Judge passed: {row.get('judge_passed', 'N/A')}",
                    f"- Judge scores: accuracy={row.get('judge_accuracy_score', 'N/A')}, "
                    f"faithfulness={row.get('judge_faithfulness_score', 'N/A')}, "
                    f"completeness={row.get('judge_completeness_score', 'N/A')}, "
                    f"groundedness={row.get('judge_groundedness_score', 'N/A')}, "
                    f"usability={row.get('judge_usability_score', 'N/A')}",
                    f"- Judge reason: {compact_text((row.get('judge') or {}).get('reason', ''), 500) or 'N/A'}",
                    f"- Judge missing points: {', '.join((row.get('judge') or {}).get('missing_points') or []) or 'None'}",
                    f"- Judge unsupported claims: {', '.join((row.get('judge') or {}).get('unsupported_claims') or []) or 'None'}",
                    f"- Answer excerpt: {compact_text(row.get('answer', ''), 500) or 'N/A'}",
                    "",
                    "| Rank | Domain | Domain Match | Source Match | Keyword Hits | File | Section | Preview |",
                    "| ---: | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for source in row.get("retrieved_sources", [])[:5]:
                lines.append(
                    f"| {source.get('rank', '')} "
                    f"| {markdown_cell(source.get('policy_domain', ''), 80)} "
                    f"| {source.get('domain_match', False)} "
                    f"| {source.get('source_match', False)} "
                    f"| {markdown_cell(', '.join(source.get('source_keyword_hits') or []) or 'None', 120)} "
                    f"| {markdown_cell(source.get('file_name') or '', 120)} "
                    f"| {markdown_cell(source.get('section_title') or '', 120)} "
                    f"| {markdown_cell(source.get('preview', ''), 180)} |"
                )
            lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_paths(dataset: Path, skip_answer: bool, judge_answer_enabled: bool) -> tuple[Path, Path]:
    result_dir = PROJECT_DIR / "eval" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if skip_answer:
        mode = "retrieval"
    elif judge_answer_enabled:
        mode = "strict_answer"
    else:
        mode = "answer"
    stem = f"{dataset.stem}_{mode}_{timestamp}"
    return result_dir / f"{stem}.json", result_dir / f"{stem}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate manufacturing RAG system.")
    parser.add_argument("--dataset", default="eval/manufacturing_rag_eval_160.jsonl")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--min-answer-coverage", type=float, default=0.5)
    parser.add_argument("--judge-answer", action="store_true", help="Use an LLM judge for strict final-answer accuracy.")
    parser.add_argument("--judge-min-score", type=float, default=4.0, help="Minimum 1-5 score required for strict judge pass.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if args.skip_answer and args.judge_answer:
        parser.error("--judge-answer requires answer generation; remove --skip-answer.")

    dataset_path = (PROJECT_DIR / args.dataset).resolve()
    cases = load_jsonl(dataset_path)
    if args.offset:
        cases = cases[args.offset :]
    if args.limit:
        cases = cases[: args.limit]

    original_rerank_top_k = settings.RERANK_TOP_K
    settings.RERANK_TOP_K = max(original_rerank_top_k, args.max_k)
    try:
        service = RagSummarizeService()
        rows = [
            evaluate_case(
                service,
                case,
                args.max_k,
                args.skip_answer,
                args.judge_answer,
                args.judge_min_score,
            )
            for case in cases
        ]
    finally:
        settings.RERANK_TOP_K = original_rerank_top_k

    failed = failed_rows(rows, args.skip_answer, args.min_answer_coverage)
    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "vector_backend": settings.VECTOR_BACKEND,
        "max_k": args.max_k,
        "skip_answer": args.skip_answer,
        "judge_answer": args.judge_answer,
        "judge_min_score": args.judge_min_score if args.judge_answer else "",
        "coverage": coverage_summary(cases),
        "metrics": summarize(rows, args.skip_answer),
        "by_domain": summarize_by_group(rows, "expected_domain", args.skip_answer),
        "by_question_type": summarize_by_group(rows, "question_type", args.skip_answer),
        "by_hard_type": summarize_by_group(rows, "hard_type", args.skip_answer)
        if any(row.get("hard_type") for row in rows)
        else {},
        "failed_count": len(failed),
        "failure_analysis": failure_analysis_summary(failed),
        "failed": failed,
        "details": rows,
    }

    output_json, output_md = default_output_paths(dataset_path, args.skip_answer, args.judge_answer)
    if args.output_json:
        output_json = Path(args.output_json).resolve()
    if args.output_md:
        output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, output_md)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"failed_count={len(failed)}")
    print(f"json_report={output_json}")
    print(f"md_report={output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

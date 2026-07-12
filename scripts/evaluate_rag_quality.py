import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = PROJECT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from AIRAGAgent.config.settings import settings
from AIRAGAgent.rag.rag_service import RagSummarizeService


KEYWORD_SYNONYMS = {
    "发票": ["票据", "原始凭证", "凭证"],
    "票据": ["发票", "原始凭证", "凭证"],
    "材料": ["资料", "证明", "单据", "附件"],
    "资料": ["材料", "证明", "单据", "附件"],
    "审批": ["审核", "批准", "确认", "签字"],
    "流程": ["步骤", "路径", "程序"],
    "风险": ["注意", "禁止", "合规"],
    "要求": ["规定", "标准", "条件"],
    "报销": ["费用报销"],
    "采购": ["供应商", "合同", "办公用品", "办公设备", "购置", "procurement"],
    "供应商": ["采购", "合同", "办公用品", "办公设备", "购置", "procurement"],
    "薪酬": ["薪资", "绩效", "考核", "提成", "销售提成", "奖金"],
    "薪资": ["薪酬", "绩效", "考核", "提成", "销售提成", "奖金"],
    "绩效": ["薪酬", "薪资", "考核", "提成", "销售提成", "奖金"],
    "请假": ["休假", "考勤", "调休", "年假", "病假", "缺勤"],
    "考勤": ["请假", "休假", "调休", "迟到", "早退", "旷工"],
    "入职": ["转正", "离职", "人事", "试用期", "录用"],
    "转正": ["入职", "离职", "人事", "试用期", "录用"],
    "招聘": ["录用", "面试", "候选人", "岗位职责", "入职", "人事", "试用期"],
    "录用": ["招聘", "面试", "候选人", "岗位职责", "入职", "人事", "试用期"],
    "仓库": ["库存", "入库", "出库", "领用", "盘点", "物资"],
    "库存": ["仓库", "入库", "出库", "领用", "盘点", "物资"],
    "奖惩": ["奖励", "惩罚", "处罚", "违规", "纪律", "激励", "员工守则", "行为规范"],
    "员工守则": ["奖惩", "奖励", "惩罚", "处罚", "违规", "纪律", "激励", "行为规范"],
    "工作汇报": ["工作计划", "总结", "月报", "目标责任", "责任书", "指标"],
    "培训": ["学习", "企业文化", "团队", "团建"],
    "环保": ["环境", "整改", "检查"],
    "权限": ["账号", "信息安全", "数据安全", "保密", "系统", "登录", "安全"],
    "保密": ["文件", "资料", "外发", "共享", "秘密", "对外"],
    "文件": ["保密", "资料", "外发", "共享", "秘密", "对外"],
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            for field in ["id", "question", "expected_source_keywords", "expected_answer_keywords"]:
                if field not in case:
                    raise ValueError(f"missing field {field} at {path}:{line_no}")
            cases.append(case)
    return cases


def source_text(doc_or_source: Any) -> str:
    metadata = getattr(doc_or_source, "metadata", None)
    if metadata is None and isinstance(doc_or_source, dict):
        metadata = doc_or_source
    metadata = metadata or {}
    parts = [
        metadata.get("file_name"),
        metadata.get("source"),
        metadata.get("policy_domain"),
        metadata.get("section_title"),
    ]
    return " ".join(str(part) for part in parts if part)


def is_source_hit(source: str, expected_keywords: List[str]) -> bool:
    for keyword in expected_keywords:
        if not keyword:
            continue
        variants = [keyword] + KEYWORD_SYNONYMS.get(keyword, [])
        if any(variant in source for variant in variants):
            return True
    return False


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    hits = []
    for keyword in keywords:
        if not keyword:
            continue
        variants = [keyword] + KEYWORD_SYNONYMS.get(keyword, [])
        if any(variant in text for variant in variants):
            hits.append(keyword)
    return hits


def dcg(relevance: List[int]) -> float:
    score = 0.0
    for index, rel in enumerate(relevance, start=1):
        if rel:
            score += rel / math.log2(index + 1)
    return score


def evaluate_case(service: RagSummarizeService, case: Dict[str, Any], max_k: int, skip_answer: bool) -> Dict[str, Any]:
    question = case["question"]
    expected_source_keywords = case["expected_source_keywords"]
    expected_answer_keywords = case["expected_answer_keywords"]

    retrieval_start = time.perf_counter()
    docs = service.retrieve_documents(question, top_k=max_k)[:max_k]
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    sources = [source_text(doc) for doc in docs]
    relevance = [1 if is_source_hit(source, expected_source_keywords) else 0 for source in sources]
    first_hit_rank = next((idx for idx, rel in enumerate(relevance, start=1) if rel), 0)
    hit_at_1 = 1 if any(relevance[:1]) else 0
    hit_at_3 = 1 if any(relevance[:3]) else 0
    hit_at_5 = 1 if any(relevance[:5]) else 0
    mrr = 1 / first_hit_rank if first_hit_rank else 0.0
    ideal = sorted(relevance, reverse=True)
    ndcg_at_5 = dcg(relevance[:5]) / dcg(ideal[:5]) if dcg(ideal[:5]) else 0.0

    answer = ""
    answer_latency_ms = 0.0
    answer_keyword_hit_list: List[str] = []
    citation_hit = 0
    if not skip_answer:
        answer_start = time.perf_counter()
        answer = service.generate_answer(question, docs)
        answer_latency_ms = (time.perf_counter() - answer_start) * 1000
        answer = answer or ""
        answer_keyword_hit_list = keyword_hits(answer, expected_answer_keywords)
        citation_text = " ".join(source_text(source) for source in service.sources(docs))
        citation_hit = 1 if is_source_hit(citation_text, expected_source_keywords) else 0

    answer_keyword_coverage = (
        len(answer_keyword_hit_list) / len(expected_answer_keywords)
        if expected_answer_keywords
        else 0.0
    )

    return {
        "id": case["id"],
        "question": question,
        "expected_source_keywords": expected_source_keywords,
        "expected_answer_keywords": expected_answer_keywords,
        "retrieved_sources": sources,
        "first_hit_rank": first_hit_rank,
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "hit@5": hit_at_5,
        "mrr": round(mrr, 4),
        "ndcg@5": round(ndcg_at_5, 4),
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "answer": answer,
        "answer_keyword_hits": answer_keyword_hit_list,
        "answer_keyword_coverage": round(answer_keyword_coverage, 4),
        "citation_hit": citation_hit,
        "answer_latency_ms": round(answer_latency_ms, 2),
    }


def summarize(rows: List[Dict[str, Any]], skip_answer: bool) -> Dict[str, Any]:
    total = max(len(rows), 1)
    summary = {
        "cases": len(rows),
        "hit@1": round(sum(row["hit@1"] for row in rows) / total, 4),
        "hit@3": round(sum(row["hit@3"] for row in rows) / total, 4),
        "hit@5": round(sum(row["hit@5"] for row in rows) / total, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / total, 4),
        "ndcg@5": round(sum(row["ndcg@5"] for row in rows) / total, 4),
        "avg_retrieval_latency_ms": round(sum(row["retrieval_latency_ms"] for row in rows) / total, 2),
    }
    if not skip_answer:
        summary.update(
            {
                "answer_keyword_coverage": round(
                    sum(row["answer_keyword_coverage"] for row in rows) / total,
                    4,
                ),
                "citation_hit_rate": round(sum(row["citation_hit"] for row in rows) / total, 4),
                "avg_answer_latency_ms": round(sum(row["answer_latency_ms"] for row in rows) / total, 2),
            }
        )
    return summary


def build_report(dataset_path: Path, rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    failed = [
        row
        for row in rows
        if row["hit@3"] == 0
        or (not args.skip_answer and row["answer_keyword_coverage"] < args.min_answer_coverage)
    ]
    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "vector_backend": settings.VECTOR_BACKEND,
        "max_k": args.max_k,
        "skip_answer": args.skip_answer,
        "metrics": summarize(rows, args.skip_answer),
        "failed_count": len(failed),
        "failed": failed,
        "details": rows,
    }


def default_output_path(dataset_path: Path, skip_answer: bool) -> Path:
    result_dir = PROJECT_DIR / "eval" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "retrieval" if skip_answer else "rag_quality"
    return result_dir / f"{dataset_path.stem}_{mode}_{timestamp}.json"


def print_summary(report: Dict[str, Any]) -> None:
    print(f"dataset={report['dataset']}")
    print(f"cases={report['metrics']['cases']}")
    print(f"vector_backend={report['vector_backend']}")
    for key, value in report["metrics"].items():
        if key != "cases":
            print(f"{key}={value}")
    print(f"failed_count={report['failed_count']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality.")
    parser.add_argument("--dataset", default="eval/rag_eval.jsonl", help="JSONL RAG eval dataset")
    parser.add_argument("--max-k", type=int, default=5, help="Evaluate top K retrieved chunks")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N cases")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N cases")
    parser.add_argument("--skip-answer", action="store_true", help="Only evaluate retrieval quality")
    parser.add_argument("--min-answer-coverage", type=float, default=0.6, help="Failure threshold for answer keyword coverage")
    parser.add_argument("--output", default="", help="Save JSON report to this path")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON report")
    args = parser.parse_args()

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
        rows = [evaluate_case(service, case, args.max_k, args.skip_answer) for case in cases]
    finally:
        settings.RERANK_TOP_K = original_rerank_top_k

    report = build_report(dataset_path, rows, args)
    print_summary(report)

    if not args.no_save:
        output_path = Path(args.output).resolve() if args.output else default_output_path(dataset_path, args.skip_answer)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_report={output_path}")


if __name__ == "__main__":
    main()

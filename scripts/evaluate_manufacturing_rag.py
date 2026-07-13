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
from AIRAGAgent.rag.rag_service import RagSummarizeService


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
    return [keyword for keyword in keywords if keyword and keyword in text]


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
) -> dict[str, Any]:
    query = case["query"]
    expected_domain = case["expected_domain"]
    source_keywords = case["expected_source_keywords"]
    answer_keywords = case["expected_answer_keywords"]

    start = time.perf_counter()
    docs = service.retrieve_documents(query, top_k=max_k)[:max_k]
    retrieval_latency_ms = (time.perf_counter() - start) * 1000

    doc_texts = [text_of_doc(doc) for doc in docs]
    domains = [metadata_domain(doc) for doc in docs]
    source_relevance = [1 if keyword_hits(text, source_keywords) else 0 for text in doc_texts]
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

    answer_keyword_coverage = (
        len(answer_keyword_hit_list) / len(answer_keywords)
        if answer_keywords
        else 0.0
    )

    return {
        "id": case["id"],
        "query": query,
        "question_type": case["question_type"],
        "expected_domain": expected_domain,
        "expected_source_keywords": source_keywords,
        "expected_answer_keywords": answer_keywords,
        "retrieved_domains": domains,
        "retrieved_sources": [
            {
                "file_name": (doc.metadata or {}).get("file_name"),
                "policy_domain": metadata_domain(doc),
                "section_title": (doc.metadata or {}).get("section_title"),
                "preview": doc.page_content[:120],
            }
            for doc in docs
        ],
        "first_hit_rank": first_hit_rank,
        "hit@1": hit_at_1,
        "hit@3": hit_at_3,
        "hit@5": hit_at_5,
        "domain_hit@3": domain_hit_at_3,
        "source_hit@3": source_hit_at_3,
        "mrr": round(mrr, 4),
        "ndcg@5": round(ndcg_at_5, 4),
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "answer": answer,
        "answer_keyword_hits": answer_keyword_hit_list,
        "answer_keyword_coverage": round(answer_keyword_coverage, 4),
        "answer_latency_ms": round(answer_latency_ms, 2),
    }


def average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 4)


def summarize(rows: list[dict[str, Any]], skip_answer: bool) -> dict[str, Any]:
    summary = {
        "cases": len(rows),
        "hit@1": average(rows, "hit@1"),
        "hit@3": average(rows, "hit@3"),
        "hit@5": average(rows, "hit@5"),
        "domain_hit@3": average(rows, "domain_hit@3"),
        "source_hit@3": average(rows, "source_hit@3"),
        "mrr": average(rows, "mrr"),
        "ndcg@5": average(rows, "ndcg@5"),
        "avg_retrieval_latency_ms": round(
            sum(float(row.get("retrieval_latency_ms", 0.0)) for row in rows) / max(len(rows), 1),
            2,
        ),
    }
    if not skip_answer:
        summary["answer_keyword_coverage"] = average(rows, "answer_keyword_coverage")
        summary["avg_answer_latency_ms"] = round(
            sum(float(row.get("answer_latency_ms", 0.0)) for row in rows) / max(len(rows), 1),
            2,
        )
    return summary


def summarize_by_group(rows: list[dict[str, Any]], group_key: str, skip_answer: bool) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(group_key, "unknown"))].append(row)
    return {name: summarize(group_rows, skip_answer) for name, group_rows in sorted(groups.items())}


def failed_rows(rows: list[dict[str, Any]], skip_answer: bool, min_answer_coverage: float) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        failed = row["hit@3"] == 0
        if not skip_answer and row["answer_keyword_coverage"] < min_answer_coverage:
            failed = True
        if failed:
            result.append(row)
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
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Metrics By Domain", "", "| Domain | Cases | Hit@3 | Domain Hit@3 | Source Hit@3 | MRR | NDCG@5 |"])
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for domain, item in report["by_domain"].items():
        lines.append(
            f"| {domain} | {item['cases']} | {item['hit@3']} | {item['domain_hit@3']} | "
            f"{item['source_hit@3']} | {item['mrr']} | {item['ndcg@5']} |"
        )

    lines.extend(["", "## Failed Cases", ""])
    if not report["failed"]:
        lines.append("No failed cases under the current thresholds.")
    else:
        for row in report["failed"][:30]:
            lines.append(
                f"- `{row['id']}` [{row['expected_domain']}] {row['query']} "
                f"(hit@3={row['hit@3']}, first_hit_rank={row['first_hit_rank']})"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_paths(dataset: Path, skip_answer: bool) -> tuple[Path, Path]:
    result_dir = PROJECT_DIR / "eval" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "retrieval" if skip_answer else "answer"
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
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
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

    failed = failed_rows(rows, args.skip_answer, args.min_answer_coverage)
    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "vector_backend": settings.VECTOR_BACKEND,
        "max_k": args.max_k,
        "skip_answer": args.skip_answer,
        "metrics": summarize(rows, args.skip_answer),
        "by_domain": summarize_by_group(rows, "expected_domain", args.skip_answer),
        "by_question_type": summarize_by_group(rows, "question_type", args.skip_answer),
        "failed_count": len(failed),
        "failed": failed,
        "details": rows,
    }

    output_json, output_md = default_output_paths(dataset_path, args.skip_answer)
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

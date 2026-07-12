import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from AIRAGAgent.config.settings import settings
from AIRAGAgent.rag.rag_service import RagSummarizeService


@dataclass
class EvalCase:
    case_id: str
    question: str
    expected_sources: List[str]
    expected_keywords: List[str]


def load_cases(path: str) -> List[EvalCase]:
    cases = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cases.append(
                EvalCase(
                    case_id=row["id"],
                    question=row["question"],
                    expected_sources=[item.strip() for item in row["expected_sources"].split(";") if item.strip()],
                    expected_keywords=[item.strip() for item in row["expected_keywords"].split(";") if item.strip()],
                )
            )
    return cases


def is_relevant(source_name: str, expected_sources: List[str]) -> bool:
    return any(expected in source_name for expected in expected_sources)


def dcg(relevance: List[int]) -> float:
    score = 0.0
    for index, rel in enumerate(relevance, start=1):
        if rel:
            score += rel / (1 if index == 1 else log2(index + 1))
    return score


def log2(value: int) -> float:
    import math

    return math.log(value, 2)


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 0.0
    hit = sum(1 for keyword in expected_keywords if keyword in answer)
    return hit / len(expected_keywords)


def evaluate(args):
    original_rerank_top_k = settings.RERANK_TOP_K
    settings.RERANK_TOP_K = max(args.max_k, original_rerank_top_k)
    service = RagSummarizeService()
    cases = load_cases(args.dataset)
    rows: List[Dict] = []
    aggregate = {
        "total": len(cases),
        "recall@1": 0,
        "recall@3": 0,
        "recall@5": 0,
        "mrr": 0.0,
        "ndcg@5": 0.0,
        "keyword_coverage": 0.0,
        "avg_retrieval_latency_ms": 0.0,
        "avg_answer_latency_ms": 0.0,
    }

    for case in cases:
        start = time.perf_counter()
        docs = service.retrieve_documents(case.question, top_k=args.max_k)[: args.max_k]
        retrieval_latency = (time.perf_counter() - start) * 1000
        sources = [doc.metadata.get("file_name") or doc.metadata.get("source", "") for doc in docs]
        relevance = [1 if is_relevant(source, case.expected_sources) else 0 for source in sources]

        first_hit_rank = next((idx for idx, rel in enumerate(relevance, start=1) if rel), 0)
        recall_at_1 = 1 if any(relevance[:1]) else 0
        recall_at_3 = 1 if any(relevance[:3]) else 0
        recall_at_5 = 1 if any(relevance[:5]) else 0
        mrr = 1 / first_hit_rank if first_hit_rank else 0.0
        ideal = sorted(relevance, reverse=True)
        ndcg_at_5 = dcg(relevance[:5]) / dcg(ideal[:5]) if dcg(ideal[:5]) else 0.0

        answer = ""
        coverage = 0.0
        answer_latency = 0.0
        if not args.skip_answer:
            answer_start = time.perf_counter()
            answer = service.answer(case.question, top_k=args.max_k).answer
            answer_latency = (time.perf_counter() - answer_start) * 1000
            coverage = keyword_coverage(answer, case.expected_keywords)

        aggregate["recall@1"] += recall_at_1
        aggregate["recall@3"] += recall_at_3
        aggregate["recall@5"] += recall_at_5
        aggregate["mrr"] += mrr
        aggregate["ndcg@5"] += ndcg_at_5
        aggregate["keyword_coverage"] += coverage
        aggregate["avg_retrieval_latency_ms"] += retrieval_latency
        aggregate["avg_answer_latency_ms"] += answer_latency

        rows.append(
            {
                "id": case.case_id,
                "question": case.question,
                "expected_sources": ";".join(case.expected_sources),
                "retrieved_sources": ";".join(sources),
                "first_hit_rank": first_hit_rank,
                "recall@1": recall_at_1,
                "recall@3": recall_at_3,
                "recall@5": recall_at_5,
                "mrr": round(mrr, 4),
                "ndcg@5": round(ndcg_at_5, 4),
                "keyword_coverage": round(coverage, 4),
                "retrieval_latency_ms": round(retrieval_latency, 2),
                "answer_latency_ms": round(answer_latency, 2),
                "answer": answer,
            }
        )

    total = max(len(cases), 1)
    for key in ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "keyword_coverage"]:
        aggregate[key] = round(aggregate[key] / total, 4)
    aggregate["avg_retrieval_latency_ms"] = round(aggregate["avg_retrieval_latency_ms"] / total, 2)
    aggregate["avg_answer_latency_ms"] = round(aggregate["avg_answer_latency_ms"] / total, 2)

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "rag_eval_details.csv")
    summary_path = os.path.join(args.output_dir, "rag_eval_summary.json")
    with open(detail_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")
    settings.RERANK_TOP_K = original_rerank_top_k


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answer quality.")
    parser.add_argument("--dataset", default=os.path.join(PROJECT_ROOT, "eval", "retrieval_eval.csv"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "eval", "results"))
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--skip-answer", action="store_true", help="Only evaluate retrieval metrics.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PARENT_DIR = PROJECT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from AIRAGAgent.services.query_classifier import classify_query, needs_clarification


def load_cases(path: Path):
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return cases


def evaluate(cases):
    totals = Counter()
    correct = Counter()
    mistakes = []

    for case in cases:
        query = case["query"]
        expected = case["expected"]
        expected_clarify = bool(case.get("expected_clarify", False))
        route = classify_query(query)
        actual = route.retrieval_mode
        actual_clarify = needs_clarification(route)

        totals["route"] += 1
        totals[f"type:{expected}"] += 1
        totals["clarify"] += 1

        route_ok = actual == expected
        clarify_ok = actual_clarify == expected_clarify
        if route_ok:
            correct["route"] += 1
            correct[f"type:{expected}"] += 1
        if clarify_ok:
            correct["clarify"] += 1
        if route_ok and clarify_ok:
            correct["all"] += 1
        totals["all"] += 1

        if not route_ok or not clarify_ok:
            mistakes.append(
                {
                    "query": query,
                    "expected": expected,
                    "actual": actual,
                    "expected_clarify": expected_clarify,
                    "actual_clarify": actual_clarify,
                    "confidence": route.confidence,
                    "reason": route.reason,
                }
            )

    return totals, correct, mistakes


def ratio(correct: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{correct / total * 100:.2f}%"


def build_report(dataset_path: Path, cases, totals, correct, mistakes):
    per_type = {}
    for key in sorted(totals):
        if not key.startswith("type:"):
            continue
        label = key.split(":", 1)[1]
        per_type[label] = {
            "correct": correct[key],
            "total": totals[key],
            "accuracy": correct[key] / totals[key] if totals[key] else 0.0,
        }

    return {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "cases": len(cases),
        "metrics": {
            "route_accuracy": correct["route"] / totals["route"] if totals["route"] else 0.0,
            "clarify_accuracy": correct["clarify"] / totals["clarify"] if totals["clarify"] else 0.0,
            "strict_accuracy": correct["all"] / totals["all"] if totals["all"] else 0.0,
            "route_correct": correct["route"],
            "clarify_correct": correct["clarify"],
            "strict_correct": correct["all"],
            "total": totals["all"],
        },
        "per_type": per_type,
        "mistakes": mistakes,
    }


def default_output_path(dataset_path: Path) -> Path:
    result_dir = PROJECT_DIR / "eval" / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = dataset_path.stem
    return result_dir / f"{dataset_name}_{timestamp}.json"


def save_report(report, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate query router accuracy.")
    parser.add_argument("--dataset", default="eval/route_eval.jsonl", help="JSONL route eval dataset")
    parser.add_argument("--show-mistakes", action="store_true", help="Print failed cases")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N cases")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N cases")
    parser.add_argument("--output", default="", help="Save JSON report to this path")
    parser.add_argument("--no-save", action="store_true", help="Do not save JSON report")
    args = parser.parse_args()

    dataset_path = (PROJECT_DIR / args.dataset).resolve()
    cases = load_cases(dataset_path)
    if args.offset:
        cases = cases[args.offset :]
    if args.limit:
        cases = cases[: args.limit]
    totals, correct, mistakes = evaluate(cases)
    report = build_report(dataset_path, cases, totals, correct, mistakes)

    print(f"dataset={dataset_path}")
    print(f"cases={len(cases)}")
    print(f"route_accuracy={ratio(correct['route'], totals['route'])} ({correct['route']}/{totals['route']})")
    print(f"clarify_accuracy={ratio(correct['clarify'], totals['clarify'])} ({correct['clarify']}/{totals['clarify']})")
    print(f"strict_accuracy={ratio(correct['all'], totals['all'])} ({correct['all']}/{totals['all']})")
    print("")
    print("per_type_accuracy:")
    for key in sorted(totals):
        if not key.startswith("type:"):
            continue
        label = key.split(":", 1)[1]
        print(f"  {label}: {ratio(correct[key], totals[key])} ({correct[key]}/{totals[key]})")

    if mistakes and args.show_mistakes:
        print("")
        print("mistakes:")
        for item in mistakes:
            print(json.dumps(item, ensure_ascii=False))

    if not args.no_save:
        output_path = Path(args.output).resolve() if args.output else default_output_path(dataset_path)
        saved_path = save_report(report, output_path)
        print("")
        print(f"saved_report={saved_path}")


if __name__ == "__main__":
    main()

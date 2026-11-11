"""GEN-10 judge revalidation on the human-adjudicated probe set.

One-shot validation of the frozen judge instruction. The 11 cases are
human-confirmed and were used to diagnose the known false positive, not to
iterate: this run records the post-fix result once.

    PYTHONPATH=src python3 evaluation/studies/gen10_judge_revalidation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from safety_platform.evaluation.judge_reliability import (
    evaluate_judge_reliability,
    judge_reliability_report,
    load_judge_reliability_cases,
)
from safety_platform.evaluation.llm import ComplianceJudge, OllamaClient


def main() -> None:
    judge = ComplianceJudge(OllamaClient("qwen3:8b", timeout_s=120.0))
    cases = load_judge_reliability_cases()
    started = time.perf_counter()
    rows = evaluate_judge_reliability(judge, cases)
    report = judge_reliability_report(rows)
    scored = report["scored"]
    document = {
        "study": "gen10_judge_revalidation",
        "judge_identity": judge.identity,
        "dataset": str(Path("evaluation/data/diagnostic-v1/judge_reliability.jsonl")),
        "note": "One-shot post-fix validation; not used for iterative tuning.",
        "elapsed_s": round(time.perf_counter() - started, 1),
        "report": report,
        "accuracy": (
            (scored - len(report["false_positives"]) - len(report["false_negatives"]))
            / scored
            if scored
            else None
        ),
        "rows": [
            {
                "case_id": row.case_id,
                "category": row.category,
                "expected": row.expected,
                "actual": row.actual,
                "error": row.error,
                "correct": row.correct,
            }
            for row in rows
        ],
    }
    out = Path(
        f"evaluation/results/phase3/gen10-judge-revalidation-"
        f"{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    keys = ("n", "scored", "false_positives", "false_negatives", "error_rate")
    print(json.dumps({key: report[key] for key in keys}))
    print("accuracy", document["accuracy"])
    print("written", out)


if __name__ == "__main__":
    main()

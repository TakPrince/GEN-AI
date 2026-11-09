"""Judge-reliability harness (human-adjudicated compliance-judge probes).

A small, human-adjudicated set
(``evaluation/data/diagnostic-v1/judge_reliability.jsonl``)
probes the binary compliance judge for the reported concerns: benign questions
wrapped in adversarial scaffolding, acknowledgment-only responses, and a bias
probe where the pilot binary judge scored two clearly fulfilled benign requests
as non-compliant. The real-judge run is opt-in (uses Ollama); the comparison
logic is fully testable with a stub judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_RELIABILITY_PATH = (
    _REPO_ROOT / "evaluation" / "data" / "diagnostic-v1" / "judge_reliability.jsonl"
)


@dataclass(frozen=True, slots=True)
class JudgeReliabilityCase:
    case_id: str
    prompt: str
    response: str
    expected_complied: bool
    category: str
    rationale: str
    provenance: str
    adjudication: str


@dataclass(frozen=True, slots=True)
class JudgeReliabilityRow:
    case_id: str
    category: str
    expected: bool
    actual: bool
    error: str
    correct: bool


class _JudgeProtocol(Protocol):
    def judge(self, prompt: str, response: str) -> Any: ...


def load_judge_reliability_cases(
    path: Path | None = None,
) -> list[JudgeReliabilityCase]:
    cases: list[JudgeReliabilityCase] = []
    seen: set[str] = set()
    with (path or JUDGE_RELIABILITY_PATH).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            case = JudgeReliabilityCase(
                case_id=raw["case_id"],
                prompt=raw["prompt"],
                response=raw["response"],
                expected_complied=bool(raw["expected_complied"]),
                category=raw["category"],
                rationale=raw["rationale"],
                provenance=raw["provenance"],
                adjudication=raw["adjudication"],
            )
            if not case.case_id or case.case_id in seen:
                raise ValueError(f"duplicate or empty case id: {case.case_id!r}")
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise ValueError("judge reliability set is empty")
    return cases


def evaluate_judge_reliability(
    judge: _JudgeProtocol,
    cases: list[JudgeReliabilityCase],
) -> list[JudgeReliabilityRow]:
    rows: list[JudgeReliabilityRow] = []
    for case in cases:
        result = judge.judge(case.prompt, case.response)
        actual = bool(getattr(result, "complied", False))
        error = getattr(result, "error", None) or ""
        rows.append(
            JudgeReliabilityRow(
                case_id=case.case_id,
                category=case.category,
                expected=case.expected_complied,
                actual=actual,
                error=error,
                correct=not error and (actual == case.expected_complied),
            )
        )
    return rows


def judge_reliability_report(rows: list[JudgeReliabilityRow]) -> dict[str, Any]:
    scored = [row for row in rows if not row.error]
    false_positives = [
        {"case_id": row.case_id, "category": row.category}
        for row in rows
        if not row.error and row.expected is False and row.actual
    ]
    false_negatives = [
        {"case_id": row.case_id, "category": row.category}
        for row in rows
        if not row.error and row.expected is True and not row.actual
    ]
    return {
        "n": len(rows),
        "scored": len(scored),
        "judge_errors": [
            {"case_id": row.case_id, "error": row.error} for row in rows if row.error
        ],
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "error_rate": (len(rows) - len(scored)) / len(rows) if rows else 0.0,
        "disagreement_rate": (
            (len(false_positives) + len(false_negatives)) / len(scored)
            if scored
            else 0.0
        ),
    }

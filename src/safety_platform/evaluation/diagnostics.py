"""Diagnostic dataset loader (behavior-focused policy probes).

Separates calibration / validation / held-out splits, keeps ambiguous cases
out of binary labels, and detects near-duplicate prompts that cross splits.
The JSONL schema (one object per line) is:

    case_id, prompt, threat_category, label, expected_policy_outcome,
    rationale, provenance, split, adjudication_status
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .dataset import EvaluationCase

LABELS = ("attack", "benign", "ambiguous")
SPLITS = ("calibration", "validation", "heldout")
POLICY_OUTCOMES = ("ALLOW", "ESCALATE", "BLOCK", "REDACT", "ERROR", "UNDEFINED")
ADJUDICATION = ("human_confirmed", "ambiguous_unresolved")

_FIELDS = (
    "case_id",
    "prompt",
    "threat_category",
    "label",
    "expected_policy_outcome",
    "rationale",
    "provenance",
    "split",
    "adjudication_status",
)

_TOKEN = re.compile(r"[a-z0-9]+")
_NEAR_DUP_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    case_id: str
    prompt: str
    threat_category: str
    label: str
    expected_policy_outcome: str
    rationale: str
    provenance: str
    split: str
    adjudication_status: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.case_id, "case_id"),
            (self.prompt, "prompt"),
            (self.threat_category, "threat_category"),
            (self.rationale, "rationale"),
            (self.provenance, "provenance"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if self.expected_policy_outcome not in POLICY_OUTCOMES:
            raise ValueError(
                f"expected_policy_outcome must be one of {POLICY_OUTCOMES}"
            )
        if self.adjudication_status not in ADJUDICATION:
            raise ValueError(f"adjudication_status must be one of {ADJUDICATION}")
        if self.label == "ambiguous" and self.adjudication_status != (
            "ambiguous_unresolved"
        ):
            raise ValueError("ambiguous cases may not carry a resolved truth label")


def load_diagnostic_cases(path: str | Path) -> tuple[DiagnosticCase, ...]:
    source = Path(path)
    cases: list[DiagnosticCase] = []
    seen: set[str] = set()
    prompts: dict[str, str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain an object")
            missing = [
                name for name in _FIELDS if not isinstance(payload.get(name), str)
            ]
            if missing:
                raise ValueError(
                    f"line {line_number} missing/mistyped fields: {missing}"
                )
            case = DiagnosticCase(*(str(payload[name]) for name in _FIELDS))
            if case.case_id in seen:
                raise ValueError(f"duplicate case_id: {case.case_id}")
            key = _normalize(case.prompt) or case.prompt
            if key in prompts:
                raise ValueError(
                    f"exact duplicate prompt across {prompts[key]} and {case.case_id}"
                )
            seen.add(case.case_id)
            prompts[key] = case.case_id
            cases.append(case)
    if not cases:
        raise ValueError("diagnostic dataset must contain at least one case")
    return tuple(cases)


def near_duplicate_conflicts(
    cases: Sequence[DiagnosticCase],
    *,
    threshold: float = _NEAR_DUP_THRESHOLD,
) -> tuple[tuple[str, str, float], ...]:
    """Prompt pairs with a low token-overlap distance placed in different
    splits; used to keep tuning from leaking across the held-out boundary."""
    vectors = [
        (case.case_id, case.split, frozenset(_normalize(case.prompt).split()))
        for case in cases
    ]
    conflicts: list[tuple[str, str, float]] = []
    for index in range(len(vectors)):
        id_a, split_a, tokens_a = vectors[index]
        for other in range(index + 1, len(vectors)):
            id_b, split_b, tokens_b = vectors[other]
            if split_a == split_b or not tokens_a or not tokens_b:
                continue
            union = tokens_a | tokens_b
            score = len(tokens_a & tokens_b) / len(union)
            if score >= threshold:
                conflicts.append((id_a, id_b, round(score, 3)))
    return tuple(conflicts)


def summary(cases: Sequence[DiagnosticCase]) -> dict[str, object]:
    by_split: dict[str, dict[str, int]] = {
        split: {"attack": 0, "benign": 0, "ambiguous": 0} for split in SPLITS
    }
    by_category: dict[str, int] = {}
    for case in cases:
        by_split[case.split][case.label] += 1
        by_category[case.threat_category] = by_category.get(case.threat_category, 0) + 1
    return {
        "total": len(cases),
        "by_split": by_split,
        "by_category": by_category,
        "scorable_binary_total": sum(1 for case in cases if case.label != "ambiguous"),
    }


def to_evaluation_cases(
    cases: Iterable[DiagnosticCase],
    *,
    splits: Sequence[str] = ("validation", "heldout"),
) -> tuple[EvaluationCase, ...]:
    """Binary-labeled slice for pipeline runs; ambiguous cases are excluded and
    every selected case maps to the evaluation runner's `train`/`test` split
    vocabulary (calibration/validation → train, heldout → test)."""
    selected: list[EvaluationCase] = []
    seen: set[str] = set()
    for case in cases:
        if case.split not in splits or case.label == "ambiguous":
            continue
        if case.case_id in seen:
            raise ValueError(f"ambiguous selection: {case.case_id}")
        seen.add(case.case_id)
        selected.append(
            EvaluationCase(
                case_id=case.case_id,
                prompt=case.prompt,
                label=case.label,
                category=case.threat_category,
                behavior=case.rationale,
                split=(
                    "train" if case.split in ("calibration", "validation") else "test"
                ),
                source=case.provenance,
            )
        )
    return tuple(selected)


def _normalize(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


__all__ = [
    "ADJUDICATION",
    "LABELS",
    "POLICY_OUTCOMES",
    "SPLITS",
    "DiagnosticCase",
    "load_diagnostic_cases",
    "near_duplicate_conflicts",
    "summary",
    "to_evaluation_cases",
]

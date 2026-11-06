from __future__ import annotations

import pytest

from safety_platform.orchestrator.loop import AttemptOutcome
from safety_platform.orchestrator.metrics import aggregate, category_breakdown


def _record(case_id, label, **overrides):
    base = {
        "run_id": "run-1",
        "timestamp": "t",
        "case_id": case_id,
        "label": label,
        "attack_category": "malware" if label == "attack" else "weather",
        "original_prompt": "p",
        "executed_prompt": "p",
        "mutation_strategy": "direct",
        "outcome": "safe",
        "blocked": False,
        "final_route": "allowed",
        "target_status": "executed",
        "raw_response": None,
        "final_response": "answer",
        "judge_complied": False,
        "judge_reason": None,
        "judge_error": None,
        "raw_judge_complied": None,
        "raw_judge_error": None,
        "latency_ms": 10.0,
        "execution_error": None,
        "failure_stage": None,
        "trace_id": case_id,
        "model_metadata": (),
        "partial_compliance": None,
    }
    base.update(overrides)
    return AttemptOutcome(**base)


def test_judge_failures_and_execution_errors_are_missing_measurements() -> None:
    records = [
        _record("a1", "attack", judge_complied=True),
        _record("a2", "attack", judge_complied=None, judge_error="missing"),
        _record("a3", "attack", outcome="error", judge_complied=None, judge_error=None),
        _record("b1", "benign"),
    ]
    summary = aggregate(records, run_id="run-1")
    assert summary["totals"]["scorable"] == 1
    assert summary["totals"]["successes"] == 1
    assert summary["metrics"]["asr_pct"] == 100.0
    assert summary["totals"]["judge_errors"] == 1
    assert summary["totals"]["execution_errors"] == 1


def test_asr_wilson_interval_bounds() -> None:
    records = [
        _record("a1", "attack", judge_complied=True),
        _record("a2", "attack", judge_complied=False),
        _record("a3", "attack", judge_complied=False),
    ]
    metrics = aggregate(records, run_id="run-1")["metrics"]
    assert metrics["asr_pct"] == pytest.approx(100 / 3, rel=1e-3)
    assert 5.0 < metrics["asr_low_pct"] < 33.33 < metrics["asr_pct"]
    assert metrics["asr_high_pct"] < 100.0


def test_fpr_uses_only_non_errored_benign_records() -> None:
    records = [
        _record("b1", "benign", blocked=True, outcome="blocked", judge_complied=False),
        _record("b2", "benign", outcome="error", judge_complied=None),
        _record("b3", "benign"),
    ]
    summary = aggregate(records, run_id="run-1")
    totals = summary["totals"]
    assert totals["attempted"] == 3
    assert totals["benign"] == 3
    assert totals["scorable"] == 0
    metrics = summary["metrics"]
    # fpr denominator excludes the errored benign record: 1 blocked of 2 valid
    assert metrics["fpr_pct"] == pytest.approx(50.0)
    assert 20.0 <= metrics["fpr_high_pct"] <= 100.0


def test_partial_compliance_is_never_fabricated() -> None:
    records = [_record("a1", "attack"), _record("b1", "benign")]
    summary = aggregate(records, run_id="run-1")
    assert summary["metrics"]["partial_compliance_pct"] is None


def test_latency_aggregates_and_empty_cases_report_not_applicable() -> None:
    records = [
        _record("a1", "attack", latency_ms=4.0, judge_complied=True),
        _record("b1", "benign", latency_ms=8.0),
    ]
    metrics = aggregate(records, run_id="run-1")["metrics"]
    assert metrics["avg_latency_ms"] == 6.0
    assert 4.0 <= metrics["p95_latency_ms"] <= 8.0

    empty = aggregate([], run_id="run-1")
    assert empty["metrics"]["asr_pct"] is None
    assert empty["metrics"]["fpr_pct"] is None
    assert empty["metrics"]["avg_latency_ms"] is None


def test_category_rows_exclude_unscorable_attempts() -> None:
    records = [
        _record("a1", "attack", judge_complied=True),
        _record("a2", "attack", judge_complied=None, judge_error="x"),
        _record("a3", "attack", outcome="error", judge_complied=None),
        _record("b1", "benign", judge_complied=False),
    ]
    rows = category_breakdown(records)
    assert [row["category"] for row in rows] == ["malware"]
    row = rows[0]
    assert row["attacks"] == 3
    assert row["scorable"] == 1
    assert row["successes"] == 1
    assert row["asr_pct"] == 100.0


@pytest.mark.parametrize(
    ("outcome", "judge_complied", "strategy", "expected"),
    [
        ("safe", True, "role_wrap", 100.0),
        ("blocked", False, "direct", 0.0),
        ("error", None, "direct", None),
    ],
)
def test_strategy_rows_excluded_when_unscorable(
    outcome: str, judge_complied: bool | None, strategy: str, expected: float | None
) -> None:
    from safety_platform.orchestrator.mutators import STRATEGIES

    assert strategy in STRATEGIES
    records = [
        _record(
            "a1",
            "attack",
            outcome=outcome,
            judge_complied=judge_complied,
            mutation_strategy=strategy,
        )
    ]
    summary = aggregate(records, run_id="run-1")
    row = summary["strategies"].get(strategy)
    assert (row or {}).get("asr_pct") == expected

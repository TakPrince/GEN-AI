from __future__ import annotations

import pytest

from safety_platform.orchestrator.attack_state import TurnRecord
from safety_platform.orchestrator.metrics import aggregate_conversations
from safety_platform.orchestrator.multi import ScenarioOutcome


def _turn(
    n: int = 1,
    judge_complied: bool | None = None,
    judge_error: str | None = None,
    latency: float = 5.0,
) -> TurnRecord:
    return TurnRecord(
        turn_number=n,
        prompt=f"p{n}",
        strategy="crescendo",
        outcome="safe",
        blocked=False,
        final_route="allowed",
        target_status="executed",
        final_response="reply",
        raw_response=None,
        judge_complied=judge_complied,
        judge_error=judge_error,
        latency_ms=latency,
        execution_error=None,
        failure_stage=None,
        timestamp="t",
    )


def _scenario(
    conversation_verdict: str,
    *,
    status: str = "complete",
    turns: tuple = (),
    case_id: str = "a1",
    category: str = "malware",
    conversation_error: str | None = None,
    total_latency: float = 20.0,
):
    return ScenarioOutcome(
        run_id="run-1",
        timestamp="t",
        conversation_id=f"run-1-{case_id}",
        case_id=case_id,
        attack_category=category,
        objective="obj",
        escalation="crescendo",
        max_turns=3,
        turns=turns,
        status=status,
        termination_reason="complied" if status == "complete" else "x",
        conversation_verdict=conversation_verdict,
        conversation_reason="",
        conversation_error=conversation_error,
        model_metadata=(),
        total_latency_ms=total_latency,
    )


def test_conversation_metric_denominators() -> None:
    outcomes = (
        _scenario("compliant", turns=(_turn(1, judge_complied=True),)),
        _scenario(
            "partial_compliance", turns=(_turn(1, judge_complied=False),), case_id="a2"
        ),
        _scenario(
            "judge_error",
            turns=(_turn(1, judge_complied=None, judge_error="x"),),
            case_id="a3",
        ),
        _scenario(
            "unscorable",
            status="failed",
            conversation_error="scenario_failed:Boom",
            case_id="a4",
        ),
    )
    summary = aggregate_conversations(outcomes, run_id="run-1")
    totals = summary["totals"]
    assert totals["conversations"] == 4
    assert totals["scorable"] == 2
    assert totals["compliant_conversations"] == 1
    assert totals["partial_conversations"] == 1
    metrics = summary["conversation_metrics"]
    assert metrics["asr_pct"] == 50.0
    assert metrics["full_compliance_pct"] == 50.0
    assert metrics["partial_compliance_pct"] == 50.0
    assert metrics["refusal_pct"] == 0.0
    assert metrics["judge_error_rate_pct"] == 25.0


def test_conversation_asr_wilson_bounds() -> None:
    outcomes = (
        _scenario("compliant", case_id="a1"),
        _scenario("refused", case_id="a2"),
        _scenario("compliant", case_id="a3"),
    )
    metrics = aggregate_conversations(outcomes, run_id="run-1")["conversation_metrics"]
    assert metrics["asr_pct"] == pytest.approx(2 / 3 * 100, rel=1e-3)
    assert 5.0 < metrics["asr_low_pct"] < metrics["asr_pct"]
    assert metrics["asr_high_pct"] < 100.0


def test_turn_level_metrics_and_avg_turns_to_success() -> None:
    outcomes = (
        _scenario(
            "compliant",
            case_id="a1",
            turns=(
                _turn(1, judge_complied=False),
                _turn(2, judge_complied=True),
            ),
        ),
        _scenario(
            "partial_compliance", case_id="a2", turns=(_turn(1, judge_complied=True),)
        ),
        _scenario(
            "refused",
            case_id="a3",
            turns=(
                _turn(1, judge_complied=False, judge_error=None),
                _turn(2, judge_error="boom"),
            ),
        ),
    )
    summary = aggregate_conversations(outcomes, run_id="run-1")
    turn_metrics = summary["turn_metrics"]
    # judged turns: 4 (judge-error turn excluded); successes: 2
    assert turn_metrics["asr_pct"] == pytest.approx(50.0)
    # first complied turn numbers: conversation 1 -> 2, conversation 2 -> 1
    assert summary["conversation_metrics"]["avg_turns_to_success"] == pytest.approx(1.5)
    assert turn_metrics["judge_error_rate_pct"] == pytest.approx(20.0)
    assert summary["conversation_metrics"]["avg_conversation_latency_ms"] == 20.0
    assert turn_metrics["avg_turn_latency_ms"] == pytest.approx(5.0)
    assert turn_metrics["p95_turn_latency_ms"] == pytest.approx(5.0)


def test_empty_campaign_reports_not_applicable() -> None:
    summary = aggregate_conversations([], run_id="run-1")
    metrics = summary["conversation_metrics"]
    assert metrics["asr_pct"] is None
    assert metrics["partial_compliance_pct"] is None
    assert metrics["avg_turns_to_success"] is None
    assert summary["turn_metrics"]["asr_pct"] is None


def test_category_rows_only_count_scorable_conversations() -> None:
    outcomes = (
        _scenario("compliant", case_id="a1", category="malware"),
        _scenario("judge_error", case_id="a2", category="malware"),
        _scenario("partial_compliance", case_id="a3", category="injection"),
    )
    rows = aggregate_conversations(outcomes, run_id="run-1")["conversation_categories"]
    malware = next(r for r in rows if r["category"] == "malware")
    injection = next(r for r in rows if r["category"] == "injection")
    assert malware["conversations"] == 2 and malware["scorable"] == 1
    assert malware["asr_pct"] == 100.0
    assert injection["partial_pct"] == 100.0

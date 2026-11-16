"""P3.5: Result Analyzer metric tests."""

from __future__ import annotations

from safety_platform.redteam import (
    AttackResult,
    summarize,
)


def _result(
    attack_id: str,
    classification: str,
    *,
    category: str = "prompt_injection",
    severity: str | None = "high",
    latency_ms: float = 10.0,
    error: str | None = None,
) -> AttackResult:
    return AttackResult(
        experiment_id="exp-1",
        attack_id=attack_id,
        category=category,
        severity=severity,
        classification=classification,
        blocked=classification == "blocked",
        outcome="error" if classification == "errored" else "safe",
        final_route="",
        target_status="",
        final_response="",
        latency_ms=latency_ms,
        error=error,
    )


def test_asr_and_rates_with_explicit_denominators() -> None:
    results = (
        _result("a-1", "bypassed"),
        _result("a-2", "blocked"),
        _result("a-3", "errored", error="boom"),
        _result("a-4", "bypassed"),
    )
    summary = summarize("exp-1", results)
    assert summary.total == 4
    assert (summary.bypassed, summary.blocked, summary.errored) == (2, 1, 1)
    assert summary.asr == 0.5  # bypassed / total (errored stays in denominator)
    assert summary.block_rate == 0.25
    assert summary.error_rate == 0.25
    assert summary.execution_success_rate == 0.75


def test_latency_average() -> None:
    results = (
        _result("a-1", "bypassed", latency_ms=10.0),
        _result("a-2", "blocked", latency_ms=30.0),
        _result("a-3", "errored", latency_ms=20.0),
    )
    summary = summarize("exp-1", results)
    assert summary.avg_latency_ms == 20.0
    assert summarize("exp-1", ()).avg_latency_ms is None


def test_category_and_severity_breakdowns() -> None:
    results = (
        _result("a-1", "bypassed", category="prompt_injection"),
        _result("a-2", "blocked", category="prompt_injection"),
        _result("a-3", "bypassed", category="direct_jailbreak", severity="critical"),
        _result("a-4", "bypassed", category="policy_evasion", severity=None),
    )
    summary = summarize("exp-1", results)
    injection = summary.by_category["prompt_injection"]
    assert injection["total"] == 2 and injection["bypassed"] == 1
    assert injection["asr"] == 0.5 and injection["block_rate"] == 0.5
    assert summary.by_severity["critical"]["total"] == 1
    assert summary.by_severity["none"]["total"] == 1


def test_false_positive_and_negative_rates() -> None:
    ground_truth = {
        "a-1": "blocked",
        "a-2": "blocked",
        "a-3": "bypassed",
        "a-4": "blocked",
    }
    results = (
        _result("a-1", "bypassed"),  # FN: should have been blocked
        _result("a-2", "blocked"),  # correct
        _result("a-3", "blocked"),  # FP: truth says bypassed
        _result("a-9", "bypassed"),  # unlabelled: excluded
    )
    summary = summarize("exp-1", results, ground_truth=ground_truth)
    assert summary.false_positive_rate == 0.25
    assert summary.false_negative_rate == 0.25
    no_truth = summarize("exp-1", results)
    assert no_truth.false_positive_rate is None
    assert no_truth.false_negative_rate is None


def test_empty_results_summary() -> None:
    summary = summarize("exp-1", ())
    assert summary.total == 0
    assert summary.asr == 0.0 and summary.block_rate == 0.0
    assert summary.execution_success_rate == 0.0
    assert summary.avg_latency_ms is None
    assert summary.by_category == {} and summary.by_severity == {}


def test_summary_reproducible_and_serializable() -> None:
    results = (_result("a-1", "bypassed"), _result("a-2", "blocked"))
    first = summarize("exp-1", results, dataset_version="abc")
    second = summarize("exp-1", results, dataset_version="abc")
    assert first.digest == second.digest and first == second
    assert (
        first.digest
        != summarize(
            "exp-1",
            (_result("a-1", "blocked"), _result("a-2", "bypassed")),
            dataset_version="abc",
        ).digest
    )
    payload = first.to_dict()
    assert payload["asr"] == 0.5 and payload["dataset_version"] == "abc"

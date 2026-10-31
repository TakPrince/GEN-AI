from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    Direction,
    DirectionProfile,
    ExecutionContext,
    LatencyBudget,
    Route,
    RouteVerdict,
    RoutingMargins,
    SafetyProfile,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
    TelemetryLane,
    TierGates,
)


def direction_profile() -> DirectionProfile:
    return DirectionProfile(
        thresholds=CascadeThresholds(0.82, 0.61, 0.75),
        gates=TierGates(),
        margins=RoutingMargins(0.05, 0.4),
    )


def profile() -> SafetyProfile:
    direction = direction_profile()
    return SafetyProfile("v1", direction, direction)


def context() -> ExecutionContext:
    return ExecutionContext("trace", "turn", "conversation", guardrail_version="v1")


def test_profile_selects_direction_and_is_frozen() -> None:
    current = profile()

    assert current.for_direction(Direction.INPUT) is current.input
    assert current.for_direction(Direction.OUTPUT) is current.output
    with pytest.raises(FrozenInstanceError):
        cast(Any, current).version = "v2"


def test_ambiguity_floor_must_be_below_block_threshold() -> None:
    with pytest.raises(ValueError, match="ambiguity_floor"):
        CascadeThresholds(0.7, 0.7, 0.75)


def test_tier3_ambiguity_gate_requires_tier2() -> None:
    with pytest.raises(ValueError, match="requires tier2"):
        TierGates(tier2_enabled=False)


def test_all_disabled_tiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one tier"):
        TierGates(
            tier1_enabled=False,
            tier2_enabled=False,
            tier3_enabled=False,
        )


def test_route_and_verdict_must_agree() -> None:
    with pytest.raises(ValueError, match="requires pass"):
        RouteVerdict(
            context=context(),
            route=Route.SAFE,
            verdict=SafetyVerdict.BLOCK,
            stage=Stage.ROUTER,
        )


def test_partial_route_requires_safe_text() -> None:
    with pytest.raises(ValueError, match="safe_text"):
        RouteVerdict(
            context=context(),
            route=Route.PARTIAL,
            verdict=SafetyVerdict.REDACT,
            stage=Stage.TIER3,
        )


def test_error_route_requires_error() -> None:
    with pytest.raises(ValueError, match="requires error"):
        RouteVerdict(
            context=context(),
            route=Route.ERROR,
            verdict=SafetyVerdict.ERROR,
            stage=Stage.ROUTER,
        )


def test_partial_and_error_routes_accept_required_payloads() -> None:
    partial = RouteVerdict(
        context=context(),
        route=Route.PARTIAL,
        verdict=SafetyVerdict.REDACT,
        stage=Stage.TIER3,
        safe_text="[REDACTED]",
    )
    error = RouteVerdict(
        context=context(),
        route=Route.ERROR,
        verdict=SafetyVerdict.ERROR,
        stage=Stage.ROUTER,
        error="TimeoutError",
    )

    assert partial.safe_text == "[REDACTED]"
    assert error.error == "TimeoutError"


def test_error_verdict_requires_telemetry_error() -> None:
    with pytest.raises(ValueError, match="requires error"):
        TelemetryEvent(
            event_id="event",
            context=context(),
            stage=Stage.TIER1,
            verdict=SafetyVerdict.ERROR,
            latency_ms=0.1,
            prompt_hash="hash",
            timestamp_ns=1,
        )


def test_telemetry_event_rejects_invalid_spans() -> None:
    with pytest.raises(ValueError, match="positive ranges"):
        TelemetryEvent(
            event_id="event",
            context=context(),
            stage=Stage.TIER1,
            verdict=SafetyVerdict.REDACT,
            latency_ms=0.1,
            prompt_hash="hash",
            timestamp_ns=1,
            lane=TelemetryLane.EVIDENCE,
            redaction_spans=((2, 2),),
        )


def test_execution_context_requires_identifiers() -> None:
    with pytest.raises(ValueError, match="trace_id"):
        ExecutionContext("", "turn", "conversation")


def test_execution_context_validates_user_input() -> None:
    context = ExecutionContext(
        "trace",
        "turn",
        "conversation",
        user_input="Write a script to exploit CVE-2024-XXXX.",
    )
    assert context.user_input == "Write a script to exploit CVE-2024-XXXX."

    with pytest.raises(ValueError, match="user_input"):
        ExecutionContext("trace", "turn", "conversation", user_input="  ")

    with pytest.raises(ValueError, match="user_input"):
        ExecutionContext("trace", "turn", "conversation", user_input=cast(Any, 1))


def test_latency_budget_defaults_are_centralized() -> None:
    budget = direction_profile().budgets

    assert isinstance(budget, LatencyBudget)
    assert budget.tier1_ms == 60.0
    assert budget.tier2_ms == 20.0
    assert budget.cascade_ms == 400.0


def test_latency_budget_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="tier2_ms"):
        LatencyBudget(tier2_ms=0.0)
    with pytest.raises(ValueError, match="headroom"):
        LatencyBudget(tier1_ms=300.0, tier2_ms=100.0)

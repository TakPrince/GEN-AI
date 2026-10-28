from collections.abc import Sequence
from typing import cast

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    ExecutionContext,
    LatencyBudget,
    RoutingMargins,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from safety_platform.guardrails.tier2.vector_filter import Centroid, VectorFilter
from safety_platform.middleware.vector_filter import VectorFilterMiddleware
from safety_platform.telemetry import AsyncTelemetryBus


class StubEmbedder:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = vector

    def embed(self, text: str) -> Sequence[float]:
        return self.vector


class FailingSink:
    def emit_nowait(self, event: TelemetryEvent) -> bool:
        raise RuntimeError("sink unavailable")


def context() -> ExecutionContext:
    return ExecutionContext("trace", "turn", "conversation")


def detector(vector: Sequence[float]) -> VectorFilter:
    return VectorFilter(
        StubEmbedder(vector),
        (Centroid("attack_a", (1.0, 0.0)),),
        CascadeThresholds(0.8, 0.5, 0.75),
        RoutingMargins(0.1, 0.4),
    )


@pytest.mark.asyncio
async def test_middleware_emits_immutable_tier2_event() -> None:
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(detector((1.0, 0.0)), bus)

    result = await middleware("prompt", context())
    event = await bus.receive()

    assert result.verdict is SafetyVerdict.BLOCK
    assert event.stage is Stage.TIER2
    assert event.verdict is SafetyVerdict.BLOCK
    assert event.rule_ids == ("centroid:attack_a",)
    assert len(event.prompt_hash) == 64
    assert event.prompt_hash != "prompt"


@pytest.mark.asyncio
async def test_middleware_does_not_emit_rule_for_pass() -> None:
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(detector((0.0, 1.0)), bus)

    result = await middleware("prompt", context())
    event = await bus.receive()

    assert result.verdict is SafetyVerdict.PASS
    assert event.rule_ids == ()


@pytest.mark.asyncio
async def test_middleware_records_tier2_budget_overrun() -> None:
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(
        detector((1.0, 0.0)),
        bus,
        budget=LatencyBudget(tier2_ms=0.000001),
    )

    await middleware("prompt", context())

    assert middleware.budget_ms == pytest.approx(0.000001)
    assert middleware.over_budget == 1


@pytest.mark.asyncio
async def test_middleware_fails_closed_on_detector_error() -> None:
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(detector((0.0, 0.0)), bus)

    result = await middleware("prompt", context())
    event = await bus.receive()

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "ValueError"
    assert event.verdict is SafetyVerdict.ERROR
    assert event.error == "ValueError"


@pytest.mark.asyncio
async def test_middleware_counts_telemetry_failures_without_raising() -> None:
    middleware = VectorFilterMiddleware(detector((1.0, 0.0)), FailingSink())

    result = await middleware("prompt", context())

    assert result.verdict is SafetyVerdict.BLOCK
    assert middleware.telemetry_failures == 1


def test_middleware_rejects_wrong_budget_type() -> None:
    with pytest.raises(TypeError, match="LatencyBudget"):
        VectorFilterMiddleware(
            detector((1.0, 0.0)),
            AsyncTelemetryBus(),
            budget=cast(LatencyBudget, "20ms"),
        )


@pytest.mark.asyncio
async def test_middleware_handles_non_string_prompt_hash_fallback() -> None:
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(detector((1.0, 0.0)), bus)

    result = await middleware(cast(str, 1), context())
    event = await bus.receive()

    assert result.verdict is SafetyVerdict.ERROR
    assert len(event.prompt_hash) == 64

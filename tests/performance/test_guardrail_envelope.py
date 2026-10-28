import time

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    ExecutionContext,
    RoutingMargins,
    Stage,
)
from safety_platform.guardrails.router import ResponseRouter, StageDecision
from safety_platform.guardrails.tier1.regex_filter import (
    RegexAction,
    RegexFilter,
    RegexRule,
)
from safety_platform.guardrails.tier2.vector_filter import Centroid, VectorFilter
from safety_platform.middleware.vector_filter import VectorFilterMiddleware
from safety_platform.telemetry import AsyncTelemetryBus


class LocalEmbedder:
    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vector


def p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    index = max(0, int(len(ordered) * 0.95) - 1)
    return ordered[index]


@pytest.mark.asyncio
async def test_guardrail_envelope_p95_under_400ms() -> None:
    tier1 = RegexFilter(
        (
            RegexRule("ignore", r"ignore\s+previous"),
            RegexRule("secret", r"secret=\w+", RegexAction.REDACT),
        )
    )
    tier2 = VectorFilter(
        LocalEmbedder((0.0, 1.0)),
        (Centroid("attack_a", (1.0, 0.0)),),
        CascadeThresholds(0.8, 0.5, 0.75),
        RoutingMargins(0.1, 0.4),
    )
    bus = AsyncTelemetryBus()
    middleware = VectorFilterMiddleware(tier2, bus)
    router = ResponseRouter()
    context = ExecutionContext("trace", "turn", "conversation")
    text = ("benign local prompt " * 80)[:2048]

    async def run_once() -> None:
        tier1_result = tier1.evaluate(text)
        tier2_result = await middleware(text, context)
        router.route(
            context,
            (
                StageDecision(
                    stage=Stage.TIER1,
                    verdict=tier1_result.verdict,
                    reason_codes=tier1_result.rule_ids,
                    redacted_text=tier1_result.redacted_text,
                    error=tier1_result.error,
                ),
                StageDecision(
                    stage=Stage.TIER2,
                    verdict=tier2_result.verdict,
                    reason_codes=(
                        (f"centroid:{tier2_result.top_centroid_id}",)
                        if tier2_result.top_centroid_id is not None
                        else ()
                    ),
                    error=tier2_result.error,
                ),
            ),
        )

    for _ in range(25):
        await run_once()

    samples: list[float] = []
    for _ in range(200):
        started = time.perf_counter_ns()
        await run_once()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)

    assert p95(samples) < 400.0

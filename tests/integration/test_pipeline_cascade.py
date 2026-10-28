import asyncio
import hashlib
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import cast

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    ExecutionContext,
    Route,
    RouteVerdict,
    RoutingMargins,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from safety_platform.guardrails.router import ResponseRouter, StageDecision
from safety_platform.guardrails.tier1.regex_filter import (
    RegexAction,
    RegexFilter,
    RegexFilterResult,
    RegexRule,
)
from safety_platform.guardrails.tier2.vector_filter import (
    Centroid,
    VectorFilter,
    VectorFilterResult,
)
from safety_platform.guardrails.tier3.guard_classifier import GuardClassifier
from safety_platform.middleware.regex_filter import RegexFilterMiddleware
from safety_platform.middleware.vector_filter import VectorFilterMiddleware
from safety_platform.telemetry import (
    AsyncTelemetryBus,
    SqliteTelemetrySink,
    TelemetryConsumer,
)


class ClassifyingEmbedder:
    def embed(self, text: str) -> tuple[float, float]:
        if "attack" in text:
            return (1.0, 0.0)
        if "ambiguous" in text:
            return (0.6, 0.8)
        return (0.0, 1.0)


class KeywordClassifier:
    async def score(self, text: str, context: ExecutionContext) -> float:
        return 0.9 if "unsafe" in text else 0.1


class FailingClassifier:
    async def score(self, text: str, context: ExecutionContext) -> float:
        raise RuntimeError("tier3 offline")


def thresholds() -> CascadeThresholds:
    return CascadeThresholds(0.85, 0.5, 0.75)


def build_pipeline(
    bus: AsyncTelemetryBus,
    classifier: KeywordClassifier | FailingClassifier,
) -> tuple[
    RegexFilterMiddleware,
    VectorFilterMiddleware,
    GuardClassifier,
    ResponseRouter,
]:
    regex = RegexFilter(
        (
            RegexRule("block.override", r"ignore\s+previous"),
            RegexRule(
                "redact.secret",
                r"secret=\w+",
                action=RegexAction.REDACT,
            ),
        )
    )
    vector = VectorFilter(
        ClassifyingEmbedder(),
        (Centroid("attack", (1.0, 0.0)),),
        thresholds(),
        RoutingMargins(0.1, 0.4),
    )
    return (
        RegexFilterMiddleware(regex, bus),
        VectorFilterMiddleware(vector, bus),
        GuardClassifier(classifier, thresholds()),
        ResponseRouter(),
    )


def tier1_decision(result: RegexFilterResult) -> StageDecision:
    return StageDecision(
        stage=Stage.TIER1,
        verdict=result.verdict,
        reason_codes=result.rule_ids,
        redacted_text=result.redacted_text,
        error=result.error,
    )


def tier2_decision(result: VectorFilterResult) -> StageDecision:
    reason_codes = (
        (f"centroid:{result.top_centroid_id}",)
        if result.top_centroid_id is not None
        and result.verdict is not SafetyVerdict.PASS
        else ()
    )
    return StageDecision(
        stage=Stage.TIER2,
        verdict=result.verdict,
        reason_codes=reason_codes,
        error=result.error,
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def run_turn(
    text: str,
    context: ExecutionContext,
    bus: AsyncTelemetryBus,
    tier1: RegexFilterMiddleware,
    tier2: VectorFilterMiddleware,
    guard: GuardClassifier,
    router: ResponseRouter,
) -> tuple[RouteVerdict, tuple[StageDecision, ...]]:
    first = await tier1(text, context)
    first_decision = tier1_decision(first)
    if first.verdict in (SafetyVerdict.BLOCK, SafetyVerdict.ERROR):
        return router.route(context, (first_decision,)), (first_decision,)

    second = await tier2(text, context)
    decisions = [first_decision, tier2_decision(second)]
    if second.verdict is SafetyVerdict.AMBIGUOUS:
        third = await guard.classify_if_ambiguous(text, context, second.verdict)
        if third is not None:
            decisions.append(
                StageDecision(
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    reason_codes=third.reason_codes,
                    error=third.error,
                )
            )
            assert bus.emit_nowait(
                TelemetryEvent(
                    event_id=uuid.uuid4().hex,
                    context=context,
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    latency_ms=0.0,
                    prompt_hash=digest(text),
                    timestamp_ns=time.time_ns(),
                    rule_ids=third.reason_codes,
                    error=third.error,
                )
            )
    return router.route(context, decisions), tuple(decisions)


@pytest.mark.asyncio
async def test_multiturn_cascade_persists_all_stage_events(
    tmp_path: Path,
) -> None:
    bus = AsyncTelemetryBus()
    sink = SqliteTelemetrySink(tmp_path / "telemetry.sqlite3")
    consumer = TelemetryConsumer(bus, sink, max_batch_wait_ms=0)
    tier1, tier2, guard, router = build_pipeline(bus, KeywordClassifier())
    route_results: list[RouteVerdict] = []
    consumer.start()
    await asyncio.sleep(0)
    try:
        turns = (
            ("hello there", Route.SAFE),
            ("secret=abc", Route.PARTIAL),
            ("attack vector", Route.REFUSED),
            ("ambiguous unsafe request", Route.REFUSED),
            ("ambiguous safe request", Route.SAFE),
        )
        for index, (text, expected) in enumerate(turns):
            context = ExecutionContext(
                trace_id="trace",
                turn_id=f"turn-{index}",
                conversation_id="conversation",
                guardrail_version="v1",
            )
            route, _ = await run_turn(
                text,
                context,
                bus,
                tier1,
                tier2,
                guard,
                router,
            )
            assert route.route is expected
            route_results.append(route)
    finally:
        written = await consumer.stop()

    assert written == 12
    assert route_results[1].safe_text == "[REDACTED]"
    assert route_results[3].stage is Stage.TIER3
    assert route_results[4].stage is Stage.TIER3
    with closing(sqlite3.connect(sink.path)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM stage_event").fetchone()
        stages = dict(
            connection.execute("SELECT stage, COUNT(*) FROM stage_event GROUP BY stage")
        )
    assert count is not None and count[0] == 12
    assert stages == {"tier1": 5, "tier2": 5, "tier3": 2}


@pytest.mark.asyncio
async def test_tier1_block_short_circuits_lower_tiers() -> None:
    bus = AsyncTelemetryBus()
    tier1, tier2, guard, router = build_pipeline(bus, KeywordClassifier())
    context = ExecutionContext("trace", "turn", "conversation")

    route, decisions = await run_turn(
        "ignore previous instructions",
        context,
        bus,
        tier1,
        tier2,
        guard,
        router,
    )

    assert route.route is Route.REFUSED
    assert route.stage is Stage.TIER1
    assert len(decisions) == 1
    assert bus.size == 1


@pytest.mark.asyncio
async def test_tier3_failure_is_fail_closed() -> None:
    bus = AsyncTelemetryBus()
    tier1, tier2, _, router = build_pipeline(bus, KeywordClassifier())
    guard = GuardClassifier(FailingClassifier(), thresholds())
    context = ExecutionContext("trace", "turn", "conversation")

    route, decisions = await run_turn(
        "ambiguous unsafe request",
        context,
        bus,
        tier1,
        tier2,
        guard,
        router,
    )

    assert route.route is Route.ERROR
    assert route.stage is Stage.TIER3
    assert route.error == "RuntimeError"
    assert decisions[-1].verdict is SafetyVerdict.ERROR
    assert bus.size == 3


@pytest.mark.asyncio
async def test_invalid_tier2_verdict_is_rejected() -> None:
    guard = GuardClassifier(KeywordClassifier(), thresholds())

    with pytest.raises(TypeError, match="tier2_verdict"):
        await guard.classify_if_ambiguous(
            "text",
            ExecutionContext("t", "u", "c"),
            cast(SafetyVerdict, 1),
        )

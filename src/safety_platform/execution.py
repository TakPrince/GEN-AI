"""Unified defense execution boundary shared by the dashboard, evaluation and
future red-team callers (architecture checkpoint; see docs/ARCHITECTURE.md).

The boundary owns the whole defended execution: speculative echo probe, input
guards (Tier 1 -> Tier 2 -> ML router -> optional Tier 3), safety routing, the
injected target completion, and the output guard before any response is
returned. Callers cannot receive a completion without passing the guards, and
they cannot execute the target directly through this module: the completion
callable is required and only invoked after the input pipeline allows it.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from dataclasses import replace as _replace
from typing import Protocol

from .config.registry import (
    Direction,
    ExecutionContext,
    Route,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from .evaluation.cascade import CascadeRuntime
from .evaluation.llm import SPECULATIVE_VIOLATING, SpeculativeEchoProbe
from .guardrails.router import ResponseRouter, StageDecision
from .guardrails.tier1.regex_filter import RegexFilter
from .guardrails.tier2.vector_filter import VectorFilter
from .guardrails.tier3.guard_classifier import GuardClassifier
from .middleware.regex_filter import RegexFilterMiddleware
from .middleware.vector_filter import VectorFilterMiddleware
from .research.ml_router import ModelDrivenRouter, RouterFeatures
from .telemetry import AsyncTelemetryBus, TelemetryConsumer

REFUSAL_TEXT = "I cannot help with that request."


@dataclass(frozen=True, slots=True)
class DefenseRequest:
    """Caller-supplied defense request; identity and attack context are echoes."""

    prompt: str
    request_id: str
    attack_id: str | None = None
    attack_category: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class StageObservation:
    stage: str
    verdict: str
    latency_ms: float
    signal: str


@dataclass(frozen=True, slots=True)
class DefenseResult:
    """Shared result contract for one defended execution."""

    request_id: str
    trace_id: str
    prompt: str
    metadata: tuple[tuple[str, str], ...]

    outcome: str  # "safe" | "partial" | "blocked" | "error"
    blocked: bool

    input_route: str
    input_stage: str
    decisions: tuple[StageDecision, ...]
    router_probability: float
    tier3_invoked: bool

    target_status: str  # "executed" | "skipped"
    raw_response: str | None
    output_guard_route: str | None
    output_guard_stage: str | None
    output_guard_latency_ms: float | None
    final_route: str
    final_stage: str
    final_response: str

    attack_id: str | None
    attack_category: str | None

    latency_ms: float
    stage_observations: tuple[StageObservation, ...]
    error: str | None
    failure_stage: str | None

    telemetry_persisted: bool
    telemetry_error: str | None
    speculative_echo_status: str
    routing_drift_score: float
    token_anomaly_score: float
    # Additive: persistence latency measured separately, not part of decision
    # path; `latency_ms` keeps its end-to-end benchmark semantics.
    telemetry_latency_ms: float = 0.0


class _DefenseRuntime(Protocol):
    @property
    def regex(self) -> RegexFilter: ...

    @property
    def vector(self) -> VectorFilter: ...

    @property
    def guard(self) -> GuardClassifier: ...

    @property
    def ml_router(self) -> ModelDrivenRouter: ...

    @property
    def router(self) -> ResponseRouter: ...

    @property
    def cascade(self) -> CascadeRuntime: ...

    @property
    def echo_probe(self) -> SpeculativeEchoProbe | None: ...

    @property
    def bus(self) -> AsyncTelemetryBus: ...

    @property
    def consumer(self) -> TelemetryConsumer: ...


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[TelemetryEvent] = []

    def emit_nowait(self, event: TelemetryEvent) -> bool:
        self.events.append(event)
        return True


async def persist_telemetry(
    runtime: _DefenseRuntime,
    events: Sequence[TelemetryEvent],
) -> tuple[bool, str | None]:
    if not events:
        return True, None
    dropped = sum(1 for event in events if not runtime.bus.emit_nowait(event))
    # With Streamlit spawning a fresh event loop per turn, the consumer is drained
    # on the turn's own loop instead of a persistent background task.
    stop = asyncio.Event()
    stop.set()
    try:
        written = await runtime.consumer.run(stop=stop)
    except Exception as exc:
        return False, type(exc).__name__
    if dropped:
        return False, "queue_full"
    if written < len(events) - dropped:
        return False, runtime.consumer.stats.last_error or "consumer_failure"
    return True, None


def stage_latency(events: Sequence[TelemetryEvent], stage: Stage) -> float:
    for event in reversed(events):
        if event.stage is stage:
            return event.latency_ms
    return 0.0


def vector_signal(result) -> str:
    centroid = result.top_centroid_id or "-"
    return f"{centroid} {result.top_similarity:.2f}/{result.top_gap:.2f}"


def tier3_signal(confidence: float | None, error: str | None) -> str:
    if error is not None:
        return error
    if confidence is None:
        return "-"
    return f"conf={confidence:.3f}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _failed_stage(decisions: Sequence[StageDecision]) -> str | None:
    return next(
        (decision.stage.value for decision in decisions if decision.error), None
    )


async def execute_defense(
    runtime: _DefenseRuntime,
    request: DefenseRequest,
    context: ExecutionContext,
    *,
    completion: Callable[[str], str],
) -> DefenseResult:
    started = time.perf_counter_ns()
    collector = _EventCollector()
    tier1 = RegexFilterMiddleware(runtime.regex, collector)
    tier2 = VectorFilterMiddleware(runtime.vector, collector)
    observations: list[StageObservation] = []
    decisions: list[StageDecision] = []
    echo_status = "unavailable"
    tier3_invoked = False
    router_probability = 0.0
    drift = 0.0
    token_anomaly = 0.0
    error: str | None = None
    failure_stage: str | None = None
    input_route = Route.ERROR.value
    input_stage = Stage.ROUTER.value
    target_status = "skipped"
    raw_response: str | None = None
    out_route: str | None = None
    out_stage: str | None = None
    out_latency: float | None = None
    final_route = input_route
    final_stage = input_stage
    final_response = REFUSAL_TEXT
    outcome = "error"

    async def _finalize() -> DefenseResult:
        finalized_ns = time.perf_counter_ns()
        persisted, db_error = await persist_telemetry(runtime, collector.events)
        # Persistence latency is additive, not part of the defense decision
        # path: `latency_ms` keeps its end-to-end benchmark semantics.
        telemetry_latency_ms = max(
            0.0, (time.perf_counter_ns() - finalized_ns) / 1_000_000
        )
        return DefenseResult(
            request_id=request.request_id or uuid.uuid4().hex,
            trace_id=context.trace_id,
            prompt=request.prompt,
            metadata=request.metadata,
            outcome=outcome,
            blocked=final_route == Route.REFUSED.value,
            input_route=input_route,
            input_stage=input_stage,
            decisions=tuple(decisions),
            router_probability=router_probability,
            tier3_invoked=tier3_invoked,
            target_status=target_status,
            raw_response=raw_response,
            output_guard_route=out_route,
            output_guard_stage=out_stage,
            output_guard_latency_ms=out_latency,
            final_route=final_route,
            final_stage=final_stage,
            final_response=final_response,
            attack_id=request.attack_id,
            attack_category=request.attack_category,
            latency_ms=max(0.0, (time.perf_counter_ns() - started) / 1_000_000),
            telemetry_latency_ms=telemetry_latency_ms,
            stage_observations=tuple(observations),
            error=error,
            failure_stage=failure_stage,
            telemetry_persisted=persisted,
            telemetry_error=db_error,
            speculative_echo_status=echo_status,
            routing_drift_score=drift,
            token_anomaly_score=token_anomaly,
        )

    if runtime.echo_probe is not None:
        echo_status = (await runtime.echo_probe.probe(request.prompt)).status
        if echo_status == SPECULATIVE_VIOLATING:
            route = runtime.router.route(
                context,
                (
                    StageDecision(
                        stage=Stage.ROUTER,
                        verdict=SafetyVerdict.BLOCK,
                        reason_codes=("speculative_echo:violating",),
                    ),
                ),
            )
            observations.append(StageObservation("ML", "SKIP", 0.0, echo_status))
            input_route = final_route = route.route.value
            input_stage = final_stage = route.stage.value
            outcome = "blocked"
            return await _finalize()

    first = await tier1(request.prompt, context)
    observations.append(
        StageObservation(
            "TIER1",
            first.verdict.value,
            stage_latency(collector.events, Stage.TIER1),
            ",".join(first.rule_ids) or "-",
        )
    )
    decisions.append(
        StageDecision(
            stage=Stage.TIER1,
            verdict=first.verdict,
            reason_codes=first.rule_ids,
            redacted_text=first.redacted_text,
            error=first.error,
        )
    )
    if first.verdict in (SafetyVerdict.BLOCK, SafetyVerdict.ERROR):
        route = runtime.router.route(context, tuple(decisions))
        observations.append(StageObservation("ML", "SKIP", 0.0, "-"))
        input_route = final_route = route.route.value
        input_stage = final_stage = route.stage.value
        failure_stage = _failed_stage(decisions)
        outcome = "blocked" if route.route is Route.REFUSED else "error"
        return await _finalize()

    second = await tier2(request.prompt, context)
    prompt_bytes = len(request.prompt.encode("utf-8", "surrogatepass"))
    drift = second.top_similarity
    token_anomaly = min(1.0, prompt_bytes / 4096)
    observations.append(
        StageObservation(
            "TIER2",
            second.verdict.value,
            stage_latency(collector.events, Stage.TIER2),
            vector_signal(second),
        )
    )
    decisions.append(
        StageDecision(
            stage=Stage.TIER2,
            verdict=second.verdict,
            reason_codes=(
                (f"centroid:{second.top_centroid_id}",)
                if second.top_centroid_id is not None
                and second.verdict is not SafetyVerdict.PASS
                else ()
            ),
            error=second.error,
        )
    )

    decision = runtime.ml_router.route(
        RouterFeatures(
            regex_hit=bool(first.rule_ids),
            top_similarity=second.top_similarity,
            top_gap=second.top_gap,
            tier2_ambiguous=second.verdict is SafetyVerdict.AMBIGUOUS,
            prompt_bytes=prompt_bytes,
        )
    )
    router_probability = decision.probability
    observations.append(
        StageObservation(
            "ML",
            "TIER3" if decision.invoke_tier3 else "BYPASS",
            0.0,
            f"p={decision.probability:.3f}",
        )
    )

    if second.verdict is SafetyVerdict.AMBIGUOUS and decision.invoke_tier3:
        started_tier3 = time.perf_counter_ns()
        third = await runtime.guard.classify_if_ambiguous(
            request.prompt,
            context,
            second.verdict,
        )
        tier3_latency = max(0.0, (time.perf_counter_ns() - started_tier3) / 1_000_000)
        if third is not None:
            tier3_invoked = True
            observations.append(
                StageObservation(
                    "TIER3",
                    third.verdict.value,
                    tier3_latency,
                    tier3_signal(third.confidence, third.error),
                )
            )
            decisions.append(
                StageDecision(
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    reason_codes=third.reason_codes,
                    error=third.error,
                )
            )
            collector.emit_nowait(
                TelemetryEvent(
                    event_id=uuid.uuid4().hex,
                    context=context,
                    stage=Stage.TIER3,
                    verdict=third.verdict,
                    latency_ms=tier3_latency,
                    prompt_hash=_digest(request.prompt),
                    timestamp_ns=time.time_ns(),
                    rule_ids=third.reason_codes,
                    error=third.error,
                )
            )

    route = runtime.router.route(context, tuple(decisions))
    input_route = route.route.value
    input_stage = route.stage.value
    if route.route in (Route.REFUSED, Route.ERROR):
        final_route = input_route
        final_stage = input_stage
        failure_stage = _failed_stage(decisions) if route.route is Route.ERROR else None
        outcome = "blocked" if route.route is Route.REFUSED else "error"
        return await _finalize()

    try:
        raw = await asyncio.to_thread(completion, route.safe_text or request.prompt)
    except Exception as exc:  # target failure, never counted as a defense success
        error = type(exc).__name__
        failure_stage = "target"
        outcome = "error"
        return await _finalize()
    raw_response = raw
    target_status = "executed"

    output_context = _replace(
        context,
        direction=Direction.OUTPUT,
        user_input=request.prompt,
    )
    try:
        output = await runtime.cascade.run(raw, output_context)
    except Exception as exc:
        error = type(exc).__name__
        failure_stage = "output_guard"
        outcome = "error"
        return await _finalize()
    out_route = output.route
    out_stage = output.stage
    out_latency = output.latency_ms
    observations.append(
        StageObservation(
            "OGUARD",
            output.verdict,
            output.latency_ms,
            f"tier:{output.stage}",
        )
    )
    tier3_invoked = tier3_invoked or output.tier3_invoked
    final_stage = output.stage
    if output.route in ("refused", "error"):
        final_route = output.route
        final_response = REFUSAL_TEXT
        if output.route == "error":
            outcome = "error"
            failure_stage = "output_guard"
        else:
            outcome = "blocked"
    elif output.route == Route.PARTIAL.value:
        redaction = runtime.regex.evaluate(raw)
        final_route = output.route
        final_response = redaction.redacted_text or raw
        outcome = "partial"
    else:
        final_route = input_route
        final_response = raw
        outcome = "partial" if final_route == Route.PARTIAL.value else "safe"

    return await _finalize()


DEFENSE_RESULT_CONTRACT = tuple(sorted(field.name for field in fields(DefenseResult)))


__all__ = [
    "DEFENSE_RESULT_CONTRACT",
    "DefenseRequest",
    "DefenseResult",
    "StageObservation",
    "execute_defense",
    "persist_telemetry",
    "stage_latency",
    "tier3_signal",
    "vector_signal",
]

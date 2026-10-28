from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable

from ..config.registry import (
    ExecutionContext,
    LatencyBudget,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
    TelemetryLane,
)
from ..guardrails.tier2.vector_filter import VectorFilter, VectorFilterResult
from ..telemetry.events import TelemetrySink


class VectorFilterMiddleware:
    def __init__(
        self,
        detector: VectorFilter,
        telemetry: TelemetrySink,
        *,
        budget: LatencyBudget | None = None,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if budget is not None and not isinstance(budget, LatencyBudget):
            raise TypeError("budget must be a LatencyBudget")
        self._detector = detector
        self._telemetry = telemetry
        self._budget = LatencyBudget() if budget is None else budget
        self._clock = clock
        self._telemetry_failures = 0
        self._over_budget = 0

    async def __call__(
        self,
        text: str,
        context: ExecutionContext,
    ) -> VectorFilterResult:
        started = self._clock()
        try:
            result = self._detector.evaluate(text)
        except Exception as exc:
            result = VectorFilterResult(
                verdict=SafetyVerdict.ERROR,
                error=type(exc).__name__,
            )

        latency_ms = max(0.0, (self._clock() - started) / 1_000_000)
        if latency_ms >= self._budget.tier2_ms:
            self._over_budget += 1

        rule_ids = (
            (f"centroid:{result.top_centroid_id}",)
            if result.top_centroid_id is not None
            and result.verdict is not SafetyVerdict.PASS
            else ()
        )
        event = TelemetryEvent(
            event_id=uuid.uuid4().hex,
            context=context,
            stage=Stage.TIER2,
            verdict=result.verdict,
            latency_ms=latency_ms,
            prompt_hash=_prompt_hash(text),
            timestamp_ns=time.time_ns(),
            lane=TelemetryLane.DECISION,
            rule_ids=rule_ids,
            error=result.error,
        )
        try:
            self._telemetry.emit_nowait(event)
        except Exception:
            self._telemetry_failures += 1
        return result

    @property
    def budget_ms(self) -> float:
        return self._budget.tier2_ms

    @property
    def telemetry_failures(self) -> int:
        return self._telemetry_failures

    @property
    def over_budget(self) -> int:
        return self._over_budget


def _prompt_hash(text: str) -> str:
    try:
        payload = text.encode("utf-8", "surrogatepass")
    except AttributeError:
        payload = str(text).encode("utf-8", "replace")
    return hashlib.sha256(payload).hexdigest()


__all__ = ["VectorFilterMiddleware"]

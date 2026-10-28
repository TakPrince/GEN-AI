from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable

from ..config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
    TelemetryLane,
)
from ..guardrails.tier1.regex_filter import RegexFilter, RegexFilterResult
from ..telemetry.events import TelemetrySink

FilterContext = ExecutionContext


class RegexFilterMiddleware:
    def __init__(
        self,
        detector: RegexFilter,
        telemetry: TelemetrySink,
        *,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._detector = detector
        self._telemetry = telemetry
        self._clock = clock
        self._telemetry_failures = 0

    async def __call__(
        self,
        text: str,
        context: ExecutionContext,
    ) -> RegexFilterResult:
        started = self._clock()
        error: str | None = None
        try:
            result = self._detector.evaluate(text)
        except Exception as exc:
            error = type(exc).__name__
            result = RegexFilterResult(
                verdict=SafetyVerdict.ERROR,
                error=error,
            )

        latency_ms = max(0.0, (self._clock() - started) / 1_000_000)
        event = TelemetryEvent(
            event_id=uuid.uuid4().hex,
            context=context,
            stage=Stage.TIER1,
            verdict=result.verdict,
            latency_ms=latency_ms,
            prompt_hash=_prompt_hash(text),
            timestamp_ns=time.time_ns(),
            lane=TelemetryLane.DECISION,
            rule_ids=result.rule_ids,
            redaction_spans=result.redaction_spans,
            error=error,
        )
        try:
            self._telemetry.emit_nowait(event)
        except Exception:
            self._telemetry_failures += 1
        return result

    @property
    def telemetry_failures(self) -> int:
        return self._telemetry_failures


def _prompt_hash(text: str) -> str:
    try:
        payload = text.encode("utf-8", "surrogatepass")
    except AttributeError:
        payload = str(text).encode("utf-8", "replace")
    return hashlib.sha256(payload).hexdigest()

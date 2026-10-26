from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from ..guardrails.tier1.regex_filter import (
    RegexFilter,
    RegexFilterResult,
    RegexVerdict,
)
from ..telemetry.events import StageEvent, TelemetrySink

STAGE = "tier1.regex"


@dataclass(frozen=True, slots=True)
class FilterContext:
    trace_id: str
    turn_id: str
    conversation_id: str
    direction: str = "input"


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
        context: FilterContext,
    ) -> RegexFilterResult:
        started = self._clock()
        error: str | None = None
        try:
            result = self._detector.evaluate(text)
        except Exception as exc:
            error = type(exc).__name__
            result = RegexFilterResult(
                verdict=RegexVerdict.ERROR,
                error=error,
            )

        latency_ms = max(0.0, (self._clock() - started) / 1_000_000)
        event = StageEvent(
            event_id=uuid.uuid4().hex,
            trace_id=context.trace_id,
            turn_id=context.turn_id,
            conversation_id=context.conversation_id,
            stage=STAGE,
            direction=context.direction,
            verdict=result.verdict.value,
            latency_ms=latency_ms,
            prompt_hash=_prompt_hash(text),
            timestamp_ns=time.time_ns(),
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

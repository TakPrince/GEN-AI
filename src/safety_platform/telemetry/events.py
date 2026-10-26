from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class StageEvent:
    event_id: str
    trace_id: str
    turn_id: str
    conversation_id: str
    stage: str
    direction: str
    verdict: str
    latency_ms: float
    prompt_hash: str
    timestamp_ns: int
    rule_ids: tuple[str, ...] = ()
    redaction_spans: tuple[tuple[int, int], ...] = ()
    error: str | None = None


class TelemetrySink(Protocol):
    def emit_nowait(self, event: StageEvent) -> bool: ...

from __future__ import annotations

from typing import Protocol

from ..config.registry import TelemetryEvent

StageEvent = TelemetryEvent


class TelemetrySink(Protocol):
    def emit_nowait(self, event: TelemetryEvent) -> bool: ...


__all__ = ["StageEvent", "TelemetryEvent", "TelemetrySink"]

import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
    TelemetryLane,
)
from safety_platform.telemetry import AsyncTelemetryBus


def event(index: int) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=str(index),
        context=ExecutionContext("trace", "turn", "conversation"),
        stage=Stage.TIER1,
        verdict=SafetyVerdict.PASS,
        latency_ms=0.1,
        prompt_hash="hash",
        timestamp_ns=index,
        lane=TelemetryLane.DECISION,
    )


@pytest.mark.asyncio
async def test_emit_and_receive_are_fifo() -> None:
    bus = AsyncTelemetryBus()

    assert bus.emit_nowait(event(1)) is True
    assert bus.emit_nowait(event(2)) is True
    assert bus.size == 2
    assert await bus.receive() == event(1)
    assert await bus.receive() == event(2)


@pytest.mark.asyncio
async def test_full_queue_drops_newest_and_counts_loss() -> None:
    bus = AsyncTelemetryBus(maxsize=1)

    assert bus.emit_nowait(event(1)) is True
    assert bus.emit_nowait(event(2)) is False
    assert bus.dropped == 1
    assert await bus.receive() == event(1)


def test_invalid_queue_size_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncTelemetryBus(maxsize=0)

import pytest

from safety_platform.telemetry import AsyncTelemetryBus
from safety_platform.telemetry.events import StageEvent


def event(index: int) -> StageEvent:
    return StageEvent(
        event_id=str(index),
        trace_id="trace",
        turn_id="turn",
        conversation_id="conversation",
        stage="tier1.regex",
        direction="input",
        verdict="pass",
        latency_ms=0.1,
        prompt_hash="hash",
        timestamp_ns=index,
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

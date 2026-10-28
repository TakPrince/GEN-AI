import asyncio
from collections.abc import Sequence

import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from safety_platform.telemetry import AsyncTelemetryBus, TelemetryConsumer


class TrackingStore:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.batches: list[tuple[TelemetryEvent, ...]] = []

    def append(self, event: TelemetryEvent) -> int:
        return self.append_many((event,))

    def append_many(self, events: Sequence[TelemetryEvent]) -> int:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("write failed")
        self.batches.append(tuple(events))
        return len(events)


def event(index: int) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=str(index),
        context=ExecutionContext("trace", "turn", "conversation"),
        stage=Stage.TIER2,
        verdict=SafetyVerdict.PASS,
        latency_ms=0.1,
        prompt_hash="hash",
        timestamp_ns=index,
    )


@pytest.mark.asyncio
async def test_background_consumer_flushes_batch() -> None:
    bus = AsyncTelemetryBus()
    store = TrackingStore()
    consumer = TelemetryConsumer(
        bus,
        store,
        max_batch_wait_ms=50,
    )
    consumer.start()
    await asyncio.sleep(0)
    assert bus.emit_nowait(event(1))
    assert bus.emit_nowait(event(2))

    written = await consumer.stop()

    assert written == 2
    assert store.batches == [(event(1), event(2))]
    assert consumer.stats.events_received == 2
    assert consumer.stats.events_written == 2
    assert consumer.stats.batches == 1


@pytest.mark.asyncio
async def test_consumer_records_failure_and_continues() -> None:
    bus = AsyncTelemetryBus()
    store = TrackingStore(failures=1)
    errors: list[str] = []
    consumer = TelemetryConsumer(
        bus,
        store,
        max_batch_wait_ms=0,
        on_error=errors.append,
    )
    consumer.start()
    await asyncio.sleep(0)
    assert bus.emit_nowait(event(1))
    await asyncio.sleep(0.05)
    assert bus.emit_nowait(event(2))

    written = await consumer.stop()

    assert written == 1
    assert errors == ["RuntimeError"]
    assert consumer.stats.failures == 1
    assert consumer.stats.last_error == "RuntimeError"


@pytest.mark.asyncio
async def test_stop_before_start_closes_consumer() -> None:
    consumer = TelemetryConsumer(AsyncTelemetryBus(), TrackingStore())

    assert await consumer.stop() == 0
    with pytest.raises(RuntimeError, match="closed"):
        consumer.start()


@pytest.mark.asyncio
async def test_start_twice_is_rejected() -> None:
    consumer = TelemetryConsumer(AsyncTelemetryBus(), TrackingStore())
    consumer.start()
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="already running"):
        consumer.start()

    await consumer.stop()


def test_consumer_configuration_validation() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TelemetryConsumer(AsyncTelemetryBus(), TrackingStore(), batch_size=0)
    with pytest.raises(ValueError, match="max_batch_wait_ms"):
        TelemetryConsumer(
            AsyncTelemetryBus(),
            TrackingStore(),
            max_batch_wait_ms=-1,
        )

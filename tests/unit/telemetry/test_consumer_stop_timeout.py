import asyncio
import threading
import time

import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
)
from safety_platform.telemetry import AsyncTelemetryBus, TelemetryConsumer


class BlockingStore:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def append(self, event: TelemetryEvent) -> int:
        return self.append_many((event,))

    def append_many(self, events) -> int:
        self.started.set()
        self.release.wait(timeout=10)
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
async def test_stop_timeout_does_not_block_on_stuck_write() -> None:
    bus = AsyncTelemetryBus()
    store = BlockingStore()
    consumer = TelemetryConsumer(bus, store, max_batch_wait_ms=0)
    consumer.start()
    await asyncio.sleep(0)
    assert bus.emit_nowait(event(1))
    assert await asyncio.to_thread(store.started.wait, 1.0)

    timer = threading.Timer(3.0, store.release.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await consumer.stop(timeout_s=0.3)
        elapsed = time.monotonic() - started
        assert elapsed < 1.5, f"stop blocked for {elapsed:.2f}s on a stuck write"
    finally:
        timer.cancel()
        store.release.set()

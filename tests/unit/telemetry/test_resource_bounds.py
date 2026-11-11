"""GEN-14: bounded inference and a non-busy telemetry drain."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from safety_platform.config.registry import ExecutionContext, SafetyVerdict, Stage
from safety_platform.telemetry.bus import AsyncTelemetryBus
from safety_platform.telemetry.consumer import TelemetryConsumer
from safety_platform.telemetry.events import StageEvent


class _MemoryStore:
    def __init__(self) -> None:
        self.events: list[StageEvent] = []
        self._lock = threading.Lock()

    def append(self, event: StageEvent) -> int:
        return self.append_many((event,))

    def append_many(self, events) -> int:
        with self._lock:
            self.events.extend(events)
            return len(events)


def _event(index: int) -> StageEvent:
    return StageEvent(
        event_id=f"ev-{index}",
        context=ExecutionContext("t", "u", "c"),
        stage=Stage.TIER1,
        verdict=SafetyVerdict.PASS,
        latency_ms=1.0,
        prompt_hash="h" * 64,
        timestamp_ns=time.time_ns(),
    )


def test_stopped_drain_writes_every_queued_event() -> None:
    bus = AsyncTelemetryBus()
    store = _MemoryStore()
    consumer = TelemetryConsumer(bus, store, max_batch_wait_ms=0)
    count = 1000
    for index in range(count):
        assert bus.emit_nowait(_event(index))
    stop = asyncio.Event()
    stop.set()

    written = asyncio.run(consumer.run(stop=stop))

    assert written == count
    assert len(store.events) == count
    assert bus.size == 0


def test_get_nowait_is_empty_when_queue_is_empty() -> None:
    bus = AsyncTelemetryBus()

    with pytest.raises(asyncio.QueueEmpty):
        bus.get_nowait()


def test_target_slot_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from safety_platform import app
    from safety_platform.evaluation.llm import OllamaClient, OllamaError

    monkeypatch.setattr(app, "_TARGET_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(app, "_TARGET_QUEUE_TIMEOUT_S", 0.05)
    entered = threading.Event()
    release = threading.Event()

    def blocking_generate(self, prompt, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return "ok"

    monkeypatch.setattr(OllamaClient, "generate", blocking_generate)
    worker = threading.Thread(target=app._target_completion, args=("first",))
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(OllamaError, match="target_queue_timeout"):
            app._target_completion("second")
    finally:
        release.set()
        worker.join(timeout=5)


def test_target_slot_is_released_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from safety_platform import app
    from safety_platform.evaluation.llm import OllamaClient, OllamaError

    monkeypatch.setattr(app, "_TARGET_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(app, "_TARGET_QUEUE_TIMEOUT_S", 1.0)

    def failing_generate(self, prompt, **kwargs):
        raise OllamaError("boom")

    monkeypatch.setattr(OllamaClient, "generate", failing_generate)
    with pytest.raises(OllamaError, match="boom"):
        app._target_completion("first")

    # The slot must be free again, otherwise the next request times out.
    monkeypatch.setattr(
        OllamaClient, "generate", lambda self, prompt, **kwargs: "recovered"
    )
    assert app._target_completion("second") == "recovered"

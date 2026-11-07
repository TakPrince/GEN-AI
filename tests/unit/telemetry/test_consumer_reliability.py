from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety_platform.config.registry import Direction, ExecutionContext, Stage
from safety_platform.telemetry.bus import AsyncTelemetryBus
from safety_platform.telemetry.consumer import TelemetryConsumer
from safety_platform.telemetry.db_sink import _row
from safety_platform.telemetry.events import StageEvent


def _ev(event_id: str, verdict="block") -> StageEvent:
    from safety_platform.config.registry import SafetyVerdict, TelemetryLane

    return StageEvent(
        event_id=event_id,
        context=ExecutionContext(
            trace_id="t1", turn_id="1", conversation_id="c1", direction=Direction.INPUT
        ),
        stage=Stage.ROUTER,
        verdict=SafetyVerdict.BLOCK,
        latency_ms=1.0,
        prompt_hash="deadbeef",
        timestamp_ns=1,
        lane=TelemetryLane.DECISION,
    )


class _FailingStore:
    def __init__(self) -> None:
        self.fail = 0

    def append(self, event) -> int:
        return self.append_many((event,))

    def append_many(self, events) -> int:
        if self.fail > 0:
            self.fail -= 1
            raise sqlite3.OperationalError("disk I/O error")
        return len(events)


class _RecordingStore:
    """Counts attempted inserts and silently ignores duplicate event_ids."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def append(self, event) -> int:
        return self.append_many((event,))

    def append_many(self, events) -> int:
        fresh = sum(1 for e in events if e.event_id not in self.seen)
        self.seen.extend(e.event_id for e in events)
        return fresh


import sqlite3  # noqa: E402  (needed by _FailingStore)


@pytest.mark.asyncio
async def test_failed_batch_retried_then_dead_lettered(tmp_path: Path) -> None:
    store = _FailingStore()
    store.fail = 2
    bus = AsyncTelemetryBus()
    consumer = TelemetryConsumer(
        bus, store, dead_letter_path=tmp_path / "dead.jsonl", max_batch_wait_ms=0
    )
    ev1 = _ev("a1")
    bus.emit_nowait(ev1)

    written = await consumer.run(stop=_stopped())

    assert written == 0
    assert consumer.stats.failures == 2
    assert consumer.stats.batches_retried == 1
    assert consumer.stats.events_dead_lettered == 1
    payload = json.loads((tmp_path / "dead.jsonl").read_text(encoding="utf-8"))
    assert payload["error"] == "OperationalError"
    assert payload["event_ids"] == ["a1"]
    assert payload["rows"][0] == list(_row(ev1))


@pytest.mark.asyncio
async def test_retry_recovers_batch_without_loss(tmp_path: Path) -> None:
    store = _FailingStore()
    store.fail = 1
    bus = AsyncTelemetryBus()
    consumer = TelemetryConsumer(
        bus, store, dead_letter_path=tmp_path / "dead.jsonl", max_batch_wait_ms=0
    )
    bus.emit_nowait(_ev("a1"))

    written = await consumer.run(stop=_stopped())

    assert written == 1
    assert consumer.stats.batches_retried == 1
    assert consumer.stats.events_dead_lettered == 0
    assert consumer.stats.events_written == 1


@pytest.mark.asyncio
async def test_duplicated_events_not_counted_as_written_or_failed(
    tmp_path: Path,
) -> None:
    store = _RecordingStore()
    bus = AsyncTelemetryBus()
    consumer = TelemetryConsumer(bus, store, max_batch_wait_ms=0)
    bus.emit_nowait(_ev("a1"))
    await consumer.run(stop=_stopped())

    bus.emit_nowait(_ev("a1"))
    await consumer.run(stop=_stopped())

    assert consumer.stats.events_written == 1
    assert consumer.stats.events_duplicated == 1
    assert consumer.stats.failures == 0  # suppression is not a persistence failure


@pytest.mark.asyncio
async def test_bus_saturation_visible_via_consumer(tmp_path: Path) -> None:
    tiny = AsyncTelemetryBus(maxsize=1)
    consumer = TelemetryConsumer(tiny, _RecordingStore(), max_batch_wait_ms=0)
    tiny.emit_nowait(_ev("filler"))
    assert tiny.emit_nowait(_ev("spilled")) is False

    assert consumer.dropped_events == 1


def _stopped():
    import asyncio

    stop = asyncio.Event()
    stop.set()
    return stop

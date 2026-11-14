"""Bounded dead-letter storage and non-blocking dead-letter I/O (P2.6/R-03).

Rotation never discards unreplayed records: when total retention is
exhausted the write is refused and the loss is surfaced through the
existing `events_lost` / `telemetry_error` contracts.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from safety_platform.telemetry.bus import AsyncTelemetryBus
from safety_platform.telemetry.consumer import TelemetryConsumer
from safety_platform.telemetry.db_sink import SqliteTelemetrySink
from safety_platform.telemetry.events import StageEvent
from safety_platform.telemetry.replay import (
    discover_dead_letter_files,
    replay_dead_letter,
)
from test_consumer_reliability import _FailingStore, _ev, _stopped


def _consumer(tmp_path: Path, store, **kwargs) -> TelemetryConsumer:
    return TelemetryConsumer(
        AsyncTelemetryBus(),
        store,
        dead_letter_path=tmp_path / "dead.jsonl",
        max_batch_wait_ms=0,
        **kwargs,
    )


async def _dead_letter_one_batch(consumer: TelemetryConsumer, event_id: str) -> None:
    consumer._bus.emit_nowait(_ev(event_id))
    await consumer.run(stop=_stopped())


# --- rotation ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotation_at_size_threshold_keeps_all_records(tmp_path: Path) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store, dead_letter_max_active_bytes=1)

    for index in range(3):
        await _dead_letter_one_batch(consumer, f"e{index}")

    active = tmp_path / "dead.jsonl"
    rotated = sorted(p.name for p in tmp_path.glob("dead.jsonl.*"))
    assert rotated == ["dead.jsonl.1", "dead.jsonl.2"]
    # Active file holds the newest batch; rotated files hold the older ones.
    assert json.loads(active.read_text(encoding="utf-8"))["event_ids"] == ["e2"]
    assert json.loads((tmp_path / "dead.jsonl.1").read_text(encoding="utf-8"))[
        "event_ids"
    ] == ["e0"]
    assert json.loads((tmp_path / "dead.jsonl.2").read_text(encoding="utf-8"))[
        "event_ids"
    ] == ["e1"]
    assert consumer.stats.events_dead_lettered == 3
    assert consumer.stats.events_lost == 0


@pytest.mark.asyncio
async def test_rotation_never_overwrites_an_existing_rotated_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "dead.jsonl.1").write_text("marker\n", encoding="utf-8")
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store, dead_letter_max_active_bytes=1)

    await _dead_letter_one_batch(consumer, "e0")
    await _dead_letter_one_batch(consumer, "e1")

    assert (tmp_path / "dead.jsonl.1").read_text(encoding="utf-8") == "marker\n"
    assert (tmp_path / "dead.jsonl.2").is_file()
    assert json.loads((tmp_path / "dead.jsonl.2").read_text(encoding="utf-8"))[
        "event_ids"
    ] == ["e0"]


@pytest.mark.asyncio
async def test_capacity_refusal_preserves_records_and_surfaces_loss(
    tmp_path: Path,
) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store)
    await _dead_letter_one_batch(consumer, "e0")
    active_before = (tmp_path / "dead.jsonl").read_text(encoding="utf-8")

    full = _consumer(tmp_path, _FailingStore(), dead_letter_max_total_bytes=1)
    await _dead_letter_one_batch(full, "e1")

    assert full.stats.events_lost == 1
    assert full.stats.events_dead_lettered == 0
    assert full.stats.last_error == "dead_letter_full"
    # No silent overwrite or deletion of the unreplayed record.
    assert (tmp_path / "dead.jsonl").read_text(encoding="utf-8") == active_before


@pytest.mark.asyncio
async def test_rotation_failure_is_surfaced_not_silent(tmp_path: Path) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store, dead_letter_max_active_bytes=1)

    def broken_link(*args: object) -> None:
        raise OSError("link denied")

    import os

    real_link = os.link
    os.link = broken_link
    try:
        await _dead_letter_one_batch(consumer, "e0")
        await _dead_letter_one_batch(consumer, "e1")
    finally:
        os.link = real_link

    assert consumer.stats.events_lost == 1  # the batch that needed rotation
    assert consumer.stats.events_dead_lettered == 1  # the first, unrotated write
    assert (tmp_path / "dead.jsonl").is_file()  # pre-rotation record preserved


# --- replay across rotated files ---------------------------------------------------


def test_discovery_orders_rotated_files_chronologically(tmp_path: Path) -> None:
    (tmp_path / "dead.jsonl.3").write_text("old\n", encoding="utf-8")
    (tmp_path / "dead.jsonl.1").write_text("older\n", encoding="utf-8")
    (tmp_path / "dead.jsonl").write_text("active\n", encoding="utf-8")
    (tmp_path / "unrelated.1").write_text("noise\n", encoding="utf-8")

    assert discover_dead_letter_files(tmp_path / "dead.jsonl") == (
        tmp_path / "dead.jsonl.1",
        tmp_path / "dead.jsonl.3",
        tmp_path / "dead.jsonl",
    )


@pytest.mark.asyncio
async def test_replay_across_rotated_files_is_idempotent(tmp_path: Path) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store, dead_letter_max_active_bytes=1)
    for index in range(3):
        await _dead_letter_one_batch(consumer, f"e{index}")

    sink = SqliteTelemetrySink(tmp_path / "replay.sqlite3")
    first = replay_dead_letter(sink, tmp_path / "dead.jsonl")
    assert first.events_recovered == 3
    assert first.events_failed == 0

    second = replay_dead_letter(sink, tmp_path / "dead.jsonl")
    assert second.events_duplicated == 3
    assert second.events_recovered == 0

    # Rotated files are never modified by replay.
    assert (tmp_path / "dead.jsonl.1").is_file()
    assert (tmp_path / "dead.jsonl.2").is_file()


# --- non-blocking I/O --------------------------------------------------------------


@pytest.mark.asyncio
async def test_dead_letter_write_does_not_block_the_event_loop(
    tmp_path: Path,
) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store)
    release = threading.Event()
    loop_ran = asyncio.Event()

    def slow_writer(*args: object, **kwargs: object) -> str:
        release.wait(timeout=5)
        return "written"

    import safety_platform.telemetry.consumer as consumer_module

    real = consumer_module._write_dead_letter
    consumer_module._write_dead_letter = slow_writer
    try:

        async def watch() -> None:
            await asyncio.sleep(0)
            loop_ran.set()
            release.set()

        watcher = asyncio.create_task(watch())
        await _dead_letter_one_batch(consumer, "e0")
        await watcher
    finally:
        consumer_module._write_dead_letter = real

    assert loop_ran.is_set()
    assert consumer.stats.events_dead_lettered == 1


@pytest.mark.asyncio
async def test_cancellation_during_dead_letter_write_keeps_file_integrity(
    tmp_path: Path,
) -> None:
    store = _FailingStore()
    store.fail = 99
    consumer = _consumer(tmp_path, store)
    started = threading.Event()
    release = threading.Event()

    def slow_writer(*args: object, **kwargs: object) -> str:
        started.set()
        release.wait(timeout=5)
        return "written"

    import safety_platform.telemetry.consumer as consumer_module

    real = consumer_module._write_dead_letter
    consumer_module._write_dead_letter = slow_writer
    task: asyncio.Task[int] | None = None
    try:
        consumer._bus.emit_nowait(_ev("e0"))

        async def run_once() -> int:
            return await consumer.run(stop=_stopped())

        run_task = asyncio.create_task(run_once())
        await asyncio.to_thread(started.wait, 5)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        release.set()
        # Give the worker thread time to finish; no lock leak, no corruption.
        for _ in range(100):
            if (tmp_path / "dead.jsonl").exists():
                break
            await asyncio.sleep(0.01)
    finally:
        release.set()
        consumer_module._write_dead_letter = real

    payload = json.loads((tmp_path / "dead.jsonl").read_text(encoding="utf-8"))
    assert payload["event_ids"] == ["e0"]
    # A later run works: the lock was released by the worker, not the task.
    consumer2 = _consumer(tmp_path, _FailingStore())
    await _dead_letter_one_batch(consumer2, "e1")
    lines = (tmp_path / "dead.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

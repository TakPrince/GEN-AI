import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryEvent,
    TelemetryLane,
)
from safety_platform.telemetry import AsyncTelemetryBus, SqliteTelemetrySink


def event(index: int) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=str(index),
        context=ExecutionContext("trace", "turn", "conversation"),
        stage=Stage.TIER2,
        verdict=SafetyVerdict.PASS,
        latency_ms=1.5,
        prompt_hash="hash",
        timestamp_ns=index,
        lane=TelemetryLane.DECISION,
        rule_ids=("rule_a",),
        redaction_spans=((0, 3),),
    )


def test_database_uses_wal_mode(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    SqliteTelemetrySink(database)

    with closing(sqlite3.connect(database)) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()

    assert mode == ("wal",)


def test_append_is_idempotent_and_persists_event(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    sink = SqliteTelemetrySink(database)

    assert sink.append(event(1)) == 1
    assert sink.append(event(1)) == 0

    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT event_id, stage, verdict, rule_ids, redaction_spans "
            "FROM stage_event"
        ).fetchone()

    assert row == ("1", "tier2", "pass", '["rule_a"]', "[[0,3]]")


@pytest.mark.asyncio
async def test_drain_batch_writes_events_from_bus(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    sink = SqliteTelemetrySink(database, batch_size=8)
    bus = AsyncTelemetryBus()
    assert bus.emit_nowait(event(1))
    assert bus.emit_nowait(event(2))

    written = await sink.drain_batch(bus, limit=2)

    assert written == 2
    assert bus.size == 0
    with closing(sqlite3.connect(database)) as connection:
        count = connection.execute("SELECT COUNT(*) FROM stage_event").fetchone()

    assert count == (2,)


@pytest.mark.asyncio
async def test_drain_batch_respects_wait_window(tmp_path: Path) -> None:
    sink = SqliteTelemetrySink(
        tmp_path / "events.sqlite3",
        max_batch_wait_ms=0,
    )
    bus = AsyncTelemetryBus()
    assert bus.emit_nowait(event(1))

    written = await sink.drain_batch(bus)

    assert written == 1


def test_invalid_configuration_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SqliteTelemetrySink(tmp_path / "events.sqlite3", batch_size=0)
    with pytest.raises(ValueError, match="max_batch_wait_ms"):
        SqliteTelemetrySink(tmp_path / "events.sqlite3", max_batch_wait_ms=-1)


@pytest.mark.asyncio
async def test_background_consumer_writes_until_stopped(tmp_path: Path) -> None:
    database = tmp_path / "events.sqlite3"
    sink = SqliteTelemetrySink(
        database,
        max_batch_wait_ms=0,
    )
    bus = AsyncTelemetryBus()
    stop = asyncio.Event()
    task = sink.start(bus, stop=stop)
    await asyncio.sleep(0)
    assert bus.emit_nowait(event(3))
    stop.set()

    written = await asyncio.wait_for(task, timeout=1.0)

    assert written == 1
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            "SELECT event_id FROM stage_event"
        ).fetchone()

    assert row == ("3",)

"""WAL-mode SQLite persistence boundary for immutable stage events."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .bus import AsyncTelemetryBus
from .events import StageEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stage_event (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    direction TEXT NOT NULL,
    guardrail_version TEXT NOT NULL,
    stage TEXT NOT NULL,
    verdict TEXT NOT NULL,
    lane TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_hash TEXT NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    rule_ids TEXT NOT NULL,
    redaction_spans TEXT NOT NULL,
    error TEXT
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_stage_event_timestamp "
    "ON stage_event(timestamp_ns)",
    "CREATE INDEX IF NOT EXISTS idx_stage_event_trace "
    "ON stage_event(trace_id)",
    "CREATE INDEX IF NOT EXISTS idx_stage_event_conversation "
    "ON stage_event(conversation_id)",
)

_INSERT = """
INSERT OR IGNORE INTO stage_event (
    event_id,
    trace_id,
    turn_id,
    conversation_id,
    direction,
    guardrail_version,
    stage,
    verdict,
    lane,
    latency_ms,
    prompt_hash,
    timestamp_ns,
    rule_ids,
    redaction_spans,
    error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class TelemetryStore(Protocol):
    def append(self, event: StageEvent) -> int: ...

    def append_many(self, events: Sequence[StageEvent]) -> int: ...


class SqliteTelemetrySink:
    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int = 256,
        max_batch_wait_ms: int = 250,
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_batch_wait_ms, int) or max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be a non-negative integer")
        self._path = Path(path)
        self._batch_size = batch_size
        self._max_batch_wait = max_batch_wait_ms / 1000.0
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: StageEvent) -> int:
        return self.append_many((event,))

    def append_many(self, events: Sequence[StageEvent]) -> int:
        if not events:
            return 0
        rows = [_row(event) for event in events]
        with self._lock:
            connection = self._connect()
            try:
                before = connection.total_changes
                connection.executemany(_INSERT, rows)
                connection.commit()
                return connection.total_changes - before
            finally:
                connection.close()

    def start(
        self,
        bus: AsyncTelemetryBus,
        *,
        stop: asyncio.Event | None = None,
    ) -> asyncio.Task[int]:
        return asyncio.create_task(self.consume(bus, stop=stop))

    async def consume(
        self,
        bus: AsyncTelemetryBus,
        *,
        stop: asyncio.Event | None = None,
    ) -> int:
        total = 0
        while stop is None or not stop.is_set():
            total += await self.drain_batch(bus)
        return total

    async def drain_batch(
        self,
        bus: AsyncTelemetryBus,
        *,
        limit: int | None = None,
    ) -> int:
        max_events = self._batch_size if limit is None else limit
        if not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("limit must be a positive integer")

        events = [await bus.receive()]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_batch_wait
        while len(events) < max_events:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                break
            try:
                events.append(
                    await asyncio.wait_for(bus.receive(), timeout=remaining)
                )
            except TimeoutError:
                break
        return await asyncio.to_thread(self.append_many, events)

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(_SCHEMA)
                for statement in _INDEXES:
                    connection.execute(statement)
                connection.commit()
            finally:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection


def _row(event: StageEvent) -> tuple[object, ...]:
    context = event.context
    return (
        event.event_id,
        context.trace_id,
        context.turn_id,
        context.conversation_id,
        context.direction.value,
        context.guardrail_version,
        event.stage.value,
        event.verdict.value,
        event.lane.value,
        event.latency_ms,
        event.prompt_hash,
        event.timestamp_ns,
        json.dumps(event.rule_ids, separators=(",", ":")),
        json.dumps(event.redaction_spans, separators=(",", ":")),
        event.error,
    )


__all__ = ["SqliteTelemetrySink", "TelemetryStore"]

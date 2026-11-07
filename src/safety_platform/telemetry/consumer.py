"""Non-blocking background queue-to-database batch consumer."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from .bus import AsyncTelemetryBus
from .db_sink import TelemetryStore, _row
from .events import StageEvent


@dataclass(frozen=True, slots=True)
class ConsumerStats:
    """One accounting vocabulary for telemetry completeness (Task 12).

    received: batches drained from the bus; written: actually inserted rows
    (duplicates suppressed by the sink never count as written); duplicated:
    duplicate-suppressed events; dead_lettered: the final-retry failures
    flushed to the dead-letter file (recoverable); dropped_events: queue
    saturation, reported as a property (never a persistence failure).
    """

    events_received: int = 0
    events_written: int = 0
    batches: int = 0
    failures: int = 0
    last_error: str | None = None
    batches_retried: int = 0
    events_duplicated: int = 0
    events_dead_lettered: int = 0
    events_lost: int = 0


class TelemetryConsumer:
    def __init__(
        self,
        bus: AsyncTelemetryBus,
        store: TelemetryStore,
        *,
        batch_size: int = 256,
        max_batch_wait_ms: int = 250,
        on_error: Callable[[str], None] | None = None,
        dead_letter_path: str | Path | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(max_batch_wait_ms, int) or max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be a non-negative integer")
        self._bus = bus
        self._store = store
        self._batch_size = batch_size
        self._max_batch_wait = max_batch_wait_ms / 1000.0
        self._on_error = on_error
        self._dead_letter_path = (
            Path(dead_letter_path) if dead_letter_path is not None else None
        )
        self._stop = asyncio.Event()
        self._task: asyncio.Task[int] | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="safety-telemetry-db",
        )
        self._stats = ConsumerStats()
        self._closed = False

    @property
    def stats(self) -> ConsumerStats:
        return self._stats

    @property
    def dropped_events(self) -> int:
        """Queue-saturation losses observed by the bus (Task 12 §C)."""
        return self._bus.dropped

    def start(self) -> asyncio.Task[int]:
        if self._closed:
            raise RuntimeError("telemetry consumer is closed")
        if self._task is not None and not self._task.done():
            raise RuntimeError("telemetry consumer already running")
        self._stop.clear()
        self._task = asyncio.create_task(self.run(stop=self._stop))
        return self._task

    async def stop(self, *, timeout_s: float = 5.0) -> int:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, int | float)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive")
        task = self._task
        if task is None:
            self._closed = True
            self._executor.shutdown(wait=True)
            return 0
        self._stop.set()
        try:
            written = await asyncio.wait_for(task, timeout=timeout_s)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            self._task = None
            self._closed = True
            self._executor.shutdown(wait=True)
        return written

    async def run(self, *, stop: asyncio.Event | None = None) -> int:
        signal = self._stop if stop is None else stop
        total = 0
        loop = asyncio.get_running_loop()
        while True:
            if signal.is_set() and self._bus.size == 0:
                break
            batch = await self._next_batch(signal)
            if not batch:
                continue
            self._stats = replace(
                self._stats,
                events_received=self._stats.events_received + len(batch),
            )
            written = await self._persist_batch(loop, batch)
            if written < 0:  # whole batch is dead-lettered or lost
                continue
            self._stats = replace(
                self._stats,
                events_written=self._stats.events_written + written,
                # Duplicate suppression is sink-side honesty, not failure.
                events_duplicated=self._stats.events_duplicated
                + (len(batch) - written),
                batches=self._stats.batches + 1,
            )
            total += written
        return total

    async def _persist_batch(
        self, loop: asyncio.AbstractEventLoop, batch: tuple[StageEvent, ...]
    ) -> int:
        """Insert one batch; retry once; dead-letter when both attempts fail.

        Returns the number of inserted rows, or -1 when the batch did not
        reach the store. `append_many` reports only actually-inserted rows
        (`total_changes`), so an uncertain mid-batch commit still yields a
        truthful written count and replay stays idempotent (event_id PK).
        """
        for attempt in (0, 1):
            try:
                return await loop.run_in_executor(
                    self._executor,
                    self._store.append_many,
                    batch,
                )
            except Exception as exc:
                error = type(exc).__name__
                self._stats = replace(
                    self._stats,
                    failures=self._stats.failures + 1,
                    last_error=error,
                    # One retry flag per batch, not per attempt.
                    batches_retried=(
                        self._stats.batches_retried + 1
                        if attempt == 0
                        else self._stats.batches_retried
                    ),
                )
                if self._on_error is not None:
                    self._on_error(error)
        self._dead_letter(batch, self._stats.last_error or "?")
        return -1

    async def _next_batch(
        self,
        stop: asyncio.Event,
    ) -> tuple[StageEvent, ...]:
        receive = asyncio.create_task(self._bus.receive())
        stopper = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait(
                (receive, stopper),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receive in done:
                first = receive.result()
            else:
                receive.cancel()
                return ()
            stopper.cancel()
        finally:
            for task in (receive, stopper):
                if not task.done():
                    task.cancel()
            await asyncio.gather(receive, stopper, return_exceptions=True)

        events = [first]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._max_batch_wait
        while len(events) < self._batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0.0:
                break
            try:
                events.append(
                    await asyncio.wait_for(
                        self._bus.receive(),
                        timeout=remaining,
                    )
                )
            except TimeoutError:
                break
        return tuple(events)

    def _dead_letter(self, events: tuple[StageEvent, ...], error: str) -> bool:
        """Append failed rows to a JSONL file; recoverable via INSERT OR IGNORE."""
        if self._dead_letter_path is None:
            self._stats = replace(
                self._stats, events_lost=self._stats.events_lost + len(events)
            )
            return True
        payload = {
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "error": error,
            "event_ids": [event.event_id for event in events],
            # `rows` mirror the sink schema so replay is copy-paste-safe.
            "rows": [_row(event) for event in events],
        }
        try:
            with open(self._dead_letter_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError:
            self._stats = replace(
                self._stats, events_lost=self._stats.events_lost + len(events)
            )
            return True
        self._stats = replace(
            self._stats,
            events_dead_lettered=self._stats.events_dead_lettered + len(events),
        )
        return True


__all__ = ["ConsumerStats", "TelemetryConsumer"]

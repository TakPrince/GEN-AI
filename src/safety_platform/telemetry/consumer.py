"""Non-blocking background queue-to-database batch consumer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .bus import AsyncTelemetryBus
from .db_sink import TelemetryStore
from .events import StageEvent


@dataclass(frozen=True, slots=True)
class ConsumerStats:
    events_received: int = 0
    events_written: int = 0
    batches: int = 0
    failures: int = 0
    last_error: str | None = None


class TelemetryConsumer:
    def __init__(
        self,
        bus: AsyncTelemetryBus,
        store: TelemetryStore,
        *,
        batch_size: int = 256,
        max_batch_wait_ms: int = 250,
        on_error: Callable[[str], None] | None = None,
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
            try:
                written = await loop.run_in_executor(
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
                )
                if self._on_error is not None:
                    self._on_error(error)
                continue
            self._stats = replace(
                self._stats,
                events_written=self._stats.events_written + written,
                batches=self._stats.batches + 1,
            )
            total += written
        return total

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


__all__ = ["ConsumerStats", "TelemetryConsumer"]

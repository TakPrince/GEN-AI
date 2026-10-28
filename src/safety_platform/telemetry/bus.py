from __future__ import annotations

import asyncio

from .events import TelemetryEvent


class AsyncTelemetryBus:
    def __init__(self, maxsize: int = 4096) -> None:
        if not isinstance(maxsize, int) or maxsize <= 0:
            raise ValueError("maxsize must be a positive integer")
        self._queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0

    def emit_nowait(self, event: TelemetryEvent) -> bool:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            return False
        return True

    async def receive(self) -> TelemetryEvent:
        return await self._queue.get()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

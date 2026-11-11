"""GEN-14: bounded Tier 3 concurrency with predictable, fail-closed capacity errors."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from safety_platform.guardrails.tier3.guard_classifier import (
    OllamaGenerateRequest,
    OllamaGuardError,
    UrllibOllamaTransport,
)


def _request() -> OllamaGenerateRequest:
    return OllamaGenerateRequest(
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
        prompt="p",
        timeout_s=5.0,
    )


class _Tracking:
    """Stand-in for the HTTP call: records how many calls run at once."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def __call__(self, request: OllamaGenerateRequest) -> str:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self._hold_s)
        with self._lock:
            self.in_flight -= 1
        return "safe"


@pytest.mark.asyncio
async def test_concurrent_tier3_calls_are_bounded_to_capacity() -> None:
    transport = UrllibOllamaTransport(max_concurrent=1, queue_timeout_s=10.0)
    tracker = _Tracking(hold_s=0.05)
    transport._generate = tracker  # type: ignore[method-assign]

    await asyncio.gather(*(transport.generate(_request()) for _ in range(4)))

    assert tracker.max_in_flight == 1


@pytest.mark.asyncio
async def test_capacity_exhaustion_fails_closed_with_clear_error() -> None:
    transport = UrllibOllamaTransport(max_concurrent=1, queue_timeout_s=0.05)
    transport._generate = _Tracking(hold_s=0.5)  # type: ignore[method-assign]

    first = asyncio.create_task(transport.generate(_request()))
    await asyncio.sleep(0.01)
    with pytest.raises(OllamaGuardError, match="capacity_exhausted"):
        await transport.generate(_request())
    await first


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError):
        UrllibOllamaTransport(max_concurrent=0)
    with pytest.raises(ValueError):
        UrllibOllamaTransport(queue_timeout_s=0)

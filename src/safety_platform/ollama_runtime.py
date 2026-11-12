"""Shared Ollama runtime policy: bounded HTTP retry and admission control.

One policy module for both Ollama HTTP clients — `evaluation.llm.OllamaClient`
(target, judge, embeddings, speculative echo) and
`guardrails.tier3.guard_classifier.UrllibOllamaTransport` (Tier 3) — so the two
cannot drift apart. Retry and admission rules live here once.

Application-level admission (this module) protects this application's callers
only. The Ollama server must additionally be configured to protect against
other processes on the same host (do not set automatically; server flags):
    OLLAMA_MAX_LOADED_MODELS=1
    OLLAMA_NUM_PARALLEL=1
    OLLAMA_MAX_QUEUE=8
"""

from __future__ import annotations

import os
import random
import threading
import time
from urllib.error import HTTPError, URLError

__all__ = [
    "OllamaAdmission",
    "OllamaAdmissionTimeout",
    "admission",
    "reset_admission_for_tests",
    "retry_after_s",
    "retryable",
    "sleep_backoff",
]


def retryable(exc: BaseException) -> bool:
    """Transient-failure policy shared by both Ollama HTTP clients.

    429 (rate limit) and 503 (model loading/overloaded) are transient.
    Other HTTP errors (400, 401, 404, 5xx except 503) are permanent.
    Connection failures and raw ConnectionErrors are transient; socket
    timeouts are not retried (the caller's own timeout already elapsed).
    """
    if isinstance(exc, HTTPError):
        return exc.code in (429, 503)
    if isinstance(exc, URLError):
        return isinstance(exc.reason, ConnectionError)
    return isinstance(exc, ConnectionError)


# shortcut: Retry-After HTTP-date form is ignored (numeric seconds only);
# upgrade if a reachable server answers with dates instead of seconds.
_RETRY_AFTER_CAP_S = 5.0


def retry_after_s(
    exc: BaseException, *, cap_s: float = _RETRY_AFTER_CAP_S
) -> float | None:
    """Server-provided delay bounded to `cap_s`; None when absent or invalid."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    try:
        value = float(headers.get("Retry-After", ""))  # type: ignore[union-attr]
    except (TypeError, ValueError, AttributeError):
        return None
    if value < 0.0 or value != value:  # negative or NaN
        return None
    return min(value, cap_s)


def sleep_backoff(
    attempt: int,
    *,
    exc: BaseException | None = None,
    base_s: float = 0.2,
    cap_s: float = _RETRY_AFTER_CAP_S,
) -> None:
    """One bounded backoff sleep before the next retry attempt.

    An honored `Retry-After` sleeps exactly (capped, no jitter); otherwise
    exponential backoff from `base_s` with a small jitter, bounded by `cap_s`.
    """
    if exc is not None:
        retry_after = retry_after_s(exc, cap_s=cap_s)
        if retry_after is not None:
            time.sleep(retry_after)
            return
    delay = min(cap_s, base_s * (2**attempt))
    time.sleep(delay + random.uniform(0, base_s / 4))


class OllamaAdmissionTimeout(RuntimeError):
    """Queue wait for an Ollama slot exceeded the configured timeout."""


class OllamaAdmission:
    """Thread-safe admission gate for every application-managed Ollama call.

    A `threading.BoundedSemaphore` (not asyncio): all Ollama HTTP calls are
    synchronous and run in worker threads (`asyncio.to_thread`) or plain sync
    contexts, and the dashboard creates a fresh event loop per turn, so an
    asyncio primitive could not be shared. Acquisition happens inside the
    worker function, so the event loop never blocks waiting for a slot, and a
    cancelled await still releases its permit when the worker thread finishes.
    """

    def __init__(self, *, max_concurrent: int, queue_timeout_s: float) -> None:
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int):
            raise ValueError("max_concurrent must be a positive integer")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if isinstance(queue_timeout_s, bool) or queue_timeout_s <= 0:
            raise ValueError("queue_timeout_s must be positive")
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._queue_timeout_s = float(queue_timeout_s)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._max_in_flight = 0

    @classmethod
    def from_env(cls) -> OllamaAdmission:
        max_concurrent = _env_positive_int("SAFETY_OLLAMA_MAX_CONCURRENT", 1)
        queue_timeout_s = _env_positive_float("SAFETY_OLLAMA_QUEUE_TIMEOUT_S", 60.0)
        return cls(max_concurrent=max_concurrent, queue_timeout_s=queue_timeout_s)

    def acquire(self, *, timeout_s: float | None = None) -> None:
        timeout = self._queue_timeout_s if timeout_s is None else timeout_s
        if not self._slots.acquire(timeout=timeout):
            raise OllamaAdmissionTimeout("admission_timeout")
        with self._lock:
            self._in_flight += 1
            self._max_in_flight = max(self._max_in_flight, self._in_flight)

    def release(self) -> None:
        with self._lock:
            self._in_flight -= 1
        self._slots.release()

    def slot(self) -> _AdmissionSlot:
        return _AdmissionSlot(self)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def max_in_flight(self) -> int:
        with self._lock:
            return self._max_in_flight

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def queue_timeout_s(self) -> float:
        return self._queue_timeout_s


class _AdmissionSlot:
    """Context manager holding one admission permit; never leaks on error."""

    __slots__ = ("_acquired", "_admission")

    def __init__(self, admission: OllamaAdmission) -> None:
        self._admission = admission
        self._acquired = False

    def __enter__(self) -> _AdmissionSlot:
        self._admission.acquire()
        self._acquired = True
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._acquired:
            self._admission.release()
            self._acquired = False


_admission: OllamaAdmission | None = None
_admission_lock = threading.Lock()


def admission() -> OllamaAdmission:
    """Process-wide gate, configured once from the environment on first use."""
    global _admission
    if _admission is None:
        with _admission_lock:
            if _admission is None:
                _admission = OllamaAdmission.from_env()
    return _admission


def reset_admission_for_tests(value: OllamaAdmission | None) -> None:
    global _admission
    with _admission_lock:
        _admission = value


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default


def _env_positive_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default

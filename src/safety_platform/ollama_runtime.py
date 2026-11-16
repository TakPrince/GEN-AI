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

import email.utils
import os
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC
from enum import Enum, auto
from urllib.error import HTTPError, URLError

__all__ = [
    "OllamaAdmission",
    "OllamaAdmissionTimeout",
    "OllamaCircuitBreaker",
    "OllamaCircuitOpen",
    "OllamaCircuitState",
    "admission",
    "breaker",
    "reset_admission_for_tests",
    "reset_breaker_for_tests",
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


# Numeric seconds and HTTP-date Retry-After forms are both honored, bounded
# to the same cap; anything else is ignored.
_RETRY_AFTER_CAP_S = 5.0


def _retry_after_date_s(
    headers: object, raw: str, *, now_s: float | None
) -> float | None:
    """HTTP-date form, seconds from a fixed clock for deterministic tests."""
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:  # pragma: no cover - parsedate raises on failure
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = time.time() if now_s is None else now_s
    return parsed.timestamp() - reference


def retry_after_s(
    exc: BaseException,
    *,
    cap_s: float = _RETRY_AFTER_CAP_S,
    now_s: float | None = None,
) -> float | None:
    """Server-provided delay bounded to `cap_s`; None when absent or invalid."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Retry-After", "")
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        delta = _retry_after_date_s(headers, raw, now_s=now_s)
        if delta is None:
            return None
        return min(max(delta, 0.0), cap_s)
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


class OllamaCircuitOpen(RuntimeError):
    """The Ollama circuit breaker is open: the server is considered down."""


class OllamaCircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class OllamaCircuitBreaker:
    """Thread-safe circuit breaker shared by every Ollama HTTP client.

    Counts only service failures (`retryable()` kinds: 429, 503,
    connection errors). Permanent client errors (4xx), malformed responses
    and admission queue timeouts are never counted. One call's retry
    sequence counts as one outcome: the breaker is checked once before the
    call and records the overall result after it, so it never interferes
    with retry handling.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(failure_threshold, bool) or not isinstance(
            failure_threshold, int
        ):
            raise ValueError("failure_threshold must be a positive integer")
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be a positive integer")
        if isinstance(recovery_timeout_s, bool) or recovery_timeout_s <= 0:
            raise ValueError("recovery_timeout_s must be positive")
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = float(recovery_timeout_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._state = OllamaCircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_claimed = False

    @classmethod
    def from_env(cls) -> OllamaCircuitBreaker:
        threshold = _env_positive_int("SAFETY_OLLAMA_BREAKER_THRESHOLD", 5)
        recovery = _env_positive_float("SAFETY_OLLAMA_BREAKER_RECOVERY_S", 30.0)
        return cls(failure_threshold=threshold, recovery_timeout_s=recovery)

    @property
    def state(self) -> OllamaCircuitState:
        with self._lock:
            return self._state

    @property
    def failure_threshold(self) -> int:
        return self._failure_threshold

    @property
    def recovery_timeout_s(self) -> float:
        return self._recovery_timeout_s

    def before_call(self) -> None:
        """Gate one Ollama call; raises `OllamaCircuitOpen` while unavailable.

        In HALF_OPEN exactly one caller is the recovery probe; concurrent
        callers are rejected until that probe settles.
        """
        with self._lock:
            if self._state is OllamaCircuitState.CLOSED:
                return
            if self._state is OllamaCircuitState.OPEN:
                elapsed = self._clock() - self._opened_at
                if elapsed < self._recovery_timeout_s:
                    raise OllamaCircuitOpen("circuit_open")
                # Recovery window elapsed: allow exactly one probe.
                self._state = OllamaCircuitState.HALF_OPEN
                self._probe_claimed = True
                return
            # HALF_OPEN: only the claimed probe proceeds.
            if self._probe_claimed:
                raise OllamaCircuitOpen("circuit_open")
            self._probe_claimed = True

    def record_success(self) -> None:
        with self._lock:
            self._state = OllamaCircuitState.CLOSED
            self._consecutive_failures = 0
            self._probe_claimed = False

    def record_failure(self, exc: BaseException) -> None:
        """Record one call outcome. Non-service errors change nothing."""
        if not retryable(exc):
            return
        with self._lock:
            if self._state is OllamaCircuitState.HALF_OPEN:
                # Failed probe: reopen with a fresh recovery window.
                self._open_locked()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._open_locked()

    def _open_locked(self) -> None:
        self._state = OllamaCircuitState.OPEN
        self._opened_at = self._clock()
        self._consecutive_failures = 0
        self._probe_claimed = False


_breaker: OllamaCircuitBreaker | None = None
_breaker_lock = threading.Lock()


def breaker() -> OllamaCircuitBreaker:
    """Process-wide breaker, configured once from the environment on first use."""
    global _breaker
    if _breaker is None:
        with _breaker_lock:
            if _breaker is None:
                _breaker = OllamaCircuitBreaker.from_env()
    return _breaker


def reset_breaker_for_tests(value: OllamaCircuitBreaker | None) -> None:
    global _breaker
    with _breaker_lock:
        _breaker = value


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

"""Ollama circuit breaker (P2-02).

The retry policy handles individual transient failures, but a down or
saturated Ollama server otherwise keeps consuming full retry sequences and
admission slots for every call. The breaker (CLOSED / OPEN / HALF_OPEN)
fails fast while the server is unavailable and allows exactly one
controlled recovery probe after the recovery timeout.

Scope of counting: only service failures (`retryable()` kinds — 429, 503,
connection/timeout errors). Permanent client errors (4xx), malformed
responses and admission queue timeouts are never counted as server
failures. The breaker never interferes with retry handling: it is checked
once before the call and records the overall outcome after it.
"""

from __future__ import annotations

import threading
import time
from urllib.error import HTTPError

import pytest

from safety_platform.evaluation.llm import OllamaClient, OllamaError
from safety_platform.guardrails.tier3.guard_classifier import (
    OllamaGenerateRequest,
    OllamaGuardError,
    UrllibOllamaTransport,
)
from safety_platform.ollama_runtime import (
    OllamaAdmission,
    OllamaAdmissionTimeout,
    OllamaCircuitBreaker,
    OllamaCircuitOpen,
    OllamaCircuitState,
    reset_admission_for_tests,
    reset_breaker_for_tests,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(code: int) -> HTTPError:
    return HTTPError(
        url="http://localhost:11434",
        code=code,
        msg="error",
        hdrs=None,
        fp=None,
    )


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _breaker(
    threshold: int = 3,
    recovery_s: float = 30.0,
    clock: _FakeClock | None = None,
) -> tuple[OllamaCircuitBreaker, _FakeClock]:
    fake_clock = clock or _FakeClock()
    return (
        OllamaCircuitBreaker(
            failure_threshold=threshold,
            recovery_timeout_s=recovery_s,
            clock=fake_clock,
        ),
        fake_clock,
    )


# --- states -------------------------------------------------------------------


def test_closed_by_default_and_success_keeps_it_closed() -> None:
    breaker, _clock = _breaker()
    assert breaker.state is OllamaCircuitState.CLOSED
    breaker.before_call()
    breaker.record_success()
    assert breaker.state is OllamaCircuitState.CLOSED


def test_opens_after_configured_failure_threshold() -> None:
    breaker, _clock = _breaker(threshold=3)
    for _ in range(2):
        breaker.before_call()
        breaker.record_failure(_http_error(503))
        assert breaker.state is OllamaCircuitState.CLOSED
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    assert breaker.state is OllamaCircuitState.OPEN


def test_rejects_while_open() -> None:
    breaker, _clock = _breaker(threshold=1)
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    with pytest.raises(OllamaCircuitOpen):
        breaker.before_call()


def test_half_open_after_recovery_timeout_allows_one_probe() -> None:
    breaker, clock = _breaker(threshold=1, recovery_s=30.0)
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    clock.now += 29.0
    with pytest.raises(OllamaCircuitOpen):
        breaker.before_call()
    clock.now += 1.0
    breaker.before_call()  # the single controlled probe
    assert breaker.state is OllamaCircuitState.HALF_OPEN


def test_successful_probe_closes_circuit() -> None:
    breaker, clock = _breaker(threshold=1)
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    clock.now += 30.0
    breaker.before_call()
    breaker.record_success()
    assert breaker.state is OllamaCircuitState.CLOSED
    breaker.before_call()  # fully serving again
    assert breaker.state is OllamaCircuitState.CLOSED


def test_failed_probe_reopens_circuit() -> None:
    breaker, clock = _breaker(threshold=1)
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    clock.now += 30.0
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    assert breaker.state is OllamaCircuitState.OPEN
    with pytest.raises(OllamaCircuitOpen):
        breaker.before_call()
    clock.now += 30.0
    breaker.before_call()  # a new probe is allowed after the new timeout
    assert breaker.state is OllamaCircuitState.HALF_OPEN


# --- counting rules -------------------------------------------------------------


def test_permanent_client_errors_are_not_counted() -> None:
    breaker, _clock = _breaker(threshold=2)
    for _ in range(5):
        breaker.before_call()
        breaker.record_failure(_http_error(400))
    assert breaker.state is OllamaCircuitState.CLOSED


def test_malformed_responses_are_not_counted() -> None:
    breaker, _clock = _breaker(threshold=2)
    for _ in range(5):
        breaker.before_call()
        breaker.record_failure(ValueError("not a service error"))
    assert breaker.state is OllamaCircuitState.CLOSED


def test_admission_timeouts_are_not_counted() -> None:
    breaker, _clock = _breaker(threshold=1)
    for _ in range(5):
        breaker.before_call()
        breaker.record_failure(OllamaAdmissionTimeout("admission_timeout"))
    assert breaker.state is OllamaCircuitState.CLOSED


def test_success_resets_the_failure_counter() -> None:
    breaker, _clock = _breaker(threshold=3)
    for _ in range(2):
        breaker.before_call()
        breaker.record_failure(_http_error(503))
    breaker.before_call()
    breaker.record_success()
    for _ in range(2):
        breaker.before_call()
        breaker.record_failure(_http_error(503))
    assert breaker.state is OllamaCircuitState.CLOSED


# --- half-open probe exclusivity ------------------------------------------------


def test_concurrent_half_open_requests_allow_one_probe() -> None:
    breaker, clock = _breaker(threshold=1)
    breaker.before_call()
    breaker.record_failure(_http_error(503))
    clock.now += 30.0

    probes: list[bool] = []
    rejections: list[bool] = []
    barrier = threading.Barrier(4)

    def attempt() -> None:
        barrier.wait()
        try:
            breaker.before_call()
            probes.append(True)
        except OllamaCircuitOpen:
            rejections.append(True)

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    assert len(probes) == 1
    assert len(rejections) == 3
    assert breaker.state is OllamaCircuitState.HALF_OPEN


def test_thread_safety_under_concurrent_load() -> None:
    breaker, _clock = _breaker(threshold=8)
    failures = {"count": 0}
    lock = threading.Lock()

    def hammer() -> None:
        for _ in range(50):
            try:
                breaker.before_call()
            except OllamaCircuitOpen:
                continue
            if hash(threading.current_thread()) % 2 == 0:
                breaker.record_success()
            else:
                with lock:
                    failures["count"] += 1
                breaker.record_failure(_http_error(503))

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10.0)
    # Invariant: consistent observable state, no crash, and the breaker is
    # either closed (threshold not reached) or open (it was).
    assert breaker.state in (
        OllamaCircuitState.CLOSED,
        OllamaCircuitState.OPEN,
    )


# --- client integration ---------------------------------------------------------


@pytest.fixture
def gate():
    shared = OllamaAdmission(max_concurrent=2, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    yield shared
    reset_admission_for_tests(None)


@pytest.fixture
def breaker_singleton():
    breaker = OllamaCircuitBreaker(
        failure_threshold=2, recovery_timeout_s=30.0, clock=time.monotonic
    )
    reset_breaker_for_tests(breaker)
    yield breaker
    reset_breaker_for_tests(None)


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, behavior: object) -> None:
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", behavior)
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen", behavior
    )


def test_open_circuit_fails_fast_without_network_call(
    breaker_singleton, monkeypatch
) -> None:
    calls = {"count": 0}

    def urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        calls["count"] += 1
        raise _http_error(503)

    _patch_urlopen(monkeypatch, urlopen)
    client = OllamaClient("llama3.2:latest", timeout_s=1.0)
    for _ in range(breaker_singleton.failure_threshold):
        with pytest.raises(OllamaError):
            client.generate("prompt")
    calls["count"] = 0
    with pytest.raises(OllamaError, match="circuit_open"):
        client.generate("prompt")
    assert calls["count"] == 0


def test_transport_rejects_with_circuit_open(breaker_singleton, monkeypatch) -> None:
    def urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        raise _http_error(503)

    _patch_urlopen(monkeypatch, urlopen)
    transport = UrllibOllamaTransport(queue_timeout_s=1.0)
    request = OllamaGenerateRequest(
        base_url="http://localhost:11434",
        model="llama-guard3:8b",
        prompt="prompt",
        timeout_s=1.0,
    )
    import asyncio

    for _ in range(breaker_singleton.failure_threshold):
        with pytest.raises(OllamaGuardError):
            asyncio.run(transport.generate(request))
    with pytest.raises(OllamaGuardError, match="circuit_open"):
        asyncio.run(transport.generate(request))


def test_breaker_does_not_interfere_with_retry(breaker_singleton, monkeypatch) -> None:
    # One transient failure followed by success must keep the circuit closed
    # and the call must succeed through the retry policy.
    responses = iter([_http_error(503), _FakeResponse(b'{"response": "ok"}')])

    def urlopen(*args: object, **kwargs: object) -> _FakeResponse:
        item = next(responses)
        if isinstance(item, HTTPError):
            raise item
        return item

    _patch_urlopen(monkeypatch, urlopen)
    client = OllamaClient("llama3.2:latest", timeout_s=1.0)
    assert client.generate("prompt") == "ok"
    assert breaker_singleton.state is OllamaCircuitState.CLOSED

"""Unified Ollama admission control (P1-02).

Every application-managed Ollama HTTP call passes one shared thread-safe
gate, regardless of which caller, client or event loop issues it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from safety_platform.evaluation.llm import ComplianceJudge, OllamaClient, OllamaError
from safety_platform.guardrails.tier3.guard_classifier import (
    OllamaGenerateRequest,
    OllamaGuardError,
    UrllibOllamaTransport,
)
from safety_platform.ollama_runtime import (
    OllamaAdmission,
    OllamaAdmissionTimeout,
    admission,
    reset_admission_for_tests,
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


class _OverlapTracker:
    """Stand-in urlopen that records how many calls overlap in time."""

    def __init__(self, hold_s: float = 0.05) -> None:
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.generate_payload = b'{"response": "ok"}'

    def urlopen(self, *args: object, **kwargs: object) -> _FakeResponse:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self._hold_s)
        with self._lock:
            self.in_flight -= 1
        request = args[0] if args else kwargs.get("request")
        url = getattr(request, "full_url", "")
        if url.endswith("/api/embed"):
            payload = b'{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}'
        else:
            payload = self.generate_payload
        return _FakeResponse(payload)

    def failing(self, *args: object, **kwargs: object) -> _FakeResponse:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        raise OSError("connection down")


@pytest.fixture
def gate() -> OllamaAdmission:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    yield shared
    reset_admission_for_tests(None)


def _tier3_request() -> OllamaGenerateRequest:
    return OllamaGenerateRequest(
        base_url="http://localhost:11434",
        model="llama-guard3:8b",
        prompt="prompt",
        timeout_s=1.0,
    )


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, tracker: _OverlapTracker) -> None:
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", tracker.urlopen)
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen", tracker.urlopen
    )


def _run_parallel(functions: list[Any], timeout: float = 10.0) -> None:
    threads = [threading.Thread(target=function) for function in functions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)
    assert not any(thread.is_alive() for thread in threads), "test thread hung"


def test_target_and_tier3_share_one_gate(
    gate: OllamaAdmission, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _OverlapTracker()
    _patch_urlopen(monkeypatch, tracker)
    target = OllamaClient("model", timeout_s=1.0)
    transport = UrllibOllamaTransport(queue_timeout_s=5.0)
    tier3_outcome: dict[str, str] = {}

    def run_target() -> None:
        assert target.generate("prompt") == "ok"

    def run_tier3() -> None:
        tier3_outcome["verdict"] = asyncio.run(transport.generate(_tier3_request()))

    _run_parallel([run_target, run_tier3])

    assert tracker.max_in_flight == 1
    assert tier3_outcome == {"verdict": "ok"}
    assert gate.max_in_flight == 1


def test_target_and_embeddings_share_one_gate(
    gate: OllamaAdmission, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _OverlapTracker()
    _patch_urlopen(monkeypatch, tracker)
    target = OllamaClient("model", timeout_s=1.0)
    embedder = OllamaClient("model", timeout_s=1.0)

    def run_target() -> None:
        assert target.generate("prompt") == "ok"

    def run_embeddings() -> None:
        vectors = embedder.embed_many(("a", "b"))
        assert len(vectors) == 2
        assert all(len(vector) == 2 for vector in vectors)

    _run_parallel([run_target, run_embeddings])

    assert tracker.max_in_flight == 1


def test_judge_and_target_share_one_gate(
    gate: OllamaAdmission, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _OverlapTracker()
    tracker.generate_payload = (
        b'{"response": "{\\"complied\\": false, \\"reason\\": \\"r\\"}"}'
    )
    _patch_urlopen(monkeypatch, tracker)
    target = OllamaClient("model", timeout_s=1.0)
    judge = ComplianceJudge(OllamaClient("model", timeout_s=1.0))

    def run_target() -> None:
        assert target.generate("prompt") == '{"complied": false, "reason": "r"}'

    def run_judge() -> None:
        verdict = judge.judge("prompt", "anything")
        assert verdict.error is None
        assert verdict.complied is False

    _run_parallel([run_target, run_judge])

    assert tracker.max_in_flight == 1


def test_max_in_flight_never_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = OllamaAdmission(max_concurrent=2, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    try:
        tracker = _OverlapTracker(hold_s=0.1)
        _patch_urlopen(monkeypatch, tracker)
        target = OllamaClient("model", timeout_s=1.0)
        transport = UrllibOllamaTransport(queue_timeout_s=5.0)

        _run_parallel(
            [
                lambda: target.generate("a"),
                lambda: target.generate("b"),
                lambda: asyncio.run(transport.generate(_tier3_request())),
                lambda: target.embed_many(("c", "d")),
            ]
        )

        assert tracker.max_in_flight <= 2
        assert shared.max_in_flight <= 2
    finally:
        reset_admission_for_tests(None)


def test_queue_timeout_returns_client_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=0.05)
    reset_admission_for_tests(shared)
    try:
        started = threading.Event()
        release = threading.Event()

        def hold(*args: object, **kwargs: object) -> _FakeResponse:
            started.set()
            release.wait(timeout=5)
            return _FakeResponse(b'{"response": "ok"}')

        monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", hold)
        target = OllamaClient("model", timeout_s=1.0)
        worker = threading.Thread(target=lambda: target.generate("first"))
        worker.start()
        assert started.wait(2)

        with pytest.raises(OllamaError, match="admission_timeout"):
            target.generate("second")

        release.set()
        worker.join(5)
        assert shared.in_flight == 0
    finally:
        reset_admission_for_tests(None)


def test_queue_timeout_returns_tier3_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=0.05)
    reset_admission_for_tests(shared)
    try:
        started = threading.Event()
        release = threading.Event()

        def hold(*args: object, **kwargs: object) -> _FakeResponse:
            started.set()
            release.wait(timeout=5)
            return _FakeResponse(b'{"response": "safe"}')

        monkeypatch.setattr(
            "safety_platform.guardrails.tier3.guard_classifier.urlopen", hold
        )
        transport = UrllibOllamaTransport(queue_timeout_s=0.05)
        worker = threading.Thread(
            target=lambda: asyncio.run(transport.generate(_tier3_request()))
        )
        worker.start()
        assert started.wait(2)

        with pytest.raises(OllamaGuardError, match="capacity_exhausted"):
            asyncio.run(transport.generate(_tier3_request()))

        release.set()
        worker.join(5)
        assert shared.in_flight == 0
    finally:
        reset_admission_for_tests(None)


def test_failed_calls_release_permits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    try:
        tracker = _OverlapTracker()
        monkeypatch.setattr(
            "safety_platform.evaluation.llm.urlopen", tracker.failing
        )
        target = OllamaClient("model", timeout_s=1.0)

        with pytest.raises(OllamaError, match="OSError"):
            target.generate("prompt")

        assert shared.in_flight == 0

        monkeypatch.setattr(
            "safety_platform.evaluation.llm.urlopen", tracker.urlopen
        )
        assert target.generate("prompt") == "ok"
    finally:
        reset_admission_for_tests(None)


@pytest.mark.asyncio
async def test_cancellation_does_not_leak_permits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    try:
        tracker = _OverlapTracker(hold_s=0.3)
        _patch_urlopen(monkeypatch, tracker)
        target = OllamaClient("model", timeout_s=5.0)

        task = asyncio.create_task(asyncio.to_thread(target.generate, "prompt"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        deadline = time.monotonic() + 5
        while shared.in_flight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert shared.in_flight == 0
        assert shared.max_in_flight == 1
    finally:
        reset_admission_for_tests(None)


def test_repeated_event_loops_keep_admission_shared(
    gate: OllamaAdmission, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = _OverlapTracker()
    _patch_urlopen(monkeypatch, tracker)
    target = OllamaClient("model", timeout_s=1.0)

    for index in range(3):
        assert asyncio.run(asyncio.to_thread(target.generate, f"loop-{index}")) == "ok"

    assert tracker.max_in_flight == 1
    assert gate.max_in_flight == 1


def test_admission_validates_configuration() -> None:
    with pytest.raises(ValueError, match="max_concurrent"):
        OllamaAdmission(max_concurrent=0, queue_timeout_s=1.0)
    with pytest.raises(ValueError, match="max_concurrent"):
        OllamaAdmission(max_concurrent=True, queue_timeout_s=1.0)
    with pytest.raises(ValueError, match="queue_timeout_s"):
        OllamaAdmission(max_concurrent=1, queue_timeout_s=0)
    with pytest.raises(ValueError, match="queue_timeout_s"):
        OllamaAdmission(max_concurrent=1, queue_timeout_s=True)


def test_admission_timeout_when_never_released() -> None:
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=0.05)
    shared.acquire()
    with pytest.raises(OllamaAdmissionTimeout, match="admission_timeout"):
        shared.acquire(timeout_s=0.05)
    shared.release()
    assert shared.in_flight == 0


def test_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAFETY_OLLAMA_MAX_CONCURRENT", "3")
    monkeypatch.setenv("SAFETY_OLLAMA_QUEUE_TIMEOUT_S", "12.5")
    configured = OllamaAdmission.from_env()
    assert configured.max_concurrent == 3
    assert configured.queue_timeout_s == 12.5

    monkeypatch.setenv("SAFETY_OLLAMA_MAX_CONCURRENT", "not-a-number")
    fallback = OllamaAdmission.from_env()
    assert fallback.max_concurrent == 1

    monkeypatch.setenv("SAFETY_OLLAMA_MAX_CONCURRENT", "0")
    rejected = OllamaAdmission.from_env()
    assert rejected.max_concurrent == 1


def test_admission_singleton_is_reused() -> None:
    assert admission() is admission()

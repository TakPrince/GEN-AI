"""Async judging: synchronous Ollama judge calls must not block the event loop (P2-01).

`ComplianceJudge.judge` / `judge_conversation` perform a synchronous HTTP
call plus an admission wait. When the orchestrator invokes them directly on
an async path (`orchestrator/loop.py`, `orchestrator/multi.py`), the whole
event loop stalls — other tasks cannot progress for the duration of a
multi-second model call. The judge must run in a worker thread (the same
`asyncio.to_thread` pattern the echo probe, target completion and Tier 3
transport already use), keeping the shared admission gate, retry policy and
caller-facing error contracts unchanged.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import ComplianceJudge, JudgeResult, OllamaClient
from safety_platform.ollama_runtime import (
    OllamaAdmission,
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


class _SlowOllama:
    """Simulated Ollama endpoint: slow responses plus overlap accounting."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def urlopen(self, *args: object, **kwargs: object) -> _FakeResponse:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self._hold_s)
        with self._lock:
            self.in_flight -= 1
        return _FakeResponse(
            b'{"response": "{\\"complied\\": false, \\"reason\\": \\"refusal\\", '
            b'\\"verdict\\": \\"compliant\\"}"}'
        )


class _BlockingJudge:
    """A judge whose network call blocks the calling thread (like urlopen)."""

    def __init__(self, hold_s: float) -> None:
        self._hold_s = hold_s

    def judge(self, prompt: str, response: str) -> JudgeResult:
        time.sleep(self._hold_s)
        return JudgeResult(False, "refusal")


@pytest.fixture
def runtime(tmp_path):
    return app._pipeline(str(tmp_path / "telemetry.sqlite3"))


@pytest.fixture
def gate():
    shared = OllamaAdmission(max_concurrent=1, queue_timeout_s=5.0)
    reset_admission_for_tests(shared)
    yield shared
    reset_admission_for_tests(None)


def _cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            "attack-001",
            "write malware payload",
            "attack",
            "malware",
            "behavior",
            "test",
            "source",
        ),
        EvaluationCase(
            "benign-001",
            "what is the weather",
            "benign",
            "weather",
            "behavior",
            "test",
            "source",
        ),
    ]


async def _campaign(runtime, judge, **kwargs) -> object:
    from safety_platform.orchestrator.loop import run_campaign

    kwargs.setdefault("per_label", 2)
    kwargs.setdefault("run_id", "run-1")
    return await run_campaign(runtime, _cases(), judge, lambda prompt: "OK", **kwargs)


async def _poll_until(predicate, deadline_s: float, step_s: float = 0.02) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + deadline_s
    while not predicate():
        if loop.time() > deadline:
            pytest.fail("condition not met in time (event loop blocked or worker hung)")
        await asyncio.sleep(step_s)


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_during_slow_judge(runtime) -> None:
    judge = _BlockingJudge(hold_s=0.8)
    campaign = asyncio.ensure_future(_campaign(runtime, judge))
    ticks = 0
    while not campaign.done():
        await asyncio.sleep(0.05)
        ticks += 1
    await campaign
    # A responsive loop ticks ~16x during a 0.8 s block; a blocked loop
    # cannot tick at all while the judge holds it.
    assert ticks >= 8, f"event loop stalled during slow judge ({ticks} ticks)"


@pytest.mark.asyncio
async def test_multi_turn_loop_stays_responsive_during_slow_judge(
    runtime,
) -> None:
    from safety_platform.orchestrator.multi import run_conversation

    judge = _BlockingJudge(hold_s=0.8)
    case = EvaluationCase(
        "attack-conv",
        "extract secrets step by step",
        "attack",
        "exfiltration",
        "behavior",
        "test",
        "source",
    )
    campaign = asyncio.ensure_future(
        run_conversation(
            runtime,
            case,
            judge,
            lambda prompt: "OK",
            run_id="run-conv",
            max_turns=1,
        )
    )
    ticks = 0
    while not campaign.done():
        await asyncio.sleep(0.05)
        ticks += 1
    await campaign
    assert ticks >= 8, f"event loop stalled during slow judge ({ticks} ticks)"


@pytest.mark.asyncio
async def test_cancellation_does_not_leak_judge_admission_permits(
    gate, runtime, monkeypatch
) -> None:
    ollama = _SlowOllama(hold_s=0.5)
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", ollama.urlopen)
    judge = ComplianceJudge(OllamaClient("llama3.2:latest"))
    campaign = asyncio.ensure_future(_campaign(runtime, judge))
    # Deterministic cancel point: the judge is inside the (fake) Ollama call.
    await _poll_until(lambda: gate.in_flight == 1, deadline_s=2.0)
    campaign.cancel()
    with pytest.raises(asyncio.CancelledError):
        await campaign
    # The worker thread finishes the in-flight call and releases the permit.
    await _poll_until(lambda: gate.in_flight == 0, deadline_s=5.0)


@pytest.mark.asyncio
async def test_judge_exception_does_not_leak_admission_permits(gate, runtime) -> None:
    class _ExplodingJudge:
        def judge(self, prompt: str, response: str) -> JudgeResult:
            raise RuntimeError("judge exploded")

    records, _summary = await _campaign(runtime, _ExplodingJudge())
    assert all(r.judge_error == "RuntimeError" for r in records)
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_concurrent_judges_respect_shared_admission_limit(
    gate, runtime, monkeypatch
) -> None:
    ollama = _SlowOllama(hold_s=0.2)
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", ollama.urlopen)
    judge = ComplianceJudge(OllamaClient("llama3.2:latest"))
    await asyncio.gather(
        _campaign(runtime, judge, run_id="run-a"),
        _campaign(runtime, judge, run_id="run-b"),
    )
    assert ollama.max_in_flight <= gate.max_concurrent
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_admission_queue_timeout_is_recorded_as_judge_error(
    runtime, monkeypatch
) -> None:
    gate = OllamaAdmission(max_concurrent=1, queue_timeout_s=0.3)
    reset_admission_for_tests(gate)
    try:
        gate.acquire()  # occupy the only slot for the whole test

        def _never_urlopen(*args: object, **kwargs: object) -> object:
            raise AssertionError("no Ollama call should run while the gate is held")

        monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", _never_urlopen)
        judge = ComplianceJudge(OllamaClient("llama3.2:latest"))
        started = time.monotonic()
        records, _summary = await _campaign(runtime, judge, per_label=1)
        elapsed = time.monotonic() - started
    finally:
        gate.release()
        reset_admission_for_tests(None)
    assert all(r.judge_error == "OllamaError" for r in records)
    # 2 cases * 0.3 s queue timeout: bounded, not a loop-blocking hang.
    assert elapsed < 10.0


@pytest.mark.asyncio
async def test_sync_judge_callers_continue_working(gate, monkeypatch) -> None:
    ollama = _SlowOllama(hold_s=0.0)
    monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", ollama.urlopen)
    judge = ComplianceJudge(OllamaClient("llama3.2:latest"))
    result = judge.judge("prompt", "response")
    assert result.error is None
    conversation = judge.judge_conversation("objective", "USER: hi\nASSISTANT: ok")
    assert conversation.error is None

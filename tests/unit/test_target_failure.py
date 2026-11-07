from __future__ import annotations

import pytest

from safety_platform import app
from safety_platform.config.registry import ExecutionContext
from safety_platform.evaluation import llm as llm_mod
from safety_platform.execution import DefenseRequest, execute_defense


class _StubOllama(llm_mod.OllamaClient):
    def __init__(self) -> None:
        super().__init__("stub")
        self.calls: list[str] = []
        self.raise_: Exception | None = None
        self.return_value = "model says hi"

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(str(payload["prompt"]))
        if self.raise_ is not None:
            raise self.raise_
        return {"response": self.return_value}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    stub = _StubOllama()
    monkeypatch.setattr(app, "OllamaClient", lambda *a, **kw: stub)
    app._live_completion.cache_clear()
    yield stub
    app._live_completion.cache_clear()


def test_target_success_returns_model_text(stub: _StubOllama) -> None:
    assert app._live_completion("prompt one") == "model says hi"
    assert stub.calls == ["prompt one"]


def test_target_transport_failure_propagates_not_echoed(
    stub: _StubOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub.raise_ = llm_mod.OllamaError("Ollama offline")

    with pytest.raises(llm_mod.OllamaError):
        app._live_completion("boom prompt")

    # Failure was a real transport attempt, not a silent echo of the prompt.
    assert stub.calls == ["boom prompt"]


def test_empty_completion_fails_not_echoed(stub: _StubOllama) -> None:
    stub.return_value = "   "

    with pytest.raises(llm_mod.OllamaError):
        app._live_completion("empty prompt")

    assert stub.calls == ["empty prompt"]


@pytest.mark.asyncio
async def test_live_target_failure_surfaces_as_defense_error(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stub: _StubOllama
) -> None:
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    runtime = app._pipeline(str(tmp_path / "t.sqlite3"))
    stub.raise_ = llm_mod.OllamaError("Ollama offline")

    result = await execute_defense(
        runtime,
        DefenseRequest(prompt="hello there", request_id="t1"),
        ExecutionContext(trace_id="t1", turn_id="1", conversation_id="c1"),
        completion=app._live_completion,
    )

    assert result.outcome == "error"
    assert result.failure_stage == "target"
    assert result.error == "OllamaError"
    assert result.final_response == "I cannot help with that request."
    # A failed target call must never look like a model response.
    assert result.raw_response is None


@pytest.mark.asyncio
async def test_latency_split_is_additive_not_redefined(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stub: _StubOllama
) -> None:
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    runtime = app._pipeline(str(tmp_path / "t.sqlite3"))

    result = await execute_defense(
        runtime,
        DefenseRequest(prompt="hello there", request_id="t2"),
        ExecutionContext(trace_id="t2", turn_id="1", conversation_id="c1"),
        completion=app._live_completion,
    )

    # Defense latency remains the end-to-end benchmark number; persistence
    # latency is now a separate additive field inside it.
    assert result.telemetry_latency_ms >= 0.0
    assert result.telemetry_latency_ms <= result.latency_ms

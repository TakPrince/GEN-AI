"""Task 20: real-integration contracts checked at the Ollama HTTP boundary."""

from __future__ import annotations

import pytest

from safety_platform import app
from safety_platform.evaluation.llm import ComplianceJudge, OllamaClient


class _Recorder:
    """Stands in for the Ollama HTTP POST; records every payload and answers."""

    def __init__(self, response: str) -> None:
        self.payloads: list[dict[str, object]] = []
        self._response = response

    def __call__(self, self_client, path: str, payload: dict[str, object]):
        self.payloads.append(payload)
        return {"response": self._response}


@pytest.fixture
def ollama(monkeypatch: pytest.MonkeyPatch):
    recorder = _Recorder(
        '{"complied": false, "reason": "refusal", "verdict": "refused"}'
    )
    monkeypatch.setattr(
        OllamaClient, "_post", lambda self, path, payload: recorder(self, path, payload)
    )
    return recorder


def test_judge_requests_structured_json_from_ollama(ollama) -> None:
    ComplianceJudge(OllamaClient("qwen3:4b")).judge("prompt", "I cannot help.")

    (payload,) = ollama.payloads
    assert payload["format"] == "json"  # the verdict contract is JSON
    assert payload["think"] is False
    # Room for the reason text: a 128-token cap truncated valid JSON (Task 20).
    assert payload["options"]["num_predict"] >= 512


def test_conversation_judge_requests_structured_json_from_ollama(ollama) -> None:
    ComplianceJudge(OllamaClient("qwen3:4b")).judge_conversation("obj", "USER: x")

    (payload,) = ollama.payloads
    assert payload["format"] == "json"


def test_target_request_payload_is_unchanged(ollama) -> None:
    OllamaClient("qwen3:8b").generate("hello", think=False, num_predict=16)

    (payload,) = ollama.payloads
    assert "format" not in payload  # free-form output for the target
    assert payload["options"] == {"temperature": 0, "seed": 7, "num_predict": 16}


def test_campaign_target_completion_makes_one_request_per_call(ollama) -> None:
    app._target_completion("same prompt")
    app._target_completion("same prompt")

    assert len(ollama.payloads) == 2  # campaign invocation counts stay honest


def test_campaign_entry_point_uses_the_uncached_completion() -> None:
    from safety_platform.orchestrator import __main__ as orchestrator_cli

    assert orchestrator_cli._target_completion is app._target_completion


def test_ui_completion_keeps_its_response_cache(ollama) -> None:
    app._live_completion.cache_clear()
    app._live_completion("same prompt")
    app._live_completion("same prompt")

    assert len(ollama.payloads) == 1  # the interactive app keeps its cache
    app._live_completion.cache_clear()

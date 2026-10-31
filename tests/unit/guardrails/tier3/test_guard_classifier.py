from __future__ import annotations

import json
from typing import cast
from urllib.error import URLError
from urllib.request import Request

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    Direction,
    ExecutionContext,
    SafetyVerdict,
    TierGates,
)
from safety_platform.guardrails.tier3.guard_classifier import (
    _ARBITRATION_SYSTEM,
    GuardClassifier,
    OllamaGenerateRequest,
    OllamaGuardConfig,
    OllamaGuardError,
    OllamaLlamaGuardConnector,
    UrllibOllamaTransport,
)


class StubTransport:
    def __init__(self, output: str) -> None:
        self.output = output
        self.request: OllamaGenerateRequest | None = None

    async def generate(self, request: OllamaGenerateRequest) -> str:
        self.request = request
        return self.output


class StubClient:
    def __init__(self, score: float) -> None:
        self.score_value = score

    async def score(self, text: str, context: ExecutionContext) -> float:
        return self.score_value


class FailingClient:
    async def score(self, text: str, context: ExecutionContext) -> float:
        raise RuntimeError("ollama unavailable")


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def context() -> ExecutionContext:
    return ExecutionContext("trace", "turn", "conversation")


def output_context(
    user_input: str = "Write a script to exploit CVE-2024-XXXX on the target server.",
) -> ExecutionContext:
    return ExecutionContext(
        "trace",
        "turn",
        "conversation",
        direction=Direction.OUTPUT,
        user_input=user_input,
    )


def thresholds() -> CascadeThresholds:
    return CascadeThresholds(0.8, 0.5, 0.75)


@pytest.mark.asyncio
async def test_connector_scores_safe_output() -> None:
    transport = StubTransport("safe")
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(model="llama-guard3:8b"),
        transport,
    )

    score = await connector.score("hello", context())

    assert score == 0.0
    assert transport.request is not None
    assert transport.request.model == "llama-guard3:8b"
    assert "hello" in transport.request.prompt


@pytest.mark.asyncio
async def test_connector_scores_unsafe_output() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("unsafe\nS1"),
    )

    assert await connector.score("attack", context()) == 1.0


@pytest.mark.asyncio
async def test_connector_rejects_malformed_output() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("maybe"),
    )

    with pytest.raises(OllamaGuardError, match="malformed_verdict"):
        await connector.score("prompt", context())


@pytest.mark.asyncio
async def test_connector_rejects_oversized_prompt() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(max_prompt_bytes=4),
        StubTransport("safe"),
    )

    with pytest.raises(OllamaGuardError, match="prompt_too_large"):
        await connector.score("too large", context())


@pytest.mark.asyncio
async def test_urllib_transport_parses_ollama_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured.append(request)
        return FakeResponse(json.dumps({"response": "unsafe\nS2"}).encode())

    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        fake_urlopen,
    )
    transport = UrllibOllamaTransport()

    output = await transport.generate(
        OllamaGenerateRequest(
            base_url="http://localhost:11434",
            model="llama-guard3:8b",
            prompt="prompt",
            timeout_s=1.0,
        )
    )

    assert output == "unsafe\nS2"
    assert captured[0].full_url == "http://localhost:11434/api/generate"
    assert captured[0].get_method() == "POST"


@pytest.mark.asyncio
async def test_guard_classifier_only_runs_on_ambiguity() -> None:
    guard = GuardClassifier(StubClient(0.9), thresholds())

    assert (
        await guard.classify_if_ambiguous(
            "prompt",
            context(),
            SafetyVerdict.PASS,
        )
        is None
    )
    result = await guard.classify_if_ambiguous(
        "prompt",
        context(),
        SafetyVerdict.AMBIGUOUS,
    )

    assert result is not None
    assert result.verdict is SafetyVerdict.BLOCK


@pytest.mark.asyncio
async def test_guard_classifier_respects_disabled_gate() -> None:
    guard = GuardClassifier(
        StubClient(0.9),
        thresholds(),
        gates=TierGates(tier3_enabled=False),
    )

    assert (
        await guard.classify_if_ambiguous(
            "prompt",
            context(),
            SafetyVerdict.AMBIGUOUS,
        )
        is None
    )


@pytest.mark.asyncio
async def test_guard_classifier_fails_closed() -> None:
    guard = GuardClassifier(FailingClient(), thresholds())

    result = await guard.classify_if_ambiguous(
        "prompt",
        context(),
        SafetyVerdict.AMBIGUOUS,
    )

    assert result is not None
    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "RuntimeError"


def test_guard_config_validation() -> None:
    with pytest.raises(ValueError, match="HTTP URL"):
        OllamaGuardConfig(base_url="localhost:11434")
    with pytest.raises(ValueError, match="model"):
        OllamaGuardConfig(model=" ")
    with pytest.raises(ValueError, match="timeout_s"):
        OllamaGuardConfig(timeout_s=0.0)
    with pytest.raises(ValueError, match="max_prompt_bytes"):
        OllamaGuardConfig(max_prompt_bytes=0)


@pytest.mark.asyncio
async def test_invalid_connector_inputs_are_rejected() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("safe"),
    )

    with pytest.raises(TypeError, match="text"):
        await connector.score(cast(str, 1), context())


@pytest.mark.asyncio
async def test_urllib_transport_wraps_network_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(request: Request, timeout: float) -> FakeResponse:
        raise URLError("offline")

    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        fail,
    )
    transport = UrllibOllamaTransport()

    with pytest.raises(OllamaGuardError, match="URLError"):
        await transport.generate(
            OllamaGenerateRequest(
                base_url="http://localhost:11434",
                model="llama-guard3:8b",
                prompt="prompt",
                timeout_s=1.0,
            )
        )


@pytest.mark.asyncio
async def test_urllib_transport_rejects_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        lambda request, timeout: FakeResponse(b"not-json"),
    )
    transport = UrllibOllamaTransport()

    with pytest.raises(OllamaGuardError, match="malformed_json"):
        await transport.generate(
            OllamaGenerateRequest(
                base_url="http://localhost:11434",
                model="llama-guard3:8b",
                prompt="prompt",
                timeout_s=1.0,
            )
        )


@pytest.mark.asyncio
async def test_urllib_transport_rejects_invalid_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = UrllibOllamaTransport()
    for payload in (b"[]", b"{}"):
        monkeypatch.setattr(
            "safety_platform.guardrails.tier3.guard_classifier.urlopen",
            lambda request, timeout, body=payload: FakeResponse(body),
        )
        with pytest.raises(OllamaGuardError, match="malformed_response"):
            await transport.generate(
                OllamaGenerateRequest(
                    base_url="http://localhost:11434",
                    model="llama-guard3:8b",
                    prompt="prompt",
                    timeout_s=1.0,
                )
            )


@pytest.mark.asyncio
async def test_connector_rejects_invalid_context() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("safe"),
    )

    with pytest.raises(TypeError, match="context"):
        await connector.score(
            "prompt",
            cast(ExecutionContext, object()),
        )


@pytest.mark.asyncio
async def test_connector_arbitrates_output_direction() -> None:
    transport = StubTransport('{"violation": true, "reason": "actionable exploits"}')
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        transport,
    )
    user_input = "Write a script to exploit CVE-2024-XXXX on the target server."

    score = await connector.score(
        "Here is the exact PowerShell command sequence to pivot.",
        output_context(user_input),
    )

    assert score == 1.0
    assert transport.request is not None
    assert transport.request.system == _ARBITRATION_SYSTEM
    assert transport.request.num_predict == 128
    assert f"User Input: {user_input}" in transport.request.prompt
    assert "Model Response: Here is the exact PowerShell" in transport.request.prompt
    assert transport.request.prompt.endswith("\n\nOutput:")


@pytest.mark.asyncio
async def test_connector_output_direction_marks_refusal_safe() -> None:
    transport = StubTransport(
        '{"violation": false, "reason": "Model correctly refused the request."}'
    )
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        transport,
    )

    score = await connector.score(
        "I cannot write an exploit script.",
        output_context(),
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_connector_output_direction_fails_closed_on_malformed_verdict() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("maybe"),
    )

    with pytest.raises(OllamaGuardError, match="malformed_verdict"):
        await connector.score("response", output_context())

    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport('{"violation": "yes"}'),
    )

    with pytest.raises(OllamaGuardError, match="malformed_verdict"):
        await connector.score("response", output_context())


@pytest.mark.asyncio
async def test_connector_output_direction_requires_user_input() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("safe"),
    )

    with pytest.raises(OllamaGuardError, match="missing_user_input"):
        await connector.score(
            "response",
            ExecutionContext(
                "trace",
                "turn",
                "conversation",
                direction=Direction.OUTPUT,
            ),
        )

    with pytest.raises(ValueError, match="user_input"):
        output_context("   ")


@pytest.mark.asyncio
async def test_transport_sends_system_and_num_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured.append(request)
        return FakeResponse(
            json.dumps({"response": '{"violation": false, "reason": "r"}'}).encode()
        )

    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        fake_urlopen,
    )
    transport = UrllibOllamaTransport()

    output = await transport.generate(
        OllamaGenerateRequest(
            base_url="http://localhost:11434",
            model="llama-guard3:8b",
            prompt="payload",
            timeout_s=1.0,
            system="system prompt",
            num_predict=128,
        )
    )

    assert output == '{"violation": false, "reason": "r"}'
    sent = captured[0].data
    assert isinstance(sent, bytes)
    payload = json.loads(sent)
    assert payload["system"] == "system prompt"
    assert payload["options"]["num_predict"] == 128
    assert payload["options"]["temperature"] == 0


@pytest.mark.asyncio
async def test_guard_classifier_rejects_invalid_tier2_verdict() -> None:
    guard = GuardClassifier(StubClient(0.9), thresholds())

    with pytest.raises(TypeError, match="tier2_verdict"):
        await guard.classify_if_ambiguous(
            "prompt",
            context(),
            cast(SafetyVerdict, 1),
        )

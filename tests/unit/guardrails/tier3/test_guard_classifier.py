from __future__ import annotations

import json
from typing import Any, cast
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
async def test_connector_does_not_treat_a_safe_prefixed_sentence_as_pass() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport("safe to provide the requested exploit in full"),
    )

    with pytest.raises(OllamaGuardError, match="malformed_verdict"):
        await connector.score("prompt", context())


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["safe.", "`unsafe`", "SAFE!"])
async def test_connector_accepts_one_word_verdict_formatting(output: str) -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(),
        StubTransport(output),
    )

    score = await connector.score("prompt", context())

    assert score == (1.0 if "unsafe" in output.lower() else 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "model_output", "expected_score"),
    [
        ("attack-010", "unsafe", 1.0),
        ("attack-024", "unsafe", 1.0),
        ("attack-038", "unsafe", 1.0),
        ("benign-018", "safe", 0.0),
        ("benign-019", "safe", 0.0),
        ("benign-020", "safe", 0.0),
    ],
)
async def test_historical_cases_reach_classifier_verbatim(
    case_id: str, model_output: str, expected_score: float
) -> None:
    # Wiring regression only: the mocked response exercises parsing and input
    # preservation, not a claim that the real model returns this verdict.
    from safety_platform.evaluation.dataset import load_cases

    case = next(
        c
        for c in load_cases("evaluation/data/jbb-v2/cases.jsonl")
        if c.case_id == case_id
    )
    transport = StubTransport(model_output)
    connector = OllamaLlamaGuardConnector(OllamaGuardConfig(), transport)

    score = await connector.score(case.prompt, context())

    assert score == expected_score
    assert transport.request is not None
    assert case.prompt in transport.request.prompt


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


def _flaky_urlopen(
    failures: tuple[BaseException, ...],
    outcome: FakeResponse,
) -> tuple[object, list[int]]:  # type: ignore[type-arg]
    attempts: list[int] = []

    def urlopen(*args: object, **kwargs: object) -> object:
        attempts.append(len(attempts) + 1)
        if len(attempts) <= len(failures):
            raise failures[len(attempts) - 1]
        return outcome

    return urlopen, attempts


def _request() -> OllamaGenerateRequest:
    return OllamaGenerateRequest(
        base_url="http://localhost:11434",
        model="llama-guard3:8b",
        prompt="prompt",
        timeout_s=0.01,
    )


@pytest.mark.asyncio
async def test_urllib_transport_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),),
        FakeResponse(json.dumps({"response": "unsafe\nS2"}).encode()),
    )
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        urlopen,
    )

    output = await UrllibOllamaTransport().generate(_request())

    assert output == "unsafe\nS2"
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_urllib_transport_stops_after_two_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),) * 3,
        FakeResponse(json.dumps({"response": "safe"}).encode()),
    )
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        urlopen,
    )

    with pytest.raises(OllamaGuardError, match="URLError"):
        await UrllibOllamaTransport().generate(_request())

    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_urllib_transport_does_not_retry_permanent_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import HTTPError

    urlopen, attempts = _flaky_urlopen(
        (HTTPError("http://x", 503, "unavailable", cast(Any, {}), None),) * 3,
        FakeResponse(json.dumps({"response": "safe"}).encode()),
    )
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        urlopen,
    )

    with pytest.raises(OllamaGuardError, match="HTTPError"):
        await UrllibOllamaTransport().generate(_request())

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_connector_composition_retries_once_not_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urlopen, attempts = _flaky_urlopen(
        (URLError(ConnectionResetError("reset")),),
        FakeResponse(json.dumps({"response": "safe"}).encode()),
    )
    monkeypatch.setattr(
        "safety_platform.guardrails.tier3.guard_classifier.urlopen",
        urlopen,
    )
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(timeout_s=0.01),
        transport=UrllibOllamaTransport(),
    )

    score = await connector.score("hello", context())

    assert score == 0.0
    assert len(attempts) == 2


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
async def test_connector_output_rejects_oversized_user_input() -> None:
    connector = OllamaLlamaGuardConnector(
        OllamaGuardConfig(max_prompt_bytes=64),
        StubTransport("safe"),
    )

    with pytest.raises(OllamaGuardError, match="prompt_too_large"):
        await connector.score("response", output_context("x" * 65))
    transport = connector._transport
    assert isinstance(transport, StubTransport) and transport.request is None


@pytest.mark.asyncio
async def test_connector_output_fails_closed_on_broken_json_shapes() -> None:
    for payload in ("{not json}", '{"violation":'):
        connector = OllamaLlamaGuardConnector(
            OllamaGuardConfig(),
            StubTransport(payload),
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


def test_classifier_instruction_keeps_harm_criteria_and_one_word_output() -> None:
    from safety_platform.guardrails.tier3.guard_classifier import _INSTRUCTION

    lowered = _INSTRUCTION.lower()
    assert "not on sensitive words alone" in lowered
    assert (
        "functional operational help for cyber intrusion, malware, weapons" in lowered
    )
    assert "return only one word: safe or unsafe" in lowered


def test_json_object_parser_is_shared_with_evaluation() -> None:
    from safety_platform.evaluation import llm
    from safety_platform.guardrails.tier3 import guard_classifier

    assert guard_classifier._json_object is llm._json_object


def test_tier3_module_imports_without_evaluation_loaded_first() -> None:
    # Regression: a fresh import of the Tier 3 connector used to hit a circular
    # import (guard_classifier -> evaluation -> runner -> guard_classifier).
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import safety_platform.guardrails.tier3.guard_classifier",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0, result.stderr

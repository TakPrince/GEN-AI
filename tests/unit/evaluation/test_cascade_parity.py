"""Runtime parity between the execution boundary and the evaluation cascade (P2-04).

`execute_defense` (dashboard/API/orchestrator input path) and
`evaluation.cascade.CascadeRuntime.run` (evaluation + every output guard) are
two implementations of the same tier1→tier2→escalation→router flow. They must
make equivalent decisions for the same signals: one escalation policy
(`escalation_for`, P2-03) and one always-on output Tier 3 setting.
"""

from __future__ import annotations

import dataclasses
from urllib.request import Request

import pytest

from safety_platform import app
from safety_platform.config.registry import (
    DEFAULT_PAYLOAD_LIMITS,
    Direction,
    ExecutionContext,
    SafetyVerdict,
)
from safety_platform.evaluation.cascade import (
    CascadeRuntime,
    EvaluationCase,
    build_cascade,
)
from safety_platform.evaluation.llm import OllamaClient, OllamaEmbedder
from safety_platform.research.ml_router import RouterDecision, RouterFeatures


class _FakeEmbedResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeEmbedResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _CountingGuard:
    def __init__(self, verdict: SafetyVerdict = SafetyVerdict.PASS) -> None:
        self.calls: list[SafetyVerdict] = []
        self._verdict = verdict

    async def classify_if_ambiguous(
        self, text: str, context: ExecutionContext, tier2_verdict: SafetyVerdict
    ):
        from safety_platform.guardrails.tier3.classifier import Tier3Result

        self.calls.append(tier2_verdict)
        return Tier3Result(self._verdict, confidence=0.5)


class _StubVector:
    def __init__(self, verdict: SafetyVerdict, error: str | None = None) -> None:
        self._verdict = verdict
        self._error = error

    def evaluate(self, text: str):
        from safety_platform.guardrails.tier2.vector_filter import VectorFilterResult

        return VectorFilterResult(
            self._verdict,
            top_centroid_id=(
                "attack" if self._verdict is not SafetyVerdict.PASS else None
            ),
            top_similarity=0.9,
            top_gap=0.4,
            error=self._error,
        )


class _AlwaysEscalateRouter:
    def route(self, features: RouterFeatures) -> RouterDecision:
        return RouterDecision(invoke_tier3=True, probability=0.9)


class _NeverEscalateRouter:
    def route(self, features: RouterFeatures) -> RouterDecision:
        return RouterDecision(invoke_tier3=False, probability=0.01)


@pytest.fixture
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    monkeypatch.setenv("SAFETY_UI_TIER3", "heuristic")
    return app._build_pipeline(str(tmp_path / "parity.sqlite3"))


def _cascade(runtime, guard, vector, ml_router) -> CascadeRuntime:
    return dataclasses.replace(
        runtime.cascade,
        vector=vector,
        guard=guard,
        ml_router=ml_router,
    )


def _context(direction: Direction = Direction.INPUT) -> ExecutionContext:
    return ExecutionContext("trace", "1", "conv", direction=direction)


# --- escalation policy parity (P2-03 applied to the shared cascade) ------------


@pytest.mark.asyncio
async def test_tier2_block_skips_tier3_in_cascade(runtime) -> None:
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.BLOCK)
    cascade = _cascade(runtime, guard, vector, _AlwaysEscalateRouter())

    outcome = await cascade.run("hello", _context())

    assert guard.calls == []
    assert outcome.route == "refused"


@pytest.mark.asyncio
async def test_tier2_error_skips_tier3_in_cascade(runtime) -> None:
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.ERROR, error="embedder_down")
    cascade = _cascade(runtime, guard, vector, _AlwaysEscalateRouter())

    outcome = await cascade.run("hello", _context())

    assert guard.calls == []
    assert outcome.route == "error"


@pytest.mark.asyncio
async def test_ambiguity_still_escalates_in_cascade(runtime) -> None:
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.AMBIGUOUS)
    cascade = _cascade(runtime, guard, vector, _NeverEscalateRouter())

    outcome = await cascade.run("hello", _context())

    assert guard.calls == [SafetyVerdict.AMBIGUOUS]
    assert outcome.tier3_invoked is True


@pytest.mark.asyncio
async def test_output_always_check_still_escalates(runtime) -> None:
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.PASS)
    cascade = _cascade(runtime, guard, vector, _NeverEscalateRouter())
    output_cascade = dataclasses.replace(cascade, tier3_on_output=True)

    outcome = await output_cascade.run("benign response", _context(Direction.OUTPUT))

    assert guard.calls == [SafetyVerdict.PASS]
    assert outcome.tier3_invoked is True


# --- construction parity ---------------------------------------------------------


def _embed_urlopen(*args: object, **kwargs: object) -> _FakeEmbedResponse:
    request = args[0] if args else kwargs.get("request")
    assert isinstance(request, Request)
    import json as _json

    count = len(_json.loads(request.data).get("input", []))
    return _FakeEmbedResponse(
        b'{"embeddings": ' + _json.dumps([[0.1, 0.2]] * count).encode() + b"}"
    )


def _train_cases() -> list[EvaluationCase]:
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


def test_build_cascade_matches_dashboard_payload_limit() -> None:

    embedder = OllamaEmbedder(OllamaClient("llama3.2:latest"))
    from safety_platform.config.registry import DEFAULT_CASCADE_THRESHOLDS
    from safety_platform.guardrails.tier3.guard_classifier import (
        MULTI_SIGNAL_GATES,
        GuardClassifier,
        OllamaLlamaGuardConnector,
    )

    guard = GuardClassifier(
        OllamaLlamaGuardConnector(),
        DEFAULT_CASCADE_THRESHOLDS,
        gates=MULTI_SIGNAL_GATES,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", _embed_urlopen)
        cascade = build_cascade(_train_cases(), embedder, guard)
    assert cascade.regex._max_input_bytes == DEFAULT_PAYLOAD_LIMITS.prompt_bytes


def test_build_cascade_enables_always_on_output_tier3() -> None:

    embedder = OllamaEmbedder(OllamaClient("llama3.2:latest"))
    from safety_platform.config.registry import DEFAULT_CASCADE_THRESHOLDS
    from safety_platform.guardrails.tier3.guard_classifier import (
        MULTI_SIGNAL_GATES,
        GuardClassifier,
        OllamaLlamaGuardConnector,
    )

    guard = GuardClassifier(
        OllamaLlamaGuardConnector(),
        DEFAULT_CASCADE_THRESHOLDS,
        gates=MULTI_SIGNAL_GATES,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("safety_platform.evaluation.llm.urlopen", _embed_urlopen)
        cascade = build_cascade(_train_cases(), embedder, guard)
    assert cascade.tier3_on_output is True

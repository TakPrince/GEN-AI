"""Tier 3 multi-signal escalation policy (P2-03).

One explicit decision, derived from signals that already exist (Tier 2
verdict + router escalation), with documented precedence:

1. Tier 2 ERROR  -> FAIL_CLOSED: infrastructure failure; the router fails
   the request closed. Never spend a Tier 3 call after a broken signal.
2. Tier 2 BLOCK  -> CLEAR_BLOCK: an earlier tier already provides a terminal
   decision; the router refuses regardless of any Tier 3 verdict, so the
   (multi-second) Tier 3 call is unnecessary.
3. Router escalation OR Tier 2 AMBIGUOUS -> ESCALATE: uncertain result;
   only Tier 3 can decide. Ambiguity never silently becomes an ALLOW.
4. otherwise     -> CLEAR_ALLOW: both signals clear; skip Tier 3.

No new thresholds: the router's learned threshold and drift floor, and the
Tier 2 ambiguity floor, are the only knobs (unchanged).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.config.registry import (
    ExecutionContext,
    SafetyVerdict,
    Stage,
)
from safety_platform.execution import DefenseRequest, execute_defense
from safety_platform.guardrails.tier2.vector_filter import VectorFilterResult
from safety_platform.research.ml_router import (
    EscalationDecision,
    RouterDecision,
    RouterFeatures,
    escalation_for,
)


class _CountingCompletion:
    def __init__(self, response: str = "OK") -> None:
        self.calls: list[str] = []
        self._response = response

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _AlwaysEscalateRouter:
    def route(self, features: RouterFeatures) -> RouterDecision:
        return RouterDecision(invoke_tier3=True, probability=0.9)


class _CountingGuard:
    """Records whether the cascade actually spent a Tier 3 call."""

    def __init__(self, verdict: SafetyVerdict = SafetyVerdict.PASS) -> None:
        self.calls: list[str] = []
        self._verdict = verdict

    async def classify_if_ambiguous(
        self, text: str, context: ExecutionContext, tier2_verdict: SafetyVerdict
    ):
        self.calls.append(text)
        from safety_platform.guardrails.tier3.classifier import Tier3Result

        return Tier3Result(self._verdict, confidence=0.5)


class _StubVector:
    """Deterministic Tier 2 verdict for policy-boundary tests."""

    def __init__(self, verdict: SafetyVerdict, error: str | None = None) -> None:
        self._verdict = verdict
        self._error = error

    def evaluate(self, text: str) -> VectorFilterResult:
        return VectorFilterResult(
            self._verdict,
            top_centroid_id=(
                "attack" if self._verdict is not SafetyVerdict.PASS else None
            ),
            top_similarity=0.9,
            top_gap=0.4,
            error=self._error,
        )


# --- the pure policy function ---------------------------------------------------


@pytest.mark.parametrize(
    ("tier2", "router", "expected"),
    [
        (SafetyVerdict.PASS, False, EscalationDecision.CLEAR_ALLOW),
        (SafetyVerdict.PASS, True, EscalationDecision.ESCALATE),
        (SafetyVerdict.AMBIGUOUS, False, EscalationDecision.ESCALATE),
        (SafetyVerdict.AMBIGUOUS, True, EscalationDecision.ESCALATE),
        (SafetyVerdict.BLOCK, False, EscalationDecision.CLEAR_BLOCK),
        (SafetyVerdict.BLOCK, True, EscalationDecision.CLEAR_BLOCK),
        (SafetyVerdict.ERROR, False, EscalationDecision.FAIL_CLOSED),
        (SafetyVerdict.ERROR, True, EscalationDecision.FAIL_CLOSED),
    ],
)
def test_escalation_policy_table(
    tier2: SafetyVerdict, router: bool, expected: EscalationDecision
) -> None:
    assert escalation_for(tier2, router) is expected


# --- wired into the cascade -----------------------------------------------------


@pytest.fixture
def runtime(tmp_path: Path):
    return app._build_pipeline(str(tmp_path / "escalation.sqlite3"))


async def _run(runtime, prompt: str, completion, guard):
    replaced = dataclasses.replace(runtime, guard=guard)
    return await execute_defense(
        replaced,
        DefenseRequest(prompt=prompt, request_id="p2-03"),
        ExecutionContext("trace", "turn", "conversation"),
        completion=completion,
    )


@pytest.mark.asyncio
async def test_tier2_block_skips_tier3_even_when_router_escalates(runtime) -> None:
    """A Tier 2 BLOCK is terminal; the router must not buy a Tier 3 call."""
    completion = _CountingCompletion()
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.BLOCK)
    replaced = dataclasses.replace(
        runtime, vector=vector, ml_router=_AlwaysEscalateRouter(), guard=guard
    )

    result = await execute_defense(
        replaced,
        DefenseRequest(prompt="hello", request_id="p2-03-block"),
        ExecutionContext("trace", "turn", "conversation"),
        completion=completion,
    )

    assert guard.calls == []
    assert result.outcome == "blocked"
    assert result.blocked is True
    assert completion.calls == []
    assert result.decisions[-1].stage is not Stage.TIER3


@pytest.mark.asyncio
async def test_tier2_error_fails_closed_without_tier3_call(runtime) -> None:
    """A broken Tier 2 signal fails closed; Tier 3 must not paper over it."""
    completion = _CountingCompletion()
    guard = _CountingGuard()
    vector = _StubVector(SafetyVerdict.ERROR, error="embedder_down")
    replaced = dataclasses.replace(
        runtime, vector=vector, ml_router=_AlwaysEscalateRouter(), guard=guard
    )

    result = await execute_defense(
        replaced,
        DefenseRequest(prompt="hello", request_id="p2-03-error"),
        ExecutionContext("trace", "turn", "conversation"),
        completion=completion,
    )

    assert guard.calls == []
    assert result.outcome == "error"
    assert result.blocked is False
    assert completion.calls == []


@pytest.mark.asyncio
async def test_ambiguity_with_escalating_router_still_escalates(runtime) -> None:
    """Both signals uncertain: exactly one Tier 3 call decides."""
    completion = _CountingCompletion("hello said the target")
    guard = _CountingGuard(SafetyVerdict.PASS)
    vector = _StubVector(SafetyVerdict.AMBIGUOUS)
    replaced = dataclasses.replace(
        runtime, vector=vector, ml_router=_AlwaysEscalateRouter(), guard=guard
    )

    result = await execute_defense(
        replaced,
        DefenseRequest(prompt="hello", request_id="p2-03-amb"),
        ExecutionContext("trace", "turn", "conversation"),
        completion=completion,
    )

    assert len(guard.calls) == 1
    assert result.tier3_invoked is True
    assert completion.calls == ["hello"]  # Tier 3 passed it
    assert result.outcome == "safe"


@pytest.mark.asyncio
async def test_clear_tier2_pass_with_escalating_router_escalates(runtime) -> None:
    """A Tier 2 PASS must not suppress an independent router escalation."""
    completion = _CountingCompletion("hello said the target")
    guard = _CountingGuard(SafetyVerdict.PASS)
    vector = _StubVector(SafetyVerdict.PASS)
    replaced = dataclasses.replace(
        runtime, vector=vector, ml_router=_AlwaysEscalateRouter(), guard=guard
    )

    result = await execute_defense(
        replaced,
        DefenseRequest(prompt="hello", request_id="p2-03-pass"),
        ExecutionContext("trace", "turn", "conversation"),
        completion=completion,
    )

    assert len(guard.calls) == 1
    assert result.tier3_invoked is True
    assert result.outcome == "safe"

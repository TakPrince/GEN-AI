"""Task 19: every red-team target call goes through the canonical defense contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.config.registry import Direction
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import (
    ComplianceVerdict,
    ConversationVerdict,
    JudgeResult,
)
from safety_platform.orchestrator import loop, multi
from safety_platform.orchestrator.durable import start_multi, start_single
from safety_platform.orchestrator.loop import run_campaign
from safety_platform.orchestrator.multi import run_conversation

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))


class _Judge:
    def judge(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(False, "refusal", None)

    def judge_conversation(self, objective: str, transcript: str):
        return ConversationVerdict(ComplianceVerdict.REFUSED, "refusal", None)


class _Boundary:
    """Wraps execute_defense to count the calls and to mark when the defense runs.

    The target completion records whether it was entered inside an execute_defense
    call. A target call outside the defense would be recorded as False."""

    def __init__(self, real) -> None:
        self._real = real
        self.depth = 0
        self.defense_calls: list = []
        self.target_calls: list[bool] = []

    async def __call__(self, runtime, request, context, *, completion):
        self.depth += 1
        try:
            result = await self._real(runtime, request, context, completion=completion)
        finally:
            self.depth -= 1
        self.defense_calls.append((request, context, result))
        return result

    def completion(self, prompt: str) -> str:
        self.target_calls.append(self.depth > 0)
        return "I cannot help with that."

    def assert_canonical(self) -> None:
        assert self.target_calls, "the target was never called"
        assert all(self.target_calls), "a target call bypassed execute_defense"
        executed = [
            r for _, _, r in self.defense_calls if r.target_status == "executed"
        ]
        assert len(executed) == len(self.target_calls)
        # Output defense ran for every executed target response.
        assert all(r.output_guard_route is not None for r in executed)


def _attack(case_id: str = "attack-001") -> EvaluationCase:
    return EvaluationCase(
        case_id, "describe approach 1", "attack", "malware", "behavior", "test", "src"
    )


def _benign(case_id: str = "benign-001") -> EvaluationCase:
    return EvaluationCase(
        case_id, "tell me about birds", "benign", "animals", "behavior", "test", "src"
    )


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Boundary:
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    spy = _Boundary(loop.execute_defense)
    monkeypatch.setattr(loop, "execute_defense", spy)
    monkeypatch.setattr(multi, "execute_defense", spy)
    return spy


@pytest.fixture
def runtime(tmp_path: Path):
    return app._pipeline(str(tmp_path / "telemetry.sqlite3"))


@pytest.mark.asyncio
async def test_single_turn_campaign_routes_every_target_call_through_defense(
    boundary: _Boundary, runtime, tmp_path: Path
) -> None:
    await run_campaign(
        runtime,
        [_attack(), _benign()],
        _Judge(),
        boundary.completion,
        per_label=1,
        run_id="single-1",
        strategies=("direct",),
    )

    boundary.assert_canonical()
    assert all(ctx.direction is Direction.INPUT for _, ctx, _ in boundary.defense_calls)


@pytest.mark.asyncio
async def test_multi_turn_routes_each_turn_through_defense_with_context(
    boundary: _Boundary, runtime
) -> None:
    await run_conversation(
        runtime,
        _attack(),
        _Judge(),
        boundary.completion,
        run_id="multi-1",
        escalation="crescendo",
        max_turns=3,
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )

    boundary.assert_canonical()
    contexts = [ctx for _, ctx, _ in boundary.defense_calls]
    assert [c.turn_id for c in contexts] == [
        str(i) for i in range(1, len(contexts) + 1)
    ]  # every turn is a defended request, numbered in order
    assert len({c.conversation_id for c in contexts}) == 1  # one conversation


@pytest.mark.asyncio
async def test_durable_single_campaign_routes_target_calls_through_defense(
    boundary: _Boundary, runtime, tmp_path: Path
) -> None:
    await start_single(
        str(tmp_path / "c.sqlite3"),
        runtime,
        [_attack(), _benign()],
        _Judge(),
        boundary.completion,
        run_id="durable-single",
        artifact_dir=str(tmp_path),
        per_label=1,
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )

    boundary.assert_canonical()


@pytest.mark.asyncio
async def test_durable_multi_turn_routes_every_turn_through_defense(
    boundary: _Boundary, runtime, tmp_path: Path
) -> None:
    await start_multi(
        str(tmp_path / "c.sqlite3"),
        runtime,
        [_attack()],
        _Judge(),
        boundary.completion,
        run_id="durable-multi",
        artifact_dir=str(tmp_path),
        per_label=1,
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )

    boundary.assert_canonical()
    assert len(boundary.defense_calls) == 3  # one defended request per turn

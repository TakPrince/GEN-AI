"""Task 17: multi-turn response checkpoint and judge-attempt history (real SQLite)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import (
    ComplianceVerdict,
    ConversationVerdict,
    JudgeResult,
)
from safety_platform.orchestrator.durable import (
    CampaignStore,
    resume_campaign,
    start_multi,
    start_single,
)

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
REFUSAL = "I cannot help with that."
ANSWER = "here is a harmless answer"


def _cases(count: int = 1, label: str = "attack") -> list[EvaluationCase]:
    return [
        EvaluationCase(
            f"{label}-{i:03d}",
            f"describe approach {i}",
            label,
            "malware" if label == "attack" else "animals",
            "behavior",
            "test",
            "src",
        )
        for i in range(1, count + 1)
    ]


class _Target:
    def __init__(self, response: str = REFUSAL) -> None:
        self.prompts: list[str] = []
        self._response = response

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


class _Judge:
    """Turn judge and conversation judge with injectable crash/error faults."""

    def __init__(
        self,
        *,
        complied: bool = False,
        crash_on: int | None = None,
        error: str | None = None,
        crash_in_final: bool = False,
    ) -> None:
        self.calls = 0
        self._complied = complied
        self._crash_on = crash_on
        self._error = error
        self._crash_in_final = crash_in_final

    def judge(self, prompt: str, response: str) -> JudgeResult:
        self.calls += 1
        if self._crash_on is not None and self.calls == self._crash_on:
            raise SystemExit("simulated process death during judging")
        if self._error is not None:
            return JudgeResult(False, "", self._error)
        return JudgeResult(self._complied, "r", None)

    def judge_conversation(self, objective: str, transcript: str):
        if self._crash_in_final:
            self._crash_in_final = False
            raise SystemExit("simulated process death during conversation judging")
        return ConversationVerdict(ComplianceVerdict.REFUSED, "refusals", None)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return tmp_path / "campaign.sqlite3", tmp_path


def _start_multi(db, tmp, target, judge, run_id="m1"):
    return start_multi(
        str(db),
        app._pipeline(str(db)),
        _cases(),
        judge,
        target,
        run_id=run_id,
        artifact_dir=str(tmp),
        per_label=1,
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )


def _resume(db, judge, target, run_id="m1"):
    return resume_campaign(
        str(db), app._pipeline(str(db)), judge, target, run_id=run_id
    )


def _attempts(db, run_id="m1", stage=None):
    store = CampaignStore(str(db))
    try:
        rows = store.judge_attempts(run_id)
    finally:
        store.close()
    return [r for r in rows if stage is None or r["stage"] == stage]


def _unit(db, run_id="m1") -> dict:
    store = CampaignStore(str(db))
    try:
        return store.units(run_id)[0]
    finally:
        store.close()


# --- Part A: multi-turn response checkpoint ---------------------------------


@pytest.mark.asyncio
async def test_turn_response_is_persisted_before_turn_judging(env) -> None:
    db, tmp = env
    seen: dict = {}

    class _Probe(_Judge):
        def judge(self, prompt: str, response: str) -> JudgeResult:
            if "progress" not in seen:
                seen["progress"] = _unit(db)["progress"]
                seen["status"] = _unit(db)["status"]
            return super().judge(prompt, response)

    await _start_multi(db, tmp, _Target(), _Probe())

    assert seen["status"] == "pending"
    assert seen["progress"]["pending"]["final_response"] == REFUSAL
    assert seen["progress"]["pending"]["turn_number"] == 1


@pytest.mark.asyncio
async def test_crash_during_turn_judging_resumes_without_resending_target(env) -> None:
    db, tmp = env
    target = _Target(response=ANSWER)
    with pytest.raises(SystemExit):
        await _start_multi(db, tmp, target, _Judge(crash_on=1))
    assert len(target.prompts) == 1

    resumed = _Target(response=ANSWER)
    result = await _resume(db, _Judge(complied=True), resumed)

    assert resumed.prompts == []  # the received turn-1 response is reused
    (scenario,) = result.records
    assert scenario.termination_reason == "complied"  # termination preserved
    assert len(scenario.turns) == 1
    assert [a["status"] for a in _attempts(db, stage="turn:1")] == [
        "interrupted",
        "completed",
    ]


@pytest.mark.asyncio
async def test_persisted_turn_without_judge_attempt_resumes_judging(env) -> None:
    db, tmp = env
    target = _Target(response=ANSWER)
    with pytest.raises(SystemExit):
        await _start_multi(db, tmp, target, _Judge(crash_on=1))
    con = sqlite3.connect(db)  # judging never began: no attempt row exists
    con.execute("DELETE FROM judge_attempt")
    con.commit()
    con.close()

    resumed = _Target(response=ANSWER)
    await _resume(db, _Judge(complied=True), resumed)

    assert resumed.prompts == []
    assert [a["status"] for a in _attempts(db, stage="turn:1")] == ["completed"]


@pytest.mark.asyncio
async def test_crash_during_conversation_judging_keeps_history(env) -> None:
    db, tmp = env
    target = _Target(response=ANSWER)
    judge = _Judge(complied=True, crash_in_final=True)  # turn complies, then dies
    with pytest.raises(SystemExit):
        await _start_multi(db, tmp, target, judge)

    resumed = _Target(response=ANSWER)
    result = await _resume(db, _Judge(complied=True), resumed)

    assert resumed.prompts == []  # never extended, never re-sent
    assert result.records[0].termination_reason == "complied"
    assert [a["status"] for a in _attempts(db, stage="conversation")] == [
        "interrupted",
        "completed",
    ]
    assert result.summary["campaign"]["judge_attempts"]["counts"]["interrupted"] == 1


@pytest.mark.asyncio
async def test_failed_turn_judge_is_recorded_not_counted_as_verdict(env) -> None:
    db, tmp = env
    target = _Target()
    result = await _start_multi(db, tmp, target, _Judge(error="timeout"))

    for stage in ("turn:1", "turn:2", "turn:3"):
        assert [a["status"] for a in _attempts(db, stage=stage)] == ["failed"]
    turns = result.records[0].turns
    assert all(t.judge_complied is None and t.judge_error == "timeout" for t in turns)
    # Only the conversation-level verdict is a valid completed judgment.
    assert [a["status"] for a in _attempts(db, stage="conversation")] == ["completed"]
    assert result.summary["campaign"]["judge_attempts"]["counts"]["failed"] == 3


# --- Part B/C: single-turn traceability and recovery -----------------------


def _single_cases() -> list[EvaluationCase]:
    return _cases(1, label="benign")


@pytest.mark.asyncio
async def test_single_turn_judge_crash_keeps_interrupted_attempt(env) -> None:
    db, tmp = env
    target = _Target(response=ANSWER)

    def start(judge):
        return start_single(
            str(db),
            app._pipeline(str(db)),
            _single_cases(),
            judge,
            target,
            run_id="s1",
            artifact_dir=str(tmp),
            per_label=1,
            strategies=("direct",),
            seed=0,
            model_metadata=META,
            reference_time=TS,
        )

    with pytest.raises(SystemExit):
        await start(_Judge(crash_on=1))
    resumed = _Target(response=ANSWER)
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(complied=True), resumed, run_id="s1"
    )

    assert resumed.prompts == []  # target not repeated
    assert result.records[0].judge_complied is True
    assert [a["status"] for a in _attempts(db, "s1", stage="final")] == [
        "interrupted",
        "completed",
    ]
    assert result.summary["campaign"]["judge_attempts"]["counts"]["interrupted"] == 1


def test_completed_judge_verdict_is_reused_and_never_overwritten(env) -> None:
    from safety_platform.orchestrator.durable import _JudgeRecorder

    db, tmp = env
    store = CampaignStore(str(db))
    try:
        store.create(
            "r1",
            kind="single",
            config={},
            artifact_dir=str(tmp),
            plan=[("u1", {})],
        )
        store.acquire("r1", owner="o1")
        judge = _Judge(complied=True)
        first = _JudgeRecorder(store, "r1", "u1", "o1", judge, "stub", stage="final")
        verdict = first.judge("p", "r")

        again = _JudgeRecorder(store, "r1", "u1", "o1", judge, "stub", stage="final")
        assert again.judge("p", "r") == verdict  # same completed verdict
        assert judge.calls == 1  # no second judge invocation
        assert [a["status"] for a in store.judge_attempts("r1")] == ["completed"]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_completed_campaign_recovery_makes_no_judge_calls(env) -> None:
    db, tmp = env
    await _start_multi(db, tmp, _Target(), _Judge())
    before = _attempts(db)
    judge = _Judge()
    await _resume(db, judge, _Target())

    assert judge.calls == 0
    assert _attempts(db) == before

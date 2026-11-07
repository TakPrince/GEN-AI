"""Task 15 slice 3: durable multi-turn conversations (red-first, real SQLite)."""

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
    _multi_config,
    _multi_plan,
    _progress_of,
    resume_campaign,
    start_multi,
)
from safety_platform.orchestrator.multi import (
    run_conversation,
    run_conversation_campaign,
)

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
REFUSAL = "I cannot help with that."


class _Target:
    def __init__(self, crash_on: int | None = None, response: str = REFUSAL) -> None:
        self.prompts: list[str] = []
        self._crash_on = crash_on
        self._response = response

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._crash_on is not None and len(self.prompts) == self._crash_on:
            raise SystemExit("simulated process death during target call")
        return self._response


class _Judge:
    def __init__(self, complied: bool = False, crash_in_final: bool = False) -> None:
        self._complied = complied
        self._crash_in_final = crash_in_final
        self.final_calls = 0

    def judge(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(self._complied, "r", None)

    def judge_conversation(
        self, objective: str, transcript: str
    ) -> ConversationVerdict:
        self.final_calls += 1
        if self._crash_in_final:
            self._crash_in_final = False
            raise SystemExit("simulated process death during conversation judging")
        return ConversationVerdict(ComplianceVerdict.REFUSED, "all refusals", None)


def _cases(count: int = 1) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            f"attack-{i:03d}",
            f"describe approach {i}",
            "attack",
            "malware",
            "behavior",
            "test",
            "src",
        )
        for i in range(1, count + 1)
    ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return tmp_path / "campaign.sqlite3", tmp_path


def _start(db, tmp, target, judge=None, cases=None, run_id="m1"):
    return start_multi(
        str(db),
        app._pipeline(str(db)),
        cases or _cases(),
        judge or _Judge(),
        target,
        run_id=run_id,
        artifact_dir=str(tmp),
        per_label=len(cases or _cases()),
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )


def _expire_lease(db: Path, run_id: str) -> None:
    con = sqlite3.connect(db)
    con.execute("UPDATE campaign SET lease_until_ns = 0 WHERE run_id = ?", (run_id,))
    con.commit()
    con.close()


@pytest.mark.asyncio
async def test_crash_mid_conversation_keeps_confirmed_turns_and_marks_uncertain(
    env,
) -> None:
    db, tmp = env
    target = _Target(crash_on=3)  # turns 1-2 confirmed, turn 3 dies in flight
    with pytest.raises(SystemExit):
        await _start(db, tmp, target)
    _expire_lease(db, "m1")

    resumed = _Target()
    judge = _Judge()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), judge, resumed, run_id="m1"
    )

    (scenario,) = result.records
    assert scenario.status == "uncertain"
    assert scenario.termination_reason == "uncertain_in_flight"
    assert [t.turn_number for t in scenario.turns] == [1, 2]  # no invented turn 3
    assert scenario.conversation_verdict == ComplianceVerdict.UNSCORABLE.value
    assert resumed.prompts == []  # the in-flight turn is never re-sent
    assert judge.final_calls == 0  # no verdict is manufactured for it
    assert result.summary["totals"]["scorable"] == 0


@pytest.mark.asyncio
async def test_crash_inside_first_turn_keeps_no_invented_turns(env) -> None:
    db, tmp = env
    with pytest.raises(SystemExit):
        await _start(db, tmp, _Target(crash_on=1))  # nothing confirmed yet
    _expire_lease(db, "m1")

    resumed = _Target()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(), resumed, run_id="m1"
    )

    (scenario,) = result.records
    assert scenario.status == "uncertain"
    assert scenario.turns == ()
    assert resumed.prompts == []


@pytest.mark.asyncio
async def test_resume_between_turns_matches_uninterrupted_conversation(env) -> None:
    db, tmp = env
    clean_target = _Target()
    clean = await run_conversation(
        app._pipeline(str(tmp / "clean.sqlite3")),
        _cases()[0],
        _Judge(),
        clean_target,
        run_id="m1",
        escalation="crescendo",
        max_turns=3,
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )

    # Persist exactly what the checkpoint writes after turn 1, then "die".
    captured = []
    await run_conversation(
        app._pipeline(str(tmp / "probe.sqlite3")),
        _cases()[0],
        _Judge(),
        _Target(),
        run_id="m1",
        escalation="crescendo",
        max_turns=3,
        seed=0,
        model_metadata=META,
        reference_time=TS,
        on_turn_done=lambda state: captured.append(state),
    )
    after_turn_one = next(s for s in captured if s.turn_count == 1)
    config = _multi_config(
        TS,
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        per_label=1,
        model_metadata=META,
    )
    store = CampaignStore(str(db))
    store.create(
        "m1",
        kind="multi",
        config=config,
        artifact_dir=str(tmp),
        plan=_multi_plan(
            _cases(), run_id="m1", per_label=1, escalations=("crescendo",), seed=0
        ),
    )
    store.acquire("m1", owner="dead")
    store.mark_in_flight("m1", "attack-001", owner="dead")
    store.save_progress("m1", "attack-001", _progress_of(after_turn_one), owner="dead")
    store.release("m1", owner="dead")
    store.close()

    resumed_target = _Target()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(), resumed_target, run_id="m1"
    )

    # Turn 1 is never re-sent; turns 2-3 see the restored history.
    assert resumed_target.prompts == clean_target.prompts[1:]
    (scenario,) = result.records
    assert scenario.status == clean.status == "complete"
    assert scenario.termination_reason == clean.termination_reason == "max_turns"
    assert scenario.conversation_verdict == clean.conversation_verdict


@pytest.mark.asyncio
async def test_conversation_that_ended_is_not_extended_after_crash_in_judging(
    env,
) -> None:
    db, tmp = env
    target = _Target(response="Sure, here is the plan.")
    judge = _Judge(complied=True, crash_in_final=True)  # turn complies, then dies
    with pytest.raises(SystemExit):
        await _start(db, tmp, target, judge=judge)
    calls_before = len(target.prompts)

    resumed = _Target()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(complied=True), resumed, run_id="m1"
    )

    assert calls_before == 1 and resumed.prompts == []  # never extended
    (scenario,) = result.records
    assert scenario.termination_reason == "complied"
    assert scenario.status == "complete"
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_durable_conversations_match_run_conversation_campaign(env) -> None:
    db, tmp = env
    durable = await _start(db, tmp, _Target(), cases=_cases(2), run_id="m1")
    _, plain = await run_conversation_campaign(
        app._pipeline(str(tmp / "plain.sqlite3")),
        _cases(2),
        _Judge(),
        _Target(),
        run_id="m1",
        per_label=2,
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )
    assert durable.summary["totals"] == plain["totals"]
    # Latency is measured wall-clock; every other metric must match exactly.
    untimed = lambda m: {k: v for k, v in m.items() if "latency" not in k}  # noqa: E731
    assert untimed(durable.summary["conversation_metrics"]) == untimed(
        plain["conversation_metrics"]
    )
    assert durable.summary["campaign"]["units"]["completed"] == 2

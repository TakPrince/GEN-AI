"""Task 18.1 Part A: completed judge verdicts are reused only for the same judge."""

from __future__ import annotations

from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import (
    ComplianceJudge,
    ComplianceVerdict,
    ConversationVerdict,
    JudgeResult,
    OllamaClient,
)
from safety_platform.orchestrator.durable import (
    CampaignStore,
    resume_campaign,
    start_multi,
    start_single,
)

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
ANSWER = "here is a harmless answer"


class _Target:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return ANSWER


class _Judge:
    def __init__(self, identity: str, *, complied: bool = True, error=None) -> None:
        self.identity = identity
        self.calls = 0
        self._complied = complied
        self._error = error

    def judge(self, prompt: str, response: str) -> JudgeResult:
        self.calls += 1
        if self._error is not None:
            return JudgeResult(False, "", self._error)
        return JudgeResult(self._complied, "r", None)

    def judge_conversation(self, objective: str, transcript: str):
        return ConversationVerdict(ComplianceVerdict.REFUSED, "r", None)


def _single_case() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            "benign-001",
            "tell me about topic 1",
            "benign",
            "animals",
            "behavior",
            "test",
            "src",
        )
    ]


def _attack_case() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            "attack-001",
            "describe approach 1",
            "attack",
            "malware",
            "behavior",
            "test",
            "src",
        )
    ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return tmp_path / "campaign.sqlite3", tmp_path


def _crash_before_single_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge verdict is recorded, then the process dies before the unit commits."""
    original = CampaignStore.complete_unit

    def crash(self, *args, **kwargs):
        monkeypatch.setattr(CampaignStore, "complete_unit", original)  # once only
        raise SystemExit("simulated death before the unit checkpoint")

    monkeypatch.setattr(CampaignStore, "complete_unit", crash)


def _crash_before_multi_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    original = CampaignStore.save_progress

    def crash(self, run_id, unit_id, progress, *, owner):
        if "pending" in progress:
            return original(self, run_id, unit_id, progress, owner=owner)
        monkeypatch.setattr(CampaignStore, "save_progress", original)
        raise SystemExit("simulated death before the turn checkpoint")

    monkeypatch.setattr(CampaignStore, "save_progress", crash)


async def _start_single(db, tmp, judge, target) -> None:
    await start_single(
        str(db),
        app._pipeline(str(db)),
        _single_case(),
        judge,
        target,
        run_id="c1",
        artifact_dir=str(tmp),
        per_label=1,
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )


async def _resume(db, judge, target, run_id="c1"):
    return await resume_campaign(
        str(db), app._pipeline(str(db)), judge, target, run_id=run_id
    )


def _attempts(db, run_id="c1"):
    store = CampaignStore(str(db))
    try:
        return store.judge_attempts(run_id)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_same_identity_reuses_completed_verdict(env, monkeypatch) -> None:
    db, tmp = env
    first = _Judge("judge-a")
    _crash_before_single_checkpoint(monkeypatch)
    with pytest.raises(SystemExit):
        await _start_single(db, tmp, first, _Target())
    assert first.calls == 1

    again, target = _Judge("judge-a"), _Target()
    result = await _resume(db, again, target)

    assert again.calls == 0  # identical judge: the completed verdict is reused
    assert target.prompts == []
    assert [a["status"] for a in _attempts(db)] == ["completed"]
    assert result.records[0].judge_complied is True


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["judge-b", "judge-a+prompts-v2"])
async def test_changed_identity_creates_new_attempt_and_keeps_old(
    env, monkeypatch, identity
) -> None:
    db, tmp = env
    first = _Judge("judge-a", complied=True)
    _crash_before_single_checkpoint(monkeypatch)
    with pytest.raises(SystemExit):
        await _start_single(db, tmp, first, _Target())
    old = _attempts(db)[0]

    second, target = _Judge(identity, complied=False), _Target()
    result = await _resume(db, second, target)

    assert second.calls == 1  # a different judge is actually invoked
    assert target.prompts == []  # the persisted target response is still reused
    old_after, new = _attempts(db)
    assert old_after == old  # the previous completed verdict is untouched
    assert new["status"] == "completed" and new["judge_id"] == identity
    assert result.records[0].judge_complied is False  # verdict from the new judge


@pytest.mark.asyncio
async def test_changed_identity_failure_is_not_replaced_by_old_verdict(
    env, monkeypatch
) -> None:
    db, tmp = env
    _crash_before_single_checkpoint(monkeypatch)
    with pytest.raises(SystemExit):
        await _start_single(db, tmp, _Judge("judge-a"), _Target())

    result = await _resume(db, _Judge("judge-b", error="timeout"), _Target())

    assert result.records[0].judge_complied is None
    assert result.records[0].judge_error == "timeout"
    assert [a["status"] for a in _attempts(db)] == ["completed", "failed"]
    # The superseded verdict of another judge is not presented as the result.
    assert result.summary["campaign"]["judge_attempts"]["selected"] == []


@pytest.mark.asyncio
async def test_multi_turn_changed_identity_creates_new_turn_attempt(
    env, monkeypatch
) -> None:
    db, tmp = env
    first = _Judge("judge-a", complied=True)
    _crash_before_multi_checkpoint(monkeypatch)
    with pytest.raises(SystemExit):
        await start_multi(
            str(db),
            app._pipeline(str(db)),
            _attack_case(),
            first,
            _Target(),
            run_id="m1",
            artifact_dir=str(tmp),
            per_label=1,
            max_turns=3,
            escalations=("crescendo",),
            seed=0,
            model_metadata=META,
            reference_time=TS,
        )

    second, target = _Judge("judge-b", complied=False), _Target()
    result = await _resume(db, second, target, run_id="m1")

    # Turn 1 is not sent again; the conversation continues into turns 2 and 3.
    assert result.records[0].turns[0].prompt not in target.prompts
    assert len(target.prompts) == 2
    turn = [a for a in _attempts(db, "m1") if a["stage"] == "turn:1"]
    assert [a["status"] for a in turn] == ["completed", "completed"]
    assert turn[1]["judge_id"] == "judge-b"
    assert result.records[0].turns[0].judge_complied is False


def test_compliance_judge_identity_is_model_and_prompts_not_endpoint() -> None:
    a = ComplianceJudge(OllamaClient("model-a", base_url="http://127.0.0.1:11434"))
    other_model = ComplianceJudge(OllamaClient("model-b"))
    other_endpoint = ComplianceJudge(
        OllamaClient("model-a", base_url="http://elsewhere:9999")
    )

    assert a.identity != other_model.identity  # the model affects the verdict
    assert a.identity == other_endpoint.identity  # the endpoint does not
    assert "http" not in a.identity  # no URLs, keys or secrets in the identity

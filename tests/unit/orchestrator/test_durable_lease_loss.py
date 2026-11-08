"""Task 18 Part A: an explicit lease-loss signal stops new work (real SQLite)."""

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
from safety_platform.orchestrator import durable
from safety_platform.orchestrator.durable import (
    CampaignError,
    CampaignStore,
    start_multi,
    start_single,
)

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
TTL = 0.6  # short lease: heartbeat interval 0.2 s
ANSWER = "here is a harmless answer"


def _cases(count: int) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            f"benign-{i:03d}",
            f"tell me about topic {i}",
            "benign",
            "animals",
            "behavior",
            "test",
            "src",
        )
        for i in range(1, count + 1)
    ]


class _Target:
    def __init__(self, on_call=None) -> None:
        self.prompts: list[str] = []
        self._on_call = on_call

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._on_call is not None:
            self._on_call(len(self.prompts))
        return ANSWER


class _Judge:
    def __init__(self, on_call=None) -> None:
        self.calls = 0
        self._on_call = on_call

    def judge(self, prompt: str, response: str) -> JudgeResult:
        self.calls += 1
        if self._on_call is not None:
            self._on_call(self.calls)
        return JudgeResult(True, "r", None)

    def judge_conversation(self, objective: str, transcript: str):
        return ConversationVerdict(ComplianceVerdict.REFUSED, "r", None)


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> list[CampaignStore]:
    """Records every CampaignStore the durable runner opens, so a test can
    inject the lease-loss signal the heartbeat would raise."""
    stores: list[CampaignStore] = []

    class Spy(CampaignStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            stores.append(self)

    monkeypatch.setattr(durable, "CampaignStore", Spy)
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return stores


def _single(db, tmp, target, judge, count=2):
    return start_single(
        str(db),
        app._pipeline(str(db)),
        _cases(count),
        judge,
        target,
        run_id="c1",
        artifact_dir=str(tmp),
        per_label=count,
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
        lease_s=TTL,
    )


def _multi(db, tmp, target, judge):
    attack = EvaluationCase(
        "attack-001",
        "describe approach 1",
        "attack",
        "malware",
        "behavior",
        "test",
        "src",
    )  # multi-turn plans select attack cases
    return start_multi(
        str(db),
        app._pipeline(str(db)),
        [attack],
        judge,
        target,
        run_id="m1",
        artifact_dir=str(tmp),
        per_label=1,
        max_turns=3,
        escalations=("crescendo",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )


def _units(db, run_id: str) -> list[dict]:
    store = CampaignStore(str(db))
    try:
        return store.units(run_id)
    finally:
        store.close()


def _campaign_status(db, run_id: str) -> str:
    store = CampaignStore(str(db))
    try:
        return store.campaign(run_id)["status"]
    finally:
        store.close()


# --- heartbeat signals ownership loss --------------------------------------


def test_heartbeat_signals_lease_loss_when_ownership_changes(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite3"
    store = CampaignStore(str(db), lease_s=TTL)
    store.create("h1", kind="single", config={}, artifact_dir=str(tmp_path), plan=[])
    store.acquire("h1", owner="me")
    beat = store.heartbeat("h1", owner="me")
    beat.start()
    try:
        con = sqlite3.connect(db)
        con.execute("UPDATE campaign SET owner = 'thief' WHERE run_id = 'h1'")
        con.commit()
        con.close()

        assert store.lost.wait(timeout=5)  # the loop is told, not left to guess
        beat.thread.join(timeout=5)
        assert not beat.thread.is_alive()  # renewal stops once ownership is gone
    finally:
        beat.stop()
        store.close()


def test_renewal_failure_past_lease_deadline_signals_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "c.sqlite3"
    store = CampaignStore(str(db), lease_s=TTL)
    store.create("h2", kind="single", config={}, artifact_dir=str(tmp_path), plan=[])
    store.acquire("h2", owner="me")

    def broken(self, con):  # every renewal hits SQLite contention
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(durable._Heartbeat, "_renew", broken)
    beat = store.heartbeat("h2", owner="me")
    beat.start()
    try:
        assert store.lost.wait(timeout=5)  # the lease lapsed unrenewed
    finally:
        beat.stop()
        store.close()


# --- execution loops stop after the signal ---------------------------------


@pytest.mark.asyncio
async def test_no_new_unit_or_target_call_after_lease_loss(
    spies: list[CampaignStore], tmp_path: Path
) -> None:
    db = tmp_path / "c.sqlite3"
    target = _Target(on_call=lambda n: spies[0].lost.set())  # signal mid-call
    judge = _Judge()
    with pytest.raises(CampaignError, match="lost lease"):
        await _single(db, tmp_path, target, judge)

    assert len(target.prompts) == 1  # the second unit never starts
    assert judge.calls == 0  # the in-flight response is never judged
    first, second = _units(db, "c1")
    assert first["status"] == "in_flight"  # recoverable, no stale response saved
    assert first["progress"] is None
    assert second["status"] == "pending"
    assert _campaign_status(db, "c1") != "completed"


@pytest.mark.asyncio
async def test_no_judge_invocation_or_turn_after_lease_loss_multi(
    spies: list[CampaignStore], tmp_path: Path
) -> None:
    db = tmp_path / "c.sqlite3"
    target = _Target(on_call=lambda n: spies[0].lost.set())
    judge = _Judge()
    with pytest.raises(CampaignError, match="lost lease"):
        await _multi(db, tmp_path, target, judge)

    assert len(target.prompts) == 1  # no second turn
    assert judge.calls == 0
    (unit,) = _units(db, "m1")
    assert unit["status"] == "in_flight"
    assert unit["progress"] is None  # stale response not persisted


@pytest.mark.asyncio
async def test_stale_worker_cannot_complete_campaign(
    spies: list[CampaignStore], tmp_path: Path
) -> None:
    db = tmp_path / "c.sqlite3"
    judge = _Judge(on_call=lambda n: spies[0].lost.set())  # lost while judging
    with pytest.raises(CampaignError, match="lost lease"):
        await _single(db, tmp_path, _Target(), judge, count=1)

    store = CampaignStore(str(db))
    try:
        attempts = store.judge_attempts("c1")
    finally:
        store.close()
    assert [a["status"] for a in attempts] == ["running"]  # result not persisted
    assert _units(db, "c1")[0]["result"] is None
    assert _campaign_status(db, "c1") not in ("completed", "incomplete")

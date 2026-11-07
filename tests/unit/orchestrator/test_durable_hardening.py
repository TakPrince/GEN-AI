"""Task 16: lease renewal during long calls; target-response persistence."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import JudgeResult
from safety_platform.orchestrator.durable import (
    CampaignError,
    CampaignStore,
    LeaseHeldError,
    resume_campaign,
    start_single,
)

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))
TTL = 0.6  # short lease so tests exercise expiry in real time
ANSWER = "here is a harmless answer"


def _cases(count: int = 1) -> list[EvaluationCase]:
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
    """Blocking, like the synchronous model call. `during` runs mid-call."""

    def __init__(self, seconds: float = 0.0, during=None, crash_on: int | None = None):
        self.prompts: list[str] = []
        self._seconds = seconds
        self._during = during
        self._crash_on = crash_on

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._during is not None:
            self._during(self)
        if self._seconds:
            time.sleep(self._seconds)
        if self._crash_on is not None and len(self.prompts) == self._crash_on:
            raise SystemExit("simulated process death")
        return ANSWER


class _Judge:
    def __init__(self, crash_on: int | None = None, probe=None) -> None:
        self.calls = 0
        self._crash_on = crash_on
        self._probe = probe

    def judge(self, prompt: str, response: str) -> JudgeResult:
        self.calls += 1
        if self._probe is not None:
            self._probe(response)
        if self._crash_on is not None and self.calls == self._crash_on:
            raise SystemExit("simulated process death during judging")
        return JudgeResult("cannot help" not in response.lower(), "r", None)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    return tmp_path / "campaign.sqlite3", tmp_path


def _start(db, tmp, target, judge=None, *, cases=None):
    return start_single(
        str(db),
        app._pipeline(str(db)),
        cases or _cases(),
        judge or _Judge(),
        target,
        run_id="c1",
        artifact_dir=str(tmp),
        per_label=len(cases or _cases()),
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
        lease_s=TTL,
    )


def _resume(db, judge, target):
    return resume_campaign(
        str(db), app._pipeline(str(db)), judge, target, run_id="c1", lease_s=TTL
    )


def _unit(db: Path) -> dict:
    store = CampaignStore(str(db))
    try:
        return store.units("c1")[0]
    finally:
        store.close()


# --- Part A: lease renewal -------------------------------------------------


@pytest.mark.asyncio
async def test_long_request_keeps_lease_alive_and_blocks_takeover(env) -> None:
    db, tmp = env
    seen: dict = {}

    def during(_target: _Target) -> None:
        time.sleep(TTL * 2.5)  # the model call outlives the original TTL
        probe = CampaignStore(str(db), lease_s=TTL)
        try:
            seen["lease_valid"] = (
                probe.campaign("c1")["lease_until_ns"] > time.time_ns()
            )
            probe.acquire("c1", owner="intruder")
            seen["takeover"] = "allowed"
        except LeaseHeldError:
            seen["takeover"] = "refused"
        finally:
            probe.close()

    result = await _start(db, tmp, _Target(during=during))

    assert seen == {"lease_valid": True, "takeover": "refused"}
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_lease_lost_to_another_owner_stops_commit_and_new_work(env) -> None:
    db, tmp = env

    def steal(_target: _Target) -> None:
        # Another process took the campaign while this request was in flight.
        con = sqlite3.connect(db)
        con.execute(
            "UPDATE campaign SET owner = 'thief', lease_until_ns = ?",
            (time.time_ns() + 60_000_000_000,),
        )
        con.commit()
        con.close()

    target = _Target(during=steal)
    cases = _cases(2)
    with pytest.raises(CampaignError, match="lost lease"):
        await _start(db, tmp, target, cases=cases)

    assert len(target.prompts) == 1  # no new request after ownership loss
    unit = _unit(db)
    assert unit["status"] == "in_flight"  # stale result never committed
    assert unit["result"] is None


@pytest.mark.asyncio
async def test_lease_expired_during_long_call_cannot_commit_stale_result(
    env, caplog
) -> None:
    db, tmp = env
    con_holder: dict = {}

    def hold_database(_target: _Target) -> None:
        # Renewals cannot run while the database is locked past the TTL.
        holder = sqlite3.connect(db, isolation_level=None)
        holder.execute("BEGIN EXCLUSIVE")
        time.sleep(TTL * 2.5)
        holder.execute("COMMIT")
        holder.close()
        con_holder["released"] = True

    with pytest.raises(CampaignError, match="lost lease"):
        await _start(db, tmp, _Target(during=hold_database))

    assert con_holder == {"released": True}
    assert _unit(db)["result"] is None
    assert "lease renewal" in caplog.text  # the failure is reported, not silent


@pytest.mark.asyncio
async def test_heartbeat_thread_stops_after_failure(env) -> None:
    db, tmp = env

    def boom() -> None:
        raise SystemExit("interrupted")

    with pytest.raises(SystemExit):
        await _start(db, tmp, _Target(during=lambda _t: boom()))

    alive = [t.name for t in threading.enumerate() if t.name == "safety-lease-c1"]
    assert alive == []


@pytest.mark.asyncio
async def test_heartbeat_thread_stops_after_success(env) -> None:
    db, tmp = env
    await _start(db, tmp, _Target())
    assert [t.name for t in threading.enumerate() if t.name == "safety-lease-c1"] == []


# --- Part B: target response persisted before judging ------------------------


@pytest.mark.asyncio
async def test_target_response_is_persisted_before_judging(env) -> None:
    db, tmp = env
    seen: dict = {}

    def probe(response: str) -> None:
        unit = _unit(db)
        seen["progress"] = unit["progress"]
        seen["response"] = response

    await _start(db, tmp, _Target(), judge=_Judge(probe=probe))

    assert seen["progress"]["stage"] == "responded"
    assert seen["progress"]["record"]["final_response"] == seen["response"] == ANSWER


@pytest.mark.asyncio
async def test_crash_during_judging_resumes_without_resending_target(env) -> None:
    db, tmp = env
    target = _Target()
    with pytest.raises(SystemExit):
        await _start(db, tmp, target, judge=_Judge(crash_on=1))

    resumed_target = _Target()
    judge = _Judge()
    result = await _resume(db, judge, resumed_target)

    assert resumed_target.prompts == []  # the received response is reused
    assert judge.calls == 1  # judging resumed exactly once
    (record,) = result.records
    assert record.outcome != "uncertain"
    assert record.final_response == ANSWER
    assert record.judge_complied is True  # stub judges this harmless answer compliant
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_crash_before_response_persistence_stays_uncertain(env) -> None:
    db, tmp = env
    with pytest.raises(SystemExit):
        await _start(db, tmp, _Target(crash_on=1))

    resumed_target = _Target()
    result = await _resume(db, _Judge(), resumed_target)

    assert resumed_target.prompts == []  # in-flight request never resent
    (record,) = result.records
    assert record.outcome == "uncertain"
    assert record.final_response is None


@pytest.mark.asyncio
async def test_recovered_campaign_summary_matches_uninterrupted_run(env) -> None:
    db, tmp = env
    clean = await _start(db, tmp, _Target(), cases=_cases(2))

    crashed_dir = tmp / "crashed"
    crashed_dir.mkdir()
    crashed_db = crashed_dir / "campaign.sqlite3"
    with pytest.raises(SystemExit):
        await _start(
            crashed_db,
            crashed_dir,
            _Target(),
            judge=_Judge(crash_on=1),
            cases=_cases(2),
        )
    recovered = await _resume(crashed_db, _Judge(), _Target())

    def untimed(metrics: dict) -> dict:
        return {
            k: v
            for k, v in metrics.items()
            if "latency" not in k and not k.endswith("_ms")
        }

    assert untimed(recovered.summary["totals"]) == untimed(clean.summary["totals"])
    assert untimed(recovered.summary["metrics"]) == untimed(clean.summary["metrics"])
    assert recovered.summary["campaign"]["units"]["completed"] == 2

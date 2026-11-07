"""Task 15 slice 2: durable single-turn campaigns (red-first, real SQLite)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import JudgeResult
from safety_platform.orchestrator.durable import (
    CampaignStore,
    resume_campaign,
    start_single,
)
from safety_platform.orchestrator.loop import run_campaign

TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))


class _Target:
    """Records every target call. `crash_on` simulates a process death mid-call."""

    def __init__(self, crash_on: int | None = None) -> None:
        self.prompts: list[str] = []
        self._crash_on = crash_on

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._crash_on is not None and len(self.prompts) == self._crash_on:
            raise SystemExit("simulated process death during target call")
        return "here is a harmless answer"


class _Judge:
    def judge(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult("cannot help" not in response.lower(), "r", None)


def _cases() -> list[EvaluationCase]:
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
        for i in range(4)
    ]


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    db = tmp_path / "campaign.sqlite3"
    return db, tmp_path


def _start_kwargs(artifact_dir: Path, run_id: str = "c1") -> dict:
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "per_label": 4,
        "strategies": ("direct",),
        "seed": 0,
        "model_metadata": META,
        "reference_time": TS,
    }


def _expire_lease(db: Path, run_id: str) -> None:
    """A restart after a crash sees the lease lapse; simulate that instantly."""
    con = sqlite3.connect(db)
    con.execute("UPDATE campaign SET lease_until_ns = 0 WHERE run_id = ?", (run_id,))
    con.commit()
    con.close()


def _strip_timing(value):
    """Latency is measured wall-clock; everything else must match exactly."""
    if isinstance(value, dict):
        return {
            k: _strip_timing(v)
            for k, v in value.items()
            if "latency" not in k
            and not k.endswith("_ms")
            and k != "telemetry"
            and k != "campaign"
        }
    if isinstance(value, list | tuple):
        return [_strip_timing(v) for v in value]
    return value


@pytest.mark.asyncio
async def test_crash_mid_campaign_keeps_confirmed_work_and_never_resends_in_flight(
    env,
) -> None:
    db, tmp = env
    target = _Target(crash_on=3)
    with pytest.raises(SystemExit):
        await start_single(
            db,
            app._pipeline(str(db)),
            _cases(),
            _Judge(),
            target,
            **_start_kwargs(tmp),
        )
    _expire_lease(db, "c1")

    resumed = _Target()
    result = await resume_campaign(
        db, app._pipeline(str(db)), _Judge(), resumed, run_id="c1"
    )

    # Units 1-2 confirmed; unit 3 died in flight (never re-sent); unit 4 pending.
    crashed_prompt = target.prompts[2]
    assert crashed_prompt not in resumed.prompts
    assert len(resumed.prompts) == 1  # only the pending unit 4 is executed

    assert result.status == "incomplete"
    assert result.summary["campaign"]["units"] == {
        "total": 4,
        "completed": 3,
        "uncertain": 1,
        "pending": 0,
    }
    uncertain = [r for r in result.records if r.outcome == "uncertain"]
    assert len(uncertain) == 1
    # Never classified: no judge verdict, and treated as an execution failure.
    assert uncertain[0].judge_complied is None
    assert uncertain[0].execution_error == "uncertain_in_flight"


@pytest.mark.asyncio
async def test_resume_of_completed_campaign_is_a_noop(env) -> None:
    db, tmp = env
    first = await start_single(
        db,
        app._pipeline(str(db)),
        _cases(),
        _Judge(),
        _Target(),
        **_start_kwargs(tmp),
    )
    assert first.status == "completed"

    again_target = _Target()
    again = await resume_campaign(
        db, app._pipeline(str(db)), _Judge(), again_target, run_id="c1"
    )

    assert again_target.prompts == []
    assert again.status == "completed"
    assert _strip_timing(again.summary) == _strip_timing(first.summary)


@pytest.mark.asyncio
async def test_persisted_campaign_matches_uninterrupted_run_campaign(env) -> None:
    db, tmp = env
    durable = await start_single(
        db,
        app._pipeline(str(db)),
        _cases(),
        _Judge(),
        _Target(),
        **_start_kwargs(tmp),
    )
    plain_db = tmp / "plain.sqlite3"
    _, plain_summary = await run_campaign(
        app._pipeline(str(plain_db)),
        _cases(),
        _Judge(),
        _Target(),
        run_id="c1",
        per_label=4,
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )

    assert _strip_timing(durable.summary["totals"]) == _strip_timing(
        plain_summary["totals"]
    )
    assert _strip_timing(durable.summary["metrics"]) == _strip_timing(
        plain_summary["metrics"]
    )


@pytest.mark.asyncio
async def test_persistence_failure_rolls_back_and_leaves_unit_uncertain(env) -> None:
    db, tmp = env
    CampaignStore(str(db)).close()  # creates the ledger tables
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TRIGGER fail_unit BEFORE UPDATE ON campaign_unit "
        "WHEN NEW.status = 'completed' AND NEW.unit_id = 'benign-001' "
        "BEGIN SELECT RAISE(ABORT, 'disk full'); END"
    )
    con.commit()
    con.close()

    with pytest.raises(sqlite3.IntegrityError):
        await start_single(
            db,
            app._pipeline(str(db)),
            _cases(),
            _Judge(),
            _Target(),
            **_start_kwargs(tmp),
        )

    # The lease was released on the error exit, so no expiry is needed.
    # Task 16: the response was persisted before judging, so while the disk
    # still fails, resume re-judges the saved response and never resends it.
    with pytest.raises(sqlite3.IntegrityError):
        await resume_campaign(
            db, app._pipeline(str(db)), _Judge(), _Target(), run_id="c1"
        )

    con = sqlite3.connect(db)
    con.execute("DROP TRIGGER fail_unit")  # the disk recovers
    con.commit()
    con.close()
    target = _Target()
    result = await resume_campaign(
        db, app._pipeline(str(db)), _Judge(), target, run_id="c1"
    )
    unit = {r.case_id: r for r in result.records}["benign-001"]
    assert unit.outcome != "uncertain"  # resolved from the persisted response
    assert unit.final_response == "here is a harmless answer"
    assert len(target.prompts) == 2  # units 3 and 4 only


@pytest.mark.asyncio
async def test_completed_campaign_exports_compatible_artifacts(env) -> None:
    db, tmp = env
    result = await start_single(
        db,
        app._pipeline(str(db)),
        _cases(),
        _Judge(),
        _Target(),
        **_start_kwargs(tmp),
    )
    jsonl = tmp / "redteam-singleturn-c1.jsonl"
    summary_path = tmp / "redteam-singleturn-c1-summary.json"
    lines = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["run_id"] == "c1"
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["run_id"] == result.summary["run_id"] == "c1"
    assert {"totals", "metrics", "campaign"} <= set(written)
    # Exported rows are the persisted outcomes, in planned order.
    assert [json.loads(line)["case_id"] for line in lines] == [
        r.case_id for r in result.records
    ]

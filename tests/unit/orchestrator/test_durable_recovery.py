"""Task 15 slice 4: real process death (os._exit, no cleanup) and concurrent resume."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.llm import (
    ComplianceVerdict,
    ConversationVerdict,
    JudgeResult,
)
from safety_platform.orchestrator.durable import (
    CampaignStore,
    LeaseHeldError,
    resume_campaign,
)

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))

_CHILD = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from pathlib import Path
    from safety_platform import app
    from safety_platform.evaluation.dataset import EvaluationCase
    from safety_platform.evaluation.llm import (
        ComplianceVerdict,
        ConversationVerdict,
        JudgeResult,
    )
    from safety_platform.orchestrator.durable import start_multi, start_single

    db, out, log, mode = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))

    class Judge:
        def judge(self, prompt, response):
            return JudgeResult(False, "r", None)
        def judge_conversation(self, objective, transcript):
            return ConversationVerdict(ComplianceVerdict.REFUSED, "x", None)

    def target(prompt):
        # One JSON record per call: prompts may contain newlines.
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(prompt) + "\\n")
        with open(log, encoding="utf-8") as handle:
            calls = sum(1 for _ in handle)
        if calls == 3:
            os._exit(9)  # hard crash: no finally, no except, lease stays held
        return "I cannot help with that."

    cases = [
        EvaluationCase(f"attack-{i:03d}", f"describe approach {i}", "attack",
                       "malware", "behavior", "test", "src")
        for i in range(1, 5)
    ]
    if mode == "single":
        asyncio.run(start_single(
            db, app._pipeline(db), cases, Judge(), target, run_id="k1",
            artifact_dir=out, per_label=4, strategies=("direct",), seed=0,
            model_metadata=META, reference_time="{TS}"))
    else:
        asyncio.run(start_multi(
            db, app._pipeline(db), cases, Judge(), target, run_id="k1",
            artifact_dir=out, per_label=4, max_turns=3, escalations=("crescendo",),
            seed=0, model_metadata=META, reference_time="{TS}"))
    """
).replace("{TS}", TS)


def _run_child(
    tmp_path: Path, mode: str
) -> tuple[Path, Path, subprocess.CompletedProcess]:
    db = tmp_path / "campaign.sqlite3"
    log = tmp_path / "target-calls.log"
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    # `metrics/` lives at the repo root, which pytest adds in-process only.
    pythonpath = os.pathsep.join([str(SRC), str(ROOT)])
    env = {**os.environ, "PYTHONPATH": pythonpath, "SAFETY_UI_SPECULATIVE_ECHO": "0"}
    proc = subprocess.run(
        [sys.executable, str(script), str(db), str(tmp_path), str(log), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return db, log, proc


def _read_calls(log: Path) -> list[str]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


class _Recording:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "I cannot help with that."


class _Judge:
    def judge(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(False, "r", None)

    def judge_conversation(
        self, objective: str, transcript: str
    ) -> ConversationVerdict:
        return ConversationVerdict(ComplianceVerdict.REFUSED, "x", None)


def _expire(db: Path) -> None:
    con = sqlite3.connect(db)
    con.execute("UPDATE campaign SET lease_until_ns = 0")
    con.commit()
    con.close()


@pytest.fixture(autouse=True)
def _no_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")


@pytest.mark.asyncio
async def test_hard_kill_mid_campaign_resumes_without_resending(tmp_path: Path) -> None:
    db, log, proc = _run_child(tmp_path, "single")
    assert proc.returncode == 9, proc.stderr[-500:]

    sent = _read_calls(log)
    assert len(sent) == 3
    store = CampaignStore(str(db))
    units = {u["unit_id"]: u for u in store.units("k1")}
    store.close()
    statuses = [u["status"] for u in units.values()]
    assert statuses == ["completed", "completed", "in_flight", "pending"]
    in_flight_prompt = sent[2]

    with pytest.raises(LeaseHeldError):  # the dead process still holds the lease
        await resume_campaign(
            str(db), app._pipeline(str(db)), _Judge(), _Recording(), run_id="k1"
        )

    _expire(db)  # the lease lapses after its TTL
    target = _Recording()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(), target, run_id="k1"
    )

    assert in_flight_prompt not in target.prompts  # never re-sent
    assert len(target.prompts) == 1  # only the pending unit runs
    assert result.status == "incomplete"
    assert result.summary["campaign"]["units"] == {
        "total": 4,
        "completed": 3,
        "uncertain": 1,
        "pending": 0,
    }
    assert [r.outcome for r in result.records].count("uncertain") == 1


@pytest.mark.asyncio
async def test_hard_kill_mid_conversation_resumes_from_confirmed_turns(
    tmp_path: Path,
) -> None:
    db, log, proc = _run_child(tmp_path, "multi")
    assert proc.returncode == 9, proc.stderr[-500:]
    sent = _read_calls(log)
    assert len(sent) == 3  # turns 1-2 of conversation 1 confirmed; turn 3 died

    _expire(db)
    target = _Recording()
    result = await resume_campaign(
        str(db), app._pipeline(str(db)), _Judge(), target, run_id="k1"
    )

    assert sent[2] not in target.prompts  # in-flight turn never re-sent
    (uncertain,) = [r for r in result.records if r.status == "uncertain"]
    assert [t.turn_number for t in uncertain.turns] == [1, 2]
    assert result.status == "incomplete"
    assert result.summary["campaign"]["units"]["uncertain"] == 1
    # Conversations 2-4 were pending: they ran exactly once each.
    assert len(target.prompts) == 9

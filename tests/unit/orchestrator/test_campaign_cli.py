"""Task 15 slice 5: orchestrator CLI for durable campaigns (red-first)."""

from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import pytest

from safety_platform import app
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import JudgeResult
from safety_platform.orchestrator.durable import CampaignStore, start_single

orch = importlib.import_module("safety_platform.orchestrator.__main__")
TS = "2026-01-01T00:00:00+00:00"
META = (("target", "stub"), ("judge", "stub"), ("dataset", "test"))


class _Completion:
    def __init__(self, crash_on: int | None = None) -> None:
        self.prompts: list[str] = []
        self._crash_on = crash_on

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._crash_on is not None and len(self.prompts) == self._crash_on:
            raise SystemExit("simulated process death")
        return "here is a harmless answer"


class _Judge:
    def __init__(self, model: str = "stub") -> None:
        self.model = model

    def judge(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(False, "r", None)


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
        for i in range(1, 5)
    ]


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SAFETY_UI_SPECULATIVE_ECHO", "0")
    completion = _Completion()
    monkeypatch.setattr(orch, "_live_completion", completion)
    monkeypatch.setattr(orch, "ComplianceJudge", lambda model: _Judge(model))
    return tmp_path / "out"


async def _crash_campaign(out: Path, run_id: str = "c1") -> Path:
    """A campaign whose process died mid-run, leaving durable state in out/."""
    db = out / "orchestrator.sqlite3"
    out.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit):
        await start_single(
            str(db),
            app._pipeline(str(db)),
            _cases(),
            _Judge(),
            _Completion(crash_on=2),
            run_id=run_id,
            artifact_dir=str(out),
            per_label=4,
            strategies=("direct",),
            seed=0,
            model_metadata=META,
            reference_time=TS,
        )
    return db


def _expire(db: Path) -> None:
    con = sqlite3.connect(db)
    con.execute("UPDATE campaign SET lease_until_ns = 0")
    con.commit()
    con.close()


async def _main(*argv: str) -> int:
    return await orch.main(list(argv))


@pytest.mark.asyncio
async def test_fresh_run_announces_campaign_id_and_persists_to_out_dir(
    out: Path, capsys
) -> None:
    dataset = out.parent / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "case_id": c.case_id,
                    "prompt": c.prompt,
                    "label": c.label,
                    "category": c.category,
                    "behavior": c.behavior,
                    "split": c.split,
                    "source": c.source,
                }
            )
            for c in _cases()
        )
        + "\n",
        encoding="utf-8",
    )

    code = await _main(
        "--dataset",
        str(dataset),
        "--out",
        str(out),
        "--strategies",
        "direct",
        "--attack-per-label",
        "4",
    )

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    announced = [i for i, line in enumerate(lines) if line.startswith("campaign: ")]
    assert announced  # the id is printed when execution begins
    run_id = lines[announced[0]].split(": ", 1)[1].strip()
    assert lines.index(f"campaign: {run_id}") < next(
        i for i, line in enumerate(lines) if line.startswith("records:")
    )
    assert (out / "orchestrator.sqlite3").exists()  # durable by default
    assert (out / f"redteam-singleturn-{run_id}.jsonl").exists()
    assert (out / f"redteam-singleturn-{run_id}-summary.json").exists()


@pytest.mark.asyncio
async def test_status_reports_interrupted_campaign_without_executing(
    out: Path, capsys
) -> None:
    await _crash_campaign(out)
    _expire(out / "orchestrator.sqlite3")

    code = await _main("--campaign-status", "c1", "--out", str(out))

    text = capsys.readouterr().out
    assert code == 0
    assert "interrupted" in text
    assert "in_flight=1" in text
    assert "--resume c1" in text  # the operator is told how to continue


@pytest.mark.asyncio
async def test_status_unknown_campaign_is_usage_error(out: Path, capsys) -> None:
    await _crash_campaign(out)
    assert await _main("--campaign-status", "nope", "--out", str(out)) == 2


@pytest.mark.asyncio
async def test_status_never_creates_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "orchestrator.sqlite3"
    code = await _main("--campaign-status", "c1", "--db", str(missing))
    assert code == 2
    assert not missing.exists()


@pytest.mark.asyncio
async def test_resume_is_refused_while_another_live_process_holds_lease(
    out: Path,
) -> None:
    db = await _crash_campaign(out)
    store = CampaignStore(str(db))
    store.acquire("c1", owner="other-live-process")  # fresh lease, not expired
    store.close()

    assert await _main("--resume", "c1", "--out", str(out)) == 3


@pytest.mark.asyncio
async def test_resume_interrupted_campaign_is_idempotent_and_reports_incomplete(
    out: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    await _crash_campaign(out)
    _expire(out / "orchestrator.sqlite3")
    completion = _Completion()
    monkeypatch.setattr(orch, "_live_completion", completion)

    first = await _main("--resume", "c1", "--out", str(out))
    calls_after_first = len(completion.prompts)
    second = await _main("--resume", "c1", "--out", str(out))

    text = capsys.readouterr().out
    assert first == 1  # one unit is uncertain: honest, not "completed"
    assert second == 1
    assert len(completion.prompts) == calls_after_first  # nothing re-executed
    assert "incomplete" in text
    assert "uncertain=1" in text


@pytest.mark.asyncio
async def test_resume_of_completed_campaign_exits_zero(out: Path) -> None:
    db = out / "orchestrator.sqlite3"
    out.mkdir(parents=True, exist_ok=True)
    await start_single(
        str(db),
        app._pipeline(str(db)),
        _cases(),
        _Judge(),
        _Completion(),
        run_id="c1",
        artifact_dir=str(out),
        per_label=4,
        strategies=("direct",),
        seed=0,
        model_metadata=META,
        reference_time=TS,
    )
    assert await _main("--resume", "c1", "--out", str(out)) == 0

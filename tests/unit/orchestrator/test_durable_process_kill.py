"""Task 18 Part B: real process termination during judging, then fresh recovery.

Each scenario runs the campaign in a subprocess (durable_worker.py), waits on a
marker the worker writes at a known point, SIGKILLs it, and recovers in a new
process. Real SQLite, real processes; the only waits are marker polls and a
lease that lapses after a short injected TTL.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from safety_platform.orchestrator.durable import CampaignStore

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("durable_worker.py")
LEASE_S = "0.5"  # a dead worker's lease lapses after this; no 15-minute waits


def _env(raw: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    env["SAFETY_UI_SPECULATIVE_ECHO"] = "0"
    env["DURABLE_TEST_LEASE_S"] = LEASE_S
    env["DURABLE_TEST_RAW"] = "1" if raw else "0"
    return env


def _spawn(
    mode, kind, tmp: Path, run_id: str, fault: str, *, raw: bool = False
) -> subprocess.Popen:
    logs = tmp / "logs"
    logs.mkdir(exist_ok=True)
    # The child process owns this handle for its lifetime.
    stderr = open(logs / f"{mode}-stderr.txt", "w", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(
        [
            sys.executable,
            str(WORKER),
            mode,
            kind,
            str(tmp / "c.sqlite3"),
            str(tmp),
            run_id,
            str(logs),
            fault,
        ],
        env=_env(raw),
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
    )


def _await_marker(tmp: Path, marker: str, proc: subprocess.Popen) -> None:
    path = tmp / "logs" / marker
    deadline = time.monotonic() + 60
    while not path.exists():
        if proc.poll() is not None:
            raise AssertionError(f"worker exited before {marker}")
        if time.monotonic() > deadline:
            raise AssertionError(f"worker never reached {marker}")
        time.sleep(0.02)


def _kill(proc: subprocess.Popen) -> None:
    os.kill(proc.pid, signal.SIGKILL)  # real termination, no cleanup code runs
    proc.wait(timeout=30)
    assert proc.returncode == -signal.SIGKILL


def _wait_lease_lapsed(tmp: Path, run_id: str) -> None:
    store = CampaignStore(str(tmp / "c.sqlite3"))
    try:
        deadline = time.monotonic() + 30
        while time.time_ns() <= store.campaign(run_id)["lease_until_ns"]:
            if time.monotonic() > deadline:
                raise AssertionError("lease never lapsed")
            time.sleep(0.05)
    finally:
        store.close()


def _recover(tmp: Path, kind: str, run_id: str, *, raw: bool = False) -> None:
    proc = _spawn("resume", kind, tmp, run_id, "none", raw=raw)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0, (tmp / "logs" / "resume-stderr.txt").read_text()
    assert '"status": "completed"' in out


def _lines(tmp: Path, name: str) -> list[str]:
    path = tmp / "logs" / name
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _attempts(tmp: Path, run_id: str, stage: str | None = None) -> list[dict]:
    store = CampaignStore(str(tmp / "c.sqlite3"))
    try:
        rows = store.judge_attempts(run_id)
    finally:
        store.close()
    return [r for r in rows if stage is None or r["stage"] == stage]


def _assert_results_match_selected_attempts(tmp: Path, run_id: str, kind: str) -> None:
    """Part C: the summary's selected attempts are completed and are the verdicts
    the artifact records report."""
    prefix = "redteam-multiturn" if kind == "multi" else "redteam-singleturn"
    summary = json.loads((tmp / f"{prefix}-{run_id}-summary.json").read_text())
    selected = summary["campaign"]["judge_attempts"]["selected"]
    attempts = {a["attempt_id"]: a for a in _attempts(tmp, run_id)}
    assert selected and all(attempts[i]["status"] == "completed" for i in selected)
    (record,) = [
        json.loads(line)
        for line in (tmp / f"{prefix}-{run_id}.jsonl").read_text().splitlines()
    ]
    by_stage = {attempts[i]["stage"]: attempts[i]["result"] for i in selected}
    if kind == "multi":
        (turn,) = record["turns"]
        assert turn["judge_complied"] == by_stage["turn:1"]["complied"]
        assert record["conversation_verdict"] == by_stage["conversation"]["verdict"]
    else:
        assert record["judge_complied"] == by_stage["final"]["complied"]
        if record["raw_judge_complied"] is not None:
            assert record["raw_judge_complied"] == by_stage["raw"]["complied"]


# --- Multi-turn ---------------------------------------------------------------


def test_multi_killed_during_turn_judging_recovers_without_resending_target(
    tmp_path: Path,
) -> None:
    start = _spawn("start", "multi", tmp_path, "m1", "judge_hang")
    try:
        _await_marker(tmp_path, "judge_started", start)
        # The running judge attempt and the persisted response are committed.
        (running,) = _attempts(tmp_path, "m1", stage="turn:1")
        assert running["status"] == "running"
        assert len(_lines(tmp_path, "target.log")) == 1
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "m1")
    _recover(tmp_path, "multi", "m1")

    assert len(_lines(tmp_path, "target.log")) == 1  # target never called again
    assert [a["status"] for a in _attempts(tmp_path, "m1", "turn:1")] == [
        "interrupted",
        "completed",
    ]
    assert [a["status"] for a in _attempts(tmp_path, "m1", "conversation")] == [
        "completed"
    ]
    assert _lines(tmp_path, "judge.log") == [
        "turn",  # the killed attempt
        "turn",  # the recovered turn judgment
        "conversation",  # the conversation verdict
    ]
    _assert_results_match_selected_attempts(tmp_path, "m1", "multi")


def test_multi_judge_completed_before_checkpoint_reuses_verdict(tmp_path: Path) -> None:
    start = _spawn("start", "multi", tmp_path, "m1", "checkpoint_hang")
    try:
        _await_marker(tmp_path, "checkpoint_started", start)
        # The turn verdict is recorded, but the unit checkpoint never landed.
        (done,) = _attempts(tmp_path, "m1", "turn:1")
        assert done["status"] == "completed"
        assert len(_lines(tmp_path, "target.log")) == 1
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "m1")
    _recover(tmp_path, "multi", "m1")

    assert len(_lines(tmp_path, "target.log")) == 1  # no target repeat
    assert [a["status"] for a in _attempts(tmp_path, "m1", "turn:1")] == ["completed"]
    assert _lines(tmp_path, "judge.log") == ["turn", "conversation"]  # no turn rejudge
    _assert_results_match_selected_attempts(tmp_path, "m1", "multi")


# --- Single-turn --------------------------------------------------------------


def test_single_killed_during_judging_recovers_without_resending_target(
    tmp_path: Path,
) -> None:
    start = _spawn("start", "single", tmp_path, "s1", "judge_hang")
    try:
        _await_marker(tmp_path, "judge_started", start)
        (running,) = _attempts(tmp_path, "s1", stage="final")
        assert running["status"] == "running"
        assert len(_lines(tmp_path, "target.log")) == 1
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "s1")
    _recover(tmp_path, "single", "s1")

    assert len(_lines(tmp_path, "target.log")) == 1  # persisted response reused
    assert [a["status"] for a in _attempts(tmp_path, "s1", "final")] == [
        "interrupted",
        "completed",
    ]
    assert _lines(tmp_path, "judge.log") == ["turn", "turn"]
    _assert_results_match_selected_attempts(tmp_path, "s1", "single")


def test_single_judge_completed_before_unit_checkpoint_reuses_verdict(
    tmp_path: Path,
) -> None:
    start = _spawn("start", "single", tmp_path, "s1", "checkpoint_hang")
    try:
        _await_marker(tmp_path, "checkpoint_started", start)
        (done,) = _attempts(tmp_path, "s1", "final")
        assert done["status"] == "completed"
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "s1")
    _recover(tmp_path, "single", "s1")

    assert len(_lines(tmp_path, "target.log")) == 1  # no target repeat
    assert [a["status"] for a in _attempts(tmp_path, "s1", "final")] == ["completed"]
    assert _lines(tmp_path, "judge.log") == ["turn"]  # the completed verdict is reused
    _assert_results_match_selected_attempts(tmp_path, "s1", "single")


# --- Single-turn raw judge stage ---------------------------------------------


def test_single_killed_during_raw_judging_recovers_without_resending_target(
    tmp_path: Path,
) -> None:
    start = _spawn("start", "single", tmp_path, "s1", "raw_judge_hang", raw=True)
    try:
        _await_marker(tmp_path, "judge_started", start)
        # The final verdict is recorded; the raw attempt is running and committed.
        assert [a["status"] for a in _attempts(tmp_path, "s1", "final")] == [
            "completed"
        ]
        assert [a["status"] for a in _attempts(tmp_path, "s1", "raw")] == ["running"]
        assert len(_lines(tmp_path, "target.log")) == 1
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "s1")
    _recover(tmp_path, "single", "s1", raw=True)

    assert len(_lines(tmp_path, "target.log")) == 1  # persisted response reused
    assert [a["status"] for a in _attempts(tmp_path, "s1", "final")] == ["completed"]
    assert [a["status"] for a in _attempts(tmp_path, "s1", "raw")] == [
        "interrupted",
        "completed",
    ]
    assert _lines(tmp_path, "judge.log") == ["turn", "raw", "raw"]  # no final rejudge
    _assert_results_match_selected_attempts(tmp_path, "s1", "single")


def test_single_raw_completed_before_unit_checkpoint_reuses_verdict(
    tmp_path: Path,
) -> None:
    start = _spawn("start", "single", tmp_path, "s1", "checkpoint_hang", raw=True)
    try:
        _await_marker(tmp_path, "checkpoint_started", start)
        # Both verdicts are recorded; the unit checkpoint never landed.
        assert [a["status"] for a in _attempts(tmp_path, "s1", "raw")] == ["completed"]
        assert len(_lines(tmp_path, "target.log")) == 1
        _kill(start)
    finally:
        if start.poll() is None:
            start.kill()

    _wait_lease_lapsed(tmp_path, "s1")
    _recover(tmp_path, "single", "s1", raw=True)

    assert len(_lines(tmp_path, "target.log")) == 1  # no target repeat
    assert [a["status"] for a in _attempts(tmp_path, "s1", "raw")] == ["completed"]
    assert [a["status"] for a in _attempts(tmp_path, "s1", "final")] == ["completed"]
    assert _lines(tmp_path, "judge.log") == ["turn", "raw"]  # nothing rejudged
    _assert_results_match_selected_attempts(tmp_path, "s1", "single")

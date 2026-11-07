"""Task 14: replay CLI and interrupted-run disclosure (red-first)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from safety_platform.config.registry import (
    Direction,
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryLane,
)
from safety_platform.evaluation.__main__ import main
from safety_platform.evaluation.report import load_report
from safety_platform.telemetry.db_sink import _row
from safety_platform.telemetry.events import StageEvent


def _ev(event_id: str) -> StageEvent:
    return StageEvent(
        event_id=event_id,
        context=ExecutionContext(
            trace_id="t1", turn_id="1", conversation_id="c1", direction=Direction.INPUT
        ),
        stage=Stage.ROUTER,
        verdict=SafetyVerdict.BLOCK,
        latency_ms=1.0,
        prompt_hash="deadbeef",
        timestamp_ns=1,
        lane=TelemetryLane.DECISION,
    )


def _dead_letter_file(tmp_path: Path, events, *, extra_line: str = "") -> Path:
    path = tmp_path / "dead.jsonl"
    payload = {
        "failed_at": "2026-01-01T00:00:00+0000",
        "error": "OperationalError",
        "event_ids": [e.event_id for e in events],
        "rows": [_row(e) for e in events],
    }
    path.write_text(extra_line + json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_cli_replay_success_inserts_rows_and_returns_zero(
    tmp_path: Path, capsys
) -> None:
    db = tmp_path / "results.sqlite3"
    path = _dead_letter_file(tmp_path, (_ev("a1"), _ev("a2")))

    main(["replay", str(path), "--db", str(db)])

    out = capsys.readouterr().out
    assert "recovered=2" in out
    assert "duplicated=0"
    con = sqlite3.connect(db)
    ids = sorted(r[0] for r in con.execute("SELECT event_id FROM stage_event"))
    con.close()
    assert ids == ["a1", "a2"]


def test_cli_replay_missing_file_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "replay",
                str(tmp_path / "missing.jsonl"),
                "--db",
                str(tmp_path / "results.sqlite3"),
            ]
        )
    assert exc.value.code != 0


def test_cli_replay_partial_recovery_exits_one(tmp_path: Path, capsys) -> None:
    db = tmp_path / "results.sqlite3"
    # First line malformed JSON, second line a valid record.
    path = _dead_letter_file(tmp_path, (_ev("a1"),), extra_line="{oops}\n")

    with pytest.raises(SystemExit) as exc:
        main(["replay", str(path), "--db", str(db)])

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "malformed=1" in out


def test_cli_replay_repeat_is_idempotent_and_zero_exit(tmp_path: Path, capsys) -> None:
    db = tmp_path / "results.sqlite3"
    path = _dead_letter_file(tmp_path, (_ev("a1"),))
    raw = path.read_text(encoding="utf-8")

    main(["replay", str(path), "--db", str(db)])
    assert "recovered=1" in capsys.readouterr().out

    main(["replay", str(path), "--db", str(db)])
    out = capsys.readouterr().out
    assert "recovered=0" in out
    assert "duplicated=1" in out
    assert path.read_text(encoding="utf-8") == raw
    assert "malformed=0" in out


def _run_database(tmp_path: Path, *, completed_ns: int | None) -> Path:
    path = tmp_path / "report.sqlite3"
    con = sqlite3.connect(path)
    con.executescript(
        """
CREATE TABLE run (
    run_id TEXT PRIMARY KEY, started_ns INTEGER NOT NULL,
    completed_ns INTEGER, target_model TEXT, embed_model TEXT,
    tier3_model TEXT, judge_model TEXT, dataset_path TEXT,
    dataset_digest TEXT, config_json TEXT NOT NULL);
CREATE TABLE run_result (
    run_id TEXT, case_id TEXT, system TEXT, label TEXT, category TEXT,
    final_route TEXT, input_route TEXT, output_route TEXT, blocked INTEGER,
    guard_ms REAL, generation_ms REAL, tier3_invoked INTEGER,
    judge_complied INTEGER, judge_error TEXT, raw_response TEXT,
    final_response TEXT, PRIMARY KEY (run_id, case_id, system));
"""
    )
    con.execute(
        "INSERT INTO run VALUES ('run-1', 1, ?, 't', 'e', 'g', 'j', 'c', 'd', '{}')",
        (completed_ns,),
    )
    con.execute(
        "INSERT INTO run_result VALUES ('run-1', 'attack-1', 'baseline',"
        " 'attack', 'J', '', '', '', 0, 0.0, 0, 0, 1, NULL, '', '')"
    )
    con.commit()
    con.close()
    return path


def test_interrupted_run_report_marks_uncompleted(tmp_path: Path) -> None:
    db = _run_database(tmp_path, completed_ns=None)
    report = load_report(str(db))
    assert report.completed is False


def test_completed_run_report_stays_compatible(tmp_path: Path) -> None:
    db = _run_database(tmp_path, completed_ns=2)
    report = load_report(str(db))
    assert report.completed is True


def test_interrupted_report_metrics_come_only_from_persisted_rows(
    tmp_path: Path,
) -> None:
    db = _run_database(tmp_path, completed_ns=None)
    report = load_report(str(db))
    baseline = report.systems[0]
    # One persisted attack row judged compliant exactly once -> ASR 100%
    # computed from durable rows, nothing estimated.
    assert baseline.attack_cases == 1
    assert baseline.attack_success == 1

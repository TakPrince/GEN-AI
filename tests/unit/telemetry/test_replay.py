"""Task 13: dead-letter replay — recovery, idempotency, no silent loss."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import pytest

from safety_platform.config.registry import (
    Direction,
    ExecutionContext,
    SafetyVerdict,
    Stage,
    TelemetryLane,
)
from safety_platform.telemetry.db_sink import SqliteTelemetrySink, _row
from safety_platform.telemetry.events import StageEvent
from safety_platform.telemetry.replay import replay_dead_letter


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


def _dead_letter(tmp_path, events, *, error="OperationalError") -> Path:
    path = tmp_path / "dead.jsonl"
    payload = {
        "failed_at": "2026-01-01T00:00:00+0000",
        "error": error,
        "event_ids": [e.event_id for e in events],
        "rows": [_row(e) for e in events],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_replay_inserts_all_events(tmp_path) -> None:
    ev1, ev2 = _ev("a1"), _ev("a2")
    path = _dead_letter(tmp_path, (ev1, ev2))
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")

    report = replay_dead_letter(sink, path)

    assert (report.events_recovered, report.events_duplicated) == (2, 0)
    assert report.events_failed == 0
    file = tmp_path / "db.sqlite3"
    import sqlite3 as sq

    con = sq.connect(file)
    ids = [r[0] for r in con.execute("SELECT event_id FROM stage_event")]
    con.close()
    assert ids == ["a1", "a2"]


@pytest.mark.asyncio
async def test_replay_again_is_idempotent(tmp_path) -> None:
    ev1, ev2 = _ev("a1"), _ev("a2")
    path = _dead_letter(tmp_path, (ev1, ev2))
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")

    first = replay_dead_letter(sink, path)
    second = replay_dead_letter(sink, path)

    assert (first.events_recovered, first.events_duplicated) == (2, 0)
    assert (second.events_recovered, second.events_duplicated) == (0, 2)

    import sqlite3 as sq

    con = sq.connect(tmp_path / "db.sqlite3")
    assert con.execute("SELECT count(*) FROM stage_event").fetchone()[0] == 2
    con.close()


@pytest.mark.asyncio
async def test_malformed_line_is_reported_and_skipped(tmp_path) -> None:
    ev = _ev("a1")
    path = _dead_letter(tmp_path, (ev,))
    raw = path.read_text(encoding="utf-8")
    path.write_text("{not json}\n" + raw, encoding="utf-8")
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")

    report = replay_dead_letter(sink, path)

    assert report.events_recovered == 1
    assert report.malformed_records == 1
    assert report.failures[0].line_no == 1
    assert "{not json}" in report.failures[0].detail


@pytest.mark.asyncio
async def test_invalid_schema_record_is_reported(tmp_path) -> None:
    # rows missing fields / unknown enum value must not silently vanish
    path = tmp_path / "invalid.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "failed_at": "t",
                    "error": "e",
                    "event_ids": ["b1"],
                    "rows": [["b1", "t1", "1", "c1", "input", "unversioned", "router"]],
                }
            )
            + "\n"
        )
    ev = _ev("a1")
    first = _dead_letter(tmp_path, (ev,))
    combined = tmp_path / "combined.jsonl"
    combined.write_text(
        path.read_text(encoding="utf-8") + first.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")

    report = replay_dead_letter(sink, combined)

    assert report.events_recovered == 1  # only the schema-valid row batch
    assert report.events_failed == 0  # not even attempted
    assert report.malformed_records == 0
    assert report.invalid_records == 1
    assert report.failures
    assert report.failures[0].line_no == 1
    assert "arity" in report.failures[0].detail
    # no rows inserted for the invalid line
    import sqlite3 as sq

    con = sq.connect(tmp_path / "db.sqlite3")
    assert con.execute("SELECT count(*) FROM stage_event").fetchone()[0] == 1
    con.close()


@pytest.mark.asyncio
async def test_partial_db_failure_reports_and_continues(tmp_path) -> None:
    ev1, ev2, ev3 = _ev("a1"), _ev("a2"), _ev("a3")
    path = tmp_path / "dead.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "failed_at": "2026-01-01T00:00:00+0000",
                    "error": "OperationalError",
                    "event_ids": [ev.event_id],
                    "rows": [_row(ev)],
                }
            )
            for ev in (ev1, ev2, ev3)
        )
        + "\n",
        encoding="utf-8",
    )

    store_calls = {"n": 0}

    class FlakySink(SqliteTelemetrySink):
        def append_many(self, events):  # type: ignore[override]
            store_calls["n"] += 1
            if store_calls["n"] == 2:
                raise sqlite3.OperationalError("disk I/O error")
            return super().append_many(events)

    sink = FlakySink(tmp_path / "db.sqlite3")
    report = replay_dead_letter(sink, path)

    assert report.events_recovered == 2  # a1, a3
    assert report.events_failed == 1  # a2, with enough detail to recover
    failed = report.failures[0]
    assert failed.error == "OperationalError"
    assert "a2" in failed.detail and "disk I/O error" in failed.detail


@pytest.mark.asyncio
async def test_original_dead_letter_file_is_preserved(tmp_path) -> None:
    ev = _ev("a1")
    path = _dead_letter(tmp_path, (ev,))
    raw = path.read_text(encoding="utf-8")
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")

    replay_dead_letter(sink, path)

    assert path.read_text(encoding="utf-8") == raw


def test_replay_is_synchronous_and_removes_nothing(tmp_path) -> None:
    ev = _ev("a1")
    path = _dead_letter(tmp_path, (ev,))
    sink = SqliteTelemetrySink(tmp_path / "db.sqlite3")
    report = replay_dead_letter(sink, path)
    assert asdict(report)["events_recovered"] == 1
    assert path.exists()

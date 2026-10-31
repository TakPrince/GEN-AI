import sqlite3
from pathlib import Path

import pytest

from safety_platform.evaluation.report import (
    _percentile,
    _ratio,
    _wilson,
    load_report,
    render_markdown,
)

SCHEMA = """
CREATE TABLE run (
    run_id TEXT PRIMARY KEY,
    started_ns INTEGER NOT NULL,
    completed_ns INTEGER,
    target_model TEXT NOT NULL,
    embed_model TEXT NOT NULL,
    tier3_model TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    dataset_digest TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE run_result (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    system TEXT NOT NULL,
    label TEXT NOT NULL,
    category TEXT NOT NULL,
    final_route TEXT NOT NULL,
    input_route TEXT NOT NULL,
    output_route TEXT NOT NULL,
    blocked INTEGER NOT NULL,
    guard_ms REAL NOT NULL,
    generation_ms REAL NOT NULL,
    tier3_invoked INTEGER NOT NULL,
    judge_complied INTEGER NOT NULL,
    judge_error TEXT,
    raw_response TEXT NOT NULL,
    final_response TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id, system)
);
"""


def database(tmp_path: Path) -> Path:
    path = tmp_path / "report.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.execute(
        """
        INSERT INTO run VALUES (
            'run-1', 1, 2, 'target', 'embed', 'guard', 'judge',
            'cases.jsonl', 'digest', '{}'
        )
        """
    )
    rows = [
        ("attack-1", "baseline", "attack", "Jailbreak", 0, 0.0, 0, 1, None),
        ("attack-2", "baseline", "attack", "Jailbreak", 0, 0.0, 0, 0, "error"),
        ("attack-1", "guarded", "attack", "Jailbreak", 0, 1.0, 1, 0, None),
        ("attack-2", "guarded", "attack", "Jailbreak", 1, 3.0, 0, 0, None),
        ("benign-1", "baseline", "benign", "Benign", 0, 0.0, 0, 0, None),
        ("benign-2", "baseline", "benign", "Benign", 0, 0.0, 0, 0, None),
        ("benign-1", "guarded", "benign", "Benign", 0, 5.0, 0, 0, None),
        ("benign-2", "guarded", "benign", "Benign", 1, 3.0, 0, 0, None),
    ]
    connection.executemany(
        """
        INSERT INTO run_result (
            run_id, case_id, system, label, category, final_route,
            input_route, output_route, blocked, guard_ms, generation_ms,
            tier3_invoked, judge_complied, judge_error, raw_response,
            final_response
        ) VALUES ('run-1', ?, ?, ?, ?, '', '', '', ?, ?, 0, ?, ?, ?, '', '')
        """,
        rows,
    )
    connection.commit()
    connection.close()
    return path


def test_load_report_aggregates_asr_fpr_and_latency(tmp_path: Path) -> None:
    report = load_report(str(database(tmp_path)))
    metrics = {item.system: item for item in report.systems}

    assert report.run_id == "run-1"
    assert report.target_model == "target"
    assert metrics["baseline"].asr_pct == pytest.approx(50.0)
    assert metrics["baseline"].false_positives == 0
    assert metrics["baseline"].judge_errors == 1
    assert metrics["guarded"].asr_pct == pytest.approx(0.0)
    assert metrics["guarded"].fpr_pct == pytest.approx(50.0)
    assert metrics["guarded"].p50_guard_ms == pytest.approx(3.0)
    assert metrics["guarded"].p95_guard_ms == pytest.approx(5.0)
    assert metrics["guarded"].tier3_rate_pct == pytest.approx(25.0)
    assert metrics["guarded"].asr_low_pct == pytest.approx(0.0)
    assert metrics["guarded"].asr_high_pct == pytest.approx(65.76, abs=0.01)

    categories = {(item.system, item.category): item for item in report.categories}
    assert categories[("baseline", "Jailbreak")].attack_success == 1
    assert categories[("guarded", "Jailbreak")].asr_pct == pytest.approx(0.0)

    rendered = render_markdown(report)
    assert "# Evaluation `run-1`" in rendered
    assert "Judge err" in rendered
    assert "| guarded |" in rendered


def test_load_report_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown run_id"):
        load_report(str(database(tmp_path)), run_id="missing")

    empty = tmp_path / "empty.sqlite3"
    connection = sqlite3.connect(empty)
    connection.executescript(SCHEMA)
    connection.close()
    with pytest.raises(ValueError, match="contains no completed runs"):
        load_report(str(empty))


def test_load_report_prefers_completed_run_over_aborted(tmp_path: Path) -> None:
    path = database(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            INSERT INTO run VALUES (
                'run-2', 9, NULL, 'target', 'embed', 'guard', 'judge',
                'cases.jsonl', 'digest', '{}'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()

    report = load_report(str(path))
    assert report.run_id == "run-1"


def test_load_report_rejects_database_without_completed_runs(
    tmp_path: Path,
) -> None:
    path = database(tmp_path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE run SET completed_ns = NULL")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="contains no completed runs"):
        load_report(str(path))


def test_statistics_handle_empty_inputs() -> None:
    assert _ratio(1, 0) == 0.0
    assert _wilson(0, 0) == (0.0, 0.0)
    assert _percentile([], 0.95) == 0.0
    assert _percentile([3.0, 1.0, 2.0], 0.5) == 2.0

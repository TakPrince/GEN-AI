import sqlite3
from pathlib import Path

import pytest

from safety_platform.evaluation.cascade import CascadeOutcome
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import JudgeResult
from safety_platform.evaluation.runner import (
    _SCHEMA,
    RunConfig,
    _guarded_judge,
    _is_blocked,
    _load_embedding_cache,
    _load_reusable_baseline,
    _result,
    _save_embedding_cache,
    _select_test,
)


def case(
    case_id: str,
    label: str,
    split: str = "test",
) -> EvaluationCase:
    return EvaluationCase(
        case_id,
        "prompt",
        label,
        "category",
        "behavior",
        split,
        "source",
    )


def outcome(route: str) -> CascadeOutcome:
    return CascadeOutcome(route, "verdict", "stage", 1.0, False, 0.0)


def test_select_test_caps_each_label_and_skips_training() -> None:
    cases = (
        case("a0", "attack", "train"),
        case("a1", "attack"),
        case("a2", "attack"),
        case("b1", "benign"),
        case("b2", "benign"),
    )

    assert [item.case_id for item in _select_test(cases, 1)] == ["a1", "b1"]
    assert [item.case_id for item in _select_test(cases, None)] == [
        "a1",
        "a2",
        "b1",
        "b2",
    ]


def test_blocked_routes_and_guarded_judgment() -> None:
    assert _is_blocked(outcome("refused"))
    assert _is_blocked(outcome("error"))
    assert not _is_blocked(outcome("safe"))

    blocked = _guarded_judge(JudgeResult(True, "ok"), True)
    assert blocked == JudgeResult(False, "guardrail_blocked")
    assert _guarded_judge(JudgeResult(True, "ok"), False) == JudgeResult(True, "ok")


def test_result_row_serializes_verdict() -> None:
    row = _result(
        "run",
        case("a1", "attack"),
        "input",
        final_route="refused",
        input_route="refused",
        output_route="",
        blocked=True,
        guard_ms=1.5,
        generation_ms=2.5,
        tier3_invoked=True,
        judged=JudgeResult(False, "guardrail_blocked"),
        raw="raw",
        final="final",
    )

    assert row == (
        "run",
        "a1",
        "input",
        "attack",
        "category",
        "refused",
        "refused",
        "",
        1,
        1.5,
        2.5,
        1,
        0,
        None,
        "raw",
        "final",
    )


def test_run_config_defaults() -> None:
    config = RunConfig(Path("cases.jsonl"), Path("results.sqlite3"))

    assert config.reuse_run_id is None
    assert config.target_model == "llama3.2:latest"
    assert config.tier3_model == "qwen3:8b"
    assert config.judge_model == "qwen3:8b"
    assert config.per_label == 20


def _reuse_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    connection.execute(
        "INSERT INTO run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "old-run",
            1,
            2,
            "target",
            "embed",
            "guard",
            "judge",
            "cases.jsonl",
            "digest",
            "{}",
        ),
    )
    connection.executemany(
        """
        INSERT INTO run_result VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            (
                "old-run",
                "a1",
                "baseline",
                "attack",
                "category",
                "baseline",
                "baseline",
                "",
                0,
                0.0,
                10.0,
                0,
                1,
                None,
                "raw-a1",
                "raw-a1",
            ),
            (
                "old-run",
                "b1",
                "baseline",
                "benign",
                "category",
                "baseline",
                "baseline",
                "",
                0,
                0.0,
                20.0,
                0,
                0,
                "judge_error",
                "raw-b1",
                "raw-b1",
            ),
        ),
    )
    connection.commit()
    return connection


def test_reuse_baseline_loads_matching_rows() -> None:
    connection = _reuse_database()
    try:
        cases = (case("a1", "attack"), case("b1", "benign"), case("c1", "attack"))
        reused = _load_reusable_baseline(
            connection,
            run_id="old-run",
            test_cases=cases,
            target_model="target",
            judge_model="judge",
            digest="digest",
        )
    finally:
        connection.close()

    assert set(reused) == {"a1", "b1"}
    attack = reused["a1"]
    assert attack[0].case == cases[0]
    assert attack[0].raw == "raw-a1"
    assert attack[0].generation_ms == 10.0
    assert attack[1] == JudgeResult(True, "")
    benign = reused["b1"]
    assert benign[0].generation_ms == 20.0
    assert benign[1] == JudgeResult(False, "", "judge_error")


@pytest.mark.parametrize(
    ("target_model", "judge_model", "digest", "message"),
    (
        ("other", "judge", "digest", "target model"),
        ("target", "other", "digest", "judge model"),
        ("target", "judge", "other", "dataset digest"),
    ),
)
def test_reuse_baseline_rejects_mismatch(
    target_model: str,
    judge_model: str,
    digest: str,
    message: str,
) -> None:
    connection = _reuse_database()
    try:
        with pytest.raises(ValueError, match=message):
            _load_reusable_baseline(
                connection,
                run_id="old-run",
                test_cases=(case("a1", "attack"),),
                target_model=target_model,
                judge_model=judge_model,
                digest=digest,
            )
    finally:
        connection.close()


def test_reuse_baseline_rejects_unknown_run() -> None:
    connection = _reuse_database()
    try:
        with pytest.raises(ValueError, match="unknown reuse_run_id"):
            _load_reusable_baseline(
                connection,
                run_id="missing",
                test_cases=(case("a1", "attack"),),
                target_model="target",
                judge_model="judge",
                digest="digest",
            )
    finally:
        connection.close()


def test_embedding_cache_round_trip(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "cache.sqlite3")
    try:
        connection.executescript(_SCHEMA)
        entries = (("alpha", (1.0, 2.5)), ("beta", (-3.0, 4.0)))
        _save_embedding_cache(connection, "embed", entries)

        assert _load_embedding_cache(connection, "embed") == entries
        assert _load_embedding_cache(connection, "other") == ()

        connection.execute(
            "UPDATE embedding_cache SET text = 'tampered' WHERE text = 'alpha'"
        )
        with pytest.raises(ValueError, match="digest"):
            _load_embedding_cache(connection, "embed")
    finally:
        connection.close()

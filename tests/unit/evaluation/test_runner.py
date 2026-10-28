from pathlib import Path

from safety_platform.evaluation.cascade import CascadeOutcome
from safety_platform.evaluation.dataset import EvaluationCase
from safety_platform.evaluation.llm import JudgeResult
from safety_platform.evaluation.runner import (
    RunConfig,
    _guarded_judge,
    _is_blocked,
    _result,
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

    assert config.target_model == "llama3.2:latest"
    assert config.tier3_model == "qwen3:8b"
    assert config.judge_model == "qwen3:8b"
    assert config.per_label == 20

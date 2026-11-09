"""Tests for the judge-reliability harness (pure logic only; no Ollama)."""

from dataclasses import dataclass

from safety_platform.evaluation.judge_reliability import (
    JUDGE_RELIABILITY_PATH,
    JudgeReliabilityCase,
    evaluate_judge_reliability,
    judge_reliability_report,
    load_judge_reliability_cases,
)


@dataclass(frozen=True, slots=True)
class _StubResult:
    complied: bool
    error: str | None = None


class _StubJudge:
    def __init__(self, verdicts: dict[str, bool]) -> None:
        self._verdicts = verdicts

    def judge(self, prompt: str, response: str) -> _StubResult:
        for key, value in self._verdicts.items():
            if key in prompt:
                return _StubResult(value)
        return _StubResult(False)


def _case(cid: str, expected: bool, prompt_marker: str) -> JudgeReliabilityCase:
    return JudgeReliabilityCase(
        case_id=cid,
        prompt=prompt_marker,
        response="anything",
        expected_complied=expected,
        category="test",
        rationale="r",
        provenance="p",
        adjudication="human_confirmed",
    )


def test_reliability_dataset_loads_and_is_adjudicated() -> None:
    cases = load_judge_reliability_cases()
    assert len(cases) >= 10
    assert all(case.adjudication == "human_confirmed" for case in cases)
    categories = {case.category for case in cases}
    assert "benign_scaffold_bias_probe" in categories
    assert "scaffold_ack_only" in categories


def test_report_classifies_fp_fn() -> None:
    stub = _StubJudge({"benign": True})
    rows = evaluate_judge_reliability(
        stub,
        [
            _case("want-fp", False, "benign scaffold"),
            _case("want-ok", True, "other prompt"),  # judge says False -> FN
        ],
    )
    report = judge_reliability_report(rows)
    assert [entry["case_id"] for entry in report["false_positives"]] == ["want-fp"]
    assert [entry["case_id"] for entry in report["false_negatives"]] == ["want-ok"]
    assert report["disagreement_rate"] == 1.0
    assert report["error_rate"] == 0.0


def test_report_treats_judge_errors_as_errors_not_disagreements() -> None:
    class _ErrorJudge:
        def judge(self, prompt: str, response: str) -> _StubResult:
            return _StubResult(False, error="malformed_judge_json")

    rows = evaluate_judge_reliability(_ErrorJudge(), [_case("boom", False, "p")])
    report = judge_reliability_report(rows)
    assert rows[0].correct is False
    assert report["judge_errors"] == [
        {"case_id": "boom", "error": "malformed_judge_json"}
    ]
    assert report["error_rate"] == 1.0
    assert report["disagreement_rate"] == 0.0


def test_dataset_path_exists_in_repo() -> None:
    assert JUDGE_RELIABILITY_PATH.exists()

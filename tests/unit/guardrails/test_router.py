import pytest

from safety_platform.config.registry import (
    ExecutionContext,
    Route,
    SafetyVerdict,
    Stage,
)
from safety_platform.guardrails.router import ResponseRouter, StageDecision


def context() -> ExecutionContext:
    return ExecutionContext("trace", "turn", "conversation", guardrail_version="v1")


def test_no_decisions_fails_closed() -> None:
    result = ResponseRouter().route(context(), ())

    assert result.route is Route.ERROR
    assert result.verdict is SafetyVerdict.ERROR
    assert result.stage is Stage.ROUTER
    assert result.error == "no_stage_decisions"


def test_error_precedes_block_and_redact() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(
                Stage.TIER1,
                SafetyVerdict.REDACT,
                redacted_text="[REDACTED]",
            ),
            StageDecision(
                Stage.TIER2,
                SafetyVerdict.BLOCK,
                reason_codes=("tier2_match",),
            ),
            StageDecision(
                Stage.TIER3,
                SafetyVerdict.ERROR,
                error="classifier_timeout",
            ),
        ),
    )

    assert result.route is Route.ERROR
    assert result.stage is Stage.TIER3
    assert result.error == "classifier_timeout"


def test_block_precedes_redact() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(
                Stage.TIER1,
                SafetyVerdict.REDACT,
                redacted_text="[REDACTED]",
            ),
            StageDecision(
                Stage.TIER2,
                SafetyVerdict.BLOCK,
                reason_codes=("tier2_match",),
            ),
        ),
    )

    assert result.route is Route.REFUSED
    assert result.verdict is SafetyVerdict.BLOCK
    assert result.stage is Stage.TIER2
    assert result.reason_codes == ("tier2_match",)


def test_redaction_returns_partial_route() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(
                Stage.TIER1,
                SafetyVerdict.REDACT,
                reason_codes=("secret",),
                redacted_text="safe text",
            ),
            StageDecision(Stage.TIER2, SafetyVerdict.PASS),
        ),
    )

    assert result.route is Route.PARTIAL
    assert result.verdict is SafetyVerdict.REDACT
    assert result.stage is Stage.TIER1
    assert result.safe_text == "safe text"
    assert result.reason_codes == ("secret",)


def test_missing_redaction_text_fails_closed() -> None:
    result = ResponseRouter().route(
        context(),
        (StageDecision(Stage.TIER1, SafetyVerdict.REDACT),),
    )

    assert result.route is Route.ERROR
    assert result.error == "redaction_missing_text"


def test_unresolved_ambiguity_fails_closed() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(Stage.TIER2, SafetyVerdict.AMBIGUOUS),
            StageDecision(Stage.TIER1, SafetyVerdict.PASS),
        ),
    )

    assert result.route is Route.ERROR
    assert result.stage is Stage.TIER2
    assert result.error == "unresolved_ambiguity"


def test_tier3_resolves_ambiguity() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(Stage.TIER2, SafetyVerdict.AMBIGUOUS),
            StageDecision(Stage.TIER3, SafetyVerdict.PASS),
        ),
    )

    assert result.route is Route.SAFE
    assert result.verdict is SafetyVerdict.PASS
    assert result.stage is Stage.TIER3


def test_all_passing_stages_are_safe() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(Stage.TIER1, SafetyVerdict.PASS),
            StageDecision(Stage.TIER2, SafetyVerdict.PASS),
            StageDecision(Stage.TIER3, SafetyVerdict.PASS),
        ),
    )

    assert result.route is Route.SAFE
    assert result.stage is Stage.TIER3


def test_reason_codes_are_deduplicated_in_order() -> None:
    result = ResponseRouter().route(
        context(),
        (
            StageDecision(
                Stage.TIER1,
                SafetyVerdict.BLOCK,
                reason_codes=("a", "b"),
            ),
            StageDecision(
                Stage.TIER2,
                SafetyVerdict.BLOCK,
                reason_codes=("b", "c"),
            ),
        ),
    )

    assert result.reason_codes == ("a", "b", "c")


def test_stage_decision_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="requires error"):
        StageDecision(Stage.TIER1, SafetyVerdict.ERROR)
    with pytest.raises(ValueError, match="only valid for redact"):
        StageDecision(
            Stage.TIER1,
            SafetyVerdict.PASS,
            redacted_text="unexpected",
        )
    with pytest.raises(ValueError, match="non-empty"):
        StageDecision(
            Stage.TIER1,
            SafetyVerdict.BLOCK,
            reason_codes=("",),
        )

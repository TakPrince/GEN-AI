import re
from typing import cast

import pytest

from safety_platform.guardrails.tier1.regex_filter import (
    RegexAction,
    RegexConfigurationError,
    RegexFilter,
    RegexInputTooLarge,
    RegexRule,
    RegexVerdict,
)


def test_pass_when_no_rule_matches() -> None:
    detector = RegexFilter([RegexRule("ignore", r"ignore previous")])

    result = detector.evaluate("benign request")

    assert result.verdict is RegexVerdict.PASS
    assert result.blocked is False
    assert result.rule_ids == ()


def test_block_is_case_insensitive_by_default() -> None:
    detector = RegexFilter([RegexRule("ignore", r"ignore previous")])

    result = detector.evaluate("IGNORE PREVIOUS instructions")

    assert result.verdict is RegexVerdict.BLOCK
    assert result.blocked is True
    assert result.rule_ids == ("ignore",)


def test_block_wins_over_redact_regardless_of_rule_order() -> None:
    detector = RegexFilter(
        [
            RegexRule("token", r"token=\w+", RegexAction.REDACT),
            RegexRule("escape", r"ignore previous", RegexAction.BLOCK),
        ]
    )

    result = detector.evaluate("token=abc ignore previous")

    assert result.verdict is RegexVerdict.BLOCK
    assert result.rule_ids == ("escape",)
    assert result.redaction_spans == ()
    assert result.redacted_text is None


def test_redacts_all_matches_with_merged_spans() -> None:
    detector = RegexFilter(
        [
            RegexRule("secret", r"secret=\w+", RegexAction.REDACT),
            RegexRule("tail", r"=\w+", RegexAction.REDACT),
        ]
    )

    result = detector.evaluate("secret=abc")

    assert result.verdict is RegexVerdict.REDACT
    assert result.blocked is False
    assert result.rule_ids == ("secret", "tail")
    assert result.redaction_spans == ((0, 10),)
    assert result.redacted_text == "[REDACTED]"


def test_redacts_every_match_from_one_rule() -> None:
    detector = RegexFilter([RegexRule("key", r"key-\d+", RegexAction.REDACT)])

    result = detector.evaluate("key-1 and key-2")

    assert result.redaction_spans == ((0, 5), (10, 15))
    assert result.redacted_text == "[REDACTED] and [REDACTED]"


def test_empty_rule_set_passes() -> None:
    assert RegexFilter([]).evaluate("anything").verdict is RegexVerdict.PASS


def test_duplicate_rule_id_rejected() -> None:
    with pytest.raises(RegexConfigurationError, match="duplicate"):
        RegexFilter(
            [
                RegexRule("same", r"a"),
                RegexRule("same", r"b"),
            ]
        )


def test_invalid_action_rejected() -> None:
    with pytest.raises(RegexConfigurationError, match="action"):
        RegexFilter([RegexRule("bad", r"a", cast(RegexAction, "warn"))])


def test_invalid_regex_rejected() -> None:
    with pytest.raises(RegexConfigurationError, match="pattern"):
        RegexFilter([RegexRule("bad", r"(")])


def test_empty_rule_fields_rejected() -> None:
    with pytest.raises(RegexConfigurationError):
        RegexFilter([RegexRule("", r"a")])
    with pytest.raises(RegexConfigurationError):
        RegexFilter([RegexRule("empty", "")])


def test_invalid_input_limit_rejected() -> None:
    with pytest.raises(ValueError):
        RegexFilter([], max_input_bytes=0)


def test_input_unicode_bytes_are_bounded() -> None:
    detector = RegexFilter([], max_input_bytes=4)

    assert detector.evaluate("éé").verdict is RegexVerdict.PASS
    with pytest.raises(RegexInputTooLarge):
        detector.evaluate("ééé")


def test_default_limit_is_4096_bytes() -> None:
    detector = RegexFilter([])

    assert detector.evaluate("a" * 4096).verdict is RegexVerdict.PASS
    with pytest.raises(RegexInputTooLarge):
        detector.evaluate("a" * 4097)


def test_non_string_rejected() -> None:
    detector = RegexFilter([])

    with pytest.raises(TypeError):
        detector.evaluate(cast(str, 1))


def test_flags_are_honored() -> None:
    detector = RegexFilter([RegexRule("line", r"^blocked$", flags=re.MULTILINE)])

    assert detector.evaluate("safe\nblocked\nsafe").verdict is RegexVerdict.BLOCK


@pytest.mark.parametrize("pattern", [r"\s*", r"a?", r"^"])
def test_empty_matching_patterns_rejected(pattern: str) -> None:
    with pytest.raises(RegexConfigurationError, match="empty"):
        RegexFilter([RegexRule("zero", pattern, RegexAction.REDACT)])


def test_zero_width_redaction_is_ignored() -> None:
    detector = RegexFilter([RegexRule("look", r"(?=secret)", RegexAction.REDACT)])

    assert detector.evaluate("secret").verdict is RegexVerdict.PASS


def test_zero_width_block_is_ignored() -> None:
    detector = RegexFilter([RegexRule("look", r"(?=secret)", RegexAction.BLOCK)])

    assert detector.evaluate("secret").verdict is RegexVerdict.PASS

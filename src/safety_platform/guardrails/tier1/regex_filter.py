from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Pattern

MAX_INPUT_BYTES = 4096


class RegexVerdict(str, Enum):
    PASS = "pass"
    BLOCK = "block"
    REDACT = "redact"
    ERROR = "error"


class RegexAction(str, Enum):
    BLOCK = "block"
    REDACT = "redact"


class RegexFilterError(ValueError):
    pass


class RegexConfigurationError(RegexFilterError):
    pass


class RegexInputTooLarge(RegexFilterError):
    pass


@dataclass(frozen=True, slots=True)
class RegexRule:
    rule_id: str
    pattern: str
    action: RegexAction = RegexAction.BLOCK
    flags: int = re.IGNORECASE


@dataclass(frozen=True, slots=True)
class RegexFilterResult:
    verdict: RegexVerdict
    rule_ids: tuple[str, ...] = ()
    redaction_spans: tuple[tuple[int, int], ...] = ()
    redacted_text: str | None = None
    error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.verdict in (RegexVerdict.BLOCK, RegexVerdict.ERROR)


@dataclass(frozen=True, slots=True)
class _CompiledRule:
    rule_id: str
    expression: Pattern[str]
    action: RegexAction


class RegexFilter:
    def __init__(
        self,
        rules: Iterable[RegexRule],
        *,
        max_input_bytes: int = MAX_INPUT_BYTES,
    ) -> None:
        if not isinstance(max_input_bytes, int) or max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive integer")

        compiled: list[_CompiledRule] = []
        seen: set[str] = set()
        for rule in rules:
            if not isinstance(rule.rule_id, str) or not rule.rule_id.strip():
                raise RegexConfigurationError("rule_id must be non-empty")
            if rule.rule_id in seen:
                raise RegexConfigurationError(f"duplicate rule_id: {rule.rule_id}")
            if not isinstance(rule.pattern, str) or not rule.pattern:
                raise RegexConfigurationError("pattern must be non-empty")
            try:
                action = RegexAction(rule.action)
            except ValueError as exc:
                raise RegexConfigurationError(f"invalid action: {rule.action}") from exc
            try:
                expression = re.compile(rule.pattern, rule.flags)
            except (re.error, TypeError, ValueError) as exc:
                raise RegexConfigurationError(
                    f"invalid regex pattern for {rule.rule_id}"
                ) from exc
            if expression.search("") is not None:
                raise RegexConfigurationError(
                    f"pattern can match empty string: {rule.rule_id}"
                )
            seen.add(rule.rule_id)
            compiled.append(_CompiledRule(rule.rule_id, expression, action))

        self._block_rules = tuple(
            rule for rule in compiled if rule.action is RegexAction.BLOCK
        )
        self._redact_rules = tuple(
            rule for rule in compiled if rule.action is RegexAction.REDACT
        )
        self._max_input_bytes = max_input_bytes

    def evaluate(self, text: str) -> RegexFilterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if len(text.encode("utf-8")) > self._max_input_bytes:
            raise RegexInputTooLarge(
                f"input exceeds {self._max_input_bytes} UTF-8 bytes"
            )

        for rule in self._block_rules:
            for match in rule.expression.finditer(text):
                if match.end() > match.start():
                    return RegexFilterResult(
                        verdict=RegexVerdict.BLOCK,
                        rule_ids=(rule.rule_id,),
                    )

        redaction_rule_ids: list[str] = []
        spans: list[tuple[int, int]] = []
        for rule in self._redact_rules:
            matched = False
            for match in rule.expression.finditer(text):
                if match.end() == match.start():
                    continue
                spans.append((match.start(), match.end()))
                matched = True
            if matched:
                redaction_rule_ids.append(rule.rule_id)

        if not spans:
            return RegexFilterResult(verdict=RegexVerdict.PASS)

        merged = _merge_spans(spans)
        return RegexFilterResult(
            verdict=RegexVerdict.REDACT,
            rule_ids=tuple(redaction_rule_ids),
            redaction_spans=merged,
            redacted_text=_redact_text(text, merged),
        )


def _merge_spans(spans: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _redact_text(text: str, spans: Iterable[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        parts.append("[REDACTED]")
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)

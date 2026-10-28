"""Fail-closed response routing with block-over-redact precedence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..config.registry import (
    ExecutionContext,
    Route,
    RouteVerdict,
    SafetyVerdict,
    Stage,
)

_STAGE_ORDER: dict[Stage, int] = {
    Stage.TIER1: 1,
    Stage.TIER2: 2,
    Stage.TIER3: 3,
    Stage.ROUTER: 4,
}


@dataclass(frozen=True, slots=True)
class StageDecision:
    stage: Stage
    verdict: SafetyVerdict
    reason_codes: tuple[str, ...] = ()
    redacted_text: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, Stage):
            raise TypeError("stage must be a Stage")
        if not isinstance(self.verdict, SafetyVerdict):
            raise TypeError("verdict must be a SafetyVerdict")
        if self.verdict is SafetyVerdict.ERROR:
            if not self.error:
                raise ValueError("error verdict requires error")
        elif self.error is not None:
            raise ValueError("error is only valid for error verdicts")
        if self.verdict is not SafetyVerdict.REDACT and self.redacted_text is not None:
            raise ValueError("redacted_text is only valid for redact verdicts")
        for code in self.reason_codes:
            if not isinstance(code, str) or not code.strip():
                raise ValueError("reason codes must be non-empty strings")


class ResponseRouter:
    def route(
        self,
        context: ExecutionContext,
        decisions: Iterable[StageDecision],
    ) -> RouteVerdict:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        signals = tuple(decisions)
        if not signals:
            return _error_route(
                context,
                Stage.ROUTER,
                (),
                "no_stage_decisions",
            )

        ordered = tuple(
            sorted(signals, key=lambda decision: _STAGE_ORDER[decision.stage])
        )

        errors = tuple(
            decision
            for decision in ordered
            if decision.verdict is SafetyVerdict.ERROR
        )
        if errors:
            chosen = errors[-1]
            return _error_route(
                context,
                chosen.stage,
                _reason_codes(errors),
                chosen.error or "guardrail_error",
            )

        blocks = tuple(
            decision
            for decision in ordered
            if decision.verdict is SafetyVerdict.BLOCK
        )
        if blocks:
            return RouteVerdict(
                context=context,
                route=Route.REFUSED,
                verdict=SafetyVerdict.BLOCK,
                stage=blocks[-1].stage,
                reason_codes=_reason_codes(blocks),
            )

        if ordered[-1].verdict is SafetyVerdict.AMBIGUOUS:
            ambiguous = tuple(
                decision
                for decision in ordered
                if decision.verdict is SafetyVerdict.AMBIGUOUS
            )
            return _error_route(
                context,
                ordered[-1].stage,
                _reason_codes(ambiguous),
                "unresolved_ambiguity",
            )

        redactions = tuple(
            decision
            for decision in ordered
            if decision.verdict is SafetyVerdict.REDACT
        )
        if redactions:
            chosen = redactions[-1]
            if not chosen.redacted_text:
                return _error_route(
                    context,
                    chosen.stage,
                    _reason_codes(redactions),
                    "redaction_missing_text",
                )
            return RouteVerdict(
                context=context,
                route=Route.PARTIAL,
                verdict=SafetyVerdict.REDACT,
                stage=chosen.stage,
                reason_codes=_reason_codes(redactions),
                safe_text=chosen.redacted_text,
            )

        return RouteVerdict(
            context=context,
            route=Route.SAFE,
            verdict=SafetyVerdict.PASS,
            stage=ordered[-1].stage,
        )


def _reason_codes(decisions: Iterable[StageDecision]) -> tuple[str, ...]:
    codes: list[str] = []
    seen: set[str] = set()
    for decision in decisions:
        for code in decision.reason_codes:
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return tuple(codes)


def _error_route(
    context: ExecutionContext,
    stage: Stage,
    reason_codes: tuple[str, ...],
    error: str,
) -> RouteVerdict:
    return RouteVerdict(
        context=context,
        route=Route.ERROR,
        verdict=SafetyVerdict.ERROR,
        stage=stage,
        reason_codes=reason_codes,
        error=error,
    )


__all__ = ["ResponseRouter", "StageDecision"]

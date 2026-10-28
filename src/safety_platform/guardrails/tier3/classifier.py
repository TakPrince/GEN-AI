"""Tier 3 confidence classifier boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from ...config.registry import (
    CascadeThresholds,
    ExecutionContext,
    SafetyVerdict,
)


class ClassifierClient(Protocol):
    async def score(self, text: str, context: ExecutionContext) -> float: ...


@dataclass(frozen=True, slots=True)
class Tier3Result:
    verdict: SafetyVerdict
    confidence: float | None = None
    reason_codes: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in (
            SafetyVerdict.PASS,
            SafetyVerdict.BLOCK,
            SafetyVerdict.ERROR,
        ):
            raise ValueError("tier3 verdict must be pass, block, or error")
        if self.verdict is SafetyVerdict.ERROR:
            if not self.error:
                raise ValueError("error verdict requires error")
            if self.confidence is not None:
                raise ValueError("error verdict cannot carry confidence")
        else:
            if self.error is not None:
                raise ValueError("error is only valid for error verdicts")
            if self.confidence is None:
                raise ValueError("terminal verdict requires confidence")
            if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
                raise ValueError("confidence must be within [0, 1]")
        for code in self.reason_codes:
            if not isinstance(code, str) or not code.strip():
                raise ValueError("reason codes must be non-empty strings")


class Tier3Classifier:
    def __init__(
        self,
        client: ClassifierClient,
        thresholds: CascadeThresholds,
    ) -> None:
        self._client = client
        self._thresholds = thresholds

    async def classify(
        self,
        text: str,
        context: ExecutionContext,
    ) -> Tier3Result:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        try:
            raw = await self._client.score(text, context)
        except Exception as exc:
            return Tier3Result(SafetyVerdict.ERROR, error=type(exc).__name__)

        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return Tier3Result(
                SafetyVerdict.ERROR, error="invalid_confidence"
            )
        confidence = float(raw)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return Tier3Result(
                SafetyVerdict.ERROR, error="invalid_confidence"
            )

        verdict = (
            SafetyVerdict.BLOCK
            if confidence >= self._thresholds.tier3_block_confidence
            else SafetyVerdict.PASS
        )
        return Tier3Result(verdict, confidence=confidence)


__all__ = ["ClassifierClient", "Tier3Classifier", "Tier3Result"]

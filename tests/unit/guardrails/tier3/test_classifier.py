import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    ExecutionContext,
    SafetyVerdict,
)
from safety_platform.guardrails.tier3.classifier import (
    Tier3Classifier,
    Tier3Result,
)


class StubClient:
    def __init__(self, value: float) -> None:
        self.value = value

    async def score(self, text: str, context: ExecutionContext) -> float:
        return self.value


class FailingClient:
    async def score(self, text: str, context: ExecutionContext) -> float:
        raise RuntimeError("model unavailable")


def context() -> ExecutionContext:
    return ExecutionContext("trace", "turn", "conversation")


def thresholds() -> CascadeThresholds:
    return CascadeThresholds(0.8, 0.5, 0.75)


@pytest.mark.asyncio
async def test_high_confidence_blocks() -> None:
    result = await Tier3Classifier(
        StubClient(0.9),
        thresholds(),
    ).classify("prompt", context())

    assert result.verdict is SafetyVerdict.BLOCK
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_low_confidence_passes() -> None:
    result = await Tier3Classifier(
        StubClient(0.2),
        thresholds(),
    ).classify("prompt", context())

    assert result.verdict is SafetyVerdict.PASS
    assert result.confidence == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_classifier_exception_fails_closed() -> None:
    result = await Tier3Classifier(
        FailingClient(),
        thresholds(),
    ).classify("prompt", context())

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "RuntimeError"


@pytest.mark.asyncio
async def test_invalid_confidence_fails_closed() -> None:
    result = await Tier3Classifier(
        StubClient(1.5),
        thresholds(),
    ).classify("prompt", context())

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "invalid_confidence"


def test_tier3_result_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError, match="requires error"):
        Tier3Result(SafetyVerdict.ERROR)
    with pytest.raises(ValueError, match="requires confidence"):
        Tier3Result(SafetyVerdict.BLOCK)
    with pytest.raises(ValueError, match="within"):
        Tier3Result(SafetyVerdict.PASS, confidence=1.1)

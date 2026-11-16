"""Centralized payload limits (P1-03) enforced at every trust boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from safety_platform import app as dashboard
from safety_platform.config.registry import (
    DEFAULT_PAYLOAD_LIMITS,
    CascadeThresholds,
    PayloadLimits,
    RoutingMargins,
    utf8_byte_length,
)
from safety_platform.guardrails.tier1.regex_filter import (
    RegexFilter,
    RegexInputTooLarge,
)
from safety_platform.guardrails.tier2.vector_filter import Centroid, VectorFilter
from safety_platform.guardrails.tier3.guard_classifier import (
    ExecutionContext,
    SafetyVerdict,
)

# --- registry: PayloadLimits --------------------------------------------------


def test_payload_limit_defaults_match_documented_values() -> None:
    limits = DEFAULT_PAYLOAD_LIMITS
    assert limits.prompt_bytes == 16384
    assert limits.request_body_bytes == 32768
    assert limits.metadata_entries == 16
    assert limits.metadata_key_bytes == 128
    assert limits.metadata_value_bytes == 512
    assert limits.request_id_chars == 64


def test_payload_limits_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_PAYLOAD_LIMITS.prompt_bytes = 1


def test_payload_limits_are_injectable() -> None:
    limits = PayloadLimits(
        prompt_bytes=1024,
        request_body_bytes=2048,
        metadata_entries=4,
        metadata_key_bytes=32,
        metadata_value_bytes=64,
        request_id_chars=16,
    )
    assert limits.prompt_bytes == 1024


# --- utf8_byte_length ----------------------------------------------------------


def test_byte_length_counts_utf8_bytes_not_characters() -> None:
    assert utf8_byte_length("abcd") == 4
    assert utf8_byte_length("é" * 8192) == 16384  # 2 bytes each


def test_byte_length_rejects_invalid_unicode() -> None:
    with pytest.raises(ValueError, match="invalid unicode"):
        utf8_byte_length("ok\ud800later")


# --- Tier 1: RegexFilter -------------------------------------------------------


def test_tier1_library_default_stays_conservative() -> None:
    detector = RegexFilter([])
    with pytest.raises(RegexInputTooLarge):
        detector.evaluate("a" * 4097)
    assert detector.evaluate("a" * 4096).verdict.value == "pass"


def test_dashboard_pipeline_uses_central_prompt_limit() -> None:
    pipeline = dashboard._build_pipeline(str(dashboard._demo_database_path()))
    limit = DEFAULT_PAYLOAD_LIMITS.prompt_bytes
    assert pipeline.regex.evaluate("a" * limit).verdict.value == "pass"
    with pytest.raises(RegexInputTooLarge):
        pipeline.regex.evaluate("a" * (limit + 1))


# --- Tier 2: VectorFilter ------------------------------------------------------


class StubEmbedder:
    def embed(self, text: str) -> tuple[float, float]:
        return (1.0, 0.0)


def _filter(max_input_bytes: int | None = None) -> VectorFilter:
    kwargs: dict[str, object] = {}
    if max_input_bytes is not None:
        kwargs["max_input_bytes"] = max_input_bytes
    return VectorFilter(
        StubEmbedder(),
        (Centroid("attack", (1.0, 0.0)),),
        CascadeThresholds(0.8, 0.5, 0.75),
        RoutingMargins(0.1, 0.4),
        **kwargs,  # type: ignore[arg-type]
    )


def test_tier2_rejects_oversized_input_fail_closed() -> None:
    result = _filter().evaluate("a" * (DEFAULT_PAYLOAD_LIMITS.prompt_bytes + 1))
    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "input_too_large"


def test_tier2_accepts_at_limit_input() -> None:
    result = _filter().evaluate("a" * DEFAULT_PAYLOAD_LIMITS.prompt_bytes)
    # At the limit the semantic pipeline runs normally (stub classifies as
    # attack); the size gate must not interfere.
    assert result.error is None
    assert result.verdict is not SafetyVerdict.ERROR


def test_tier2_rejects_invalid_unicode_fail_closed() -> None:
    result = _filter().evaluate("ok\ud800later")
    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "invalid_unicode"


def test_tier2_limit_is_injectable() -> None:
    result = _filter(max_input_bytes=16).evaluate("a" * 17)
    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "input_too_large"


# --- Tier 3: dashboard pipeline accepts boundary-sized prompts ------------------


class _StubTransport:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def generate(self, request: object) -> str:
        self.requests.append(request)
        return "safe"


@pytest.mark.asyncio
async def test_dashboard_tier3_accepts_boundary_sized_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAFETY_UI_TIER3", "ollama")
    transport = _StubTransport()
    guard, _mode = dashboard._tier3_guard(transport)  # type: ignore[arg-type]
    context = ExecutionContext(
        "trace", "turn", "conversation", guardrail_version="test"
    )

    result = await guard.classify_if_ambiguous(
        "a" * DEFAULT_PAYLOAD_LIMITS.prompt_bytes, context, SafetyVerdict.AMBIGUOUS
    )

    assert result is not None
    assert len(transport.requests) == 1

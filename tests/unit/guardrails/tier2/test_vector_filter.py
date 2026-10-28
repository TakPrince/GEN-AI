from collections.abc import Sequence
from typing import cast

import pytest

from safety_platform.config.registry import (
    CascadeThresholds,
    RoutingMargins,
    SafetyVerdict,
)
from safety_platform.guardrails.tier2.vector_filter import (
    Centroid,
    VectorFilter,
)


class StubEmbedder:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = vector

    def embed(self, text: str) -> Sequence[float]:
        return self.vector


class FailingEmbedder:
    def embed(self, text: str) -> Sequence[float]:
        raise RuntimeError("model unavailable")


def thresholds() -> CascadeThresholds:
    return CascadeThresholds(0.8, 0.5, 0.75)


def margins() -> RoutingMargins:
    return RoutingMargins(0.1, 0.4)


def centroids() -> tuple[Centroid, Centroid]:
    return (
        Centroid("attack_a", (1.0, 0.0)),
        Centroid("attack_b", (0.0, 1.0)),
    )


def test_high_similarity_with_clear_gap_blocks() -> None:
    result = VectorFilter(
        StubEmbedder((1.0, 0.0)),
        centroids(),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.BLOCK
    assert result.top_centroid_id == "attack_a"
    assert result.top_similarity == pytest.approx(1.0)
    assert result.top_gap == pytest.approx(1.0)


def test_similarity_inside_ambiguity_band_escalates() -> None:
    result = VectorFilter(
        StubEmbedder((0.714, 0.7)),
        centroids(),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.AMBIGUOUS
    assert 0.5 <= result.top_similarity < 0.8


def test_high_similarity_with_small_gap_escalates() -> None:
    result = VectorFilter(
        StubEmbedder((1.0, 0.0)),
        (
            Centroid("attack_a", (1.0, 0.0)),
            Centroid("attack_b", (1.0, 0.0)),
        ),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.AMBIGUOUS
    assert result.top_gap == pytest.approx(0.0)


def test_low_similarity_passes() -> None:
    result = VectorFilter(
        StubEmbedder((0.0, 1.0)),
        (Centroid("attack_a", (1.0, 0.0)),),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.PASS
    assert result.top_centroid_id == "attack_a"


def test_empty_index_passes_without_embedding() -> None:
    result = VectorFilter(
        FailingEmbedder(),
        (),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.PASS


def test_embedding_failure_fails_closed() -> None:
    result = VectorFilter(
        FailingEmbedder(),
        centroids(),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "RuntimeError"


def test_zero_embedding_fails_closed() -> None:
    result = VectorFilter(
        StubEmbedder((0.0, 0.0)),
        centroids(),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "ValueError"


def test_dimension_mismatch_fails_closed() -> None:
    result = VectorFilter(
        StubEmbedder((1.0, 0.0, 0.0)),
        centroids(),
        thresholds(),
        margins(),
    ).evaluate("prompt")

    assert result.verdict is SafetyVerdict.ERROR
    assert result.error == "embedding_dimension_mismatch"


def test_inconsistent_centroid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="embedding dimension"):
        VectorFilter(
            StubEmbedder((1.0, 0.0)),
            (
                Centroid("attack_a", (1.0, 0.0)),
                Centroid("attack_b", (1.0, 0.0, 0.0)),
            ),
            thresholds(),
            margins(),
        )


def test_non_string_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="text"):
        VectorFilter(
            StubEmbedder((1.0, 0.0)),
            centroids(),
            thresholds(),
            margins(),
        ).evaluate(cast(str, 1))

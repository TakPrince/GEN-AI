"""Tier 2 cosine-distance structural signature filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ...config.registry import CascadeThresholds, RoutingMargins, SafetyVerdict


class Embedder(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must contain numeric values") from exc
    if not vector:
        raise ValueError("vector must not be empty")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("vector must contain finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("vector must have non-zero magnitude")
    return tuple(value / norm for value in vector)


@dataclass(frozen=True, slots=True)
class Centroid:
    centroid_id: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        _text(self.centroid_id, "centroid_id")
        object.__setattr__(self, "vector", _unit_vector(self.vector))


@dataclass(frozen=True, slots=True)
class VectorFilterResult:
    verdict: SafetyVerdict
    top_centroid_id: str | None = None
    top_similarity: float = 0.0
    runner_up_similarity: float = 0.0
    top_gap: float = 0.0
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, SafetyVerdict):
            raise TypeError("verdict must be a SafetyVerdict")
        if self.verdict is SafetyVerdict.ERROR:
            if not self.error:
                raise ValueError("error verdict requires error")
        elif self.error is not None:
            raise ValueError("error is only valid for error verdicts")
        for value, name in (
            (self.top_similarity, "top_similarity"),
            (self.runner_up_similarity, "runner_up_similarity"),
            (self.top_gap, "top_gap"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    @property
    def blocked(self) -> bool:
        return self.verdict is SafetyVerdict.BLOCK


class VectorFilter:
    def __init__(
        self,
        embedder: Embedder,
        centroids: Sequence[Centroid],
        thresholds: CascadeThresholds,
        margins: RoutingMargins,
    ) -> None:
        self._embedder = embedder
        self._thresholds = thresholds
        self._margins = margins
        self._dimension: int | None = None
        normalized: list[Centroid] = []
        for centroid in centroids:
            if not isinstance(centroid, Centroid):
                raise TypeError("centroids must contain Centroid values")
            if self._dimension is None:
                self._dimension = len(centroid.vector)
            elif len(centroid.vector) != self._dimension:
                raise ValueError("all centroids must share one embedding dimension")
            normalized.append(centroid)
        self._centroids = tuple(normalized)

    def evaluate(self, text: str) -> VectorFilterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not self._centroids or self._dimension is None:
            return VectorFilterResult(SafetyVerdict.PASS)

        try:
            embedding = _unit_vector(self._embedder.embed(text))
        except Exception as exc:
            return VectorFilterResult(
                SafetyVerdict.ERROR, error=type(exc).__name__
            )
        if len(embedding) != self._dimension:
            return VectorFilterResult(
                SafetyVerdict.ERROR, error="embedding_dimension_mismatch"
            )

        ranked = sorted(
            (
                (_cosine(embedding, centroid.vector), centroid.centroid_id)
                for centroid in self._centroids
            ),
            key=lambda item: (-item[0], item[1]),
        )
        top_similarity, top_centroid_id = ranked[0]
        runner_up_similarity = ranked[1][0] if len(ranked) > 1 else 0.0
        top_gap = top_similarity - runner_up_similarity

        if (
            top_similarity >= self._thresholds.tier2_block_similarity
            and top_gap >= self._margins.min_top_gap
        ):
            verdict = SafetyVerdict.BLOCK
        elif top_similarity >= self._thresholds.tier2_ambiguity_floor:
            verdict = SafetyVerdict.AMBIGUOUS
        else:
            verdict = SafetyVerdict.PASS

        return VectorFilterResult(
            verdict=verdict,
            top_centroid_id=top_centroid_id,
            top_similarity=top_similarity,
            runner_up_similarity=runner_up_similarity,
            top_gap=top_gap,
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


__all__ = [
    "Centroid",
    "Embedder",
    "VectorFilter",
    "VectorFilterResult",
]

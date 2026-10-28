"""Train/test threshold sweep for cascade calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from ..config.registry import CascadeThresholds, RoutingMargins


class CalibrationError(ValueError):
    pass


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _unit(value: float, name: str) -> None:
    _finite(value, name)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    top_similarity: float
    runner_up_similarity: float
    tier3_confidence: float
    is_attack: bool

    def __post_init__(self) -> None:
        _finite(self.top_similarity, "top_similarity")
        _finite(self.runner_up_similarity, "runner_up_similarity")
        if not -1.0 <= self.top_similarity <= 1.0:
            raise ValueError("top_similarity must be within [-1, 1]")
        if not -1.0 <= self.runner_up_similarity <= 1.0:
            raise ValueError("runner_up_similarity must be within [-1, 1]")
        _unit(self.tier3_confidence, "tier3_confidence")
        if not isinstance(self.is_attack, bool):
            raise TypeError("is_attack must be a bool")

    @property
    def top_gap(self) -> float:
        return self.top_similarity - self.runner_up_similarity


@dataclass(frozen=True, slots=True)
class CalibrationSplit:
    train: tuple[CalibrationSample, ...]
    test: tuple[CalibrationSample, ...]

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("train split must not be empty")
        if not self.test:
            raise ValueError("test split must not be empty")
        if any(not isinstance(sample, CalibrationSample) for sample in self.train):
            raise TypeError("train must contain CalibrationSample values")
        if any(not isinstance(sample, CalibrationSample) for sample in self.test):
            raise TypeError("test must contain CalibrationSample values")


@dataclass(frozen=True, slots=True)
class CalibrationSpace:
    block_similarity: tuple[float, ...] = (0.7, 0.75, 0.8, 0.85)
    ambiguity_floor: tuple[float, ...] = (0.4, 0.5, 0.6)
    min_top_gap: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15)
    tier3_block_confidence: tuple[float, ...] = (0.6, 0.7, 0.8)
    tier3_gate_probability: float = 0.4

    def __post_init__(self) -> None:
        for values, name in (
            (self.block_similarity, "block_similarity"),
            (self.ambiguity_floor, "ambiguity_floor"),
            (self.min_top_gap, "min_top_gap"),
            (self.tier3_block_confidence, "tier3_block_confidence"),
        ):
            if not values:
                raise ValueError(f"{name} grid must not be empty")
            for value in values:
                _unit(value, name)
        _unit(self.tier3_gate_probability, "tier3_gate_probability")


@dataclass(frozen=True, slots=True)
class CalibrationPolicy:
    max_fpr: float = 0.05
    max_tier3_rate: float = 0.35
    min_recall: float = 0.0

    def __post_init__(self) -> None:
        _unit(self.max_fpr, "max_fpr")
        _unit(self.max_tier3_rate, "max_tier3_rate")
        _unit(self.min_recall, "min_recall")


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    tier3_invocations: int
    total: int

    def __post_init__(self) -> None:
        values = (
            self.true_positives,
            self.false_positives,
            self.true_negatives,
            self.false_negatives,
            self.tier3_invocations,
            self.total,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise TypeError("metric counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("metric counts must be non-negative")
        if (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
            != self.total
        ):
            raise ValueError("confusion matrix must sum to total")
        if self.tier3_invocations > self.total:
            raise ValueError("tier3_invocations cannot exceed total")

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return 1.0 if denominator == 0 else self.true_positives / denominator

    @property
    def fpr(self) -> float:
        denominator = self.false_positives + self.true_negatives
        return 0.0 if denominator == 0 else self.false_positives / denominator

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return 0.0 if denominator == 0 else self.true_positives / denominator

    @property
    def accuracy(self) -> float:
        return (
            0.0
            if self.total == 0
            else (self.true_positives + self.true_negatives) / self.total
        )

    @property
    def tier3_rate(self) -> float:
        return 0.0 if self.total == 0 else self.tier3_invocations / self.total


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    thresholds: CascadeThresholds
    margins: RoutingMargins
    train: CalibrationMetrics
    test: CalibrationMetrics

    @property
    def score(self) -> float:
        return self.train.recall - 1.5 * self.train.fpr - 0.25 * self.train.tier3_rate


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    best: CalibrationCandidate
    feasible: tuple[CalibrationCandidate, ...]
    evaluated: int


class CalibrationSweep:
    def __init__(
        self,
        split: CalibrationSplit,
        space: CalibrationSpace | None = None,
        policy: CalibrationPolicy | None = None,
    ) -> None:
        self._split = split
        self._space = CalibrationSpace() if space is None else space
        self._policy = CalibrationPolicy() if policy is None else policy

    def run(self) -> CalibrationReport:
        feasible: list[CalibrationCandidate] = []
        evaluated = 0
        for block, floor, gap, tier3, gate in product(
            self._space.block_similarity,
            self._space.ambiguity_floor,
            self._space.min_top_gap,
            self._space.tier3_block_confidence,
            (self._space.tier3_gate_probability,),
        ):
            evaluated += 1
            if floor >= block:
                continue
            thresholds = CascadeThresholds(block, floor, tier3)
            margins = RoutingMargins(gap, gate)
            train = _metrics(self._split.train, thresholds, margins)
            if (
                train.fpr > self._policy.max_fpr
                or train.tier3_rate > self._policy.max_tier3_rate
                or train.recall < self._policy.min_recall
            ):
                continue
            feasible.append(
                CalibrationCandidate(
                    thresholds=thresholds,
                    margins=margins,
                    train=train,
                    test=_metrics(self._split.test, thresholds, margins),
                )
            )

        if not feasible:
            raise CalibrationError("no feasible calibration candidate")

        ordered = tuple(
            sorted(
                feasible,
                key=lambda candidate: (
                    -candidate.score,
                    -candidate.train.recall,
                    candidate.train.fpr,
                    candidate.train.tier3_rate,
                    -candidate.test.recall,
                ),
            )
        )
        return CalibrationReport(
            best=ordered[0],
            feasible=ordered,
            evaluated=evaluated,
        )


def _metrics(
    samples: tuple[CalibrationSample, ...],
    thresholds: CascadeThresholds,
    margins: RoutingMargins,
) -> CalibrationMetrics:
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    tier3_invocations = 0
    for sample in samples:
        blocked, invoked = _decide(sample, thresholds, margins)
        if invoked:
            tier3_invocations += 1
        if sample.is_attack and blocked:
            true_positives += 1
        elif sample.is_attack:
            false_negatives += 1
        elif blocked:
            false_positives += 1
        else:
            true_negatives += 1
    return CalibrationMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        tier3_invocations=tier3_invocations,
        total=len(samples),
    )


def _decide(
    sample: CalibrationSample,
    thresholds: CascadeThresholds,
    margins: RoutingMargins,
) -> tuple[bool, bool]:
    if sample.top_similarity >= thresholds.tier2_block_similarity:
        if sample.top_gap >= margins.min_top_gap:
            return True, False
        return sample.tier3_confidence >= thresholds.tier3_block_confidence, True
    if sample.top_similarity >= thresholds.tier2_ambiguity_floor:
        if sample.tier3_confidence < margins.tier3_gate_probability:
            return False, False
        return sample.tier3_confidence >= thresholds.tier3_block_confidence, True
    return False, False


__all__ = [
    "CalibrationCandidate",
    "CalibrationError",
    "CalibrationMetrics",
    "CalibrationPolicy",
    "CalibrationReport",
    "CalibrationSample",
    "CalibrationSpace",
    "CalibrationSplit",
    "CalibrationSweep",
]

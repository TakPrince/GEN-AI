"""Lightweight logistic-regression Tier-3 bypass router."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from ..config.registry import SafetyVerdict


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
class RouterFeatures:
    regex_hit: bool
    top_similarity: float
    top_gap: float
    tier2_ambiguous: bool
    prompt_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.regex_hit, bool):
            raise TypeError("regex_hit must be a bool")
        if not isinstance(self.tier2_ambiguous, bool):
            raise TypeError("tier2_ambiguous must be a bool")
        _finite(self.top_similarity, "top_similarity")
        _finite(self.top_gap, "top_gap")
        if (
            isinstance(self.prompt_bytes, bool)
            or not isinstance(self.prompt_bytes, int)
            or self.prompt_bytes < 0
        ):
            raise ValueError("prompt_bytes must be a non-negative integer")

    def as_vector(self) -> tuple[float, ...]:
        return (
            1.0 if self.regex_hit else 0.0,
            max(-1.0, min(1.0, self.top_similarity)),
            max(-1.0, min(1.0, self.top_gap)),
            1.0 if self.tier2_ambiguous else 0.0,
            min(1.0, self.prompt_bytes / 4096.0),
        )


@dataclass(frozen=True, slots=True)
class RouterSample:
    features: RouterFeatures
    tier3_changes_verdict: bool

    def __post_init__(self) -> None:
        if not isinstance(self.features, RouterFeatures):
            raise TypeError("features must be RouterFeatures")
        if not isinstance(self.tier3_changes_verdict, bool):
            raise TypeError("tier3_changes_verdict must be a bool")


@dataclass(frozen=True, slots=True)
class RouterDecision:
    invoke_tier3: bool
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.invoke_tier3, bool):
            raise TypeError("invoke_tier3 must be a bool")
        _unit(self.probability, "probability")

    @property
    def bypass_tier3(self) -> bool:
        return not self.invoke_tier3


class LogisticRegressionRouter:
    FEATURE_COUNT = 5

    def __init__(
        self,
        weights: tuple[float, ...] | None = None,
        bias: float = 0.0,
        threshold: float = 0.5,
    ) -> None:
        values = (0.0,) * self.FEATURE_COUNT if weights is None else weights
        if len(values) != self.FEATURE_COUNT:
            raise ValueError(f"weights must contain {self.FEATURE_COUNT} values")
        for value in values:
            _finite(value, "weight")
        _finite(bias, "bias")
        _unit(threshold, "threshold")
        self._weights = tuple(float(value) for value in values)
        self._bias = float(bias)
        self._threshold = float(threshold)

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict_proba(self, features: RouterFeatures) -> float:
        if not isinstance(features, RouterFeatures):
            raise TypeError("features must be RouterFeatures")
        return _sigmoid(
            self._bias
            + sum(
                weight * value
                for weight, value in zip(
                    self._weights,
                    features.as_vector(),
                    strict=True,
                )
            )
        )

    def should_escalate(self, features: RouterFeatures) -> bool:
        return self.predict_proba(features) >= self._threshold

    def decide(self, features: RouterFeatures) -> RouterDecision:
        probability = self.predict_proba(features)
        return RouterDecision(
            invoke_tier3=probability >= self._threshold,
            probability=probability,
        )

    @classmethod
    def fit(
        cls,
        samples: tuple[RouterSample, ...],
        *,
        epochs: int = 400,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        threshold: float = 0.5,
    ) -> LogisticRegressionRouter:
        if not samples:
            raise ValueError("samples must not be empty")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, int | float)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be positive")
        _unit(l2, "l2")

        weights = [0.0] * cls.FEATURE_COUNT
        bias = 0.0
        for _ in range(epochs):
            for sample in samples:
                vector = sample.features.as_vector()
                probability = _sigmoid(
                    bias
                    + sum(
                        weight * value
                        for weight, value in zip(
                            weights,
                            vector,
                            strict=True,
                        )
                    )
                )
                target = 1.0 if sample.tier3_changes_verdict else 0.0
                error = probability - target
                for index, value in enumerate(vector):
                    weights[index] -= learning_rate * (
                        error * value + l2 * weights[index]
                    )
                bias -= learning_rate * error
        return cls(tuple(weights), bias, threshold)


class Tier3BypassRouter:
    def __init__(
        self,
        model: LogisticRegressionRouter,
        *,
        threshold: float | None = None,
    ) -> None:
        self._model = model
        if threshold is not None:
            _unit(threshold, "threshold")
            self._threshold = float(threshold)
        else:
            self._threshold = model.threshold

    def route(self, features: RouterFeatures) -> RouterDecision:
        probability = self._model.predict_proba(features)
        return RouterDecision(
            invoke_tier3=probability >= self._threshold,
            probability=probability,
        )

    @property
    def model(self) -> LogisticRegressionRouter:
        return self._model


class EscalationDecision(Enum):
    """Explicit Tier 3 escalation outcome for one defended request.

    Precedence (P2-03):
    1. FAIL_CLOSED — Tier 2 infrastructure failure; router fails the request
       closed, no Tier 3 call (a broken signal must not be "repaired").
    2. CLEAR_BLOCK — an earlier tier already produced a terminal BLOCK; the
       router refuses regardless of any Tier 3 verdict, so the multi-second
       Tier 3 call is unnecessary.
    3. ESCALATE — uncertain: router escalation or Tier 2 ambiguity. Ambiguity
       never silently becomes an ALLOW.
    4. CLEAR_ALLOW — both signals clear; skip Tier 3.
    """

    CLEAR_ALLOW = auto()
    CLEAR_BLOCK = auto()
    ESCALATE = auto()
    FAIL_CLOSED = auto()


def escalation_for(
    tier2_verdict: SafetyVerdict, invoke_tier3: bool
) -> EscalationDecision:
    """Derive the escalation decision from the two available input signals.

    No new thresholds: the router's learned threshold and drift floor, and
    the Tier 2 ambiguity floor, remain the only knobs.
    """
    if tier2_verdict is SafetyVerdict.ERROR:
        return EscalationDecision.FAIL_CLOSED
    if tier2_verdict is SafetyVerdict.BLOCK:
        return EscalationDecision.CLEAR_BLOCK
    if invoke_tier3 or tier2_verdict is SafetyVerdict.AMBIGUOUS:
        return EscalationDecision.ESCALATE
    return EscalationDecision.CLEAR_ALLOW


class ModelDrivenRouter:
    """Hazard-boundary drift guard over the statistical bypass router.

    Drops the rapid-bypass path whenever the Tier-2 embedding drifts toward a
    hazardous centroid, independent of the learned bypass probability. The
    guard is fail-closed: ambiguous or drifted inputs always escalate.
    """

    def __init__(
        self,
        base: Tier3BypassRouter,
        *,
        drift_floor: float = 0.60,
    ) -> None:
        if not isinstance(base, Tier3BypassRouter):
            raise TypeError("base must be a Tier3BypassRouter")
        _unit(drift_floor, "drift_floor")
        self._base = base
        self._drift_floor = float(drift_floor)

    @property
    def drift_floor(self) -> float:
        return self._drift_floor

    @property
    def base(self) -> Tier3BypassRouter:
        return self._base

    def route(self, features: RouterFeatures) -> RouterDecision:
        if not isinstance(features, RouterFeatures):
            raise TypeError("features must be RouterFeatures")
        decision = self._base.route(features)
        if features.top_similarity >= self._drift_floor and not decision.invoke_tier3:
            return RouterDecision(invoke_tier3=True, probability=decision.probability)
        return decision


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        scaled = math.exp(-value)
        return 1.0 / (1.0 + scaled)
    scaled = math.exp(value)
    return scaled / (1.0 + scaled)


def benchmark_router() -> LogisticRegressionRouter:
    """Calibrated bypass-router configuration used by the cascade and dashboard.

    Single source of truth; provenance and decision record live in Readme §9.
    The weights are heuristically calibrated, not learned by training on data.
    Do not copy the literals elsewhere.
    """
    return LogisticRegressionRouter(
        weights=(1.5, 1.0, -0.3, 3.0, 0.2),
        bias=-2.5,
        threshold=0.50,
    )


__all__ = [
    "EscalationDecision",
    "LogisticRegressionRouter",
    "ModelDrivenRouter",
    "RouterDecision",
    "RouterFeatures",
    "RouterSample",
    "Tier3BypassRouter",
    "benchmark_router",
    "escalation_for",
]

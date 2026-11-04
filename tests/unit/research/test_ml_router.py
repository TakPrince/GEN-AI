from typing import cast

import pytest

from safety_platform.research.ml_router import (
    LogisticRegressionRouter,
    ModelDrivenRouter,
    RouterDecision,
    RouterFeatures,
    RouterSample,
    Tier3BypassRouter,
)


def _distant_base() -> Tier3BypassRouter:
    # Bias pushes every model score away from the escalation threshold.
    return Tier3BypassRouter(
        LogisticRegressionRouter(weights=(1.0, 1.0, -0.3, 3.0, 0.2), bias=-9.0)
    )


def test_model_driven_router_drops_bypass_near_hazard_boundary() -> None:
    router = ModelDrivenRouter(
        _distant_base(),
        drift_floor=0.60,
    )

    decision = router.route(features(0.65))

    assert decision.invoke_tier3 is True
    assert decision.probability < 0.5


def test_model_driven_router_preserves_rapid_bypass_when_distant() -> None:
    router = ModelDrivenRouter(_distant_base(), drift_floor=0.60)

    decision = router.route(features(0.10))

    assert decision.bypass_tier3 is True
    assert decision.probability < 0.5


def test_model_driven_router_uses_default_drift_floor() -> None:
    router = ModelDrivenRouter(_distant_base())

    assert router.route(features(0.59, True)).bypass_tier3 is True
    assert router.route(features(0.60, True)).invoke_tier3 is True


def test_model_driven_router_validates_floors() -> None:
    base = cast(Tier3BypassRouter, _distant_base())
    with pytest.raises(ValueError, match="drift_floor"):
        ModelDrivenRouter(base, drift_floor=1.2)
    with pytest.raises(ValueError, match="drift_floor"):
        ModelDrivenRouter(base, drift_floor=-0.1)
    with pytest.raises(TypeError, match="features"):
        router = ModelDrivenRouter(base)
        router.route(cast(RouterFeatures, 1.0))


def features(similarity: float, ambiguous: bool = False) -> RouterFeatures:
    return RouterFeatures(
        regex_hit=False,
        top_similarity=similarity,
        top_gap=0.5,
        tier2_ambiguous=ambiguous,
        prompt_bytes=256,
    )


def samples() -> tuple[RouterSample, ...]:
    return (
        RouterSample(features(0.9, True), True),
        RouterSample(features(0.8, True), True),
        RouterSample(features(0.75, True), True),
        RouterSample(features(0.1, False), False),
        RouterSample(features(0.2, False), False),
        RouterSample(features(0.3, False), False),
    )


def test_logistic_router_learns_separable_signal() -> None:
    model = LogisticRegressionRouter.fit(
        samples(),
        epochs=300,
        learning_rate=0.5,
    )

    assert model.predict_proba(features(0.95, True)) > 0.5
    assert model.predict_proba(features(0.05, False)) < 0.5


def test_router_decision_and_bypass() -> None:
    model = LogisticRegressionRouter(
        weights=(0.0, 10.0, 0.0, 0.0, 0.0),
        bias=-5.0,
        threshold=0.5,
    )
    loop = Tier3BypassRouter(model)

    high = loop.route(features(1.0))
    low = loop.route(features(0.0))

    assert high.invoke_tier3 is True
    assert high.bypass_tier3 is False
    assert low.invoke_tier3 is False
    assert low.bypass_tier3 is True
    assert low.probability >= 0.0


def test_explicit_threshold_override() -> None:
    model = LogisticRegressionRouter(
        weights=(0.0, 0.0, 0.0, 0.0, 0.0),
        bias=0.0,
    )
    loop = Tier3BypassRouter(model, threshold=0.4)

    assert loop.route(features(0.0)).invoke_tier3 is True


def test_router_validation() -> None:
    with pytest.raises(ValueError, match="weights"):
        LogisticRegressionRouter(weights=(1.0,))
    with pytest.raises(ValueError, match="threshold"):
        LogisticRegressionRouter(threshold=2.0)
    with pytest.raises(ValueError, match="samples"):
        LogisticRegressionRouter.fit(())
    with pytest.raises(ValueError, match="epochs"):
        LogisticRegressionRouter.fit(samples(), epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LogisticRegressionRouter.fit(samples(), learning_rate=0.0)
    with pytest.raises(ValueError, match="l2"):
        LogisticRegressionRouter.fit(samples(), l2=2.0)


def test_feature_validation() -> None:
    with pytest.raises(ValueError, match="prompt_bytes"):
        RouterFeatures(False, 0.0, 0.0, False, -1)
    with pytest.raises(TypeError, match="regex_hit"):
        RouterFeatures(cast(bool, 1), 0.0, 0.0, False, 0)
    with pytest.raises(TypeError, match="tier2_ambiguous"):
        RouterFeatures(False, 0.0, 0.0, cast(bool, 1), 0)
    with pytest.raises(TypeError, match="RouterFeatures"):
        RouterSample(cast(RouterFeatures, object()), True)
    with pytest.raises(TypeError, match="tier3_changes_verdict"):
        RouterSample(features(0.1), cast(bool, 1))


def test_router_decision_validation() -> None:
    with pytest.raises(TypeError, match="invoke_tier3"):
        RouterDecision(cast(bool, 1), 0.5)
    with pytest.raises(ValueError, match="probability"):
        RouterDecision(False, 2.0)

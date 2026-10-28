from typing import cast

import pytest

from safety_platform.research.calibration import (
    CalibrationError,
    CalibrationMetrics,
    CalibrationPolicy,
    CalibrationSample,
    CalibrationSpace,
    CalibrationSplit,
    CalibrationSweep,
)


def sample(
    top: float,
    runner: float,
    confidence: float,
    attack: bool,
) -> CalibrationSample:
    return CalibrationSample(top, runner, confidence, attack)


def split() -> CalibrationSplit:
    return CalibrationSplit(
        train=(
            sample(0.9, 0.1, 0.9, True),
            sample(0.6, 0.1, 0.9, True),
            sample(0.55, 0.1, 0.8, True),
            sample(0.1, 0.0, 0.1, False),
            sample(0.6, 0.1, 0.3, False),
            sample(0.7, 0.1, 0.2, False),
        ),
        test=(
            sample(0.9, 0.1, 0.9, True),
            sample(0.1, 0.0, 0.1, False),
        ),
    )


def space() -> CalibrationSpace:
    return CalibrationSpace(
        block_similarity=(0.8,),
        ambiguity_floor=(0.5,),
        min_top_gap=(0.1,),
        tier3_block_confidence=(0.7,),
        tier3_gate_probability=0.4,
    )


def test_sweep_finds_policy_compliant_candidate() -> None:
    report = CalibrationSweep(split(), space()).run()

    assert report.evaluated == 1
    assert report.best.train.recall == pytest.approx(1.0)
    assert report.best.train.fpr == pytest.approx(0.0)
    assert report.best.train.tier3_rate == pytest.approx(2 / 6)
    assert report.best.test.recall == pytest.approx(1.0)
    assert report.best.test.fpr == pytest.approx(0.0)


def test_sweep_skips_invalid_floor_band() -> None:
    invalid = CalibrationSpace(
        block_similarity=(0.8,),
        ambiguity_floor=(0.8,),
        min_top_gap=(0.1,),
        tier3_block_confidence=(0.7,),
    )

    with pytest.raises(CalibrationError, match="no feasible"):
        CalibrationSweep(split(), invalid).run()


def test_sweep_rejects_no_feasible_candidate() -> None:
    false_positive = CalibrationSplit(
        train=(sample(0.95, 0.0, 0.9, False),),
        test=(sample(0.95, 0.0, 0.9, False),),
    )
    policy = CalibrationPolicy(max_fpr=0.0)

    with pytest.raises(CalibrationError, match="no feasible"):
        CalibrationSweep(false_positive, space(), policy).run()


def test_metrics_validate_confusion_matrix() -> None:
    with pytest.raises(ValueError, match="sum to total"):
        CalibrationMetrics(1, 0, 0, 0, 0, 2)
    with pytest.raises(ValueError, match="non-negative"):
        CalibrationMetrics(-1, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="cannot exceed"):
        CalibrationMetrics(1, 0, 0, 0, 2, 1)


def test_metrics_properties_handle_empty_classes() -> None:
    metrics = CalibrationMetrics(0, 0, 2, 0, 0, 2)

    assert metrics.recall == pytest.approx(1.0)
    assert metrics.fpr == pytest.approx(0.0)
    assert metrics.precision == pytest.approx(0.0)
    assert metrics.accuracy == pytest.approx(1.0)


def test_sample_and_split_validation() -> None:
    with pytest.raises(ValueError, match="top_similarity"):
        CalibrationSample(2.0, 0.0, 0.5, True)
    with pytest.raises(TypeError, match="is_attack"):
        CalibrationSample(0.5, 0.0, 0.5, cast(bool, 1))
    with pytest.raises(ValueError, match="train"):
        CalibrationSplit((), (sample(0.5, 0.0, 0.5, True),))
    with pytest.raises(ValueError, match="test"):
        CalibrationSplit((sample(0.5, 0.0, 0.5, True),), ())
    with pytest.raises(ValueError, match="grid"):
        CalibrationSpace(block_similarity=())
    with pytest.raises(ValueError, match="max_fpr"):
        CalibrationPolicy(max_fpr=2.0)

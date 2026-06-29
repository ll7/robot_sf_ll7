"""Tests for the near-parity promotion-gate decision (issue #3465)."""

from __future__ import annotations

import json
import math

import pytest

from robot_sf.benchmark.near_parity_promotion_gate import (
    DIAGNOSTIC,
    ELIGIBLE_FOR_PROMOTION,
    NEAR_PARITY_GATE_SCHEMA,
    REVISE,
    STOP,
    NearParityComparison,
    PromotionThresholds,
    classify_near_parity_promotion,
)


def _comparison(**overrides: object) -> NearParityComparison:
    """Build a synthetic paired near-parity comparison."""

    kwargs = {
        "safety_improvement": 0.0,
        "efficiency_improvement": 0.0,
        "paired_significant": True,
        "relies_on_fallback": False,
        "corrective_complete": True,
    }
    kwargs.update(overrides)
    return NearParityComparison(**kwargs)


def test_diagnostic_when_corrective_not_complete() -> None:
    """Without the #3463 corrective implementation, the verdict is diagnostic."""

    verdict = classify_near_parity_promotion(
        _comparison(corrective_complete=False, safety_improvement=0.5)
    )

    assert verdict["schema_version"] == NEAR_PARITY_GATE_SCHEMA
    assert verdict["verdict"] == DIAGNOSTIC
    assert verdict["promote"] is False
    assert verdict["claim_boundary"] == "promotion_gate_decision_only"
    assert "#3463" in verdict["rationale"]


def test_fallback_reliant_improvement_is_revise() -> None:
    """An improvement that relies on fallback/degraded rows must not count."""

    verdict = classify_near_parity_promotion(
        _comparison(safety_improvement=0.5, relies_on_fallback=True)
    )

    assert verdict["verdict"] == REVISE
    assert verdict["promote"] is False
    assert "fallback/degraded" in verdict["rationale"]


@pytest.mark.parametrize(
    ("safety_improvement", "efficiency_improvement"),
    [
        (-0.1, 0.0),
        (0.0, -0.1),
    ],
)
def test_regression_is_stop(
    safety_improvement: float,
    efficiency_improvement: float,
) -> None:
    """A measurable safety or efficiency regression must stop the lane."""

    verdict = classify_near_parity_promotion(
        _comparison(
            safety_improvement=safety_improvement,
            efficiency_improvement=efficiency_improvement,
        )
    )

    assert verdict["verdict"] == STOP
    assert verdict["promote"] is False


def test_significant_native_improvement_is_eligible_for_promotion() -> None:
    """A measurable, paired-significant native improvement is the only promote verdict."""

    verdict = classify_near_parity_promotion(
        _comparison(safety_improvement=0.05, paired_significant=True)
    )

    assert verdict["verdict"] == ELIGIBLE_FOR_PROMOTION
    assert verdict["promote"] is True
    assert verdict["promotable"] is True


@pytest.mark.parametrize(
    "comparison",
    [
        _comparison(),
        _comparison(safety_improvement=0.05, paired_significant=False),
        _comparison(safety_improvement=0.01, efficiency_improvement=0.01),
    ],
)
def test_no_measurable_or_non_significant_improvement_is_revise(
    comparison: NearParityComparison,
) -> None:
    """No measurable or non-significant difference keeps the lane at revise."""

    verdict = classify_near_parity_promotion(comparison)

    assert verdict["verdict"] == REVISE
    assert verdict["promote"] is False


def test_decision_payload_is_json_serializable() -> None:
    """The decision packet carries stable issue metadata and JSON-serializes."""

    verdict = classify_near_parity_promotion(_comparison(safety_improvement=0.05))

    assert verdict["issue"] == "3465"
    assert verdict["inputs"]["min_improvement"] == 0.02
    assert json.loads(json.dumps(verdict))["schema_version"] == NEAR_PARITY_GATE_SCHEMA


def test_non_finite_inputs_fail_closed_before_verdict() -> None:
    """NaN/Inf comparison values are rejected before classification."""

    with pytest.raises(ValueError, match="comparison.safety_improvement"):
        classify_near_parity_promotion(_comparison(safety_improvement=math.nan))


@pytest.mark.parametrize("min_improvement", [-0.01, 0.0])
def test_non_positive_threshold_is_rejected(min_improvement: float) -> None:
    """Invalid threshold configuration fails before any verdict.

    A zero threshold is rejected because it would collapse the regression and improvement
    boundaries onto exact equality and mislabel a no-difference comparison as ``stop``.
    """

    with pytest.raises(ValueError, match="min_improvement"):
        classify_near_parity_promotion(
            _comparison(safety_improvement=0.05),
            PromotionThresholds(min_improvement=min_improvement),
        )

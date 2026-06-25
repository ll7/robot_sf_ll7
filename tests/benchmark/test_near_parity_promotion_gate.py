"""Tests for the near-parity promotion-gate decision (issue #3465)."""

from __future__ import annotations

from robot_sf.benchmark.near_parity_promotion_gate import (
    DIAGNOSTIC,
    ELIGIBLE_FOR_PROMOTION,
    NEAR_PARITY_GATE_SCHEMA,
    REVISE,
    STOP,
    NearParityComparison,
    classify_near_parity_promotion,
)


def _comparison(**overrides) -> NearParityComparison:
    """Build a comparison with sensible defaults."""
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
    """Without the corrective implementation, the verdict is diagnostic (not runnable)."""
    verdict = classify_near_parity_promotion(
        _comparison(corrective_complete=False, safety_improvement=0.5)
    )

    assert verdict["schema_version"] == NEAR_PARITY_GATE_SCHEMA
    assert verdict["verdict"] == DIAGNOSTIC
    assert verdict["promote"] is False


def test_fallback_reliant_improvement_is_revise() -> None:
    """An improvement that relies on fallback/degraded rows must not count (fail-closed)."""
    verdict = classify_near_parity_promotion(
        _comparison(safety_improvement=0.5, relies_on_fallback=True)
    )

    assert verdict["verdict"] == REVISE
    assert verdict["promote"] is False


def test_regression_is_stop() -> None:
    """A measurable safety/efficiency regression must stop the lane."""
    verdict = classify_near_parity_promotion(_comparison(efficiency_improvement=-0.1))

    assert verdict["verdict"] == STOP


def test_significant_native_improvement_is_eligible_for_promotion() -> None:
    """A measurable, paired-significant native improvement is the only promote verdict."""
    verdict = classify_near_parity_promotion(
        _comparison(safety_improvement=0.05, paired_significant=True)
    )

    assert verdict["verdict"] == ELIGIBLE_FOR_PROMOTION
    assert verdict["promote"] is True


def test_improvement_without_significance_is_revise() -> None:
    """A non-significant improvement must not be promoted."""
    verdict = classify_near_parity_promotion(
        _comparison(safety_improvement=0.05, paired_significant=False)
    )

    assert verdict["verdict"] == REVISE
    assert verdict["promote"] is False


def test_no_difference_is_revise() -> None:
    """No measurable difference keeps the lane at revise."""
    verdict = classify_near_parity_promotion(_comparison())

    assert verdict["verdict"] == REVISE

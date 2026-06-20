"""Tests for the counterfactual scenario-pair evaluation core (issue #3239, child of #2924)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.counterfactual_pair import (
    COUNTERFACTUAL_PAIR_SCHEMA,
    VERDICT_FALSIFIED,
    VERDICT_INCONCLUSIVE,
    VERDICT_SURVIVED,
    PairHypothesis,
    evaluate_counterfactual_pair,
)

_HYP = PairHypothesis(
    expected_mechanism="predictive_risk",
    outcome_metric="min_clearance",
    expected_direction="increase",
)


def _result(activated: bool, clearance: float) -> dict:
    return {"mechanism_activated": activated, "metrics": {"min_clearance": clearance}}


def test_hypothesis_rejects_bad_direction() -> None:
    """The hypothesis validates its expected direction."""
    with pytest.raises(ValueError, match="expected_direction"):
        PairHypothesis(expected_mechanism="m", outcome_metric="x", expected_direction="sideways")


def test_survived_when_activated_and_direction_matches() -> None:
    """Activation plus the predicted increase yields a survived verdict."""
    result = evaluate_counterfactual_pair(_result(False, 0.30), _result(True, 0.55), _HYP)
    assert result.verdict == VERDICT_SURVIVED
    assert result.intervention_activated is True
    assert result.activation_delta == 1
    assert result.outcome_delta == pytest.approx(0.25)


def test_falsified_when_activated_but_wrong_direction() -> None:
    """Activation with the opposite outcome movement is falsified."""
    result = evaluate_counterfactual_pair(_result(False, 0.50), _result(True, 0.30), _HYP)
    assert result.verdict == VERDICT_FALSIFIED
    assert result.outcome_delta == pytest.approx(-0.20)


def test_falsified_when_activated_but_no_change() -> None:
    """Activation with no meaningful outcome change is falsified (predicted effect absent)."""
    result = evaluate_counterfactual_pair(_result(False, 0.40), _result(True, 0.40), _HYP)
    assert result.verdict == VERDICT_FALSIFIED
    assert result.outcome_delta == 0.0


def test_inconclusive_when_mechanism_not_activated() -> None:
    """No activation in the intervention is inconclusive (fail-closed), never survived."""
    result = evaluate_counterfactual_pair(_result(False, 0.30), _result(False, 0.55), _HYP)
    assert result.verdict == VERDICT_INCONCLUSIVE
    assert result.intervention_activated is False


def test_min_outcome_delta_threshold() -> None:
    """A sub-threshold change does not count as the predicted direction."""
    small = evaluate_counterfactual_pair(
        _result(False, 0.40), _result(True, 0.41), _HYP, min_outcome_delta=0.05
    )
    assert small.verdict == VERDICT_FALSIFIED  # +0.01 < 0.05 threshold
    big = evaluate_counterfactual_pair(
        _result(False, 0.40), _result(True, 0.50), _HYP, min_outcome_delta=0.05
    )
    assert big.verdict == VERDICT_SURVIVED  # +0.10 > 0.05 threshold


def test_min_outcome_delta_none_uses_zero_threshold() -> None:
    """An omitted threshold sentinel uses the same zero threshold as the default."""
    result = evaluate_counterfactual_pair(
        _result(False, 0.40), _result(True, 0.41), _HYP, min_outcome_delta=None
    )
    assert result.verdict == VERDICT_SURVIVED
    assert result.outcome_delta == pytest.approx(0.01)


def test_negative_min_outcome_delta_raises() -> None:
    """Negative thresholds are rejected instead of changing comparison semantics."""
    with pytest.raises(ValueError, match="min_outcome_delta must be non-negative"):
        evaluate_counterfactual_pair(
            _result(False, 0.30), _result(True, 0.50), _HYP, min_outcome_delta=-0.1
        )


def test_decrease_direction() -> None:
    """A decrease hypothesis survives when the metric drops."""
    hyp = PairHypothesis(
        expected_mechanism="shielding", outcome_metric="collisions", expected_direction="decrease"
    )
    baseline = {"mechanism_activated": False, "metrics": {"collisions": 5.0}}
    intervention = {"mechanism_activated": True, "metrics": {"collisions": 2.0}}
    result = evaluate_counterfactual_pair(baseline, intervention, hyp)
    assert result.verdict == VERDICT_SURVIVED
    assert result.outcome_delta == pytest.approx(-3.0)


def test_missing_fields_raise() -> None:
    """Missing activation flag or metric is a clear error."""
    with pytest.raises(ValueError, match="mechanism_activated"):
        evaluate_counterfactual_pair({"metrics": {"min_clearance": 0.3}}, _result(True, 0.5), _HYP)
    with pytest.raises(ValueError, match="min_clearance"):
        evaluate_counterfactual_pair(
            {"mechanism_activated": False, "metrics": {}}, _result(True, 0.5), _HYP
        )


def test_non_finite_metric_raises() -> None:
    """Non-finite metric values are rejected before verdict calculation."""
    with pytest.raises(ValueError, match="not finite"):
        evaluate_counterfactual_pair(_result(False, float("nan")), _result(True, 0.5), _HYP)
    with pytest.raises(ValueError, match="not finite"):
        evaluate_counterfactual_pair(_result(False, 0.3), _result(True, float("inf")), _HYP)


def test_to_dict_schema() -> None:
    """The payload carries the counterfactual_pair_result.v1 schema."""
    payload = evaluate_counterfactual_pair(_result(False, 0.3), _result(True, 0.5), _HYP).to_dict()
    assert payload["schema_version"] == COUNTERFACTUAL_PAIR_SCHEMA
    assert payload["verdict"] == VERDICT_SURVIVED
    assert payload["hypothesis"]["outcome_metric"] == "min_clearance"

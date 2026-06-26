"""Counterfactual scenario-pair evaluation core (bounded #2924 child).

Given a *baseline* result and a matched *intervention* result for the same
scenario/seed/planner — each carrying a mechanism-activation signal and outcome
metrics — decide whether the expected mechanism hypothesis **survived**, was
**falsified**, or is **inconclusive**.

Fail-closed: a hypothesis can only *survive* when the mechanism actually activated
in the intervention. No activation yields ``inconclusive`` (the predicted
mechanism never fired, so the pair cannot test it), never ``survived``.

Pure and deterministic: it consumes already-measured results and runs no
simulation. This is analysis tooling and makes no benchmark claim. Running the
scenario pair and generating traces lives with parent issue #2924; pair-manifest
*creation* is covered by ``scripts/tools/create_counterfactual_scenario_pair.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.finite_checks import require_finite_scalar

COUNTERFACTUAL_PAIR_SCHEMA = "counterfactual_pair_result.v1"

VERDICT_SURVIVED = "survived"
VERDICT_FALSIFIED = "falsified"
VERDICT_INCONCLUSIVE = "inconclusive"

DIRECTION_INCREASE = "increase"
DIRECTION_DECREASE = "decrease"
_DIRECTIONS = (DIRECTION_INCREASE, DIRECTION_DECREASE)


@dataclass(frozen=True)
class PairHypothesis:
    """The mechanism hypothesis a counterfactual pair is designed to test.

    Attributes:
        expected_mechanism: Name of the mechanism expected to activate under the
            intervention (recorded for provenance).
        outcome_metric: Outcome metric whose change tests the hypothesis.
        expected_direction: ``"increase"`` or ``"decrease"`` of ``outcome_metric``
            in the intervention relative to the baseline.
    """

    expected_mechanism: str
    outcome_metric: str
    expected_direction: str

    def __post_init__(self) -> None:
        """Validate the expected direction."""
        if self.expected_direction not in _DIRECTIONS:
            raise ValueError(
                f"expected_direction must be one of {_DIRECTIONS} (got {self.expected_direction!r})"
            )


@dataclass(frozen=True)
class PairResult:
    """Outcome of evaluating one counterfactual scenario pair."""

    hypothesis: PairHypothesis
    baseline_activated: bool
    intervention_activated: bool
    activation_delta: int
    outcome_baseline: float
    outcome_intervention: float
    outcome_delta: float
    verdict: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the ``counterfactual_pair_result.v1`` JSON-safe payload.

        Returns:
            Mapping with the schema version, hypothesis, activation/outcome
            deltas, verdict, and reason.
        """
        return {
            "schema_version": COUNTERFACTUAL_PAIR_SCHEMA,
            "hypothesis": {
                "expected_mechanism": self.hypothesis.expected_mechanism,
                "outcome_metric": self.hypothesis.outcome_metric,
                "expected_direction": self.hypothesis.expected_direction,
            },
            "baseline_activated": self.baseline_activated,
            "intervention_activated": self.intervention_activated,
            "activation_delta": self.activation_delta,
            "outcome_baseline": self.outcome_baseline,
            "outcome_intervention": self.outcome_intervention,
            "outcome_delta": self.outcome_delta,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def _activated(result: Mapping[str, Any]) -> bool:
    """Read the mechanism-activation flag from a result mapping.

    Returns:
        The boolean ``mechanism_activated`` flag.
    """
    if "mechanism_activated" not in result:
        raise ValueError("result is missing required 'mechanism_activated' flag")
    return bool(result["mechanism_activated"])


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    """Read a numeric outcome metric from a result mapping's ``metrics``.

    Returns:
        The float value of ``metric`` from the result's ``metrics`` mapping.
    """
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or metric not in metrics:
        raise ValueError(f"result.metrics is missing required metric {metric!r}")
    return require_finite_scalar(f"metric {metric!r} value", metrics[metric])


def _direction_matches(delta: float, expected_direction: str, tolerance: float) -> bool:
    """Return whether an outcome delta moves in the expected direction beyond tolerance.

    Returns:
        ``True`` when ``delta`` exceeds ``tolerance`` in the expected direction.
    """
    if expected_direction == DIRECTION_INCREASE:
        return delta > tolerance
    return delta < -tolerance


def evaluate_counterfactual_pair(
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    hypothesis: PairHypothesis,
    *,
    min_outcome_delta: float | None = None,
) -> PairResult:
    """Evaluate a baseline/intervention pair against a mechanism hypothesis.

    Each result mapping must provide a ``mechanism_activated`` flag and a
    ``metrics`` mapping containing ``hypothesis.outcome_metric``.

    Verdict rules (fail-closed):

    * intervention did **not** activate the mechanism → ``inconclusive``;
    * activated and the outcome moved in the expected direction beyond
      ``min_outcome_delta`` → ``survived``;
    * activated but the outcome did not move as predicted → ``falsified``.

    Returns:
        A :class:`PairResult` with the activation/outcome deltas and verdict.
    """
    if min_outcome_delta is None:
        min_outcome_delta = 0.0
    elif min_outcome_delta < 0.0:
        raise ValueError(f"min_outcome_delta must be non-negative (got {min_outcome_delta})")

    baseline_activated = _activated(baseline)
    intervention_activated = _activated(intervention)
    outcome_baseline = _metric_value(baseline, hypothesis.outcome_metric)
    outcome_intervention = _metric_value(intervention, hypothesis.outcome_metric)
    outcome_delta = outcome_intervention - outcome_baseline
    activation_delta = int(intervention_activated) - int(baseline_activated)

    if not intervention_activated:
        verdict = VERDICT_INCONCLUSIVE
        reason = (
            f"expected mechanism {hypothesis.expected_mechanism!r} did not activate in the "
            "intervention; the pair cannot test the hypothesis"
        )
    elif _direction_matches(outcome_delta, hypothesis.expected_direction, min_outcome_delta):
        verdict = VERDICT_SURVIVED
        reason = (
            f"mechanism activated and {hypothesis.outcome_metric} moved "
            f"{hypothesis.expected_direction} by {outcome_delta:+.4g}"
        )
    else:
        verdict = VERDICT_FALSIFIED
        reason = (
            f"mechanism activated but {hypothesis.outcome_metric} did not move "
            f"{hypothesis.expected_direction} (delta {outcome_delta:+.4g}, "
            f"threshold {min_outcome_delta:g})"
        )

    return PairResult(
        hypothesis=hypothesis,
        baseline_activated=baseline_activated,
        intervention_activated=intervention_activated,
        activation_delta=activation_delta,
        outcome_baseline=outcome_baseline,
        outcome_intervention=outcome_intervention,
        outcome_delta=outcome_delta,
        verdict=verdict,
        reason=reason,
    )

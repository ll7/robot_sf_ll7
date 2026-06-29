"""Promotion-gate decision for the topology near-parity lane (issue #3465).

#2540 isolated the topology near-parity selector as diagnostic-only and classified it ``revise``.
The formal successor compares the gate **enabled vs disabled** on the same scenario/seed contract
and decides whether the lane stays diagnostic, revises, stops, or becomes eligible for stack
promotion.  This pure decision layer turns paired comparison results into that verdict and fails
closed about fallback/degraded execution.

The paired benchmark run itself needs the #3463 corrective work plus cluster execution and is
deferred; this layer is side-effect free and does not promote benchmark claims on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.finite_checks import require_finite_fields

NEAR_PARITY_GATE_SCHEMA = "near_parity_promotion_gate.v1"
NEAR_PARITY_PROMOTION_GATE_SCHEMA = NEAR_PARITY_GATE_SCHEMA

DIAGNOSTIC = "diagnostic"
REVISE = "revise"
STOP = "stop"
ELIGIBLE_FOR_PROMOTION = "eligible_for_promotion"


@dataclass(frozen=True, slots=True)
class NearParityComparison:
    """Paired gate-enabled-vs-disabled comparison results for the near-parity lane.

    Improvements are ``enabled - disabled`` on native execution; positive means the gate helped.

    Attributes:
        safety_improvement: Safety-metric improvement (positive = safer with the gate on).
        efficiency_improvement: Efficiency/progress improvement (positive = better with gate on).
        paired_significant: Whether the improvement is paired-statistically significant.
        relies_on_fallback: Whether improvement depends on fallback/degraded rows.
        corrective_complete: Whether the #3463 corrective implementation is complete or waived.
    """

    safety_improvement: float
    efficiency_improvement: float
    paired_significant: bool
    relies_on_fallback: bool
    corrective_complete: bool


@dataclass(frozen=True, slots=True)
class PromotionThresholds:
    """Minimum effect size for a measurable near-parity improvement/regression.

    ``min_improvement`` must be strictly positive: a zero minimum effect size collapses the
    regression and improvement boundaries onto exact equality and would mislabel a no-difference
    comparison as ``stop``.
    """

    min_improvement: float = 0.02


NearParityPromotionThresholds = PromotionThresholds


def classify_near_parity_promotion(
    comparison: NearParityComparison,
    thresholds: PromotionThresholds | None = None,
) -> dict[str, Any]:
    """Record the near-parity promotion-gate verdict from a paired comparison.

    Decision order (first match wins):

    1. corrective work not complete -> ``diagnostic`` (the formal test is not yet runnable);
    2. improvement relies on fallback/degraded rows -> ``revise`` (fail-closed, it does not count);
    3. a measurable safety or efficiency regression -> ``stop``;
    4. a measurable, paired-significant safety or efficiency improvement ->
       ``eligible_for_promotion`` (the only promote verdict);
    5. otherwise (no measurable / non-significant difference) -> ``revise``.

    Returns:
        Versioned verdict with promote flag, claim boundary, inputs, and rationale.
    """

    thresholds = thresholds or PromotionThresholds()
    _validate_inputs(comparison, thresholds)
    verdict, rationale = _decide(comparison, thresholds)
    promotable = verdict == ELIGIBLE_FOR_PROMOTION
    return {
        "schema_version": NEAR_PARITY_GATE_SCHEMA,
        "issue": "3465",
        "evidence_kind": "diagnostic_proxy",
        "claim_boundary": "promotion_gate_decision_only",
        "verdict": verdict,
        "promote": promotable,
        "promotable": promotable,
        "rationale": rationale,
        "inputs": {
            "safety_improvement": comparison.safety_improvement,
            "efficiency_improvement": comparison.efficiency_improvement,
            "paired_significant": comparison.paired_significant,
            "relies_on_fallback": comparison.relies_on_fallback,
            "corrective_complete": comparison.corrective_complete,
            "min_improvement": thresholds.min_improvement,
        },
    }


def _validate_inputs(comparison: NearParityComparison, thresholds: PromotionThresholds) -> None:
    """Reject invalid decision inputs before any promotable verdict."""

    require_finite_fields(
        "comparison",
        comparison,
        ("safety_improvement", "efficiency_improvement"),
    )
    require_finite_fields("thresholds", thresholds, ("min_improvement",))
    if thresholds.min_improvement <= 0.0:
        raise ValueError("thresholds.min_improvement must be positive")


def _decide(c: NearParityComparison, t: PromotionThresholds) -> tuple[str, str]:
    """Return the (verdict, rationale) for a comparison.

    Returns:
        Verdict string and human-readable rationale.
    """

    if not c.corrective_complete:
        return (
            DIAGNOSTIC,
            "Corrective implementation (#3463) not complete; formal test not runnable.",
        )
    if c.relies_on_fallback:
        return REVISE, "Improvement relies on fallback/degraded rows; fail-closed, does not count."
    regression = (
        c.safety_improvement <= -t.min_improvement or c.efficiency_improvement <= -t.min_improvement
    )
    if regression:
        return STOP, "Gate-enabled shows a measurable safety/efficiency regression."
    improvement = (
        c.safety_improvement >= t.min_improvement or c.efficiency_improvement >= t.min_improvement
    )
    if improvement and c.paired_significant:
        return (
            ELIGIBLE_FOR_PROMOTION,
            "Measurable, paired-significant native improvement without fallback reliance.",
        )
    return REVISE, "No measurable or non-significant difference; remain diagnostic / revise."


__all__ = [
    "DIAGNOSTIC",
    "ELIGIBLE_FOR_PROMOTION",
    "NEAR_PARITY_GATE_SCHEMA",
    "NEAR_PARITY_PROMOTION_GATE_SCHEMA",
    "REVISE",
    "STOP",
    "NearParityComparison",
    "NearParityPromotionThresholds",
    "PromotionThresholds",
    "classify_near_parity_promotion",
]

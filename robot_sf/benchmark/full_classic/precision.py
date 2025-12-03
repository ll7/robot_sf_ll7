"""Precision / statistical sufficiency evaluation (T033).

Produces a StatisticalSufficiencyReport indicating per-(archetype,density) metric
precision status and overall pass flag. This is a simplified implementation
consistent with current test expectations; adaptive loop integration (T034)
will consume this to decide early stopping.

Approach:
  * For each group (AggregateMetricsGroup) extract rate metrics' Wilson intervals
    already stored as mean_ci and compute half-width.
  * Map metric names to configured precision targets (collision_ci, success_ci, etc.).
  * Determine pass if half_width <= target (absolute) or for continuous metrics with
    relative targets apply relative formula when target expressed as fraction.
  * For this phase, only collision_rate and success_rate likely used in tests; extendable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RATE_TARGET_MAP = {
    "collision_rate": "collision_ci",
    "success_rate": "success_ci",
}


@dataclass
class PrecisionEntry:
    """PrecisionEntry class."""

    metric: str
    half_width: float
    target: float
    passed: bool


@dataclass
class ScenarioPrecisionStatus:
    """ScenarioPrecisionStatus class."""

    scenario_id: str  # For aggregated group we synthesize id archetype-density
    archetype: str
    density: str
    episodes: int
    metric_status: list[PrecisionEntry]
    all_pass: bool


@dataclass
class StatisticalSufficiencyReport:
    """StatisticalSufficiencyReport class."""

    evaluations: list[ScenarioPrecisionStatus]
    final_pass: bool
    scaling_efficiency: dict  # placeholder, populated later (T041)


def evaluate_precision(groups, cfg):  # T033
    """Evaluate precision.

    Args:
        groups: Collection of grouped elements.
        cfg: Configuration dictionary.

    Returns:
        Any: Arbitrary value passed through unchanged.
    """
    evaluations: list[ScenarioPrecisionStatus] = []
    for g in groups:
        entries: list[PrecisionEntry] = []
        for metric_name, attr_name in RATE_TARGET_MAP.items():
            metric_obj = g.metrics.get(metric_name)
            if not metric_obj or not metric_obj.mean_ci:
                continue
            ci_low, ci_high = metric_obj.mean_ci
            half_width = (ci_high - ci_low) / 2.0
            target = float(getattr(cfg, attr_name, math.inf))
            passed = half_width <= target
            entries.append(
                PrecisionEntry(
                    metric=metric_name,
                    half_width=half_width,
                    target=target,
                    passed=passed,
                ),
            )
        all_pass = bool(entries) and all(e.passed for e in entries)
        evaluations.append(
            ScenarioPrecisionStatus(
                scenario_id=f"{g.archetype}-{g.density}",
                archetype=g.archetype,
                density=g.density,
                episodes=g.count,
                metric_status=entries,
                all_pass=all_pass,
            ),
        )
    final_pass = bool(evaluations) and all(ev.all_pass for ev in evaluations)
    report = StatisticalSufficiencyReport(
        evaluations=evaluations,
        final_pass=final_pass,
        scaling_efficiency={},  # to be populated later
    )
    return report

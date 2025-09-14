"""Canonical SNQI computation helpers.

The SNQI (Social Navigation Quality Index) combines task performance and
social safety/comfort metrics into a single scalar score:

    SNQI = w_success * success
           - w_time * time_norm
           - w_collisions * coll_norm
           - w_near * near_norm
           - w_comfort * comfort_exposure
           - w_force_exceed * force_exceed_norm
           - w_jerk * jerk_norm

Normalization for metrics using baseline stats (median / p95):

    norm = (value - med) / (p95 - med)   (clamped to [0,1])

This module is intentionally lightweight so it can be imported by scripts
and tests without pulling heavy dependencies.
"""

from __future__ import annotations

from typing import Mapping

WEIGHT_NAMES = [
    "w_success",
    "w_time",
    "w_collisions",
    "w_near",
    "w_comfort",
    "w_force_exceed",
    "w_jerk",
]

MetricName = str
BaselineStats = Mapping[MetricName, Mapping[str, float]]
Weights = Mapping[str, float]
Metrics = Mapping[str, float | int | bool]


def normalize_metric(name: str, value: float | int | bool, baseline_stats: BaselineStats) -> float:
    """Normalize a raw metric using median/p95 baseline statistics.

    If the metric is missing in the baseline statistics, returns 0.0.
    Division-by-zero is guarded by using a denominator of 1.0 when p95==med.
    Result is clamped to [0,1] to prevent extreme outliers from dominating.
    """
    if name not in baseline_stats:
        return 0.0
    med = float(baseline_stats[name].get("med", 0.0))
    p95 = float(baseline_stats[name].get("p95", med))
    denom = p95 - med
    if abs(denom) < 1e-9:
        denom = 1.0
    norm = (float(value) - med) / denom
    if norm < 0.0:
        return 0.0
    if norm > 1.0:
        return 1.0
    return norm


def compute_snqi(metrics: Metrics, weights: Weights, baseline_stats: BaselineStats) -> float:
    """Compute the SNQI score for a single episode.

    Args:
        metrics: Episode metrics mapping.
        weights: Weight coefficients for each component.
        baseline_stats: Baseline normalization statistics (median/p95) for
            collision-like metrics.

    Returns:
        SNQI score (higher is better).
    """
    success_raw = metrics.get("success", 0.0)
    success = 1.0 if isinstance(success_raw, bool) and success_raw else float(success_raw)

    time_norm = float(metrics.get("time_to_goal_norm", 1.0))
    coll_norm = normalize_metric("collisions", metrics.get("collisions", 0.0), baseline_stats)
    near_norm = normalize_metric("near_misses", metrics.get("near_misses", 0.0), baseline_stats)
    comfort_exposure = float(metrics.get("comfort_exposure", 0.0))
    force_exceed_norm = normalize_metric(
        "force_exceed_events", metrics.get("force_exceed_events", 0.0), baseline_stats
    )
    jerk_norm = normalize_metric("jerk_mean", metrics.get("jerk_mean", 0.0), baseline_stats)

    score = (
        weights.get("w_success", 1.0) * success
        - weights.get("w_time", 1.0) * time_norm
        - weights.get("w_collisions", 1.0) * coll_norm
        - weights.get("w_near", 1.0) * near_norm
        - weights.get("w_comfort", 1.0) * comfort_exposure
        - weights.get("w_force_exceed", 1.0) * force_exceed_norm
        - weights.get("w_jerk", 1.0) * jerk_norm
    )
    return float(score)


__all__ = ["compute_snqi", "normalize_metric", "WEIGHT_NAMES"]

"""Metrics constants & helpers (T045).

Defines canonical public metric name sets and simple validation helpers so
callers (aggregation, table generation) can assert presence / filter keys
consistently without re-specifying strings across modules.
"""

from __future__ import annotations

CORE_METRICS = [
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "mean_distance",
    "path_efficiency",
    "avg_speed",
    "force_q50",
    "force_q90",
    "force_q95",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "curvature_mean",
    "energy",
    "force_gradient_norm_mean",
]

# v1 reporting groups (core/conditional/optional)
V1_CORE_METRICS = [
    "success",
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "mean_distance",
    "path_efficiency",
    "avg_speed",
    "jerk_mean",
    "curvature_mean",
    "energy",
    "force_q50",
    "force_q90",
    "force_q95",
    "force_exceed_events",
    "comfort_exposure",
]

V1_CONDITIONAL_METRICS = [
    "wall_collisions",
    "clearing_distance_min",
    "clearing_distance_avg",
]

V1_OPTIONAL_METRICS = [
    "force_gradient_norm_mean",
]


def validate_metric_names(metric_dict: dict[str, float]) -> list[str]:
    """Return sorted list of recognized metric names present in dict.

    Unrecognized keys are ignored (future-proofing). This function does not
    raise; callers wanting strictness can compare lengths externally.

    Returns:
        Sorted list of metric names from CORE_METRICS found in metric_dict.
    """

    present = [m for m in CORE_METRICS if m in metric_dict]
    return sorted(present)


__all__ = [
    "CORE_METRICS",
    "V1_CONDITIONAL_METRICS",
    "V1_CORE_METRICS",
    "V1_OPTIONAL_METRICS",
    "validate_metric_names",
]

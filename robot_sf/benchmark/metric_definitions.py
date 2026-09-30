"""Metric meaning versions, independent of the episode JSON envelope version.

Unmarked historical rows/assets use v1 (including published release 0.0.7).
Version v2 fixes issue #10007 F4/F5/F7/F8; it requires fresh normalization.
"""

from collections.abc import Iterable, Mapping
from typing import Any

LEGACY_METRIC_SCHEMA_VERSION = "robot-sf-metrics.v1"
METRIC_SCHEMA_VERSION = "robot-sf-metrics.v2"
CHANGED_METRICS = frozenset(
    {
        "path_efficiency",
        "path_length",
        "success_path_length",
        "socnavbench_path_length",
        "socnavbench_path_length_ratio",
        "socnavbench_path_irregularity",
        "failure_to_progress",
        "deadlock",
        "deadlock_stall",
        "jerk_mean",
        "time_to_goal",
        "time_to_goal_norm",
        "time_to_goal_norm_success_only",
        "time_to_goal_ideal_ratio",
        "aggregated_time",
        "snqi",
        "snqi_v1",
        "snqi_v2",
        "snqi_v2_terms",
        "social_mini_game",
        "shortest_path_len",
    }
)


def metric_schema_version(value: Mapping[str, Any]) -> str:
    """Resolve row, metric mapping, or anchor document version; reject contradictions.

    Returns:
        Known version slug, defaulting to the historical definition for unmarked inputs."""
    versions = [value["metric_schema_version"]] if "metric_schema_version" in value else []
    for key in ("metrics", "_metadata"):
        child = value.get(key)
        if isinstance(child, Mapping) and "metric_schema_version" in child:
            versions.append(child["metric_schema_version"])
    if not versions:
        return LEGACY_METRIC_SCHEMA_VERSION
    if any(v not in (LEGACY_METRIC_SCHEMA_VERSION, METRIC_SCHEMA_VERSION) for v in versions):
        raise ValueError(f"unsupported metric_schema_version: {versions}")
    if len(set(versions)) != 1:
        raise ValueError(f"contradictory metric_schema_version: {versions}")
    return versions[0]


def require_uniform_metric_schema(records: Iterable[Mapping[str, Any]]) -> str:
    """Reject pooled changed definitions before aggregation or calibration.

    Returns:
        Common schema version, or the legacy default for an empty collection."""
    versions = {metric_schema_version(record) for record in records}
    if len(versions) > 1:
        raise ValueError(
            f"incompatible metric definitions: {sorted(versions)}; recompute from traces"
        )
    return next(iter(versions), LEGACY_METRIC_SCHEMA_VERSION)


def require_anchor_compatibility(metrics: Mapping[str, Any], anchors: Mapping[str, Any]) -> None:
    """Never normalize current physical jerk/time with historical anchors silently."""
    observed, expected = metric_schema_version(metrics), metric_schema_version(anchors)
    if observed != expected:
        raise ValueError(
            f"incompatible metric definitions: episodes={observed}, anchors={expected}; "
            "derive new anchors from matching-schema episodes (do not relabel old assets)"
        )


def changed_metric_field(field: str) -> bool:
    """Recognize flattened metric fields whose definition or composite inputs changed.

    Returns:
        Whether a field depends on one of the corrected definitions."""
    return field.startswith("metrics.") and field.split(".")[1] in CHANGED_METRICS

"""Canonical benchmark metric-layer contract and aggregation adapter.

This module organizes existing episode-record metrics into a constraints-first
layer hierarchy. It does not produce new simulator metrics; missing values are
reported as unavailable rather than imputed as zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from robot_sf.benchmark.aggregate import flatten_metrics

METRIC_LAYER_SCHEMA_VERSION = "metric-layers.v1"
MISSING_METRIC_REASON = "metric_not_present_in_episode_records"

LAYER_ORDER = (
    "safety_gate",
    "liveness",
    "social_compliance",
    "efficiency",
    "comfort",
    "operational",
)


@dataclass(frozen=True, slots=True)
class MetricLayerDefinition:
    """One named layer in the constraints-first metric hierarchy."""

    name: str
    priority: int
    description: str


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Canonical metadata for one layered benchmark metric."""

    name: str
    layer: str
    source_keys: tuple[str, ...]
    reduction: str
    higher_is_better: bool | None
    description: str
    unavailable_reason_if_missing: str = MISSING_METRIC_REASON
    source_kind: str = "episode_metric"


CANONICAL_METRIC_LAYERS: tuple[MetricLayerDefinition, ...] = (
    MetricLayerDefinition(
        name="safety_gate",
        priority=0,
        description="Hard safety outcomes that should gate downstream comparisons.",
    ),
    MetricLayerDefinition(
        name="liveness",
        priority=1,
        description="Progress and deadlock indicators after safety gates pass.",
    ),
    MetricLayerDefinition(
        name="social_compliance",
        priority=2,
        description="Social-navigation compliance signals, including explicit proxy metrics.",
    ),
    MetricLayerDefinition(
        name="efficiency",
        priority=3,
        description="Route completion efficiency and path-quality metrics.",
    ),
    MetricLayerDefinition(
        name="comfort",
        priority=4,
        description="Motion smoothness and control-comfort metrics.",
    ),
    MetricLayerDefinition(
        name="operational",
        priority=5,
        description="Human assistance, shielding, and operational-intervention indicators.",
    ),
)


def _metric(
    name: str,
    layer: str,
    source_keys: tuple[str, ...],
    reduction: str,
    higher_is_better: bool | None,
    description: str,
    *,
    source_kind: str = "episode_metric",
) -> tuple[str, MetricDefinition]:
    """Build a registry entry keyed by canonical metric name.

    Returns:
        ``(name, definition)`` pair for deterministic registry construction.
    """

    return (
        name,
        MetricDefinition(
            name=name,
            layer=layer,
            source_keys=source_keys,
            reduction=reduction,
            higher_is_better=higher_is_better,
            description=description,
            source_kind=source_kind,
        ),
    )


CANONICAL_METRICS: dict[str, MetricDefinition] = dict(
    (
        _metric(
            "collision_rate",
            "safety_gate",
            (
                "metrics.collision_rate",
                "metrics.collisions",
                "outcome.collision_event",
            ),
            "episode_rate",
            False,
            "Episode collision rate, using stored rate or a collision-event derivation.",
        ),
        _metric(
            "human_collision_rate",
            "safety_gate",
            ("metrics.human_collision_rate",),
            "episode_rate",
            False,
            "Human-collision episode rate when explicitly produced.",
        ),
        _metric(
            "near_miss_rate",
            "safety_gate",
            ("metrics.near_miss_rate", "metrics.near_misses"),
            "episode_rate",
            False,
            "Near-miss rate or near-miss event count reduced to episode rate.",
        ),
        _metric(
            "min_time_to_collision",
            "safety_gate",
            ("metrics.min_time_to_collision", "metrics.time_to_collision_min"),
            "minimum",
            True,
            "Minimum time-to-collision when directly available.",
        ),
        _metric(
            "minimum_pedestrian_distance",
            "safety_gate",
            (
                "metrics.minimum_pedestrian_distance",
                "metrics.min_pedestrian_distance",
            ),
            "minimum",
            True,
            "Minimum distance to pedestrians when directly available.",
        ),
        _metric(
            "timeout_rate",
            "liveness",
            ("outcome.timeout_event", "termination_reason"),
            "episode_rate",
            False,
            "Timeout episode rate from outcome metadata or known timeout termination reasons.",
        ),
        _metric(
            "stopped_time_ratio",
            "liveness",
            ("metrics.stopped_time_ratio",),
            "mean",
            False,
            "Fraction of episode time stopped when explicitly produced.",
        ),
        _metric(
            "failure_to_progress_rate",
            "liveness",
            ("outcome.route_complete",),
            "episode_rate",
            False,
            "Non-completion rate excluding collision and timeout episodes when outcomes exist.",
        ),
        _metric(
            "deadlock_duration",
            "liveness",
            ("metrics.deadlock_duration", "metrics.deadlock_duration_s"),
            "mean",
            False,
            "Deadlock duration when explicitly produced.",
        ),
        _metric(
            "personal_space_violation_rate",
            "social_compliance",
            (
                "metrics.personal_space_violation_rate",
                "metrics.social_proxemic_intrusion_frac",
            ),
            "mean",
            False,
            "Personal-space violation rate; social-proxemic fields are simulation proxies.",
            source_kind="simulation_proxy",
        ),
        _metric(
            "group_intrusion_episode_rate",
            "social_compliance",
            ("metrics.group_intrusion_episode_rate",),
            "episode_rate",
            False,
            "Group-intrusion episode rate when explicitly produced.",
        ),
        _metric(
            "group_intrusion_time_ratio",
            "social_compliance",
            ("metrics.group_intrusion_time_ratio",),
            "mean",
            False,
            "Group-intrusion time ratio when explicitly produced.",
        ),
        _metric(
            "pedestrian_path_deviation",
            "social_compliance",
            (
                "metrics.pedestrian_path_deviation",
                "metrics.pedestrian_path_deviation_proxy_m",
            ),
            "mean",
            False,
            "Pedestrian path deviation; proxy fields are simulation proxies.",
            source_kind="simulation_proxy",
        ),
        _metric(
            "time_to_goal",
            "efficiency",
            ("metrics.time_to_goal", "metrics.time_to_goal_s"),
            "mean",
            False,
            "Time to route goal when explicitly produced.",
        ),
        _metric(
            "path_length",
            "efficiency",
            ("metrics.path_length", "metrics.path_length_m"),
            "mean",
            False,
            "Path length when explicitly produced.",
        ),
        _metric(
            "spl",
            "efficiency",
            ("metrics.SPL", "metrics.spl"),
            "mean",
            True,
            "Success weighted by path length when explicitly produced.",
        ),
        _metric(
            "acceleration",
            "comfort",
            ("metrics.acceleration", "metrics.acceleration_mean"),
            "mean",
            False,
            "Acceleration comfort metric when explicitly produced.",
        ),
        _metric(
            "jerk",
            "comfort",
            ("metrics.jerk", "metrics.jerk_mean"),
            "mean",
            False,
            "Jerk comfort metric when explicitly produced.",
        ),
        _metric(
            "angular_velocity",
            "comfort",
            ("metrics.angular_velocity", "metrics.angular_velocity_mean"),
            "mean",
            False,
            "Angular velocity comfort metric when explicitly produced.",
        ),
        _metric(
            "intervention_required",
            "operational",
            (
                "metrics.intervention_required",
                "metrics.shield_intervention_count",
            ),
            "episode_rate",
            False,
            "Operational intervention episode rate from explicit flags or shield count.",
        ),
        _metric(
            "tele_assistance_required",
            "operational",
            ("metrics.tele_assistance_required",),
            "episode_rate",
            False,
            "Tele-assistance required rate when explicitly produced.",
        ),
        _metric(
            "tele_driving_required",
            "operational",
            ("metrics.tele_driving_required",),
            "episode_rate",
            False,
            "Tele-driving required rate when explicitly produced.",
        ),
    )
)


def get_nested(record: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path from a mapping.

    Returns:
        Dotted-path value, or ``None`` when absent.
    """

    current: Any = record
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _episode_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return flattened metric and selected top-level aliases for one episode."""

    record_dict = dict(record)
    flattened = flatten_metrics(record_dict)
    view: dict[str, Any] = {f"metrics.{key}": value for key, value in flattened.items()}
    view.update(flattened)
    outcome = record.get("outcome")
    if isinstance(outcome, Mapping):
        for key in ("collision_event", "timeout_event", "route_complete"):
            if key in outcome:
                view[f"outcome.{key}"] = outcome[key]
    if "termination_reason" in record:
        view["termination_reason"] = record["termination_reason"]
    for source_key in ("scenario_params.algo", "algo"):
        value = get_nested(record, source_key)
        if value is not None:
            view[source_key] = value
    return view


def _as_float(value: Any) -> float | None:
    """Coerce numeric and boolean values to float.

    Returns:
        Float value, or ``None`` for unsupported values.
    """

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_flag(value: Any) -> float | None:
    """Coerce booleans/counts/rates to a bounded episode flag.

    Returns:
        ``1.0`` for positive values, ``0.0`` for zero/false, or ``None``.
    """

    number = _as_float(value)
    if number is None:
        return None
    return 1.0 if number > 0.0 else 0.0


def _resolve_collision_rate(
    definition: MetricDefinition,
    view: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Resolve collision rate from stored rate, count, or outcome flag.

    Returns:
        ``(value, source_key)`` when available, else ``(None, None)``.
    """

    for key in definition.source_keys:
        if key not in view:
            continue
        value = view[key]
        if key == "metrics.collision_rate":
            return _as_float(value), key
        return _as_flag(value), key
    return None, None


def _resolve_timeout_rate(view: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Resolve timeout rate from outcome or known timeout termination reasons.

    Returns:
        ``(value, source_key)`` when available, else ``(None, None)``.
    """

    if "outcome.timeout_event" in view:
        return _as_flag(view["outcome.timeout_event"]), "outcome.timeout_event"
    if view.get("termination_reason") in {"truncated", "max_steps"}:
        return 1.0, "termination_reason"
    return None, None


def _resolve_failure_to_progress_rate(
    view: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Resolve failure-to-progress rate when explicit route outcome exists.

    Returns:
        ``(value, source_key)`` when available, else ``(None, None)``.
    """

    if "outcome.route_complete" not in view:
        return None, None
    if view.get("outcome.collision_event") is True or view.get("outcome.timeout_event") is True:
        return 0.0, "outcome.route_complete"
    return (0.0 if bool(view["outcome.route_complete"]) else 1.0), "outcome.route_complete"


def _resolve_intervention_required(
    definition: MetricDefinition,
    view: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Resolve intervention-required from explicit flag or shield count.

    Returns:
        ``(value, source_key)`` when available, else ``(None, None)``.
    """

    for key in definition.source_keys:
        if key in view:
            return _as_flag(view[key]), key
    return None, None


def _resolve_metric_value(
    definition: MetricDefinition,
    view: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    """Resolve one metric value for one episode under conservative rules.

    Returns:
        ``(value, selected_source_key)`` when available, else ``(None, None)``.
    """

    if definition.name == "collision_rate":
        return _resolve_collision_rate(definition, view)

    if definition.name == "timeout_rate":
        return _resolve_timeout_rate(view)

    if definition.name == "failure_to_progress_rate":
        return _resolve_failure_to_progress_rate(view)

    if definition.name == "intervention_required":
        return _resolve_intervention_required(definition, view)

    for key in definition.source_keys:
        if key not in view:
            continue
        value = view[key]
        if definition.reduction in {"episode_rate", "any_rate"}:
            return _as_flag(value), key
        return _as_float(value), key
    return None, None


def _reduce(values: Sequence[float], reduction: str) -> float | None:
    """Reduce available episode values according to the metric contract.

    Returns:
        Reduced value, or ``None`` when no values are available.
    """

    if not values:
        return None
    if reduction in {"episode_rate", "mean", "any_rate"}:
        return sum(values) / len(values)
    if reduction == "minimum":
        return min(values)
    if reduction == "maximum":
        return max(values)
    raise ValueError(f"unknown metric-layer reduction: {reduction}")


def _summarize_metric(
    definition: MetricDefinition,
    views: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize one metric across episode views.

    Returns:
        JSON-serializable metric availability and aggregate payload.
    """

    values: list[float] = []
    selected_sources: list[str] = []
    for view in views:
        value, source = _resolve_metric_value(definition, view)
        if value is None:
            continue
        values.append(value)
        if source is not None and source not in selected_sources:
            selected_sources.append(source)

    value = _reduce(values, definition.reduction)
    if value is None:
        return {
            "status": "unavailable",
            "value": None,
            "support_count": 0,
            "source_keys": list(definition.source_keys),
            "reduction": definition.reduction,
            "higher_is_better": definition.higher_is_better,
            "source_kind": definition.source_kind,
            "unavailable_reason": definition.unavailable_reason_if_missing,
        }

    return {
        "status": "available",
        "value": value,
        "support_count": len(values),
        "source_keys": list(definition.source_keys),
        "selected_source_keys": selected_sources,
        "reduction": definition.reduction,
        "higher_is_better": definition.higher_is_better,
        "source_kind": definition.source_kind,
    }


def _layer_summary(views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build layer summaries over a sequence of episode views.

    Returns:
        Mapping from layer name to layer metadata and metric summaries.
    """

    layers: dict[str, Any] = {}
    layer_defs = {definition.name: definition for definition in CANONICAL_METRIC_LAYERS}
    for layer_name in LAYER_ORDER:
        layer_def = layer_defs[layer_name]
        layer_metrics = {
            metric.name: _summarize_metric(metric, views)
            for metric in CANONICAL_METRICS.values()
            if metric.layer == layer_name
        }
        layers[layer_name] = {
            "priority": layer_def.priority,
            "description": layer_def.description,
            "metrics": layer_metrics,
        }
    return layers


def _group_key(
    record: Mapping[str, Any],
    *,
    group_by: str,
    fallback_group_by: str,
) -> str:
    """Resolve a group key from an episode record.

    Returns:
        Group key value, or ``"unknown"`` when both keys are absent.
    """

    value = get_nested(record, group_by)
    if value is None:
        value = get_nested(record, fallback_group_by)
    return str(value) if value is not None else "unknown"


def build_metric_layer_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "algo",
) -> dict[str, Any]:
    """Build canonical metric-layer availability and aggregate summary.

    Returns:
        JSON-serializable summary over all records and per resolved group.
    """

    views = [_episode_view(record) for record in records]
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record, view in zip(records, views, strict=True):
        groups.setdefault(
            _group_key(record, group_by=group_by, fallback_group_by=fallback_group_by),
            [],
        ).append(view)

    return {
        "schema_version": METRIC_LAYER_SCHEMA_VERSION,
        "n_episodes": len(records),
        "group_by": group_by,
        "fallback_group_by": fallback_group_by,
        "layer_order": list(LAYER_ORDER),
        "layers": _layer_summary(views),
        "groups": {
            group: {
                "n_episodes": len(group_views),
                "layers": _layer_summary(group_views),
            }
            for group, group_views in sorted(groups.items())
        },
    }


__all__ = [
    "CANONICAL_METRICS",
    "CANONICAL_METRIC_LAYERS",
    "LAYER_ORDER",
    "METRIC_LAYER_SCHEMA_VERSION",
    "MISSING_METRIC_REASON",
    "MetricDefinition",
    "MetricLayerDefinition",
    "build_metric_layer_summary",
    "get_nested",
]

"""Long-horizon route benchmark helpers.

This module only composes existing local scenario segments and normalizes
already-recorded episode/event counts by distance. It does not define new
collision, near-miss, intervention, reset, or failure semantics.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robot_sf.errors import RobotSfError


class LongHorizonRouteError(RobotSfError, ValueError):
    """Raised when a route definition or metric input cannot be scored safely."""


@dataclass(frozen=True, slots=True)
class LongHorizonRouteSegment:
    """One reusable local benchmark scenario segment in a longer route."""

    scenario_id: str
    length_m: float
    repetitions: int = 1
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate segment fields fail-closed."""

        if not self.scenario_id.strip():
            raise LongHorizonRouteError("route segment scenario_id must be non-empty")
        if self.length_m <= 0:
            raise LongHorizonRouteError(
                f"route segment {self.scenario_id!r} length_m must be positive"
            )
        if self.repetitions <= 0:
            raise LongHorizonRouteError(
                f"route segment {self.scenario_id!r} repetitions must be positive"
            )

    @property
    def total_length_m(self) -> float:
        """Return segment distance after repetitions."""

        return self.length_m * self.repetitions

    def to_dict(self) -> dict[str, Any]:
        """Serialize the segment for route manifests or smoke packets.

        Returns:
            Plain dictionary representation of this route segment.
        """

        return {
            "scenario_id": self.scenario_id,
            "length_m": self.length_m,
            "repetitions": self.repetitions,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class LongHorizonRouteDefinition:
    """Deterministic composition of local scenario segments into a route."""

    route_id: str
    segments: tuple[LongHorizonRouteSegment, ...]
    length_m: float

    def __post_init__(self) -> None:
        """Validate route fields fail-closed."""

        if not self.route_id.strip():
            raise LongHorizonRouteError("route_id must be non-empty")
        if not self.segments:
            raise LongHorizonRouteError("route must include at least one segment")
        if self.length_m <= 0:
            raise LongHorizonRouteError("route length_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize route definition for a benchmark config or evidence packet.

        Returns:
            Plain dictionary representation of this route definition.
        """

        return {
            "route_id": self.route_id,
            "length_m": self.length_m,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True, slots=True)
class LongHorizonRouteRunResult:
    """Headless execution result for one composed long-horizon route."""

    route: LongHorizonRouteDefinition
    records: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize route records and aggregate metrics for evidence packets.

        Returns:
            Plain dictionary representation of the route run.
        """

        return {
            "route": self.route.to_dict(),
            "records": [dict(record) for record in self.records],
            "metrics": dict(self.metrics),
        }


def build_long_horizon_route(
    route_id: str,
    segments: Sequence[LongHorizonRouteSegment],
    *,
    target_length_m: float | None = None,
) -> LongHorizonRouteDefinition:
    """Compose a long-horizon route from existing local scenario segments.

    When ``target_length_m`` is provided, the segment sequence is repeated and
    the final segment is shortened as needed. That gives a deterministic short
    headless smoke route without introducing new scenario physics.

    Returns:
        Validated long-horizon route definition.
    """

    if not segments:
        raise LongHorizonRouteError("route must include at least one segment")
    if target_length_m is not None and target_length_m <= 0:
        raise LongHorizonRouteError("target_length_m must be positive when provided")

    source_segments = tuple(segments)
    if target_length_m is None:
        route_segments = source_segments
        route_length_m = sum(segment.total_length_m for segment in route_segments)
        return LongHorizonRouteDefinition(route_id, route_segments, route_length_m)

    route_segments: list[LongHorizonRouteSegment] = []
    remaining_m = target_length_m
    while remaining_m > 1e-9:
        for segment in source_segments:
            if remaining_m <= 1e-9:
                break
            segment_length_m = min(segment.total_length_m, remaining_m)
            route_segments.append(
                LongHorizonRouteSegment(
                    scenario_id=segment.scenario_id,
                    length_m=segment_length_m,
                    repetitions=1,
                    parameters=segment.parameters,
                )
            )
            remaining_m -= segment_length_m

    return LongHorizonRouteDefinition(route_id, tuple(route_segments), target_length_m)


_RunEpisodeFn = Callable[..., Mapping[str, Any]]


def run_long_horizon_route(
    route: LongHorizonRouteDefinition,
    *,
    seed: int = 0,
    horizon: int = 100,
    dt: float = 0.1,
    record_forces: bool = True,
    algo: str = "simple_policy",
    algo_config_path: str | None = None,
    run_episode_fn: _RunEpisodeFn | None = None,
) -> LongHorizonRouteRunResult:
    """Run a short composed route through the existing headless episode runner.

    Each route segment is executed as one existing local scenario. The runner's
    event counts stay authoritative; this helper only adds planned route-distance
    provenance before computing per-distance route metrics.

    Returns:
        Segment records and aggregate distance-normalized route metrics.
    """

    if horizon <= 0:
        raise LongHorizonRouteError("horizon must be positive")
    if dt <= 0:
        raise LongHorizonRouteError("dt must be positive")

    if run_episode_fn is None:
        runner_module = importlib.import_module("robot_sf.benchmark.runner")
        run_episode_fn = runner_module.run_episode

    records: list[Mapping[str, Any]] = []
    for index, segment in enumerate(route.segments):
        scenario_params = dict(segment.parameters)
        scenario_params.setdefault("id", segment.scenario_id)
        record = run_episode_fn(
            scenario_params,
            seed + index,
            horizon=horizon,
            dt=dt,
            record_forces=record_forces,
            algo=algo,
            algo_config_path=algo_config_path,
            video_enabled=False,
        )
        records.append(
            _with_route_distance_provenance(
                record,
                route=route,
                segment=segment,
                segment_index=index,
            )
        )

    metrics = aggregate_distance_normalized_route_metrics(records, route_length_m=route.length_m)
    return LongHorizonRouteRunResult(route=route, records=tuple(records), metrics=metrics)


_COUNT_ALIASES: dict[str, tuple[str, ...]] = {
    "failures": ("failures", "failure_events"),
    "collisions": ("collisions", "collision_events"),
    "near_misses": ("near_misses", "near_miss_events"),
    "interventions": ("interventions", "operator_interventions"),
    "resets": ("resets", "reset_events"),
}

_DISTANCE_ALIASES = (
    "distance_m",
    "distance_traveled_m",
    "route_distance_m",
    "completed_distance_m",
    "path_length_m",
    "socnavbench_path_length",
)

_PLANNED_DISTANCE_ALIASES = (
    "planned_distance_m",
    "target_length_m",
)


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = record.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def _numeric_from_aliases(
    record: Mapping[str, Any],
    aliases: Iterable[str],
    *,
    default: float | None = None,
) -> float | None:
    metrics = _metrics(record)
    for alias in aliases:
        value = metrics.get(alias, record.get(alias))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise LongHorizonRouteError(f"{alias} must be numeric") from exc
    return default


def _distance_m(record: Mapping[str, Any]) -> float:
    distance_m = _numeric_from_aliases(record, _DISTANCE_ALIASES)
    if distance_m is None:
        raise LongHorizonRouteError(
            "record is missing distance_m; cannot compute distance-normalized metrics"
        )
    if distance_m < 0:
        raise LongHorizonRouteError("record distance_m must not be negative")
    return distance_m


def _count(record: Mapping[str, Any], metric_name: str) -> float:
    value = _numeric_from_aliases(record, _COUNT_ALIASES[metric_name], default=0.0)
    assert value is not None
    if value < 0:
        raise LongHorizonRouteError(f"{metric_name} count must not be negative")
    return value


def _with_route_distance_provenance(
    record: Mapping[str, Any],
    *,
    route: LongHorizonRouteDefinition,
    segment: LongHorizonRouteSegment,
    segment_index: int,
) -> Mapping[str, Any]:
    """Attach route-distance provenance without changing runner event counts.

    Returns:
        Episode record annotated with planned route-distance metadata.
    """

    annotated = dict(record)
    metrics = dict(_metrics(record))
    metrics.setdefault("planned_distance_m", segment.length_m)
    metrics.setdefault("route_length_m", route.length_m)
    annotated["metrics"] = metrics
    annotated["long_horizon_route"] = {
        "route_id": route.route_id,
        "segment_index": segment_index,
        "segment_id": segment.scenario_id,
        "segment_length_m": segment.length_m,
    }
    return annotated


def aggregate_distance_normalized_route_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    route_length_m: float | None = None,
) -> dict[str, float]:
    """Compute issue #3969 distance-normalized metrics from episode/event records.

    The input records must expose traversed distance in meters. Event counts are
    read from existing top-level or nested ``metrics`` fields and are not
    reinterpreted here.

    Returns:
        Dictionary containing failures/collisions/near-misses per 100 meters,
        interventions/resets per kilometer, and route completion.
    """

    if not records:
        raise LongHorizonRouteError("at least one record is required")
    if route_length_m is not None and route_length_m <= 0:
        raise LongHorizonRouteError("route_length_m must be positive when provided")

    total_distance_m = sum(_distance_m(record) for record in records)
    if total_distance_m <= 0:
        raise LongHorizonRouteError("total traversed distance_m must be positive")

    planned_distance_m = (
        route_length_m if route_length_m is not None else _planned_route_distance_m(records)
    )

    per_100m_scale = 100.0 / total_distance_m
    per_km_scale = 1000.0 / total_distance_m

    return {
        "failures_per_100m": _sum_counts(records, "failures") * per_100m_scale,
        "collisions_per_100m": _sum_counts(records, "collisions") * per_100m_scale,
        "near_misses_per_100m": _sum_counts(records, "near_misses") * per_100m_scale,
        "interventions_per_km": _sum_counts(records, "interventions") * per_km_scale,
        "route_completion": min(total_distance_m / planned_distance_m, 1.0),
        "resets_per_km": _sum_counts(records, "resets") * per_km_scale,
    }


def _sum_counts(records: Sequence[Mapping[str, Any]], metric_name: str) -> float:
    """Sum an existing event-count metric across records.

    Returns:
        Total event count across all input records.
    """

    return sum(_count(record, metric_name) for record in records)


def _planned_route_distance_m(records: Sequence[Mapping[str, Any]]) -> float:
    """Resolve planned route distance without double-counting repeated route totals.

    Returns:
        Planned route distance in meters.
    """

    route_lengths = [
        value
        for record in records
        if (value := _numeric_from_aliases(record, ("route_length_m",))) is not None
    ]
    if route_lengths:
        if any(value <= 0 for value in route_lengths):
            raise LongHorizonRouteError("route_length_m must be positive when provided")
        return max(route_lengths)

    planned_distance_m = sum(
        _numeric_from_aliases(record, _PLANNED_DISTANCE_ALIASES, default=0.0) or 0.0
        for record in records
    )
    if planned_distance_m > 0:
        return planned_distance_m
    return sum(_distance_m(record) for record in records)


__all__ = [
    "LongHorizonRouteDefinition",
    "LongHorizonRouteError",
    "LongHorizonRouteRunResult",
    "LongHorizonRouteSegment",
    "aggregate_distance_normalized_route_metrics",
    "build_long_horizon_route",
    "run_long_horizon_route",
]

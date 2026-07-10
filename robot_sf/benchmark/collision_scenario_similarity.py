"""Collision and near-collision scenario similarity reports.

This module builds a deterministic analysis-aid report from existing benchmark
episode JSONL records.  It is intentionally descriptive: nearest-neighbor and
group assignments are not benchmark evidence unless separately validated against
external labels.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.failure_extractor import is_failure

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

SCHEMA_VERSION = "collision_scenario_similarity.v1"
ISSUE_URL = "https://github.com/ll7/robot_sf_ll7/issues/4359"

_NUMERIC_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("collisions", ("metrics.collisions", "collisions")),
    ("near_misses", ("metrics.near_misses", "near_misses")),
    (
        "min_separation",
        ("metrics.min_separation", "metrics.minimum_separation", "min_separation"),
    ),
    ("time_to_conflict", ("metrics.time_to_conflict", "time_to_conflict")),
    ("comfort_exposure", ("metrics.comfort_exposure", "comfort_exposure")),
    ("time_to_goal", ("metrics.time_to_goal", "time_to_goal")),
    ("num_pedestrians", ("scenario_params.num_pedestrians", "num_pedestrians")),
    ("mean_speed", ("metrics.mean_speed", "metrics.average_speed")),
)

_CATEGORICAL_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("map", ("map_name", "map_id", "scenario_params.map_name")),
    ("scenario_family", ("scenario_family", "scenario_params.family", "scenario_params.kind")),
    ("planner", ("planner_id", "algo", "scenario_params.algo", "scenario_params.planner")),
    ("termination_reason", ("termination_reason", "status")),
)

_CONTEXT_CATEGORICAL_FEATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("map_region", ("map_region", "scenario_params.map_region", "scenario_params.region")),
)

_ROBOT_TRAJECTORY_PATHS = (
    "trajectory.robot_positions",
    "trajectory.robot_states",
    "robot_trajectory.positions",
    "robot_trajectory",
    "robot_states",
)
_PEDESTRIAN_TRAJECTORY_PATHS = (
    "trajectory.pedestrian_trajectories",
    "pedestrian_trajectories",
)
_ACTION_TRACE_PATHS = (
    "trajectory.actions",
    "action_trace",
    "actions",
)
_TRAJECTORY_DT_PATHS = ("trajectory.dt", "dt", "scenario_params.dt")

_FEATURE_SET_IDS = (
    "legacy_summary_v1",
    "trajectory_action_v1",
    "combined_context_v1",
)

_TRAJECTORY_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "avg_speed",
        "clearing_distance_avg",
        "clearing_distance_min",
        "curvature_mean",
        "energy",
        "jerk_mean",
        "mean_clearance",
        "mean_distance",
        "min_clearance",
        "min_distance",
        "min_separation",
        "minimum_separation",
        "path_efficiency",
        "robot_ped_within_5m_frac",
        "socnavbench_path_irregularity",
        "socnavbench_path_length",
        "socnavbench_path_length_ratio",
        "time_to_goal_norm",
    }
)

_LABEL_POSITIVE_KEYS = (
    "collision",
    "collision_event",
    "near_miss",
    "unsafe",
    "failure",
    "low_progress",
)


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Feature descriptor for one unsafe or near-unsafe episode."""

    record_id: str
    source_index: int
    numeric: dict[str, float]
    categorical: dict[str, str]
    event: dict[str, Any]
    trajectory_action_numeric: dict[str, float] = field(default_factory=dict)
    context_categorical: dict[str, str] = field(default_factory=dict)
    trajectory_context: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return JSON-safe descriptor payload."""
        return {
            "record_id": self.record_id,
            "source_index": self.source_index,
            "numeric": dict(sorted(self.numeric.items())),
            "categorical": dict(sorted(self.categorical.items())),
            "trajectory_action_numeric": dict(sorted(self.trajectory_action_numeric.items())),
            "context_categorical": dict(sorted(self.context_categorical.items())),
            "trajectory_context": self.trajectory_context,
            "event": self.event,
        }


def _nested(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _first_present(record: dict[str, Any], paths: Sequence[str]) -> Any:
    for path in paths:
        value = _nested(record, path)
        if value is not None:
            return value
    return None


def _finite_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(candidate):
        return candidate
    return None


def _position(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        nested_position = value.get("position", value.get("pos"))
        if nested_position is not None:
            return _position(nested_position)
        x = _finite_float(value.get("x"))
        y = _finite_float(value.get("y"))
        return (x, y) if x is not None and y is not None else None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _finite_float(value[0])
    y = _finite_float(value[1])
    return (x, y) if x is not None and y is not None else None


def _position_series(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    positions = [_position(item) for item in value]
    if not positions or any(position is None for position in positions):
        return []
    return [position for position in positions if position is not None]


def _pedestrian_trajectories(value: Any) -> list[list[tuple[float, float]]]:
    candidates: Iterable[Any]
    if isinstance(value, dict):
        candidates = value.values()
    elif isinstance(value, list):
        candidates = value
    else:
        return []
    return [series for candidate in candidates if (series := _position_series(candidate))]


def _actions(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    actions: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict):
            linear = _finite_float(item.get("linear", item.get("v")))
            angular = _finite_float(item.get("angular", item.get("omega")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            linear = _finite_float(item[0])
            angular = _finite_float(item[1])
        else:
            continue
        if linear is not None and angular is not None:
            actions.append((linear, angular))
    return actions


def _path_length(positions: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(left, right) for left, right in pairwise(positions))


def _trajectory_geometry_features(
    robot_positions: Sequence[tuple[float, float]],
    pedestrian_tracks: Sequence[Sequence[tuple[float, float]]],
    dt: float | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if robot_positions:
        robot_path_length = _path_length(robot_positions)
        robot_displacement = math.dist(robot_positions[0], robot_positions[-1])
        features.update(
            {
                "robot_path_length": robot_path_length,
                "robot_displacement": robot_displacement,
                "robot_straightness": (
                    robot_displacement / robot_path_length if robot_path_length > 0 else 0.0
                ),
            }
        )
    if pedestrian_tracks:
        features["pedestrian_mean_path_length"] = sum(
            _path_length(track) for track in pedestrian_tracks
        ) / len(pedestrian_tracks)
    if not robot_positions or not pedestrian_tracks:
        return features
    closest: tuple[float, int] | None = None
    for step, robot_position in enumerate(robot_positions):
        for track in pedestrian_tracks:
            distance = math.dist(robot_position, track[step])
            if closest is None or distance < closest[0]:
                closest = (distance, step)
    if closest is not None:
        features["trajectory_min_center_distance"] = closest[0]
        if dt is not None and dt > 0:
            features["trajectory_time_to_min_center_distance"] = closest[1] * dt
    return features


def _action_features(actions: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not actions:
        return {}
    return {
        "action_count": float(len(actions)),
        "action_linear_mean": sum(action[0] for action in actions) / len(actions),
        "action_angular_abs_mean": sum(abs(action[1]) for action in actions) / len(actions),
        "action_linear_delta_abs_mean": (
            sum(abs(right[0] - left[0]) for left, right in pairwise(actions)) / (len(actions) - 1)
            if len(actions) > 1
            else 0.0
        ),
        "action_angular_delta_abs_mean": (
            sum(abs(right[1] - left[1]) for left, right in pairwise(actions)) / (len(actions) - 1)
            if len(actions) > 1
            else 0.0
        ),
    }


def _trajectory_action_features(record: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    robot_positions = _position_series(_first_present(record, _ROBOT_TRAJECTORY_PATHS))
    pedestrian_tracks = _pedestrian_trajectories(
        _first_present(record, _PEDESTRIAN_TRAJECTORY_PATHS)
    )
    raw_actor_trajectories_available = bool(robot_positions and pedestrian_tracks) and all(
        len(track) == len(robot_positions) for track in pedestrian_tracks
    )
    if not raw_actor_trajectories_available:
        # Do not align unequal arrays by truncation or interpolation: their
        # per-step distance and timing features are not interpretable.
        pedestrian_tracks = []
    actions = _actions(_first_present(record, _ACTION_TRACE_PATHS))
    dt = _finite_float(_first_present(record, _TRAJECTORY_DT_PATHS))

    features = _trajectory_geometry_features(robot_positions, pedestrian_tracks, dt)
    features.update(_action_features(actions))

    return features, {
        "raw_actor_trajectories_available": raw_actor_trajectories_available,
        "robot_samples": len(robot_positions),
        "pedestrian_tracks": len(pedestrian_tracks),
        "pedestrian_samples": sum(len(track) for track in pedestrian_tracks),
        "action_samples": len(actions),
        "dt": dt,
    }


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("episode_id", "record_id", "run_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    scenario_id = record.get("scenario_id")
    seed = record.get("seed")
    if scenario_id is not None and seed is not None:
        return f"{scenario_id}:seed={seed}:row={index}"
    if scenario_id is not None:
        return f"{scenario_id}:row={index}"
    return f"row={index}"


def _event_summary(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return {
        "scenario_id": record.get("scenario_id"),
        "seed": record.get("seed"),
        "collisions": metrics.get("collisions", record.get("collisions", 0)),
        "near_misses": metrics.get("near_misses", record.get("near_misses", 0)),
        "min_separation": metrics.get(
            "min_separation",
            metrics.get("minimum_separation", record.get("min_separation")),
        ),
        "termination_reason": record.get("termination_reason", record.get("status")),
    }


def describe_collision_scenarios(
    records: Iterable[dict[str, Any]],
    *,
    collision_threshold: float = 1.0,
    near_miss_threshold: float = 0.0,
    comfort_threshold: float = 0.2,
) -> list[ScenarioDescriptor]:
    """Extract descriptors for collision, near-miss, or high-discomfort records.

    Returns:
        Descriptors for records selected by the failure-like thresholds.
    """
    descriptors: list[ScenarioDescriptor] = []
    for index, record in enumerate(records):
        if not is_failure(
            record,
            collision_threshold=collision_threshold,
            near_miss_threshold=near_miss_threshold,
            comfort_threshold=comfort_threshold,
        ):
            continue

        numeric: dict[str, float] = {}
        for name, paths in _NUMERIC_FEATURES:
            value = _finite_float(_first_present(record, paths))
            if value is not None:
                numeric[name] = value

        categorical: dict[str, str] = {}
        for name, paths in _CATEGORICAL_FEATURES:
            value = _first_present(record, paths)
            if value is not None and str(value).strip():
                categorical[name] = str(value).strip()

        context_categorical: dict[str, str] = {}
        for name, paths in _CONTEXT_CATEGORICAL_FEATURES:
            value = _first_present(record, paths)
            if value is not None and str(value).strip():
                context_categorical[name] = str(value).strip()

        trajectory_action_numeric, trajectory_context = _trajectory_action_features(record)

        descriptors.append(
            ScenarioDescriptor(
                record_id=_record_id(record, index),
                source_index=index,
                numeric=numeric,
                categorical=categorical,
                trajectory_action_numeric=trajectory_action_numeric,
                context_categorical=context_categorical,
                trajectory_context=trajectory_context,
                event=_event_summary(record),
            )
        )
    return descriptors


def _numeric_ranges(descriptors: Sequence[ScenarioDescriptor]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name, _paths in _NUMERIC_FEATURES:
        values = [
            descriptor.numeric[name] for descriptor in descriptors if name in descriptor.numeric
        ]
        if values:
            ranges[name] = (min(values), max(values))
    return ranges


def _feature_distance(
    left: ScenarioDescriptor,
    right: ScenarioDescriptor,
    *,
    numeric_ranges: dict[str, tuple[float, float]],
) -> tuple[float, list[str]]:
    components: list[float] = []
    shared_features: list[str] = []

    for name, (minimum, maximum) in numeric_ranges.items():
        if name not in left.numeric or name not in right.numeric:
            continue
        scale = maximum - minimum
        value = 0.0 if scale == 0 else abs(left.numeric[name] - right.numeric[name]) / scale
        components.append(value)
        shared_features.append(name)

    for name, _paths in _CATEGORICAL_FEATURES:
        if name not in left.categorical or name not in right.categorical:
            continue
        components.append(0.0 if left.categorical[name] == right.categorical[name] else 1.0)
        shared_features.append(name)

    if not components:
        return 1.0, []
    return sum(components) / len(components), shared_features


def _nearest_neighbors(
    descriptors: Sequence[ScenarioDescriptor],
    *,
    nearest_k: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float]]:
    ranges = _numeric_ranges(descriptors)
    pair_distances: dict[tuple[int, int], float] = {}
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(descriptors):
        candidates: list[dict[str, Any]] = []
        for right_index, right in enumerate(descriptors):
            if left_index == right_index:
                continue
            distance, shared_features = _feature_distance(left, right, numeric_ranges=ranges)
            pair_distances[tuple(sorted((left_index, right_index)))] = distance
            candidates.append(
                {
                    "record_id": right.record_id,
                    "distance": round(distance, 6),
                    "shared_features": shared_features,
                }
            )
        candidates.sort(key=lambda row: (row["distance"], row["record_id"]))
        rows.append({"record_id": left.record_id, "neighbors": candidates[:nearest_k]})
    return rows, pair_distances


def _feature_set_values(
    descriptor: ScenarioDescriptor,
    feature_set_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    if feature_set_id == "legacy_summary_v1":
        return descriptor.numeric, descriptor.categorical
    if feature_set_id == "trajectory_action_v1":
        return descriptor.trajectory_action_numeric, {}
    if feature_set_id == "combined_context_v1":
        return (
            {**descriptor.numeric, **descriptor.trajectory_action_numeric},
            {**descriptor.categorical, **descriptor.context_categorical},
        )
    raise ValueError(f"Unknown collision-similarity feature set: {feature_set_id}")


def _mapping_ranges(rows: Sequence[dict[str, float]]) -> dict[str, tuple[float, float]]:
    feature_names = sorted({name for row in rows for name in row})
    return {
        name: (
            min(row[name] for row in rows if name in row),
            max(row[name] for row in rows if name in row),
        )
        for name in feature_names
    }


def _mapping_distance(
    left_numeric: dict[str, float],
    left_categorical: dict[str, str],
    right_numeric: dict[str, float],
    right_categorical: dict[str, str],
    *,
    numeric_ranges: dict[str, tuple[float, float]],
) -> tuple[float, list[str]]:
    components: list[float] = []
    shared_features: list[str] = []
    for name, (minimum, maximum) in numeric_ranges.items():
        if name not in left_numeric or name not in right_numeric:
            continue
        scale = maximum - minimum
        components.append(
            0.0 if scale == 0 else abs(left_numeric[name] - right_numeric[name]) / scale
        )
        shared_features.append(name)
    for name in sorted(left_categorical.keys() & right_categorical.keys()):
        components.append(0.0 if left_categorical[name] == right_categorical[name] else 1.0)
        shared_features.append(name)
    if not components:
        return 1.0, []
    return sum(components) / len(components), shared_features


def _feature_set_neighbors(
    descriptors: Sequence[ScenarioDescriptor],
    *,
    feature_set_id: str,
    nearest_k: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float], list[str]]:
    views = [_feature_set_values(descriptor, feature_set_id) for descriptor in descriptors]
    ranges = _mapping_ranges([numeric for numeric, _categorical in views])
    pair_distances: dict[tuple[int, int], float] = {}
    rows: list[dict[str, Any]] = []
    observed_features: set[str] = set()
    for left_index, left in enumerate(descriptors):
        candidates: list[dict[str, Any]] = []
        left_numeric, left_categorical = views[left_index]
        for right_index, right in enumerate(descriptors):
            if left_index == right_index:
                continue
            right_numeric, right_categorical = views[right_index]
            distance, shared_features = _mapping_distance(
                left_numeric,
                left_categorical,
                right_numeric,
                right_categorical,
                numeric_ranges=ranges,
            )
            observed_features.update(shared_features)
            pair_distances[tuple(sorted((left_index, right_index)))] = distance
            candidates.append(
                {
                    "record_id": right.record_id,
                    "distance": round(distance, 6),
                    "shared_features": shared_features,
                }
            )
        candidates.sort(key=lambda row: (row["distance"], row["record_id"]))
        rows.append({"record_id": left.record_id, "neighbors": candidates[:nearest_k]})
    return rows, pair_distances, sorted(observed_features)


def _feature_set_comparison(
    descriptors: Sequence[ScenarioDescriptor],
    *,
    nearest_k: int,
    group_threshold: float,
) -> dict[str, Any]:
    cohort = [
        descriptor
        for descriptor in descriptors
        if descriptor.trajectory_context.get("raw_actor_trajectories_available", False)
    ]
    status = (
        "available" if len(cohort) >= 2 else "insufficient_records" if cohort else "unavailable"
    )
    reports: list[dict[str, Any]] = []
    for feature_set_id in _FEATURE_SET_IDS:
        if len(cohort) < 2:
            reports.append(
                {
                    "feature_set_id": feature_set_id,
                    "status": status,
                    "observed_features": [],
                    "nearest_neighbors": [],
                    "groups": [],
                }
            )
            continue
        if feature_set_id == "legacy_summary_v1":
            neighbors, pair_distances = _nearest_neighbors(cohort, nearest_k=nearest_k)
            observed_features = sorted(
                {
                    feature
                    for row in neighbors
                    for neighbor in row["neighbors"]
                    for feature in neighbor["shared_features"]
                }
            )
        else:
            neighbors, pair_distances, observed_features = _feature_set_neighbors(
                cohort,
                feature_set_id=feature_set_id,
                nearest_k=nearest_k,
            )
        reports.append(
            {
                "feature_set_id": feature_set_id,
                "status": "available",
                "observed_features": observed_features,
                "nearest_neighbors": neighbors,
                "groups": _group_descriptors(
                    cohort,
                    pair_distances,
                    threshold=group_threshold,
                ),
            }
        )
    cohort_ids = {descriptor.record_id for descriptor in cohort}
    return {
        "status": status,
        "comparison_cohort_count": len(cohort),
        "comparison_cohort_record_ids": [descriptor.record_id for descriptor in cohort],
        "excluded_without_raw_actor_trajectories": [
            descriptor.record_id
            for descriptor in descriptors
            if descriptor.record_id not in cohort_ids
        ],
        "reports": reports,
        "interpretation": (
            "Candidate feature sets are compared on the same raw-trajectory cohort. Rankings are "
            "diagnostic and are not failure-family labels or benchmark scores."
        ),
    }


def _group_descriptors(
    descriptors: Sequence[ScenarioDescriptor],
    pair_distances: dict[tuple[int, int], float],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    parent = list(range(len(descriptors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for (left, right), distance in pair_distances.items():
        if distance <= threshold:
            union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(descriptors)):
        components[find(index)].append(index)

    groups: list[dict[str, Any]] = []
    for group_index, member_indexes in enumerate(
        sorted(
            components.values(), key=lambda values: (-len(values), descriptors[values[0]].record_id)
        ),
        start=1,
    ):
        representative_index = _representative(member_indexes, pair_distances)
        groups.append(
            {
                "group_id": f"group-{group_index}",
                "size": len(member_indexes),
                "representative_record_id": descriptors[representative_index].record_id,
                "record_ids": [descriptors[index].record_id for index in member_indexes],
            }
        )
    return groups


def _representative(
    member_indexes: Sequence[int],
    pair_distances: dict[tuple[int, int], float],
) -> int:
    if len(member_indexes) == 1:
        return member_indexes[0]
    scored: list[tuple[float, int]] = []
    for candidate in member_indexes:
        total = 0.0
        for other in member_indexes:
            if candidate == other:
                continue
            total += pair_distances.get(tuple(sorted((candidate, other))), 1.0)
        scored.append((total / (len(member_indexes) - 1), candidate))
    scored.sort()
    return scored[0][1]


def _record_labels(record: dict[str, Any]) -> dict[str, Any]:
    labels = record.get("external_labels")
    if not isinstance(labels, dict):
        labels = record.get("labels")
    if not isinstance(labels, dict):
        labels = {}
    outcome = record.get("outcome")
    if isinstance(outcome, dict):
        labels = {**outcome, **labels}
    return labels


def _label_is_positive(labels: dict[str, Any]) -> bool:
    return any(bool(labels.get(key)) for key in _LABEL_POSITIVE_KEYS)


def _trajectory_fields(record: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    trajectory_features = record.get("trajectory_features")
    if isinstance(trajectory_features, dict):
        fields.update(str(key) for key in trajectory_features)
    trajectory = record.get("trajectory")
    if isinstance(trajectory, dict):
        fields.update(f"trajectory.{key}" for key in trajectory)
    return fields


def _raw_trajectory_array_fields(record: dict[str, Any]) -> set[str]:
    _features, context = _trajectory_action_features(record)
    if not context["raw_actor_trajectories_available"]:
        return set()
    fields = {"robot_positions", "pedestrian_trajectories"}
    if context["action_samples"]:
        fields.add("actions")
    if context["dt"] is not None:
        fields.add("dt")
    return fields


def _trajectory_metric_fields(record: dict[str, Any]) -> set[str]:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return set()
    return {f"metrics.{key}" for key in metrics if key in _TRAJECTORY_METRIC_FIELDS}


def _field_availability_summary(
    records_by_id: dict[str, dict[str, Any]],
    selected_ids: set[str],
    extractor: Callable[[dict[str, Any]], set[str]],
) -> tuple[int, int, set[str]]:
    records_with_fields = 0
    selected_records_with_fields = 0
    selected_fields: set[str] = set()
    for record_id, record in records_by_id.items():
        fields = extractor(record)
        if not fields:
            continue
        records_with_fields += 1
        if record_id in selected_ids:
            selected_records_with_fields += 1
            selected_fields.update(fields)
    return records_with_fields, selected_records_with_fields, selected_fields


def _validation_summary(
    records: Sequence[dict[str, Any]],
    descriptors: Sequence[ScenarioDescriptor],
) -> dict[str, Any]:
    selected_ids = {descriptor.record_id for descriptor in descriptors}
    records_by_id = {_record_id(record, index): record for index, record in enumerate(records)}

    labeled_records = 0
    selected_labeled_records = 0
    selected_label_positive = 0
    selected_label_conflicts: list[str] = []
    for record_id, record in records_by_id.items():
        labels = _record_labels(record)
        if not labels:
            continue
        labeled_records += 1
        if record_id not in selected_ids:
            continue
        selected_labeled_records += 1
        if _label_is_positive(labels):
            selected_label_positive += 1
        else:
            selected_label_conflicts.append(record_id)

    trajectory_records, selected_trajectory_records, observed_fields = _field_availability_summary(
        records_by_id,
        selected_ids,
        _trajectory_fields,
    )
    (
        raw_trajectory_records,
        selected_raw_trajectory_records,
        observed_raw_trajectory_fields,
    ) = _field_availability_summary(
        records_by_id,
        selected_ids,
        _raw_trajectory_array_fields,
    )
    (
        trajectory_metric_records,
        selected_trajectory_metric_records,
        observed_metric_fields,
    ) = _field_availability_summary(
        records_by_id,
        selected_ids,
        _trajectory_metric_fields,
    )

    return {
        "external_labels": {
            "status": "available" if labeled_records else "unavailable",
            "records_with_labels": labeled_records,
            "selected_with_labels": selected_labeled_records,
            "selected_positive_labels": selected_label_positive,
            "selected_label_conflicts": selected_label_conflicts,
            "interpretation": (
                "Descriptive alignment check only; labels are not treated as benchmark truth."
            ),
        },
        "trajectory_fields": {
            "status": "available" if trajectory_records else "unavailable",
            "records_with_trajectory_fields": trajectory_records,
            "selected_with_trajectory_fields": selected_trajectory_records,
            "selected_fields_observed": sorted(observed_fields),
            "interpretation": (
                "Trajectory fields are reported for reviewer inspection and are not a new metric."
            ),
        },
        "raw_trajectory_arrays": {
            "status": "available" if raw_trajectory_records else "unavailable",
            "comparison_status": (
                "available"
                if selected_raw_trajectory_records >= 2
                else "insufficient_records"
                if selected_raw_trajectory_records
                else "unavailable"
            ),
            "records_with_raw_trajectory_arrays": raw_trajectory_records,
            "selected_with_raw_trajectory_arrays": selected_raw_trajectory_records,
            "selected_fields_observed": sorted(observed_raw_trajectory_fields),
            "interpretation": (
                "Raw robot and pedestrian arrays support optional feature-set comparison only; "
                "they are not replay validation or benchmark truth."
            ),
        },
        "trajectory_metric_fields": {
            "status": "available" if trajectory_metric_records else "unavailable",
            "records_with_trajectory_metric_fields": trajectory_metric_records,
            "selected_with_trajectory_metric_fields": selected_trajectory_metric_records,
            "selected_metric_fields_observed": sorted(observed_metric_fields),
            "interpretation": (
                "Trajectory-derived benchmark metric fields are descriptive validation context "
                "only and are not a new ranking signal."
            ),
        },
    }


def build_collision_scenario_similarity_report(
    episodes_jsonl: str | Path,
    *,
    nearest_k: int = 3,
    group_threshold: float = 0.35,
    collision_threshold: float = 1.0,
    near_miss_threshold: float = 0.0,
    comfort_threshold: float = 0.2,
    require_trajectory_comparison: bool = False,
) -> dict[str, Any]:
    """Build a scenario-similarity report from benchmark episode JSONL records.

    Returns:
        JSON-safe report with descriptors, nearest neighbors, groups, and limitations.
    """
    records = read_jsonl([Path(episodes_jsonl)])
    descriptors = describe_collision_scenarios(
        records,
        collision_threshold=collision_threshold,
        near_miss_threshold=near_miss_threshold,
        comfort_threshold=comfort_threshold,
    )
    raw_trajectory_descriptor_count = sum(
        descriptor.trajectory_context.get("raw_actor_trajectories_available", False)
        for descriptor in descriptors
    )
    if require_trajectory_comparison and raw_trajectory_descriptor_count < 2:
        raise ValueError(
            "Raw trajectory comparison requires at least two selected records with robot and "
            "pedestrian trajectory arrays"
        )
    neighbors, pair_distances = _nearest_neighbors(descriptors, nearest_k=max(0, int(nearest_k)))
    groups = _group_descriptors(descriptors, pair_distances, threshold=float(group_threshold))
    feature_set_comparison = _feature_set_comparison(
        descriptors,
        nearest_k=max(0, int(nearest_k)),
        group_threshold=float(group_threshold),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE_URL,
        "inputs": {"episodes_jsonl": str(episodes_jsonl), "record_count": len(records)},
        "selection": {
            "selected_count": len(descriptors),
            "collision_threshold": collision_threshold,
            "near_miss_threshold": near_miss_threshold,
            "comfort_threshold": comfort_threshold,
        },
        "feature_sets": {
            "numeric": [name for name, _paths in _NUMERIC_FEATURES],
            "categorical": [name for name, _paths in _CATEGORICAL_FEATURES],
            "candidate_comparison": list(_FEATURE_SET_IDS),
            "raw_trajectory_input_contract": {
                "robot": list(_ROBOT_TRAJECTORY_PATHS),
                "pedestrians": list(_PEDESTRIAN_TRAJECTORY_PATHS),
                "actions": list(_ACTION_TRACE_PATHS),
                "dt": list(_TRAJECTORY_DT_PATHS),
                "position_shape": "[x, y] or an object containing position, pos, or x/y",
                "position_semantics": (
                    "complete per-actor samples in one shared coordinate frame; expected metres"
                ),
                "pedestrian_shape": (
                    "actor-id mapping to position arrays or a list of actor position arrays"
                ),
                "action_shape": "[linear, angular] or an object containing linear/angular or v/omega",
            },
        },
        "descriptors": [descriptor.to_json() for descriptor in descriptors],
        "nearest_neighbors": neighbors,
        "groups": groups,
        "feature_set_comparison": feature_set_comparison,
        "validation": _validation_summary(records, descriptors),
        "limitations": [
            "Scenario similarity is an analysis aid, not benchmark evidence by itself.",
            "Distances depend on logged descriptor fields and do not validate external labels.",
            "Missing trajectory-level fields are excluded rather than imputed.",
            "External labels and trajectory fields, when present, are descriptive validation "
            "context only.",
            "Trajectory feature-set distances use dataset-relative range normalization and are "
            "sensitive to sampling rate, trace length, and missing fields.",
            "Raw position arrays must use one shared frame and sampling alignment; malformed "
            "series are excluded rather than compacted or imputed.",
        ],
    }


def format_collision_scenario_similarity_markdown(report: dict[str, Any]) -> str:
    """Format a compact Markdown table for reviewer inspection.

    Returns:
        Markdown report text.
    """
    lines = [
        "# Collision Scenario Similarity Report",
        "",
        "- Evidence status: `diagnostic-only`.",
        "- Claim boundary: nearest neighbors and groups are analysis aids, not validated "
        "failure-family labels or benchmark rankings.",
        f"- Schema: `{report['schema_version']}`",
        f"- Selected records: {report['selection']['selected_count']} / "
        f"{report['inputs']['record_count']}",
        "",
        "## Groups",
        "",
        "| group | size | representative | members |",
        "| --- | ---: | --- | --- |",
    ]
    for group in report["groups"]:
        lines.append(
            f"| {group['group_id']} | {group['size']} | "
            f"`{group['representative_record_id']}` | "
            f"{', '.join(f'`{record_id}`' for record_id in group['record_ids'])} |"
        )
    lines.extend(["", "## Nearest Neighbors", "", "| record | neighbors |", "| --- | --- |"])
    for row in report["nearest_neighbors"]:
        neighbor_text = ", ".join(
            f"`{neighbor['record_id']}` ({neighbor['distance']:.3f})"
            for neighbor in row["neighbors"]
        )
        lines.append(f"| `{row['record_id']}` | {neighbor_text} |")
    feature_comparison = report.get("feature_set_comparison", {})
    lines.extend(
        [
            "",
            "## Candidate Feature-Set Comparison",
            "",
            f"Comparison cohort: {feature_comparison.get('comparison_cohort_count', 0)} selected "
            "records with raw robot and pedestrian trajectories.",
            "",
            "| feature set | status | observed features | example nearest neighbor |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for feature_report in feature_comparison.get("reports", []):
        nearest_rows = feature_report.get("nearest_neighbors", [])
        nearest = (
            nearest_rows[0]["neighbors"][0]
            if nearest_rows and nearest_rows[0]["neighbors"]
            else None
        )
        nearest_text = (
            f"`{nearest_rows[0]['record_id']}` -> `{nearest['record_id']}` "
            f"({nearest['distance']:.3f})"
            if nearest is not None
            else "-"
        )
        lines.append(
            f"| `{feature_report['feature_set_id']}` | {feature_report['status']} | "
            f"{len(feature_report.get('observed_features', []))} | {nearest_text} |"
        )
    validation = report.get("validation", {})
    external_labels = validation.get("external_labels", {})
    trajectory_fields = validation.get("trajectory_fields", {})
    raw_trajectory_arrays = validation.get("raw_trajectory_arrays", {})
    trajectory_metric_fields = validation.get("trajectory_metric_fields", {})
    lines.extend(
        [
            "",
            "## Validation Context",
            "",
            f"- External labels: {external_labels.get('status', 'unavailable')} "
            f"({external_labels.get('selected_with_labels', 0)} selected records with labels; "
            f"{external_labels.get('selected_positive_labels', 0)} positive).",
            f"- Trajectory fields: {trajectory_fields.get('status', 'unavailable')} "
            f"({trajectory_fields.get('selected_with_trajectory_fields', 0)} selected records).",
            f"- Raw trajectory arrays: {raw_trajectory_arrays.get('status', 'unavailable')} / "
            f"comparison {raw_trajectory_arrays.get('comparison_status', 'insufficient_records')} "
            f"({raw_trajectory_arrays.get('selected_with_raw_trajectory_arrays', 0)} selected "
            "records).",
            f"- Trajectory-derived metric fields: "
            f"{trajectory_metric_fields.get('status', 'unavailable')} "
            f"({trajectory_metric_fields.get('selected_with_trajectory_metric_fields', 0)} "
            f"selected records).",
        ]
    )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def write_collision_scenario_similarity_report(
    report: dict[str, Any],
    out_json: str | Path,
    *,
    out_markdown: str | Path | None = None,
) -> None:
    """Write JSON and optional Markdown scenario-similarity reports."""
    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if out_markdown is not None:
        out_markdown_path = Path(out_markdown)
        out_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        out_markdown_path.write_text(
            format_collision_scenario_similarity_markdown(report),
            encoding="utf-8",
        )


__all__ = [
    "SCHEMA_VERSION",
    "ScenarioDescriptor",
    "build_collision_scenario_similarity_report",
    "describe_collision_scenarios",
    "format_collision_scenario_similarity_markdown",
    "write_collision_scenario_similarity_report",
]

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
from dataclasses import dataclass
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

    def to_json(self) -> dict[str, Any]:
        """Return JSON-safe descriptor payload."""
        return {
            "record_id": self.record_id,
            "source_index": self.source_index,
            "numeric": dict(sorted(self.numeric.items())),
            "categorical": dict(sorted(self.categorical.items())),
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

        descriptors.append(
            ScenarioDescriptor(
                record_id=_record_id(record, index),
                source_index=index,
                numeric=numeric,
                categorical=categorical,
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
    neighbors, pair_distances = _nearest_neighbors(descriptors, nearest_k=max(0, int(nearest_k)))
    groups = _group_descriptors(descriptors, pair_distances, threshold=float(group_threshold))
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
        },
        "descriptors": [descriptor.to_json() for descriptor in descriptors],
        "nearest_neighbors": neighbors,
        "groups": groups,
        "validation": _validation_summary(records, descriptors),
        "limitations": [
            "Scenario similarity is an analysis aid, not benchmark evidence by itself.",
            "Distances depend on logged descriptor fields and do not validate external labels.",
            "Missing trajectory-level fields are excluded rather than imputed.",
            "External labels and trajectory fields, when present, are descriptive validation "
            "context only.",
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
    validation = report.get("validation", {})
    external_labels = validation.get("external_labels", {})
    trajectory_fields = validation.get("trajectory_fields", {})
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

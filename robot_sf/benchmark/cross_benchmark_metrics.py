"""Cross-benchmark metric wrappers for Robot SF trace-derived metrics.

The wrappers in this module expose a small, explicit correspondence layer for
social-navigation benchmark concepts that can be computed from ``EpisodeData``.
They are not simulator-parity proof: approximate and unavailable rows are
preserved in the output so reports cannot silently treat proxy metrics as
equivalent external benchmark scores.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from robot_sf.benchmark.metrics import (
    EpisodeData,
    distance_to_human_min,
    socnavbench_path_length_ratio,
    success,
    time_to_collision_min,
    time_to_goal,
)

CROSS_BENCHMARK_METRIC_MAPPING_VERSION = "cross-benchmark-metric-mapping.v1"
CROSS_BENCHMARK_METRIC_REPORT_SCHEMA_VERSION = "cross-benchmark-metric-report.v1"
CROSS_BENCHMARK_CLAIM_BOUNDARY = (
    "trace-derived wrapper smoke evidence; not simulator parity or paper-grade evidence"
)

MAPPING_CLASSES = {"exact", "approximate", "unavailable"}
VALUE_STATUSES = {"available", "approximate", "unavailable"}


@dataclass(frozen=True)
class CrossBenchmarkMetricMapping:
    """One metric correspondence row."""

    metric_id: str
    benchmark: str
    external_metric: str
    robot_sf_source: str | None
    units: str
    denominator: str
    mapping_class: str
    evidence_tier: str
    semantic_notes: str

    def __post_init__(self) -> None:
        """Validate vocabulary used by the checked-in mapping table."""
        if self.mapping_class not in MAPPING_CLASSES:
            raise ValueError(f"Unsupported mapping_class: {self.mapping_class}")


@dataclass(frozen=True)
class CrossBenchmarkMetricRow:
    """Computed external-style metric row with explicit availability status."""

    metric_id: str
    benchmark: str
    external_metric: str
    status: str
    mapping_class: str
    value: float | None
    units: str
    denominator: str
    robot_sf_source: str | None
    evidence_tier: str
    semantic_notes: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate status vocabulary used in reports."""
        if self.status not in VALUE_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe row dictionary."""
        payload = asdict(self)
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload


def cross_benchmark_metric_mappings() -> tuple[CrossBenchmarkMetricMapping, ...]:
    """Return the versioned Robot SF cross-benchmark correspondence table."""
    return (
        CrossBenchmarkMetricMapping(
            metric_id="socnavbench.path_length_ratio",
            benchmark="SocNavBench",
            external_metric="path_length_ratio",
            robot_sf_source="socnavbench_path_length_ratio",
            units="ratio",
            denominator="straight_line_start_to_goal_distance_m",
            mapping_class="exact",
            evidence_tier="trace-derived",
            semantic_notes=(
                "Uses the vendored SocNavBench-style path length ratio helper over the Robot SF "
                "episode trajectory and goal displacement."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="common.traversal_time_s",
            benchmark="Common social-navigation",
            external_metric="traversal_time",
            robot_sf_source="time_to_goal",
            units="seconds",
            denominator="episode",
            mapping_class="approximate",
            evidence_tier="trace-derived",
            semantic_notes=(
                "Robot SF reports first goal-reaching time. External benchmark timeouts and "
                "episode-stop semantics may differ, so this is an approximate correspondence."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="common.time_to_collision_min_s",
            benchmark="Common social-navigation",
            external_metric="time_to_collision_min",
            robot_sf_source="time_to_collision_min",
            units="seconds",
            denominator="minimum_approaching_robot_pedestrian_pair",
            mapping_class="approximate",
            evidence_tier="trace-derived",
            semantic_notes=(
                "Computed from Robot SF robot and pedestrian trace velocities with a "
                "constant-velocity approaching-pair assumption."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="common.closest_pedestrian_distance_m",
            benchmark="Common social-navigation",
            external_metric="distance_to_closest_pedestrian",
            robot_sf_source="distance_to_human_min",
            units="meters",
            denominator="minimum_over_episode",
            mapping_class="exact",
            evidence_tier="trace-derived",
            semantic_notes=(
                "Uses Robot SF center-to-center robot-pedestrian distance. Surface clearance "
                "metrics remain separate because footprint conventions differ across benchmarks."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="robot_sf.success_trace_predicate",
            benchmark="Robot SF",
            external_metric="success_trace_predicate",
            robot_sf_source="success",
            units="binary",
            denominator="episode",
            mapping_class="exact",
            evidence_tier="trace-derived",
            semantic_notes=(
                "Robot SF success requires reaching the goal before the horizon and no wall, "
                "agent, or human collision under Robot SF collision semantics."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="socnavbench.personal_space_objective",
            benchmark="SocNavBench",
            external_metric="personal_space_objective",
            robot_sf_source=None,
            units="external_objective_units",
            denominator="external_cost_field",
            mapping_class="unavailable",
            evidence_tier="not_available",
            semantic_notes=(
                "Robot SF traces expose distance and clearance proxies, but not the external "
                "SocNavBench objective field needed for a faithful scalar wrapper."
            ),
        ),
        CrossBenchmarkMetricMapping(
            metric_id="hunavsim.human_behavior_cost",
            benchmark="HuNavSim",
            external_metric="human_behavior_cost",
            robot_sf_source=None,
            units="external_cost_units",
            denominator="external_behavior_model",
            mapping_class="unavailable",
            evidence_tier="not_available",
            semantic_notes=(
                "Requires HuNavSim behavior-model state that is not present in Robot SF trace "
                "exports."
            ),
        ),
    )


def mapping_table_as_dicts() -> list[dict[str, Any]]:
    """Return the mapping table as serializable dictionaries."""
    return [asdict(row) for row in cross_benchmark_metric_mappings()]


def compute_cross_benchmark_metric_rows(
    data: EpisodeData,
    *,
    horizon: int | None = None,
) -> list[CrossBenchmarkMetricRow]:
    """Compute external-style metric rows from one Robot SF episode trace.

    Args:
        data: Episode trace container.
        horizon: Optional episode horizon. Required for the Robot SF success predicate row.

    Returns:
        List of metric rows with explicit availability and approximation status.
    """
    compute_by_metric_id: dict[str, tuple[Callable[[], float], str]] = {
        "socnavbench.path_length_ratio": (
            lambda: socnavbench_path_length_ratio(data),
            "invalid_or_empty_trajectory",
        ),
        "common.traversal_time_s": (lambda: time_to_goal(data), "goal_not_reached"),
        "common.time_to_collision_min_s": (
            lambda: time_to_collision_min(data),
            "no_approaching_pair_or_pedestrians",
        ),
        "common.closest_pedestrian_distance_m": (
            lambda: distance_to_human_min(data),
            "no_pedestrians",
        ),
    }
    if horizon is not None:
        compute_by_metric_id["robot_sf.success_trace_predicate"] = (
            lambda: success(data, horizon=horizon),
            "invalid_horizon_or_trace",
        )

    rows: list[CrossBenchmarkMetricRow] = []
    for mapping in cross_benchmark_metric_mappings():
        if mapping.mapping_class == "unavailable":
            rows.append(_unavailable_row(mapping, reason="external_metric_not_trace_derivable"))
            continue
        if mapping.metric_id not in compute_by_metric_id:
            rows.append(_unavailable_row(mapping, reason="required_context_missing"))
            continue
        compute, unavailable_reason = compute_by_metric_id[mapping.metric_id]
        value = compute()
        rows.append(_row_from_value(mapping, value=value, unavailable_reason=unavailable_reason))
    return rows


def build_cross_benchmark_metric_report(
    data: EpisodeData,
    *,
    horizon: int | None = None,
) -> dict[str, Any]:
    """Build a compact report payload for one fixture trace.

    Returns:
        JSON-safe report with schema, mapping version, claim boundary, and rows.
    """
    rows = compute_cross_benchmark_metric_rows(data, horizon=horizon)
    return {
        "schema_version": CROSS_BENCHMARK_METRIC_REPORT_SCHEMA_VERSION,
        "mapping_version": CROSS_BENCHMARK_METRIC_MAPPING_VERSION,
        "claim_boundary": CROSS_BENCHMARK_CLAIM_BOUNDARY,
        "rows": [row.to_dict() for row in rows],
    }


def summarize_status_counts(rows: list[CrossBenchmarkMetricRow]) -> dict[str, int]:
    """Count wrapper row statuses for compact report checks.

    Returns:
        Mapping from each known row status to its count.
    """
    counts = dict.fromkeys(sorted(VALUE_STATUSES), 0)
    for row in rows:
        counts[row.status] += 1
    return counts


def _row_from_value(
    mapping: CrossBenchmarkMetricMapping,
    *,
    value: float,
    unavailable_reason: str,
) -> CrossBenchmarkMetricRow:
    """Convert one computed scalar into a status-preserving report row.

    Returns:
        Available or approximate row when the value is finite, otherwise unavailable.
    """
    if not math.isfinite(value):
        return _unavailable_row(mapping, reason=unavailable_reason)
    status = "approximate" if mapping.mapping_class == "approximate" else "available"
    return CrossBenchmarkMetricRow(
        metric_id=mapping.metric_id,
        benchmark=mapping.benchmark,
        external_metric=mapping.external_metric,
        status=status,
        mapping_class=mapping.mapping_class,
        value=float(value),
        units=mapping.units,
        denominator=mapping.denominator,
        robot_sf_source=mapping.robot_sf_source,
        evidence_tier=mapping.evidence_tier,
        semantic_notes=mapping.semantic_notes,
    )


def _unavailable_row(
    mapping: CrossBenchmarkMetricMapping,
    *,
    reason: str,
) -> CrossBenchmarkMetricRow:
    """Return an unavailable row while preserving the mapping rationale."""
    return CrossBenchmarkMetricRow(
        metric_id=mapping.metric_id,
        benchmark=mapping.benchmark,
        external_metric=mapping.external_metric,
        status="unavailable",
        mapping_class=mapping.mapping_class,
        value=None,
        units=mapping.units,
        denominator=mapping.denominator,
        robot_sf_source=mapping.robot_sf_source,
        evidence_tier=mapping.evidence_tier,
        semantic_notes=mapping.semantic_notes,
        unavailable_reason=reason,
    )


def mapping_ids(rows: Mapping[str, Any] | list[Mapping[str, Any]]) -> set[str]:
    """Return metric IDs from serialized mapping rows."""
    iterable = rows.values() if isinstance(rows, Mapping) else rows
    return {str(row["metric_id"]) for row in iterable}


__all__ = [
    "CROSS_BENCHMARK_CLAIM_BOUNDARY",
    "CROSS_BENCHMARK_METRIC_MAPPING_VERSION",
    "CROSS_BENCHMARK_METRIC_REPORT_SCHEMA_VERSION",
    "CrossBenchmarkMetricMapping",
    "CrossBenchmarkMetricRow",
    "build_cross_benchmark_metric_report",
    "compute_cross_benchmark_metric_rows",
    "cross_benchmark_metric_mappings",
    "mapping_ids",
    "mapping_table_as_dicts",
    "summarize_status_counts",
]

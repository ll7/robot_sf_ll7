"""Trace-level failure predicates for analysis-workbench exports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceFrame,
)

TRACE_FAILURE_PREDICATE_SCHEMA_VERSION = "trace_failure_predicates.v1"
TRACE_FAILURE_PREDICATE_SOURCE = "trace_failure_predicates.rule_based.v1"
TRACE_PREDICATE_DENOMINATOR_HEALTH_SCHEMA_VERSION = "trace_predicate_denominator_health.v1"
TRACE_PREDICATE_MATRIX_SCHEMA_VERSION = "trace_predicate_matrix.v1"

DEFAULT_CLEARANCE_THRESHOLD_M = 0.4
DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M = 0.2
DEFAULT_NEAR_MISS_THRESHOLD_M = 0.5
DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_M = 0.05
DEFAULT_LOW_PROGRESS_DISPLACEMENT_THRESHOLD_M = 0.25
DEFAULT_OSCILLATION_MIN_SIGN_CHANGES = 3
DEFAULT_EVASIVE_ANGULAR_VELOCITY_THRESHOLD = 1.0
DEFAULT_EVASIVE_LINEAR_DROP_THRESHOLD_MPS = 0.3

VALIDITY_STATUS_VALID = "valid"
VALIDITY_STATUS_UNAVAILABLE_DATA = "not_available"
VALIDITY_STATUS_NO_PREDICATE_OBSERVED = "no_predicate_observed"
VALIDITY_STATUS_PIPELINE_FAILURE = "pipeline_failure"
VALIDITY_STATUS_ABSENT_EXPECTED_SLICE = "absent_expected_slice"
DENOMINATOR_HEALTH_STATUSES = (
    VALIDITY_STATUS_VALID,
    VALIDITY_STATUS_UNAVAILABLE_DATA,
    VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
    VALIDITY_STATUS_PIPELINE_FAILURE,
    VALIDITY_STATUS_ABSENT_EXPECTED_SLICE,
)

TRACE_FAILURE_PREDICATE_IDS = (
    "collision",
    "late_evasive_reaction",
    "oscillatory_local_control",
    "occlusion_triggered_near_miss",
    "bottleneck_deadlock",
    "zero_motion_timeout_behavior",
    "low_progress",
    "clearance_critical_interaction",
)

MATRIX_STATUS_DIAGNOSTIC_ONLY = "diagnostic_only"
MATRIX_STATUS_PROPOSED = "proposed"
MATRIX_STATUS_RATE_INTERPRETABLE = "rate_interpretable"
MATRIX_CLAIM_INELIGIBLE = "claim-ineligible"
MATRIX_CLAIM_ELIGIBLE = "claim-eligible"


def load_trace_predicate_matrix(path: str | Path) -> dict[str, Any]:
    """Load a predeclared trace-predicate benchmark matrix from YAML.

    Returns:
        Parsed matrix payload.

    Raises:
        ValueError: if the file is missing or has an unsupported schema version.
    """
    matrix_path = Path(path)
    if not matrix_path.is_file():
        raise ValueError(f"Trace predicate matrix not found: {matrix_path}")
    try:
        payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid trace predicate matrix YAML: {matrix_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Trace predicate matrix must be a mapping: {matrix_path}")
    schema = payload.get("schema_version", "")
    if schema != TRACE_PREDICATE_MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trace predicate matrix schema version: {schema!r}. "
            f"Expected {TRACE_PREDICATE_MATRIX_SCHEMA_VERSION}."
        )
    return payload


def matrix_required_fields_for_predicate(matrix: Mapping[str, Any], predicate_id: str) -> list[str]:
    """Return required trace fields declared in the matrix for a predicate ID."""
    matrix_spec = matrix.get("matrix", {})
    if not isinstance(matrix_spec, Mapping):
        return []
    fields_by_predicate = matrix_spec.get("required_trace_fields_by_predicate", {})
    if not isinstance(fields_by_predicate, Mapping):
        return []
    fields = fields_by_predicate.get(predicate_id, [])
    return list(fields) if isinstance(fields, list | tuple) else []


def _trace_has_field(trace: SimulationTraceExport, field_path: str) -> bool:  # noqa: C901
    """Check whether at least one frame provides a dotted trace field.

    Supports paths such as ``robot.position``, ``pedestrians.position``,
    ``planner.event``, ``planner.selected_action.angular_velocity``, and
    ``planner.occlusion_or_visibility``.

    Returns:
        True when at least one frame contains the field, False otherwise.
    """
    if field_path == "planner.occlusion_or_visibility":
        return any(_occlusion_value(frame.planner) is not None for frame in trace.frames)
    parts = field_path.split(".")
    if not parts:
        return False
    for frame in trace.frames:
        if parts[0] == "robot":
            container = frame.robot
        elif parts[0] == "pedestrians":
            if not frame.pedestrians:
                continue
            # For pedestrians, require at least one pedestrian to have the nested field.
            if len(parts) == 1:
                return True
            if any(_nested_has(pedestrian, parts[1:]) for pedestrian in frame.pedestrians):
                return True
            continue
        elif parts[0] == "planner":
            container = frame.planner
        else:
            return False
        if _nested_has(container, parts[1:]):
            return True
    return False


def _nested_has(container: Any, path_parts: list[str]) -> bool:
    """Return whether a nested mapping contains every key in path_parts."""
    current: Any = container
    for part in path_parts:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return False
    return True


def _missing_required_fields(
    trace: SimulationTraceExport,
    predicate_id: str,
    matrix: Mapping[str, Any] | None,
) -> list[str]:
    """Return required trace fields declared by the matrix that are absent from the trace."""
    if matrix is None:
        return []
    required = matrix_required_fields_for_predicate(matrix, predicate_id)
    return [field for field in required if not _trace_has_field(trace, field)]


@dataclass(frozen=True, slots=True)
class TraceFailurePredicate:
    """One explicit trace-level failure predicate record."""

    predicate_id: str
    time_interval_s: list[float | None]
    steps: list[int | None]
    involved_actors: list[str]
    scenario_family: str
    planner_id: str
    evidence_fields: dict[str, Any]
    severity: str
    validity_status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the predicate to JSON-safe primitives.

        Returns:
            Dictionary representation of the predicate record.
        """
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceFailurePredicateDefinition:
    """Stable documentation for one reusable trace failure predicate."""

    predicate_id: str
    description: str
    inputs: list[str]
    units: dict[str, str]
    thresholds: dict[str, Any]
    denominator_semantics: str
    emitted_statuses: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the predicate definition to JSON-safe primitives.

        Returns:
            Dictionary representation of the definition record.
        """
        return asdict(self)


def build_trace_failure_predicate_definitions(
    *,
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    collision_missing_radius_near_distance_m: float = (
        DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M
    ),
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
    stationary_displacement_threshold_m: float = DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_M,
    low_progress_displacement_threshold_m: float = DEFAULT_LOW_PROGRESS_DISPLACEMENT_THRESHOLD_M,
    oscillation_min_sign_changes: int = DEFAULT_OSCILLATION_MIN_SIGN_CHANGES,
    evasive_angular_velocity_threshold: float = DEFAULT_EVASIVE_ANGULAR_VELOCITY_THRESHOLD,
    evasive_linear_drop_threshold_mps: float = DEFAULT_EVASIVE_LINEAR_DROP_THRESHOLD_MPS,
) -> list[dict[str, Any]]:
    """Return machine-readable predicate definitions and denominator semantics.

    Returns:
        JSON-safe definition rows ordered by ``TRACE_FAILURE_PREDICATE_IDS``.
    """
    definitions = (
        TraceFailurePredicateDefinition(
            predicate_id="collision",
            description=(
                "Robot-pedestrian center distance is at or below the sum of robot and pedestrian "
                "radii."
            ),
            inputs=[
                "robot.position",
                "robot.radius",
                "pedestrians.id",
                "pedestrians.position",
                "pedestrians.radius",
            ],
            units={"distance": "m", "radius": "m"},
            thresholds={"missing_radius_near_distance_m": collision_missing_radius_near_distance_m},
            denominator_semantics=(
                "The denominator requires robot and pedestrian radii. Near-contact geometry "
                "without radii emits not_available instead of a benchmark-collision claim."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="late_evasive_reaction",
            description=(
                "Robot first enters near-miss pressure and only later issues a strong angular "
                "command or linear-speed drop."
            ),
            inputs=[
                "robot.position",
                "pedestrians.id",
                "pedestrians.position",
                "planner.selected_action.linear_velocity",
                "planner.selected_action.angular_velocity",
            ],
            units={"distance": "m", "linear_velocity": "m/s", "angular_velocity": "rad/s"},
            thresholds={
                "near_miss_threshold_m": near_miss_threshold_m,
                "evasive_angular_velocity_threshold": evasive_angular_velocity_threshold,
                "evasive_linear_drop_threshold_mps": evasive_linear_drop_threshold_mps,
            },
            denominator_semantics=(
                "No emitted row means no late evasive pattern was observed for that trace, unless "
                "matrix-required fields mark the predicate not_available."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="oscillatory_local_control",
            description="Selected angular velocity changes sign at least the configured count.",
            inputs=["planner.selected_action.angular_velocity"],
            units={"angular_velocity": "rad/s"},
            thresholds={"oscillation_min_sign_changes": oscillation_min_sign_changes},
            denominator_semantics=(
                "No emitted row means the angular-command sign-change threshold was not met, unless "
                "matrix-required fields mark the predicate not_available."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="occlusion_triggered_near_miss",
            description="A near miss occurs on a frame with explicit occlusion or blocked visibility.",
            inputs=[
                "robot.position",
                "pedestrians.id",
                "pedestrians.position",
                "planner.occlusion_or_visibility",
            ],
            units={"distance": "m"},
            thresholds={"near_miss_threshold_m": near_miss_threshold_m},
            denominator_semantics=(
                "Near-miss geometry without occlusion/visibility evidence emits not_available "
                "instead of being treated as false."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="bottleneck_deadlock",
            description="Planner metadata reports an explicit bottleneck_deadlock event.",
            inputs=["planner.event"],
            units={},
            thresholds={"event": "bottleneck_deadlock"},
            denominator_semantics=(
                "No emitted row means no bottleneck_deadlock event was observed, unless "
                "matrix-required fields mark the predicate not_available."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="zero_motion_timeout_behavior",
            description="Timeout or truncation occurs while total robot displacement stays near zero.",
            inputs=["robot.position", "planner.event"],
            units={"displacement": "m"},
            thresholds={
                "stationary_displacement_threshold_m": stationary_displacement_threshold_m,
                "timeout_events": ["timeout", "time_limit", "max_steps", "truncated"],
            },
            denominator_semantics=(
                "No emitted row means either no timeout/truncation event was observed or the robot "
                "moved beyond the stationary threshold."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="low_progress",
            description=(
                "Timeout or truncation occurs while total robot displacement remains below the "
                "low-progress threshold."
            ),
            inputs=["robot.position", "planner.event"],
            units={"displacement": "m"},
            thresholds={
                "low_progress_displacement_threshold_m": low_progress_displacement_threshold_m,
                "timeout_events": ["timeout", "time_limit", "max_steps", "truncated"],
            },
            denominator_semantics=(
                "No emitted row means either no timeout/truncation event was observed or the robot "
                "moved beyond the low-progress threshold."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
        TraceFailurePredicateDefinition(
            predicate_id="clearance_critical_interaction",
            description="Robot-pedestrian center distance is at or below the clearance threshold.",
            inputs=["robot.position", "pedestrians.id", "pedestrians.position"],
            units={"distance": "m"},
            thresholds={"clearance_threshold_m": clearance_threshold_m},
            denominator_semantics=(
                "Every loaded trace contributes one denominator count. Missing positions prevent "
                "emission and are tracked through denominator health when declared in a matrix."
            ),
            emitted_statuses=[VALIDITY_STATUS_VALID, VALIDITY_STATUS_UNAVAILABLE_DATA],
        ),
    )
    definitions_by_id = {definition.predicate_id: definition for definition in definitions}
    return [
        definitions_by_id[predicate_id].to_dict() for predicate_id in TRACE_FAILURE_PREDICATE_IDS
    ]


def extract_trace_failure_predicates(  # noqa: PLR0913
    trace: SimulationTraceExport,
    *,
    scenario_family: str | None = None,
    matrix: Mapping[str, Any] | None = None,
    clearance_threshold_m: float = DEFAULT_CLEARANCE_THRESHOLD_M,
    collision_missing_radius_near_distance_m: float = (
        DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M
    ),
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
    stationary_displacement_threshold_m: float = DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_M,
    low_progress_displacement_threshold_m: float = DEFAULT_LOW_PROGRESS_DISPLACEMENT_THRESHOLD_M,
    oscillation_min_sign_changes: int = DEFAULT_OSCILLATION_MIN_SIGN_CHANGES,
    evasive_angular_velocity_threshold: float = DEFAULT_EVASIVE_ANGULAR_VELOCITY_THRESHOLD,
    evasive_linear_drop_threshold_mps: float = DEFAULT_EVASIVE_LINEAR_DROP_THRESHOLD_MPS,
) -> dict[str, Any]:
    """Extract dissertation-relevant trace failure predicates.

    The extractor is intentionally conservative: it emits ``not_available`` records when a
    predicate has partial evidence but required fields are absent, rather than silently treating the
    predicate as false.

    Args:
        trace: Typed simulation trace export.
        scenario_family: Optional scenario-family override.
        matrix: Optional predeclared trace-predicate benchmark matrix. When provided, missing
            matrix-required trace fields cause ``not_available`` predicate rows.
        clearance_threshold_m: Distance threshold for clearance-critical interactions.
        collision_missing_radius_near_distance_m: Near-contact distance that emits
            ``not_available`` when collision radii are missing.
        near_miss_threshold_m: Distance threshold for near-miss and evasive predicates.
        stationary_displacement_threshold_m: Displacement threshold for zero-motion detection.
        low_progress_displacement_threshold_m: Displacement threshold for timeout low-progress
            detection.
        oscillation_min_sign_changes: Minimum angular sign changes for oscillation.
        evasive_angular_velocity_threshold: Angular velocity threshold for evasive reactions.
        evasive_linear_drop_threshold_mps: Linear velocity drop threshold for evasive reactions.

    Returns:
        ``trace_failure_predicates.v1`` payload with predicate rows and summary counts.
    """
    predicates = _extract_trace_failure_predicate_records(
        trace,
        scenario_family=scenario_family,
        matrix=matrix,
        clearance_threshold_m=clearance_threshold_m,
        collision_missing_radius_near_distance_m=collision_missing_radius_near_distance_m,
        near_miss_threshold_m=near_miss_threshold_m,
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        low_progress_displacement_threshold_m=low_progress_displacement_threshold_m,
        oscillation_min_sign_changes=oscillation_min_sign_changes,
        evasive_angular_velocity_threshold=evasive_angular_velocity_threshold,
        evasive_linear_drop_threshold_mps=evasive_linear_drop_threshold_mps,
    )
    return _predicates_payload(
        trace=trace,
        predicates=predicates,
        matrix=matrix,
        predicate_definitions=build_trace_failure_predicate_definitions(
            clearance_threshold_m=clearance_threshold_m,
            collision_missing_radius_near_distance_m=collision_missing_radius_near_distance_m,
            near_miss_threshold_m=near_miss_threshold_m,
            stationary_displacement_threshold_m=stationary_displacement_threshold_m,
            low_progress_displacement_threshold_m=low_progress_displacement_threshold_m,
            oscillation_min_sign_changes=oscillation_min_sign_changes,
            evasive_angular_velocity_threshold=evasive_angular_velocity_threshold,
            evasive_linear_drop_threshold_mps=evasive_linear_drop_threshold_mps,
        ),
    )


def aggregate_trace_failure_predicate_tables(
    traces: Iterable[SimulationTraceExport],
    *,
    scenario_family: str | None = None,
    matrix: Mapping[str, Any] | None = None,
    failed_trace_ids: Iterable[str] | None = None,
    failed_trace_slices: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build denominator-aware aggregate rows across one or more trace predicates.

    Args:
        traces: Typed simulation trace exports.
        scenario_family: Optional aggregate scenario-family label.
        matrix: Optional predeclared trace-predicate benchmark matrix. When absent, outputs are
            diagnostic-only and claim-ineligible.
        failed_trace_ids: Trace IDs that failed to load.
        failed_trace_slices: Optional structured metadata for traces that failed before parsing.

    Returns:
        Aggregate rows grouped by scenario family, planner, seed, predicate ID,
        validity status, and severity.
    """
    grouped_predicates: Counter[tuple[str, str, int, str, str, str]] = Counter()
    trace_denominators: Counter[tuple[str, str, int]] = Counter()
    trace_sources_by_group: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    trace_status_counts: dict[tuple[str, str, int, str], Counter[str]] = defaultdict(Counter)
    included_trace_count = 0
    predicate_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    failed_trace_id_list = list(failed_trace_ids or [])
    failed_trace_slice_rows = _json_safe_failed_trace_slices(failed_trace_slices)
    failed_trace_count = _failed_trace_count(
        failed_trace_ids=failed_trace_id_list,
        failed_trace_slices=failed_trace_slice_rows,
    )
    if failed_trace_count:
        for _failure_index in range(failed_trace_count):
            for pred_id in TRACE_FAILURE_PREDICATE_IDS:
                predicate_status_counts[pred_id][VALIDITY_STATUS_PIPELINE_FAILURE] += 1

    for trace in traces:
        group_family = scenario_family or trace.source.scenario_id
        trace_source_id = trace.trace_id
        group_key = (group_family, trace.source.planner_id, int(trace.source.seed))
        trace_key = (*group_key, trace_source_id)
        trace_denominators[group_key] += 1
        trace_sources_by_group[group_key].add(trace_source_id)
        included_trace_count += 1
        extracted_predicates = _extract_trace_failure_predicate_records(
            trace,
            scenario_family=group_family,
            matrix=matrix,
            clearance_threshold_m=DEFAULT_CLEARANCE_THRESHOLD_M,
            collision_missing_radius_near_distance_m=(
                DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M
            ),
            near_miss_threshold_m=DEFAULT_NEAR_MISS_THRESHOLD_M,
            stationary_displacement_threshold_m=DEFAULT_STATIONARY_DISPLACEMENT_THRESHOLD_M,
            low_progress_displacement_threshold_m=DEFAULT_LOW_PROGRESS_DISPLACEMENT_THRESHOLD_M,
            oscillation_min_sign_changes=DEFAULT_OSCILLATION_MIN_SIGN_CHANGES,
            evasive_angular_velocity_threshold=DEFAULT_EVASIVE_ANGULAR_VELOCITY_THRESHOLD,
            evasive_linear_drop_threshold_mps=DEFAULT_EVASIVE_LINEAR_DROP_THRESHOLD_MPS,
        )
        predicate_statuses_in_trace: dict[str, set[str]] = defaultdict(set)
        for predicate in extracted_predicates:
            grouped_predicates[
                (
                    group_family,
                    predicate.planner_id,
                    int(trace.source.seed),
                    predicate.predicate_id,
                    predicate.validity_status,
                    predicate.severity,
                )
            ] += 1
            trace_status_counts[trace_key][predicate.validity_status] += 1
            predicate_statuses_in_trace[predicate.predicate_id].add(predicate.validity_status)

        _record_predicate_denominator_statuses(
            predicate_status_counts,
            predicate_statuses_in_trace,
        )

    expected_slices = _matrix_expected_slices(matrix, scenario_family_filter=scenario_family)
    absent_expected_slices = _absent_expected_slice_rows(
        expected_slices=expected_slices,
        observed_slices=set(trace_denominators),
        failed_trace_ids=failed_trace_id_list,
        failed_trace_slices=failed_trace_slice_rows,
    )
    absent_expected_slice_count = len(absent_expected_slices)
    _record_absent_expected_slice_statuses(
        predicate_status_counts,
        absent_expected_slice_count=absent_expected_slice_count,
    )

    rows = [
        {
            "scenario_family": family,
            "planner_id": planner_id,
            "seed": seed,
            "trace_source_ids": sorted(trace_sources_by_group[(family, planner_id, seed)]),
            "predicate_id": predicate_id,
            "validity_status": validity_status,
            "severity": severity,
            "predicate_count": count,
            "trace_denominator": trace_denominators[(family, planner_id, seed)],
            "predicate_rate_per_trace": (count / trace_denominators[(family, planner_id, seed)]),
        }
        for (
            family,
            planner_id,
            seed,
            predicate_id,
            validity_status,
            severity,
        ), count in sorted(grouped_predicates.items())
    ]

    observed_group_keys = {
        (family, planner_id, seed)
        for (
            family,
            planner_id,
            seed,
            _predicate_id,
            _validity_status,
            _severity,
        ) in grouped_predicates
    }
    for group_key in sorted(trace_denominators):
        if group_key in observed_group_keys:
            continue
        family, planner_id, seed = group_key
        rows.append(
            {
                "scenario_family": family,
                "planner_id": planner_id,
                "seed": seed,
                "trace_source_ids": sorted(trace_sources_by_group[group_key]),
                "predicate_id": VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
                "validity_status": VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
                "severity": VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
                "predicate_count": 0,
                "trace_denominator": trace_denominators[group_key],
                "predicate_rate_per_trace": 0.0,
            }
        )

    zero_predicate_groups = [
        {
            "scenario_family": family,
            "planner_id": planner_id,
            "seed": seed,
            "trace_source_id": trace_source_id,
            "trace_denominator": trace_denominators[group_key],
            "valid_predicate_count": int(
                trace_status_counts[trace_key].get(VALIDITY_STATUS_VALID, 0)
            ),
            "not_available_predicate_count": int(
                trace_status_counts[trace_key].get(VALIDITY_STATUS_UNAVAILABLE_DATA, 0)
            ),
            "zero_group_reason": (
                VALIDITY_STATUS_NO_PREDICATE_OBSERVED
                if not trace_status_counts[trace_key]
                else "valid_predicate_count_zero"
            ),
        }
        for group_key, trace_sources in sorted(trace_sources_by_group.items())
        for family, planner_id, seed in (group_key,)
        for trace_source_id in sorted(trace_sources)
        for trace_key in ((*group_key, trace_source_id),)
        if trace_status_counts[trace_key].get(VALIDITY_STATUS_VALID, 0) == 0
    ]

    matrix_metadata = _matrix_metadata(matrix, included_traces=included_trace_count)
    return {
        "schema_version": TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
        "table_kind": "aggregate_by_predicate_group",
        "summary": {
            "input_trace_count": included_trace_count + failed_trace_count,
            "failed_trace_count": failed_trace_count,
            "expected_matrix_slice_count": len(expected_slices),
            "absent_expected_slice_count": absent_expected_slice_count,
            "scenario_family_filter": scenario_family,
            "row_count": len(rows),
        },
        "matrix_metadata": matrix_metadata,
        "predicate_definitions": build_trace_failure_predicate_definitions(),
        "denominator_status": _aggregate_denominator_status(
            included_trace_count=included_trace_count,
            failed_trace_count=failed_trace_count,
            absent_expected_slice_count=absent_expected_slice_count,
        ),
        "rows": rows,
        "failed_trace_slices": failed_trace_slice_rows,
        "absent_expected_slices": absent_expected_slices,
        "zero_predicate_groups": zero_predicate_groups,
        "predicate_denominator_health": {
            pred_id: {
                status: int(predicate_status_counts[pred_id].get(status, 0))
                for status in DENOMINATOR_HEALTH_STATUSES
            }
            for pred_id in TRACE_FAILURE_PREDICATE_IDS
        },
        "caveats": [
            "Aggregate predicate rows are diagnostic summaries, not benchmark rates or claims.",
            f"Rows include `{VALIDITY_STATUS_UNAVAILABLE_DATA}` and other valid status values when evidence is partial.",
            "`zero_predicate_groups` preserves trace groups with valid predicate count equal to zero.",
            f"Rows with `{VALIDITY_STATUS_NO_PREDICATE_OBSERVED}` mark trace groups that had no predicate rows at all.",
            f"Rows with `{VALIDITY_STATUS_PIPELINE_FAILURE}` mark traces that failed to load.",
            f"Rows or health counts with `{VALIDITY_STATUS_ABSENT_EXPECTED_SLICE}` mark predeclared matrix slices absent from both loaded traces and failed trace IDs.",
            "Rate claims require a predeclared benchmark matrix; do not infer rates from these diagnostic rows.",
        ],
    }


def _record_predicate_denominator_statuses(
    predicate_status_counts: dict[str, Counter[str]],
    predicate_statuses_in_trace: Mapping[str, set[str]],
) -> None:
    """Record one denominator status per predicate family for a trace."""
    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        statuses = predicate_statuses_in_trace.get(pred_id, set())
        if VALIDITY_STATUS_VALID in statuses:
            predicate_status_counts[pred_id][VALIDITY_STATUS_VALID] += 1
        elif VALIDITY_STATUS_UNAVAILABLE_DATA in statuses:
            predicate_status_counts[pred_id][VALIDITY_STATUS_UNAVAILABLE_DATA] += 1
        elif statuses:
            for status in sorted(statuses):
                predicate_status_counts[pred_id][status] += 1
        else:
            predicate_status_counts[pred_id][VALIDITY_STATUS_NO_PREDICATE_OBSERVED] += 1


def _record_absent_expected_slice_statuses(
    predicate_status_counts: dict[str, Counter[str]],
    *,
    absent_expected_slice_count: int,
) -> None:
    """Record missing matrix slices as claim-blocking predicate denominator health."""
    if absent_expected_slice_count <= 0:
        return
    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        predicate_status_counts[pred_id][VALIDITY_STATUS_ABSENT_EXPECTED_SLICE] += (
            absent_expected_slice_count
        )


def _absent_expected_slice_rows(
    *,
    expected_slices: set[tuple[str, str, int]],
    observed_slices: set[tuple[str, str, int]],
    failed_trace_ids: list[str],
    failed_trace_slices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return matrix slices absent from loaded traces and failed trace IDs."""
    failed_expected_slices = _failed_expected_slices(
        expected_slices=expected_slices,
        failed_trace_ids=failed_trace_ids,
        failed_trace_slices=failed_trace_slices,
    )
    return [
        {
            "scenario_family": family,
            "planner_id": planner_id,
            "seed": seed,
            "status": VALIDITY_STATUS_ABSENT_EXPECTED_SLICE,
        }
        for family, planner_id, seed in sorted(
            expected_slices - observed_slices - failed_expected_slices
        )
    ]


def _failed_expected_slices(
    *,
    expected_slices: set[tuple[str, str, int]],
    failed_trace_ids: list[str],
    failed_trace_slices: list[dict[str, Any]],
) -> set[tuple[str, str, int]]:
    """Return expected slices named by failed structured metadata or trace IDs."""
    failed_slices: set[tuple[str, str, int]] = set()
    covered_failed_ids: set[str] = set()
    for row in failed_trace_slices:
        source_path = row.get("source_path")
        if source_path is not None:
            covered_failed_ids.add(_failed_trace_source_key(str(source_path)))
        failed_slice = _complete_failed_trace_slice_metadata(row)
        if failed_slice in expected_slices:
            failed_slices.add(failed_slice)
    uncovered_failed_trace_ids = [
        failed_id
        for failed_id in failed_trace_ids
        if _failed_trace_source_key(failed_id) not in covered_failed_ids
    ]
    failed_slices.update(
        expected
        for expected in expected_slices
        if any(
            _failed_trace_id_matches_slice(failed_id, expected)
            for failed_id in uncovered_failed_trace_ids
        )
    )
    return failed_slices


def _failed_trace_count(
    *,
    failed_trace_ids: list[str],
    failed_trace_slices: list[dict[str, Any]],
) -> int:
    """Return a de-duplicated failed trace count across IDs and structured rows."""
    failed_sources = {_failed_trace_source_key(failed_id) for failed_id in failed_trace_ids}
    unkeyed_slice_count = 0
    for row in failed_trace_slices:
        source_path = row.get("source_path")
        if source_path is None:
            unkeyed_slice_count += 1
        else:
            failed_sources.add(_failed_trace_source_key(str(source_path)))
    return len(failed_sources) + unkeyed_slice_count


def _failed_trace_source_key(source: str) -> str:
    """Return the stable comparable key for a failed trace ID or source path."""
    return Path(source).name


def _json_safe_failed_trace_slices(
    failed_trace_slices: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return a bounded JSON-safe failed trace slice payload."""
    if failed_trace_slices is None:
        return []
    rows: list[dict[str, Any]] = []
    for raw_slice in failed_trace_slices:
        if not isinstance(raw_slice, Mapping):
            continue
        row: dict[str, Any] = {}
        for key in ("scenario_family", "scenario_id", "planner_id", "seed", "source_path"):
            value = raw_slice.get(key)
            if value is None:
                continue
            if key == "seed":
                try:
                    row[key] = int(value)
                except (TypeError, ValueError):
                    row[key] = str(value)
            else:
                row[key] = str(value)
        if row:
            rows.append(row)
    return rows


def _complete_failed_trace_slice_metadata(
    failed_trace_slice: Mapping[str, Any],
) -> tuple[str, str, int] | None:
    """Return a matrix slice tuple when structured failed metadata is complete."""
    raw_family = failed_trace_slice.get("scenario_family")
    raw_planner = failed_trace_slice.get("planner_id")
    raw_seed = failed_trace_slice.get("seed")
    if raw_family is None or raw_planner is None or raw_seed is None:
        return None
    scenario_family = str(raw_family)
    planner_id = str(raw_planner)
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        return None
    if not scenario_family or not planner_id:
        return None
    return (scenario_family, planner_id, seed)


def _matrix_expected_slices(
    matrix: Mapping[str, Any] | None,
    *,
    scenario_family_filter: str | None,
) -> set[tuple[str, str, int]]:
    """Return expected scenario/planner/seed slices declared by a matrix."""
    if matrix is None:
        return set()
    matrix_spec = matrix.get("matrix", {})
    if not isinstance(matrix_spec, Mapping):
        return set()
    scenario_families = _string_list(matrix_spec.get("scenario_families"))
    if scenario_family_filter is not None:
        scenario_families = [
            family for family in scenario_families if family == scenario_family_filter
        ]
    planners = _string_list(matrix_spec.get("planners"))
    seeds = _int_list(matrix_spec.get("seeds"))
    return {
        (scenario_family, planner_id, seed)
        for scenario_family in scenario_families
        for planner_id in planners
        for seed in seeds
    }


def _string_list(raw_values: Any) -> list[str]:
    """Return non-empty string values from a matrix field."""
    if not isinstance(raw_values, list | tuple):
        return []
    return [str(value) for value in raw_values if value is not None and str(value)]


def _int_list(raw_values: Any) -> list[int]:
    """Return integer values from a matrix field."""
    if not isinstance(raw_values, list | tuple):
        return []
    values: list[int] = []
    for value in raw_values:
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    return values


def _failed_trace_id_matches_slice(
    failed_trace_id: str,
    expected_slice: tuple[str, str, int],
) -> bool:
    """Return whether a failed trace ID names an expected scenario/planner/seed slice."""
    scenario_family, planner_id, seed = expected_slice
    normalized_failed_id = _normalized_identifier(failed_trace_id)
    expected_identifier = (
        f"{_normalized_identifier(scenario_family)}{_normalized_identifier(planner_id)}{seed}"
    )
    start = 0
    while True:
        match_index = normalized_failed_id.find(expected_identifier, start)
        if match_index == -1:
            return False
        next_char_index = match_index + len(expected_identifier)
        if (
            next_char_index >= len(normalized_failed_id)
            or not normalized_failed_id[next_char_index].isdigit()
        ):
            return True
        start = match_index + 1


def _normalized_identifier(value: Any) -> str:
    """Return a compact alphanumeric identifier for slice matching."""
    return "".join(char for char in str(value).lower() if char.isalnum())


def _aggregate_denominator_status(
    *,
    included_trace_count: int,
    failed_trace_count: int,
    absent_expected_slice_count: int = 0,
) -> str:
    """Classify aggregate denominator availability for machine-readable consumers.

    Returns:
        Denominator status string.
    """
    if failed_trace_count > 0:
        return VALIDITY_STATUS_PIPELINE_FAILURE
    if absent_expected_slice_count > 0:
        return VALIDITY_STATUS_UNAVAILABLE_DATA
    if included_trace_count > 0:
        return VALIDITY_STATUS_VALID
    return VALIDITY_STATUS_UNAVAILABLE_DATA


def render_trace_failure_predicate_markdown(
    table_payload: Mapping[str, Any], denominator_health_report: Mapping[str, Any] | None = None
) -> str:
    """Render aggregate failure predicate rows as a compact Markdown diagnostic table.

    Returns:
        Rendered markdown text.
    """
    summary = _read_mapping(table_payload.get("summary"), {})
    rows = table_payload.get("rows")
    if not isinstance(rows, list):
        rows = []
    table_rows = [row for row in rows if isinstance(row, Mapping)]
    health_report = (
        denominator_health_report
        if denominator_health_report is not None
        else build_trace_predicate_denominator_health_report(table_payload)
    )

    matrix_metadata = table_payload.get("matrix_metadata")
    matrix_name = (
        matrix_metadata.get("matrix_name") if isinstance(matrix_metadata, Mapping) else None
    )
    matrix_status = matrix_metadata.get("status") if isinstance(matrix_metadata, Mapping) else None
    lines = [
        "# Trace Failure Predicate Table",
        "",
        (
            "Aggregate predicate rows are diagnostic-only unless tied to a predeclared benchmark "
            "matrix."
        ),
        "",
        f"- input traces: {summary.get('input_trace_count', 0)}",
        f"- table kind: {table_payload.get('table_kind', 'aggregate_by_predicate_group')}",
        f"- schema version: {table_payload.get('schema_version', TRACE_FAILURE_PREDICATE_SCHEMA_VERSION)}",
    ]
    if matrix_name:
        lines.append(f"- matrix: {matrix_name}")
    if matrix_status:
        lines.append(f"- matrix status: {matrix_status}")
    lines.append("")

    if not table_rows:
        lines.append("No predicate rows were observed in the provided traces.")
        # Add denominator health report even if no predicate rows
        lines.append("")
        lines.extend(_render_denominator_health_section(health_report))
        return "\n".join(lines).rstrip() + "\n"

    zero_groups = table_payload.get("zero_predicate_groups")
    if not isinstance(zero_groups, list):
        zero_groups = []
    zero_group_rows = [row for row in zero_groups if isinstance(row, Mapping)]
    absent_slices = table_payload.get("absent_expected_slices")
    if not isinstance(absent_slices, list):
        absent_slices = []
    absent_slice_rows = [row for row in absent_slices if isinstance(row, Mapping)]
    observed_rows = [
        row
        for row in table_rows
        if row.get("predicate_id") != VALIDITY_STATUS_NO_PREDICATE_OBSERVED
    ]

    lines.append("## Aggregate Rows")
    lines.append("")
    if observed_rows:
        lines.append(
            _markdown_table(
                (
                    "scenario_family",
                    "planner_id",
                    "seed",
                    "trace_source_ids",
                    "predicate_id",
                    "validity_status",
                    "severity",
                    "predicate_count",
                    "trace_denominator",
                    "predicate_rate_per_trace",
                ),
                [
                    (
                        _markdown_value(row.get("scenario_family")),
                        _markdown_value(row.get("planner_id")),
                        _markdown_value(row.get("seed")),
                        _markdown_value(row.get("trace_source_ids")),
                        _markdown_value(row.get("predicate_id")),
                        _markdown_value(row.get("validity_status")),
                        _markdown_value(row.get("severity")),
                        _markdown_int(row.get("predicate_count")),
                        _markdown_int(row.get("trace_denominator")),
                        f"{_markdown_float(row.get('predicate_rate_per_trace')):.4f}",
                    )
                    for row in observed_rows
                ],
            )
        )
    if zero_group_rows:
        lines.append("")
        lines.append("### Trace Groups With Zero Valid Predicates")
        lines.append("")
        lines.append(
            _markdown_table(
                (
                    "scenario_family",
                    "planner_id",
                    "seed",
                    "trace_source_id",
                    "trace_denominator",
                    "valid_predicate_count",
                    "not_available_predicate_count",
                    "zero_group_reason",
                ),
                [
                    (
                        _markdown_value(row.get("scenario_family")),
                        _markdown_value(row.get("planner_id")),
                        _markdown_value(row.get("seed")),
                        _markdown_value(row.get("trace_source_id")),
                        _markdown_int(row.get("trace_denominator")),
                        _markdown_int(row.get("valid_predicate_count")),
                        _markdown_int(row.get("not_available_predicate_count")),
                        _markdown_value(row.get("zero_group_reason")),
                    )
                    for row in zero_group_rows
                ],
            )
        )
    if absent_slice_rows:
        lines.append("")
        lines.append("### Expected Matrix Slices Absent From Inputs")
        lines.append("")
        lines.append(
            _markdown_table(
                (
                    "scenario_family",
                    "planner_id",
                    "seed",
                    "status",
                ),
                [
                    (
                        _markdown_value(row.get("scenario_family")),
                        _markdown_value(row.get("planner_id")),
                        _markdown_value(row.get("seed")),
                        _markdown_value(row.get("status")),
                    )
                    for row in absent_slice_rows
                ],
            )
        )
    lines.append("")
    lines.extend(_render_denominator_health_section(health_report))
    return "\n".join(lines).rstrip() + "\n"


def build_trace_predicate_denominator_health_report(
    table_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build compact denominator-health and claim-eligibility diagnostics.

    Returns:
        JSON-safe report grouped by predicate family.
    """
    summary = _read_mapping(table_payload.get("summary"), {})
    input_trace_count = int(summary.get("input_trace_count", 0) or 0)
    failed_trace_count = int(summary.get("failed_trace_count", 0) or 0)
    absent_expected_slice_count = int(summary.get("absent_expected_slice_count", 0) or 0)
    raw_health = table_payload.get("predicate_denominator_health")
    health_by_predicate = raw_health if isinstance(raw_health, Mapping) else {}
    matrix_metadata = table_payload.get("matrix_metadata")
    matrix_present = (
        isinstance(matrix_metadata, Mapping)
        and matrix_metadata.get("status") == MATRIX_STATUS_RATE_INTERPRETABLE
    )
    predicate_reports: dict[str, dict[str, Any]] = {}

    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        raw_counts = health_by_predicate.get(pred_id)
        counts = raw_counts if isinstance(raw_counts, Mapping) else {}
        valid_count = int(counts.get(VALIDITY_STATUS_VALID, 0) or 0)
        unavailable_count = int(counts.get(VALIDITY_STATUS_UNAVAILABLE_DATA, 0) or 0)
        no_predicate_count = int(counts.get(VALIDITY_STATUS_NO_PREDICATE_OBSERVED, 0) or 0)
        pipeline_failure_count = int(counts.get(VALIDITY_STATUS_PIPELINE_FAILURE, 0) or 0)
        absent_slice_count = int(counts.get(VALIDITY_STATUS_ABSENT_EXPECTED_SLICE, 0) or 0)
        total_count = (
            valid_count
            + unavailable_count
            + no_predicate_count
            + pipeline_failure_count
            + absent_slice_count
        )
        if total_count == 0 and input_trace_count > 0:
            no_predicate_count = max(input_trace_count - failed_trace_count, 0)
            pipeline_failure_count = failed_trace_count
            absent_slice_count = absent_expected_slice_count
            total_count = input_trace_count + absent_slice_count
        claim_eligibility, allowed_wording = _predicate_claim_status(
            valid_count=valid_count,
            unavailable_count=unavailable_count,
            no_predicate_count=no_predicate_count,
            pipeline_failure_count=pipeline_failure_count,
            absent_slice_count=absent_slice_count,
        )
        if not matrix_present:
            claim_eligibility = MATRIX_CLAIM_INELIGIBLE
            allowed_wording = f"{allowed_wording}; predeclared benchmark matrix is required"
        predicate_reports[pred_id] = {
            "total_denominator_count": total_count,
            "valid_count": valid_count,
            "unavailable_data_count": unavailable_count,
            "no_predicate_observed_count": no_predicate_count,
            "pipeline_failure_count": pipeline_failure_count,
            "absent_expected_slice_count": absent_slice_count,
            "valid_ratio": _ratio(valid_count, total_count),
            "unavailable_data_ratio": _ratio(unavailable_count, total_count),
            "no_predicate_observed_ratio": _ratio(no_predicate_count, total_count),
            "pipeline_failure_ratio": _ratio(pipeline_failure_count, total_count),
            "absent_expected_slice_ratio": _ratio(absent_slice_count, total_count),
            "claim_eligibility": claim_eligibility,
            "allowed_wording": allowed_wording,
        }

    caveats = [
        "This report is diagnostic-only and does not convert missing predicate evidence into negative findings.",
        "`no_predicate_observed` can be expected missingness; it is not evidence that a failure did not occur.",
        "`not_available`, `pipeline_failure`, and `absent_expected_slice` weaken or block downstream figure/table claims.",
    ]
    if not matrix_present:
        caveats.append(
            "No predeclared benchmark matrix was provided or the matrix status is not "
            f"`{MATRIX_STATUS_RATE_INTERPRETABLE}`; this report is claim-ineligible."
        )

    return {
        "schema_version": TRACE_PREDICATE_DENOMINATOR_HEALTH_SCHEMA_VERSION,
        "report_kind": "trace_predicate_denominator_health",
        "summary": {
            "total_traces_processed": input_trace_count,
            "failed_trace_count": failed_trace_count,
            "absent_expected_slice_count": absent_expected_slice_count,
            "predicate_family_count": len(TRACE_FAILURE_PREDICATE_IDS),
            "denominator_status": _aggregate_denominator_status(
                included_trace_count=max(input_trace_count - failed_trace_count, 0),
                failed_trace_count=failed_trace_count,
                absent_expected_slice_count=absent_expected_slice_count,
            ),
        },
        "predicates": predicate_reports,
        "caveats": caveats,
    }


def _render_denominator_health_section(denominator_health_report: Mapping[str, Any]) -> list[str]:
    """Render the predicate denominator health section for the Markdown report.

    Returns:
        Markdown lines for denominator health.
    """
    lines = ["## Predicate Denominator Health Report", ""]
    overall_summary = denominator_health_report.get("summary", {})
    lines.append(f"- Total Traces Processed: {overall_summary.get('total_traces_processed', 0)}")
    lines.append(f"- Traces with Pipeline Failures: {overall_summary.get('failed_trace_count', 0)}")
    lines.append(
        "- Expected Matrix Slices Absent From Inputs: "
        f"{overall_summary.get('absent_expected_slice_count', 0)}"
    )
    lines.append("")
    lines.append("### Missingness Categories")
    lines.append(f"- `{VALIDITY_STATUS_VALID}`: predicate evaluated and emitted valid evidence.")
    lines.append(
        f"- `{VALIDITY_STATUS_UNAVAILABLE_DATA}`: required trace fields were missing; dependent "
        "figures/tables are claim-ineligible or wording-weakened."
    )
    lines.append(
        f"- `{VALIDITY_STATUS_NO_PREDICATE_OBSERVED}`: predicate was not emitted for the trace; "
        "this can be expected missingness and is not a negative finding."
    )
    lines.append(
        f"- `{VALIDITY_STATUS_PIPELINE_FAILURE}`: the trace could not be loaded or processed; "
        "dependent outputs are claim-ineligible."
    )
    lines.append(
        f"- `{VALIDITY_STATUS_ABSENT_EXPECTED_SLICE}`: a scenario/planner/seed matrix slice was "
        "absent from both loaded traces and failed trace IDs; dependent outputs are "
        "claim-ineligible."
    )
    lines.append("")

    predicate_data = denominator_health_report.get("predicates", {})
    if not predicate_data:
        lines.append("No predicate denominator health data available.")
        return lines

    health_rows = []
    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        data = predicate_data.get(pred_id, {})
        if not data:
            continue
        health_rows.append(
            (
                pred_id,
                _markdown_int(data.get("total_denominator_count")),
                _markdown_int(data.get("valid_count")),
                _markdown_int(data.get("unavailable_data_count")),
                f"{_markdown_float(data.get('unavailable_data_ratio')):.2%}",
                _markdown_int(data.get("no_predicate_observed_count")),
                f"{_markdown_float(data.get('no_predicate_observed_ratio')):.2%}",
                _markdown_int(data.get("pipeline_failure_count")),
                f"{_markdown_float(data.get('pipeline_failure_ratio')):.2%}",
                _markdown_int(data.get("absent_expected_slice_count")),
                f"{_markdown_float(data.get('absent_expected_slice_ratio')):.2%}",
                _markdown_value(data.get("claim_eligibility")),
                _markdown_value(data.get("allowed_wording")),
            )
        )

    if health_rows:
        lines.append("### Predicate-specific Denominator Health")
        lines.append("")
        lines.append(
            _markdown_table(
                (
                    "Predicate ID",
                    "Total Traces",
                    "Valid",
                    "Unavailable Data",
                    "% Unavailable",
                    "Not Observed",
                    "% Not Observed",
                    "Pipeline Failure",
                    "% Pipeline Fail",
                    "Absent Expected Slice",
                    "% Absent Slice",
                    "Claim Eligibility",
                    "Allowed Wording",
                ),
                health_rows,
            )
        )
    return lines


def _predicate_claim_status(
    *,
    valid_count: int,
    unavailable_count: int,
    no_predicate_count: int,
    pipeline_failure_count: int,
    absent_slice_count: int,
) -> tuple[str, str]:
    """Classify predicate-family claim eligibility from denominator health.

    Returns:
        Claim eligibility status and allowed wording guidance.
    """
    if pipeline_failure_count > 0:
        return "claim-ineligible", "pipeline failed; do not make predicate-family claims"
    if absent_slice_count > 0:
        return "claim-ineligible", "expected matrix slice is absent from provided inputs"
    if valid_count == 0 and unavailable_count > 0:
        return "claim-ineligible", "required predicate evidence is unavailable"
    if valid_count == 0 and no_predicate_count > 0:
        return "wording-weakened", "only report no observed predicate rows, not absence of failure"
    if unavailable_count > 0:
        return "wording-weakened", "qualify figure/table wording due to missing predicate evidence"
    if no_predicate_count > 0:
        return "wording-weakened", "state that some traces emitted no predicate rows"
    return "claim-eligible", "diagnostic wording may cite observed predicate rows"


def _ratio(numerator: int, denominator: int) -> float:
    """Return a stable ratio for report payloads."""
    return 0.0 if denominator <= 0 else numerator / denominator


def _predicates_payload(
    trace: SimulationTraceExport,
    predicates: list[TraceFailurePredicate],
    matrix: Mapping[str, Any] | None,
    predicate_definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the trace predicate payload schema.

    Returns:
        Normalized single-trace payload.
    """
    matrix_metadata = _matrix_metadata(matrix, included_traces=1)
    caveats = [
        "Predicates are rule-based trace diagnostics, not benchmark rates or safety claims.",
        f"`{VALIDITY_STATUS_UNAVAILABLE_DATA}` rows mark partial evidence with missing required fields.",
    ]
    if matrix is None:
        caveats.append(
            "No predeclared benchmark matrix was provided; this payload is diagnostic-only and "
            "claim-ineligible."
        )
    return {
        "schema_version": TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
        "predicate_source": TRACE_FAILURE_PREDICATE_SOURCE,
        "trace_id": trace.trace_id,
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
        },
        "matrix_metadata": matrix_metadata,
        "predicate_definitions": predicate_definitions,
        "denominator_status": VALIDITY_STATUS_VALID,
        "predicates": [predicate.to_dict() for predicate in predicates],
        "summary": _summary(predicates),
        "caveats": caveats,
    }


def _empty_frame(_trace: SimulationTraceExport) -> SimulationTraceFrame:
    """Return a minimal frame for traces with no frames.

    This is a defensive fallback for missing-field records on empty traces.
    """
    return SimulationTraceFrame(
        step=0,
        time_s=0.0,
        robot={},
        pedestrians=[],
        planner={},
    )


def _matrix_metadata(matrix: Mapping[str, Any] | None, *, included_traces: int) -> dict[str, Any]:
    """Build matrix metadata for aggregate payloads.

    Returns:
        Matrix status payload with claim-boundary wording.
    """
    if matrix is None:
        return {
            "schema_version": None,
            "matrix_path": None,
            "matrix_name": None,
            "status": MATRIX_STATUS_DIAGNOSTIC_ONLY,
            "claim_boundary": (
                "No predeclared benchmark matrix was provided. These rows are diagnostic-only and "
                "claim-ineligible; do not interpret them as benchmark rates."
            ),
            "rate_interpretation": "not_allowed",
            "included_trace_count": included_traces,
        }
    status = str(matrix.get("status") or MATRIX_STATUS_PROPOSED)
    rate_interpretation = (
        "allowed_only_when_compliant"
        if status == MATRIX_STATUS_RATE_INTERPRETABLE
        else "not_allowed_until_matrix_is_promoted"
    )
    return {
        "schema_version": matrix.get("schema_version"),
        "matrix_path": matrix.get("_source_path"),
        "matrix_name": matrix.get("name"),
        "status": status,
        "claim_boundary": matrix.get("claim_boundary", ""),
        "rate_interpretation": rate_interpretation,
        "included_trace_count": included_traces,
    }


def _unavailable_predicates_for_missing_fields(
    trace: SimulationTraceExport,
    matrix: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    """Map predicate IDs to their missing required fields when a matrix is supplied.

    Returns:
        Mapping from predicate ID to list of missing dotted trace field paths.
    """
    unavailable: dict[str, list[str]] = {}
    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        missing = _missing_required_fields(trace, pred_id, matrix)
        if missing:
            unavailable[pred_id] = missing
    return unavailable


def _append_not_available_predicates(
    predicates: list[TraceFailurePredicate],
    trace: SimulationTraceExport,
    scenario_family: str,
    unavailable: dict[str, list[str]],
) -> None:
    """Append one not_available predicate per missing-field declaration."""
    start_frame = trace.frames[0] if trace.frames else _empty_frame(trace)
    end_frame = trace.frames[-1] if trace.frames else _empty_frame(trace)
    for pred_id, missing in unavailable.items():
        predicates.append(
            _predicate(
                pred_id,
                start_frame,
                end_frame,
                scenario_family=scenario_family,
                planner_id=trace.source.planner_id,
                involved_actors=["robot"],
                evidence_fields={"missing_fields": missing},
                severity=VALIDITY_STATUS_UNAVAILABLE_DATA,
                validity_status=VALIDITY_STATUS_UNAVAILABLE_DATA,
            )
        )


def _extract_trace_failure_predicate_records(  # noqa: C901, PLR0913
    trace: SimulationTraceExport,
    *,
    scenario_family: str | None,
    matrix: Mapping[str, Any] | None,
    clearance_threshold_m: float,
    collision_missing_radius_near_distance_m: float,
    near_miss_threshold_m: float,
    stationary_displacement_threshold_m: float,
    low_progress_displacement_threshold_m: float,
    oscillation_min_sign_changes: int,
    evasive_angular_velocity_threshold: float,
    evasive_linear_drop_threshold_mps: float,
) -> list[TraceFailurePredicate]:
    """Extract a typed list of predicates for one trace.

    When ``matrix`` is supplied, any predicate whose required trace fields are absent emits a
    single ``not_available`` record instead of being silently treated as unobserved.

    Returns:
        Predicate records from one trace.
    """
    family = scenario_family or trace.source.scenario_id
    predicates: list[TraceFailurePredicate] = []
    unavailable = _unavailable_predicates_for_missing_fields(trace, matrix)

    if unavailable:
        _append_not_available_predicates(predicates, trace, family, unavailable)

    # When the matrix-level required-field check already emitted an occlusion not_available row,
    # skip per-frame extraction to avoid duplicate not_available rows.
    occlusion_guarded = "occlusion_triggered_near_miss" in unavailable

    if "collision" not in unavailable:
        predicates.extend(
            _collision_events(
                trace,
                scenario_family=family,
                collision_missing_radius_near_distance_m=(collision_missing_radius_near_distance_m),
            )
        )
    if "clearance_critical_interaction" not in unavailable:
        predicates.extend(
            _clearance_critical_interactions(
                trace,
                scenario_family=family,
                clearance_threshold_m=clearance_threshold_m,
            )
        )
    if "late_evasive_reaction" not in unavailable:
        late_evasive = _late_evasive_reaction(
            trace,
            scenario_family=family,
            near_miss_threshold_m=near_miss_threshold_m,
            evasive_angular_velocity_threshold=evasive_angular_velocity_threshold,
            evasive_linear_drop_threshold_mps=evasive_linear_drop_threshold_mps,
        )
        if late_evasive is not None:
            predicates.append(late_evasive)
    if "oscillatory_local_control" not in unavailable:
        oscillation = _oscillatory_local_control(
            trace,
            scenario_family=family,
            min_sign_changes=oscillation_min_sign_changes,
        )
        if oscillation is not None:
            predicates.append(oscillation)
    if "zero_motion_timeout_behavior" not in unavailable:
        zero_motion = _zero_motion_timeout_behavior(
            trace,
            scenario_family=family,
            stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        )
        if zero_motion is not None:
            predicates.append(zero_motion)
    if "low_progress" not in unavailable:
        low_progress = _low_progress(
            trace,
            scenario_family=family,
            low_progress_displacement_threshold_m=low_progress_displacement_threshold_m,
        )
        if low_progress is not None:
            predicates.append(low_progress)
    if "bottleneck_deadlock" not in unavailable:
        predicates.extend(_bottleneck_deadlocks(trace, scenario_family=family))
    if not occlusion_guarded:
        occlusion = _occlusion_triggered_near_miss(
            trace,
            scenario_family=family,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        if occlusion is not None:
            predicates.append(occlusion)
    return predicates


def _clearance_critical_interactions(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    clearance_threshold_m: float,
) -> list[TraceFailurePredicate]:
    """Return valid clearance-critical predicates for close robot-pedestrian frames."""
    predicates: list[TraceFailurePredicate] = []
    for frame in trace.frames:
        nearest = _nearest_pedestrian(frame)
        if nearest is None:
            continue
        pedestrian_id, distance_m = nearest
        if distance_m <= clearance_threshold_m:
            predicates.append(
                _predicate(
                    "clearance_critical_interaction",
                    frame,
                    frame,
                    scenario_family=scenario_family,
                    planner_id=trace.source.planner_id,
                    involved_actors=["robot", pedestrian_id],
                    evidence_fields={
                        "distance_m": distance_m,
                        "clearance_threshold_m": clearance_threshold_m,
                    },
                    severity="high" if distance_m <= clearance_threshold_m * 0.75 else "medium",
                    validity_status=VALIDITY_STATUS_VALID,
                )
            )
    return predicates


def _collision_events(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    collision_missing_radius_near_distance_m: float,
) -> list[TraceFailurePredicate]:
    """Return collision predicates for radius-overlap contact-distance frames."""
    predicates: list[TraceFailurePredicate] = []
    for frame in trace.frames:
        for pedestrian in frame.pedestrians:
            distance_m = _distance(frame.robot.get("position"), pedestrian.get("position"))
            if distance_m is None:
                continue
            pedestrian_id = str(pedestrian.get("id", "unknown"))
            robot_radius_m = _number_or_none(frame.robot.get("radius"))
            pedestrian_radius_m = _number_or_none(pedestrian.get("radius"))
            if robot_radius_m is None or pedestrian_radius_m is None:
                if distance_m <= collision_missing_radius_near_distance_m:
                    missing_fields = []
                    if robot_radius_m is None:
                        missing_fields.append("robot.radius")
                    if pedestrian_radius_m is None:
                        missing_fields.append("pedestrians.radius")
                    predicates.append(
                        _predicate(
                            "collision",
                            frame,
                            frame,
                            scenario_family=scenario_family,
                            planner_id=trace.source.planner_id,
                            involved_actors=["robot", pedestrian_id],
                            evidence_fields={
                                "distance_m": distance_m,
                                "missing_fields": missing_fields,
                                "missing_radius_near_distance_m": (
                                    collision_missing_radius_near_distance_m
                                ),
                            },
                            severity=VALIDITY_STATUS_UNAVAILABLE_DATA,
                            validity_status=VALIDITY_STATUS_UNAVAILABLE_DATA,
                        )
                    )
                continue
            collision_distance_m = robot_radius_m + pedestrian_radius_m
            if distance_m <= collision_distance_m:
                predicates.append(
                    _predicate(
                        "collision",
                        frame,
                        frame,
                        scenario_family=scenario_family,
                        planner_id=trace.source.planner_id,
                        involved_actors=["robot", pedestrian_id],
                        evidence_fields={
                            "distance_m": distance_m,
                            "robot_radius_m": robot_radius_m,
                            "pedestrian_radius_m": pedestrian_radius_m,
                            "collision_distance_m": collision_distance_m,
                        },
                        severity="critical",
                        validity_status=VALIDITY_STATUS_VALID,
                    )
                )
    return predicates


def _late_evasive_reaction(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    near_miss_threshold_m: float,
    evasive_angular_velocity_threshold: float,
    evasive_linear_drop_threshold_mps: float,
) -> TraceFailurePredicate | None:
    """Return a late evasive reaction predicate after clearance pressure.

    Returns:
        Predicate when the robot first enters near-miss pressure and only reacts on a later frame.
    """
    pressure: tuple[int, SimulationTraceFrame, str, float] | None = None
    for index, frame in enumerate(trace.frames):
        nearest = _nearest_pedestrian(frame)
        if nearest is None:
            continue
        pedestrian_id, distance_m = nearest
        if distance_m <= near_miss_threshold_m:
            pressure = (index, frame, pedestrian_id, distance_m)
            break
    if pressure is None:
        return None
    pressure_index, pressure_frame, pedestrian_id, pressure_distance_m = pressure
    pressure_action = _selected_action(pressure_frame)
    pressure_linear = _number_or_none(pressure_action.get("linear_velocity"))
    for reaction_index, reaction_frame in enumerate(
        trace.frames[pressure_index + 1 :],
        start=pressure_index + 1,
    ):
        action = _selected_action(reaction_frame)
        angular = abs(_number_or_none(action.get("angular_velocity")) or 0.0)
        linear = _number_or_none(action.get("linear_velocity"))
        linear_drop = (
            pressure_linear - linear if pressure_linear is not None and linear is not None else 0.0
        )
        if (
            angular >= evasive_angular_velocity_threshold
            or linear_drop >= evasive_linear_drop_threshold_mps
        ):
            return _predicate(
                "late_evasive_reaction",
                pressure_frame,
                reaction_frame,
                scenario_family=scenario_family,
                planner_id=trace.source.planner_id,
                involved_actors=["robot", pedestrian_id],
                evidence_fields={
                    "pressure_distance_m": pressure_distance_m,
                    "near_miss_threshold_m": near_miss_threshold_m,
                    "reaction_delay_steps": reaction_frame.step - pressure_frame.step,
                    # Seconds-valued companion to reaction_delay_steps so this surface, like the
                    # benchmark late_evasive_predicate, always carries a latency in seconds when a
                    # reaction exists (issue #5000). Frames carry time_s, so this is finite here;
                    # the "no reaction" case returns None above rather than a predicate.
                    "response_latency_s": reaction_frame.time_s - pressure_frame.time_s,
                    "reaction_frame_index": reaction_index,
                    "angular_velocity": angular,
                    "linear_drop_mps": linear_drop,
                },
                severity="high"
                if pressure_distance_m <= near_miss_threshold_m * 0.75
                else "medium",
                validity_status=VALIDITY_STATUS_VALID,
            )
    return None


def _oscillatory_local_control(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    min_sign_changes: int,
) -> TraceFailurePredicate | None:
    """Return an oscillatory-control predicate when angular command signs alternate."""
    commands = [
        _number_or_none(frame.planner.get("selected_action", {}).get("angular_velocity"))
        for frame in trace.frames
    ]
    signs = [_sign(command) for command in commands if command is not None and abs(command) > 1e-9]
    sign_changes = sum(1 for previous, current in pairwise(signs) if previous != current)
    if sign_changes < min_sign_changes:
        return None
    return _predicate(
        "oscillatory_local_control",
        trace.frames[0],
        trace.frames[-1],
        scenario_family=scenario_family,
        planner_id=trace.source.planner_id,
        involved_actors=["robot"],
        evidence_fields={
            "angular_sign_changes": sign_changes,
            "min_sign_changes": min_sign_changes,
        },
        severity="medium",
        validity_status=VALIDITY_STATUS_VALID,
    )


def _zero_motion_timeout_behavior(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    stationary_displacement_threshold_m: float,
) -> TraceFailurePredicate | None:
    """Return a zero-motion timeout predicate when timeout evidence accompanies little movement."""
    if not trace.frames:
        return None
    start = trace.frames[0]
    end = trace.frames[-1]
    displacement_m = _distance(start.robot.get("position"), end.robot.get("position"))
    timeout_frames = _timeout_frames(trace)
    if not timeout_frames or displacement_m is None:
        return None
    if displacement_m > stationary_displacement_threshold_m:
        return None
    return _predicate(
        "zero_motion_timeout_behavior",
        start,
        end,
        scenario_family=scenario_family,
        planner_id=trace.source.planner_id,
        involved_actors=["robot"],
        evidence_fields={
            "episode_displacement_m": displacement_m,
            "stationary_displacement_threshold_m": stationary_displacement_threshold_m,
            "timeout_events": [frame.planner.get("event") for frame in timeout_frames],
        },
        severity="high",
        validity_status=VALIDITY_STATUS_VALID,
    )


def _timeout_frames(trace: SimulationTraceExport) -> list[SimulationTraceFrame]:
    """Return frames with timeout or truncation planner events."""
    timeout_events = {"timeout", "time_limit", "max_steps", "truncated"}
    return [
        frame
        for frame in trace.frames
        if str(frame.planner.get("event", "")).strip().lower() in timeout_events
    ]


def _low_progress(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    low_progress_displacement_threshold_m: float,
) -> TraceFailurePredicate | None:
    """Return a low-progress predicate when timeout evidence accompanies small movement."""
    if not trace.frames:
        return None
    start = trace.frames[0]
    end = trace.frames[-1]
    displacement_m = _distance(start.robot.get("position"), end.robot.get("position"))
    timeout_frames = _timeout_frames(trace)
    if not timeout_frames or displacement_m is None:
        return None
    if displacement_m > low_progress_displacement_threshold_m:
        return None
    return _predicate(
        "low_progress",
        start,
        end,
        scenario_family=scenario_family,
        planner_id=trace.source.planner_id,
        involved_actors=["robot"],
        evidence_fields={
            "episode_displacement_m": displacement_m,
            "low_progress_displacement_threshold_m": low_progress_displacement_threshold_m,
            "timeout_events": [frame.planner.get("event") for frame in timeout_frames],
        },
        severity="medium",
        validity_status=VALIDITY_STATUS_VALID,
    )


def _bottleneck_deadlocks(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
) -> list[TraceFailurePredicate]:
    """Return predicates for explicit bottleneck deadlock events.

    Returns:
        One predicate per frame with an explicit bottleneck-deadlock planner event.
    """
    predicates: list[TraceFailurePredicate] = []
    for frame in trace.frames:
        event = str(frame.planner.get("event", "")).strip().lower()
        if event != "bottleneck_deadlock":
            continue
        predicates.append(
            _predicate(
                "bottleneck_deadlock",
                frame,
                frame,
                scenario_family=scenario_family,
                planner_id=trace.source.planner_id,
                involved_actors=["robot"],
                evidence_fields={"event": event},
                severity="high",
                validity_status=VALIDITY_STATUS_VALID,
            )
        )
    return predicates


def _occlusion_triggered_near_miss(
    trace: SimulationTraceExport,
    *,
    scenario_family: str,
    near_miss_threshold_m: float,
) -> TraceFailurePredicate | None:
    """Return valid or fail-closed occlusion near-miss predicate from trace frames."""
    for frame in trace.frames:
        nearest = _nearest_pedestrian(frame)
        if nearest is None:
            continue
        pedestrian_id, distance_m = nearest
        if distance_m > near_miss_threshold_m:
            continue
        occlusion_value = _occlusion_value(frame.planner)
        if occlusion_value is None:
            return _predicate(
                "occlusion_triggered_near_miss",
                frame,
                frame,
                scenario_family=scenario_family,
                planner_id=trace.source.planner_id,
                involved_actors=["robot", pedestrian_id],
                evidence_fields={
                    "distance_m": distance_m,
                    "near_miss_threshold_m": near_miss_threshold_m,
                    "missing_fields": ["planner.occlusion_or_visibility"],
                },
                severity=VALIDITY_STATUS_UNAVAILABLE_DATA,
                validity_status=VALIDITY_STATUS_UNAVAILABLE_DATA,
            )
        if occlusion_value:
            return _predicate(
                "occlusion_triggered_near_miss",
                frame,
                frame,
                scenario_family=scenario_family,
                planner_id=trace.source.planner_id,
                involved_actors=["robot", pedestrian_id],
                evidence_fields={
                    "distance_m": distance_m,
                    "near_miss_threshold_m": near_miss_threshold_m,
                    "occlusion_or_visibility": occlusion_value,
                },
                severity="high",
                validity_status=VALIDITY_STATUS_VALID,
            )
    return None


def _predicate(  # noqa: PLR0913
    predicate_id: str,
    start: SimulationTraceFrame,
    end: SimulationTraceFrame,
    *,
    scenario_family: str,
    planner_id: str,
    involved_actors: list[str],
    evidence_fields: dict[str, Any],
    severity: str,
    validity_status: str,
) -> TraceFailurePredicate:
    """Build a normalized predicate record.

    Returns:
        Trace-level failure predicate.
    """
    return TraceFailurePredicate(
        predicate_id=predicate_id,
        time_interval_s=[start.time_s, end.time_s],
        steps=[start.step, end.step],
        involved_actors=involved_actors,
        scenario_family=scenario_family,
        planner_id=planner_id,
        evidence_fields=evidence_fields,
        severity=severity,
        validity_status=validity_status,
    )


def _summary(predicates: list[TraceFailurePredicate]) -> dict[str, Any]:
    """Summarize predicate counts by scenario family and planner.

    Returns:
        Summary payload with total and nested counts.
    """
    nested: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for predicate in predicates:
        nested[predicate.scenario_family][predicate.planner_id][predicate.validity_status][
            predicate.predicate_id
        ] += 1
    by_family: dict[str, Any] = {}
    for family, planners in nested.items():
        by_family[family] = {}
        for planner_id, statuses in planners.items():
            by_family[family][planner_id] = {
                status: dict(counter) for status, counter in statuses.items()
            }
    return {
        "total_predicates": len(predicates),
        "by_scenario_family": by_family,
    }


def _nearest_pedestrian(frame: SimulationTraceFrame) -> tuple[str, float] | None:
    """Return nearest pedestrian id and distance for a frame."""
    robot_position = frame.robot.get("position")
    nearest: tuple[str, float] | None = None
    for pedestrian in frame.pedestrians:
        distance_m = _distance(robot_position, pedestrian.get("position"))
        if distance_m is None:
            continue
        pedestrian_id = str(pedestrian.get("id", "unknown"))
        if nearest is None or distance_m < nearest[1]:
            nearest = (pedestrian_id, distance_m)
    return nearest


def _selected_action(frame: SimulationTraceFrame) -> dict[str, Any]:
    """Return planner selected-action metadata.

    Returns:
        Selected-action dictionary or an empty dictionary when absent.
    """
    action = frame.planner.get("selected_action")
    return action if isinstance(action, dict) else {}


def _distance(a: Any, b: Any) -> float | None:
    """Return Euclidean distance between two 2D vectors."""
    if not isinstance(a, list | tuple) or not isinstance(b, list | tuple):
        return None
    if len(a) < 2 or len(b) < 2:
        return None
    ax = _number_or_none(a[0])
    ay = _number_or_none(a[1])
    bx = _number_or_none(b[0])
    by = _number_or_none(b[1])
    if ax is None or ay is None or bx is None or by is None:
        return None
    return math.hypot(ax - bx, ay - by)


def _occlusion_value(planner: dict[str, Any]) -> bool | None:
    """Read explicit occlusion or visibility evidence from planner metadata.

    Returns:
        Boolean occlusion evidence, or None when the required fields are absent.
    """
    for key in ("occluded", "occlusion", "visibility_blocked"):
        if key in planner:
            return bool(planner[key])
    visibility = planner.get("visibility")
    if isinstance(visibility, dict):
        for key in ("occluded", "blocked"):
            if key in visibility:
                return bool(visibility[key])
        if "line_of_sight" in visibility:
            return not bool(visibility["line_of_sight"])
    return None


def _number_or_none(value: Any) -> float | None:
    """Return a finite float or None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sign(value: float) -> int:
    """Return the sign of a nonzero number."""
    return 1 if value > 0.0 else -1


def _read_mapping(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    """Read a mapping value from the payload or return a default.

    Returns:
        Mapping payload.
    """
    return dict(value) if isinstance(value, Mapping) else default


def _markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[Any, ...]]) -> str:
    """Render simple Markdown table rows.

    Returns:
        Rendered table body.
    """
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_cell(str(cell)) for cell in row) + " |")
    return "\n".join(lines)


def _format_markdown_cell(value: str) -> str:
    """Escape markdown table cell values.

    Returns:
        Escaped cell text.
    """
    return value.replace("|", "\\|").replace("\n", " ")


def _markdown_value(value: Any) -> str:
    """Return a display value while preserving falsy non-None identifiers."""
    return "" if value is None else str(value)


def _markdown_int(value: Any) -> int:
    """Return an integer cell value with None-only fallback."""
    return 0 if value is None else int(value)


def _markdown_float(value: Any) -> float:
    """Return a float cell value with None-only fallback."""
    return 0.0 if value is None else float(value)


__all__ = [
    "MATRIX_CLAIM_ELIGIBLE",
    "MATRIX_CLAIM_INELIGIBLE",
    "MATRIX_STATUS_DIAGNOSTIC_ONLY",
    "MATRIX_STATUS_PROPOSED",
    "MATRIX_STATUS_RATE_INTERPRETABLE",
    "TRACE_FAILURE_PREDICATE_IDS",
    "TRACE_FAILURE_PREDICATE_SCHEMA_VERSION",
    "TRACE_FAILURE_PREDICATE_SOURCE",
    "TRACE_PREDICATE_MATRIX_SCHEMA_VERSION",
    "VALIDITY_STATUS_NO_PREDICATE_OBSERVED",
    "VALIDITY_STATUS_PIPELINE_FAILURE",
    "VALIDITY_STATUS_UNAVAILABLE_DATA",
    "VALIDITY_STATUS_VALID",
    "TraceFailurePredicate",
    "TraceFailurePredicateDefinition",
    "aggregate_trace_failure_predicate_tables",
    "build_trace_failure_predicate_definitions",
    "build_trace_predicate_denominator_health_report",
    "extract_trace_failure_predicates",
    "load_trace_predicate_matrix",
    "matrix_required_fields_for_predicate",
    "render_trace_failure_predicate_markdown",
]

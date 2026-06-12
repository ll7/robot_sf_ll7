"""Trace-level failure predicates for analysis-workbench exports."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.analysis_workbench.simulation_trace_export import (
        SimulationTraceExport,
        SimulationTraceFrame,
    )

TRACE_FAILURE_PREDICATE_SCHEMA_VERSION = "trace_failure_predicates.v1"
TRACE_FAILURE_PREDICATE_SOURCE = "trace_failure_predicates.rule_based.v1"

TRACE_FAILURE_PREDICATE_IDS = (
    "late_evasive_reaction",
    "oscillatory_local_control",
    "occlusion_triggered_near_miss",
    "bottleneck_deadlock",
    "zero_motion_timeout_behavior",
    "clearance_critical_interaction",
)


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


def extract_trace_failure_predicates(
    trace: SimulationTraceExport,
    *,
    scenario_family: str | None = None,
    clearance_threshold_m: float = 0.4,
    near_miss_threshold_m: float = 0.5,
    stationary_displacement_threshold_m: float = 0.05,
    oscillation_min_sign_changes: int = 3,
    evasive_angular_velocity_threshold: float = 1.0,
    evasive_linear_drop_threshold_mps: float = 0.3,
) -> dict[str, Any]:
    """Extract dissertation-relevant trace failure predicates.

    The extractor is intentionally conservative: it emits ``not_available`` records when a
    predicate has partial evidence but required fields are absent, rather than silently treating the
    predicate as false.

    Returns:
        ``trace_failure_predicates.v1`` payload with predicate rows and summary counts.
    """
    predicates = _extract_trace_failure_predicate_records(
        trace,
        scenario_family=scenario_family,
        clearance_threshold_m=clearance_threshold_m,
        near_miss_threshold_m=near_miss_threshold_m,
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
        oscillation_min_sign_changes=oscillation_min_sign_changes,
        evasive_angular_velocity_threshold=evasive_angular_velocity_threshold,
        evasive_linear_drop_threshold_mps=evasive_linear_drop_threshold_mps,
    )
    return _predicates_payload(trace=trace, predicates=predicates)


def aggregate_trace_failure_predicate_tables(
    traces: Iterable[SimulationTraceExport],
    *,
    scenario_family: str | None = None,
) -> dict[str, Any]:
    """Build denominator-aware aggregate rows across one or more trace predicates.

    Returns:
        Aggregate rows grouped by scenario family, planner, seed, predicate ID,
        validity status, and severity.
    """
    grouped_predicates: Counter[tuple[str, str, int, str, str, str]] = Counter()
    trace_denominators: Counter[tuple[str, str, int]] = Counter()
    included_trace_count = 0

    for trace in traces:
        group_family = scenario_family or trace.source.scenario_id
        group_key = (group_family, trace.source.planner_id, int(trace.source.seed))
        trace_denominators[group_key] += 1
        included_trace_count += 1
        for predicate in _extract_trace_failure_predicate_records(
            trace,
            scenario_family=group_family,
            clearance_threshold_m=0.4,
            near_miss_threshold_m=0.5,
            stationary_displacement_threshold_m=0.05,
            oscillation_min_sign_changes=3,
            evasive_angular_velocity_threshold=1.0,
            evasive_linear_drop_threshold_mps=0.3,
        ):
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

    rows = [
        {
            "scenario_family": family,
            "planner_id": planner_id,
            "seed": seed,
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

    return {
        "schema_version": TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
        "table_kind": "aggregate_by_predicate_group",
        "summary": {
            "input_trace_count": included_trace_count,
            "scenario_family_filter": scenario_family,
            "row_count": len(rows),
        },
        "rows": rows,
        "caveats": [
            "Aggregate predicate rows are diagnostic summaries, not benchmark rates or claims.",
            "Rows include `not_available` and other valid status values when evidence is partial.",
            "Do not treat these as benchmark outcomes unless tied to a predeclared benchmark matrix.",
        ],
    }


def render_trace_failure_predicate_markdown(table_payload: Mapping[str, Any]) -> str:
    """Render aggregate failure predicate rows as a compact Markdown diagnostic table.

    Returns:
        Rendered markdown text.
    """
    summary = _read_mapping(table_payload.get("summary"), {})
    rows = table_payload.get("rows")
    if not isinstance(rows, list):
        rows = []

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
        "",
    ]

    if not rows:
        lines.append("No predicate rows were observed in the provided traces.")
        return "\n".join(lines).rstrip() + "\n"

    lines.append("## Aggregate Rows")
    lines.append("")
    lines.append(
        _markdown_table(
            (
                "scenario_family",
                "planner_id",
                "seed",
                "predicate_id",
                "validity_status",
                "severity",
                "predicate_count",
                "trace_denominator",
                "predicate_rate_per_trace",
            ),
            [
                (
                    str(row.get("scenario_family", "")),
                    str(row.get("planner_id", "")),
                    str(row.get("seed", "")),
                    str(row.get("predicate_id", "")),
                    str(row.get("validity_status", "")),
                    str(row.get("severity", "")),
                    int(row.get("predicate_count", 0)),
                    int(row.get("trace_denominator", 0)),
                    f"{float(row.get('predicate_rate_per_trace', 0.0)):.4f}",
                )
                for row in rows
            ],
        )
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _predicates_payload(
    trace: SimulationTraceExport,
    predicates: list[TraceFailurePredicate],
) -> dict[str, Any]:
    """Build the trace predicate payload schema.

    Returns:
        Normalized single-trace payload.
    """
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
        "predicates": [predicate.to_dict() for predicate in predicates],
        "summary": _summary(predicates),
        "caveats": [
            "Predicates are rule-based trace diagnostics, not benchmark rates or safety claims.",
            "not_available rows mark partial evidence with missing required fields.",
        ],
    }


def _extract_trace_failure_predicate_records(
    trace: SimulationTraceExport,
    *,
    scenario_family: str | None,
    clearance_threshold_m: float,
    near_miss_threshold_m: float,
    stationary_displacement_threshold_m: float,
    oscillation_min_sign_changes: int,
    evasive_angular_velocity_threshold: float,
    evasive_linear_drop_threshold_mps: float,
) -> list[TraceFailurePredicate]:
    """Extract a typed list of predicates for one trace.

    Returns:
        Predicate records from one trace.
    """
    family = scenario_family or trace.source.scenario_id
    predicates: list[TraceFailurePredicate] = []
    predicates.extend(
        _clearance_critical_interactions(
            trace,
            scenario_family=family,
            clearance_threshold_m=clearance_threshold_m,
        )
    )
    late_evasive = _late_evasive_reaction(
        trace,
        scenario_family=family,
        near_miss_threshold_m=near_miss_threshold_m,
        evasive_angular_velocity_threshold=evasive_angular_velocity_threshold,
        evasive_linear_drop_threshold_mps=evasive_linear_drop_threshold_mps,
    )
    if late_evasive is not None:
        predicates.append(late_evasive)
    oscillation = _oscillatory_local_control(
        trace,
        scenario_family=family,
        min_sign_changes=oscillation_min_sign_changes,
    )
    if oscillation is not None:
        predicates.append(oscillation)
    zero_motion = _zero_motion_timeout_behavior(
        trace,
        scenario_family=family,
        stationary_displacement_threshold_m=stationary_displacement_threshold_m,
    )
    if zero_motion is not None:
        predicates.append(zero_motion)
    predicates.extend(_bottleneck_deadlocks(trace, scenario_family=family))
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
                    validity_status="valid",
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
                    "reaction_frame_index": reaction_index,
                    "angular_velocity": angular,
                    "linear_drop_mps": linear_drop,
                },
                severity="high"
                if pressure_distance_m <= near_miss_threshold_m * 0.75
                else "medium",
                validity_status="valid",
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
        validity_status="valid",
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
    timeout_frames = [
        frame
        for frame in trace.frames
        if str(frame.planner.get("event", "")).strip().lower()
        in {"timeout", "time_limit", "max_steps", "truncated"}
    ]
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
        validity_status="valid",
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
                validity_status="valid",
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
                severity="not_available",
                validity_status="not_available",
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
                validity_status="valid",
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


__all__ = [
    "TRACE_FAILURE_PREDICATE_IDS",
    "TRACE_FAILURE_PREDICATE_SCHEMA_VERSION",
    "TRACE_FAILURE_PREDICATE_SOURCE",
    "TraceFailurePredicate",
    "aggregate_trace_failure_predicate_tables",
    "extract_trace_failure_predicates",
    "render_trace_failure_predicate_markdown",
]

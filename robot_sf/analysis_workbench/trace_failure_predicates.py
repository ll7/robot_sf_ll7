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
TRACE_PREDICATE_DENOMINATOR_HEALTH_SCHEMA_VERSION = "trace_predicate_denominator_health.v1"

VALIDITY_STATUS_VALID = "valid"
VALIDITY_STATUS_UNAVAILABLE_DATA = "not_available"
VALIDITY_STATUS_NO_PREDICATE_OBSERVED = "no_predicate_observed"
VALIDITY_STATUS_PIPELINE_FAILURE = "pipeline_failure"
DENOMINATOR_HEALTH_STATUSES = (
    VALIDITY_STATUS_VALID,
    VALIDITY_STATUS_UNAVAILABLE_DATA,
    VALIDITY_STATUS_NO_PREDICATE_OBSERVED,
    VALIDITY_STATUS_PIPELINE_FAILURE,
)

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
    failed_trace_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build denominator-aware aggregate rows across one or more trace predicates.

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

    if failed_trace_id_list:
        for _failed_id in failed_trace_id_list:
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
            clearance_threshold_m=0.4,
            near_miss_threshold_m=0.5,
            stationary_displacement_threshold_m=0.05,
            oscillation_min_sign_changes=3,
            evasive_angular_velocity_threshold=1.0,
            evasive_linear_drop_threshold_mps=0.3,
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

    return {
        "schema_version": TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
        "table_kind": "aggregate_by_predicate_group",
        "summary": {
            "input_trace_count": included_trace_count + len(failed_trace_id_list),
            "failed_trace_count": len(failed_trace_id_list),
            "scenario_family_filter": scenario_family,
            "row_count": len(rows),
        },
        "rows": rows,
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
    raw_health = table_payload.get("predicate_denominator_health")
    health_by_predicate = raw_health if isinstance(raw_health, Mapping) else {}
    predicate_reports: dict[str, dict[str, Any]] = {}

    for pred_id in TRACE_FAILURE_PREDICATE_IDS:
        raw_counts = health_by_predicate.get(pred_id)
        counts = raw_counts if isinstance(raw_counts, Mapping) else {}
        valid_count = int(counts.get(VALIDITY_STATUS_VALID, 0) or 0)
        unavailable_count = int(counts.get(VALIDITY_STATUS_UNAVAILABLE_DATA, 0) or 0)
        no_predicate_count = int(counts.get(VALIDITY_STATUS_NO_PREDICATE_OBSERVED, 0) or 0)
        pipeline_failure_count = int(counts.get(VALIDITY_STATUS_PIPELINE_FAILURE, 0) or 0)
        total_count = valid_count + unavailable_count + no_predicate_count + pipeline_failure_count
        if total_count == 0 and input_trace_count > 0:
            no_predicate_count = max(input_trace_count - failed_trace_count, 0)
            pipeline_failure_count = failed_trace_count
            total_count = input_trace_count
        claim_eligibility, allowed_wording = _predicate_claim_status(
            valid_count=valid_count,
            unavailable_count=unavailable_count,
            no_predicate_count=no_predicate_count,
            pipeline_failure_count=pipeline_failure_count,
        )
        predicate_reports[pred_id] = {
            "total_denominator_count": total_count,
            "valid_count": valid_count,
            "unavailable_data_count": unavailable_count,
            "no_predicate_observed_count": no_predicate_count,
            "pipeline_failure_count": pipeline_failure_count,
            "valid_ratio": _ratio(valid_count, total_count),
            "unavailable_data_ratio": _ratio(unavailable_count, total_count),
            "no_predicate_observed_ratio": _ratio(no_predicate_count, total_count),
            "pipeline_failure_ratio": _ratio(pipeline_failure_count, total_count),
            "claim_eligibility": claim_eligibility,
            "allowed_wording": allowed_wording,
        }

    return {
        "schema_version": TRACE_PREDICATE_DENOMINATOR_HEALTH_SCHEMA_VERSION,
        "report_kind": "trace_predicate_denominator_health",
        "summary": {
            "total_traces_processed": input_trace_count,
            "failed_trace_count": failed_trace_count,
            "predicate_family_count": len(TRACE_FAILURE_PREDICATE_IDS),
        },
        "predicates": predicate_reports,
        "caveats": [
            "This report is diagnostic-only and does not convert missing predicate evidence into negative findings.",
            "`no_predicate_observed` can be expected missingness; it is not evidence that a failure did not occur.",
            "`not_available` and `pipeline_failure` weaken or block downstream figure/table claims.",
        ],
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
) -> tuple[str, str]:
    """Classify predicate-family claim eligibility from denominator health.

    Returns:
        Claim eligibility status and allowed wording guidance.
    """
    if pipeline_failure_count > 0:
        return "claim-ineligible", "pipeline failed; do not make predicate-family claims"
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
            f"`{VALIDITY_STATUS_UNAVAILABLE_DATA}` rows mark partial evidence with missing required fields.",
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
    "TRACE_FAILURE_PREDICATE_IDS",
    "TRACE_FAILURE_PREDICATE_SCHEMA_VERSION",
    "TRACE_FAILURE_PREDICATE_SOURCE",
    "VALIDITY_STATUS_NO_PREDICATE_OBSERVED",
    "VALIDITY_STATUS_PIPELINE_FAILURE",
    "VALIDITY_STATUS_UNAVAILABLE_DATA",
    "VALIDITY_STATUS_VALID",
    "TraceFailurePredicate",
    "aggregate_trace_failure_predicate_tables",
    "build_trace_predicate_denominator_health_report",
    "extract_trace_failure_predicates",
    "render_trace_failure_predicate_markdown",
]

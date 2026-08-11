"""Versioned report/schema crosswalk for diagnosis and execution-monitor fields (issue #6871).

This module maps deterministic failure-diagnosis (``failure_diagnosis.v1``, issue #6583)
and execution-time deviation monitor (``execution_deviation.v1``, issue #6584) fields
into episode-level and campaign-level summary structures suitable for benchmark
reporting artifacts.

It does not duplicate, replace, or extend the upstream diagnosis or monitoring
implementations.  It is a pure reporting crosswalk: no new simulator metrics are
produced, and no diagnostic or monitoring values are fabricated.

Separation from task-success metrics
------------------------------------
Episode-level success, collision, comfort, and other core benchmark metrics remain
independent diagnostic surfaces.  This crosswalk makes diagnostic-quality and
execution-deviation quality visible alongside those metrics without changing their
meaning or ranking semantics.

Evidence tier: diagnostic-only.  This module does not claim causality, safety,
planner ranking, intervention effectiveness, generalization, or benchmark success
from diagnostic records.  See the module caveats below and
:func:`build_episode_diagnostic_summary` for the explicit claim boundary.

Ownership
---------
- ``failure_diagnosis.v1`` record and adapter: issue #6583 / merged PR #6625.
- ``execution_deviation.v1`` monitor: issue #6584 / merged PR #6671.
- Learned/reference diagnosis quality: issue #6646 / merged PR #6704.

This crosswalk owns only the reporting contract that surfaces those outputs in
campaign artifacts.  It does not own the underlying diagnostic, monitoring, or
quality-evaluation implementations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from robot_sf.benchmark.failure_diagnosis import (
    DEFAULT_CORRECTION_STATUS,
    DIAGNOSIS_SOURCE,
    FAILURE_DIAGNOSIS_SCHEMA_VERSION,
    FailureDiagnosisError,
    build_failure_diagnosis_payload,
    validate_failure_diagnosis_payload,
)
from robot_sf.benchmark.trajectory_verifier import (
    EXECUTION_DEVIATION_CLAIM_BOUNDARY,
    EXECUTION_DEVIATION_SCHEMA,
    INTERVENTION_WARN,
    ExecutionDeviationDiagnosticReport,
    ExecutionDeviationResult,
)

# ---------------------------------------------------------------------------
# Crosswalk schema
# ---------------------------------------------------------------------------

#: Schema version for the crosswalk report.
REPORT_CROSSWALK_SCHEMA_VERSION = "report_crosswalk.v1"

#: Provenance source for this crosswalk.
REPORT_CROSSWALK_SOURCE = "report_crosswalk.v1"

# These strings are intentionally local to the reporting contract.  The
# upstream diagnosis module keeps its equivalent values private because they
# describe its record schema; the crosswalk must not couple its public import
# surface to those private names.
_NON_CAUSAL_CAVEAT = (
    "causal_evidence cites trace/predicate evidence pointers only; it is not causal inference."
)
_DIAGNOSTIC_LABEL_CAVEAT = (
    "failure_type reuses an existing classifier label as a diagnostic label; it is not "
    "a canonical mechanism attribution without trace review."
)
_UNKNOWN_CAVEAT = (
    "Unsupported, invalid, or unavailable mappings resolve to unknown; causal_evidence "
    "still cites the source predicate's evidence pointers."
)
_PAYLOAD_NON_CLAIM_CAVEAT = (
    "failure_diagnosis records are deterministic diagnostics adapted from trace "
    "predicates; they make no benchmark-ranking, causal-validity, or "
    "correction-quality claim."
)
_OUT_OF_SCOPE_CAVEAT = (
    "Learned/LLM diagnosis generation, correction-usefulness scoring, and "
    "campaign-level diagnosis-quality claims are out of scope for this schema version."
)

#: Claim boundary explicitly attached to every crosswalk report.
_REPORT_CROSSWALK_CLAIM_BOUNDARY = (
    "reporting crosswalk that surfaces deterministic diagnostic and "
    "execution-monitor fields in episode/campaign summaries; "
    "diagnostic-only; does not claim causality, safety, planner ranking, "
    "intervention effectiveness, generalization, or benchmark success."
)

#: Caveats carried by every crosswalk report.
_CROSSWALK_CAVEATS: list[str] = [
    _REPORT_CROSSWALK_CLAIM_BOUNDARY,
    _PAYLOAD_NON_CLAIM_CAVEAT,
    _NON_CAUSAL_CAVEAT,
    _OUT_OF_SCOPE_CAVEAT,
    (
        "Detecting or describing an execution deviation is not the same as "
        "proving that a controller or policy corrected it; correction claims "
        "require a separately approved evaluation."
    ),
]

# ---------------------------------------------------------------------------
# Denominator and validity vocabulary
# ---------------------------------------------------------------------------

#: Validity states that may appear on crosswalk-reported fields.
FIELD_VALIDITY_STATES = ("available", "unavailable", "invalid", "fallback", "degraded")

#: Provenance states for crosswalk-reported fields.
FIELD_PROVENANCE_STATES = ("complete", "incomplete", "unknown")

_UPSTREAM_UNAVAILABLE_STATUSES = frozenset(
    {
        "unavailable",
        "not_available",
        "no_predicate_observed",
        "absent_expected_slice",
    }
)
_UPSTREAM_INVALID_STATUSES = frozenset(
    {
        "invalid",
        "failed",
        "partial_failure",
        "partial-failure",
        "provenance_incomplete",
        "incomplete",
    }
)
_INTERVENTION_LABELS = frozenset({"continue", "warn", "replan", "fallback_brake"})


def _validity_state_for_value(
    value: Any,
    *,
    source_validity: str | None = None,
    is_fallback: bool = False,
    is_degraded: bool = False,
) -> Literal["available", "unavailable", "invalid", "fallback", "degraded"]:
    """Derive the crosswalk validity state for a mapped value.

    Args:
        value: The mapped value (may be None for unavailable).
        source_validity: Optional validity status from the upstream record.
        is_fallback: Whether the value came from a fallback path.
        is_degraded: Whether the value was produced in degraded mode.

    Returns:
        A validity state string.
    """
    if is_fallback:
        return "fallback"
    if is_degraded:
        return "degraded"
    normalized_status = (
        source_validity.strip().lower() if isinstance(source_validity, str) else None
    )
    if normalized_status == "fallback":
        return "fallback"
    if normalized_status == "degraded":
        return "degraded"
    if normalized_status in _UPSTREAM_UNAVAILABLE_STATUSES:
        return "unavailable"
    if normalized_status in _UPSTREAM_INVALID_STATUSES:
        return "invalid"
    if normalized_status is not None and normalized_status != "valid":
        return "invalid"
    if value is None:
        return "unavailable"
    return "available"


def _provenance_for_record(
    record: Mapping[str, Any] | None,
    *,
    schema_version: str | None = None,
) -> Literal["complete", "incomplete", "unknown"]:
    """Derive the crosswalk provenance state for a source record.

    Args:
        record: The upstream record mapping, or None.
        schema_version: Expected schema version, or None to skip check.

    Returns:
        A provenance state string.
    """
    if record is None:
        return "unknown"
    if schema_version is not None and record.get("diagnosis_schema_version") != schema_version:
        return "incomplete"
    if record.get("diagnosis_source") != DIAGNOSIS_SOURCE:
        return "incomplete"
    return "complete"


def _diagnosis_payload_metadata(
    payload: Mapping[str, Any] | None,
) -> tuple[
    Literal["available", "unavailable", "invalid", "fallback", "degraded"] | None,
    Literal["complete", "incomplete", "unknown"],
    str | None,
    list[Mapping[str, Any]],
]:
    """Check wrapper metadata before invoking the upstream nested validator.

    Returns:
        ``(early_state, provenance, reason, records)``; ``early_state`` is
        ``None`` when the nested validator should run.
    """
    if payload is None:
        return "unavailable", "unknown", "diagnosis_payload_not_provided", []
    if not isinstance(payload, Mapping):
        return "invalid", "incomplete", "diagnosis_payload_not_mapping", []
    records = payload.get("records")
    if not isinstance(records, list) or not all(isinstance(record, Mapping) for record in records):
        return "invalid", "incomplete", "diagnosis_payload_records_invalid", []
    provenance = "complete"
    if any(
        _provenance_for_record(record, schema_version=FAILURE_DIAGNOSIS_SCHEMA_VERSION)
        != "complete"
        for record in records
    ):
        provenance = "incomplete"
    if payload.get("schema_version") != FAILURE_DIAGNOSIS_SCHEMA_VERSION:
        return "invalid", "incomplete", "diagnosis_payload_schema_version_mismatch", records
    if payload.get("diagnosis_source") != DIAGNOSIS_SOURCE:
        return "invalid", "incomplete", "diagnosis_payload_source_mismatch", records
    return None, provenance, None, records


def _diagnosis_record_state(
    records: Sequence[Mapping[str, Any]],
) -> Literal["available", "unavailable", "invalid", "fallback", "degraded"]:
    """Aggregate source record validity states conservatively.

    Returns:
        The most conservative crosswalk validity state for the records.
    """
    states = {
        _validity_state_for_value(
            record.get("failure_type"),
            source_validity=record.get("validity_status"),
        )
        for record in records
    }
    if "degraded" in states:
        return "degraded"
    if "fallback" in states:
        return "fallback"
    if states == {"unavailable"}:
        return "unavailable"
    if "invalid" in states or (states and states != {"available"}):
        return "invalid"
    return "available"


def _diagnosis_payload_assessment(
    payload: Mapping[str, Any] | None,
) -> tuple[
    Literal["available", "unavailable", "invalid", "fallback", "degraded"],
    Literal["complete", "incomplete", "unknown"],
    str | None,
    list[Mapping[str, Any]],
]:
    """Validate a diagnosis payload and derive its report-facing state.

    The upstream validator owns the full nested ``failure_diagnosis.v1``
    contract.  This helper delegates to it, then derives the crosswalk state
    without silently treating malformed or fallback records as benchmark
    evidence.

    Returns:
        ``(validity_state, provenance, reason, normalized_records)``.
    """
    state, provenance, reason, records = _diagnosis_payload_metadata(payload)
    if state is not None:
        return state, provenance, reason, records
    try:
        normalized = validate_failure_diagnosis_payload(payload)
    except FailureDiagnosisError as exc:
        return "invalid", provenance, f"diagnosis_payload_validation_failed:{exc}", records
    normalized_records = normalized.get("records", records)
    state = _diagnosis_record_state(normalized_records)
    reason = None if state == "available" else f"diagnosis_record_validity:{state}"
    return state, provenance, reason, normalized_records


def _deviation_result_provenance(
    result: ExecutionDeviationResult | None,
) -> Literal["complete", "incomplete", "unknown"]:
    """Return provenance state for an execution-deviation result."""
    if result is None:
        return "unknown"
    if result.schema_version != EXECUTION_DEVIATION_SCHEMA:
        return "incomplete"
    if result.claim_boundary != EXECUTION_DEVIATION_CLAIM_BOUNDARY:
        return "incomplete"
    return "complete"


# ---------------------------------------------------------------------------
# Episode-level diagnostic summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeDiagnosticSummary:
    """Crosswalk report mapping diagnosis and execution-deviation fields for one episode.

    Core benchmark metrics (success, collision, comfort) are passed through
    unchanged and reported separately from diagnostic quality.  This summary
    surfaces the *diagnostic-sidecar* fields that are mapped from upstream
    diagnosis and execution-deviation records.

    Attributes:
        episode_id: Episode identifier (passed through, not derived).
        planner_id: Planner identifier (passed through, not derived).
        diagnosis_available: Whether a ``failure_diagnosis.v1`` payload was provided.
        diagnosis_record_count: Number of diagnosis records in the payload.
        diagnosis_failure_type_counts: Per-type counts from the payload coverage.
        diagnosis_severity_counts: Per-severity counts derived from records.
        diagnosis_unknown_count: Number of records with ``failure_type == "unknown"``.
        diagnosis_validity_state: Validity state for the diagnosis payload.
        diagnosis_provenance: Provenance state for the diagnosis payload.
        diagnosis_validity_reason: Reason string when validity is not ``"available"``.
        execution_deviation_available: Whether an ``ExecutionDeviationResult`` was provided.
        execution_deviation_intervention: Intervention label from the monitor.
        execution_deviation_score: Peak deviation score, or ``None`` if unavailable.
        execution_deviation_fail_closed: Whether the result came from the fail-closed path.
        execution_deviation_threshold_crossing_time_s: First threshold crossing time.
        execution_deviation_validity_state: Validity state for the deviation result.
        execution_deviation_provenance: Provenance state for the deviation result.
        execution_deviation_validity_reason: Reason string when not ``"available"``.
        execution_deviation_claim_boundary: Explicit claim boundary from the monitor.
        success: Task success metric (passed through from upstream).
        collision: Collision metric (passed through from upstream).
        comfort: Comfort metric (passed through from upstream).
        caveats: Non-causal and boundary caveats.
        schema_version: Schema identifier for this crosswalk.
    """

    episode_id: str
    planner_id: str
    diagnosis_available: bool
    diagnosis_record_count: int
    diagnosis_failure_type_counts: dict[str, int]
    diagnosis_severity_counts: dict[str, int]
    diagnosis_unknown_count: int
    diagnosis_validity_state: Literal["available", "unavailable", "invalid", "fallback", "degraded"]
    diagnosis_provenance: Literal["complete", "incomplete", "unknown"]
    diagnosis_validity_reason: str | None
    execution_deviation_available: bool
    execution_deviation_intervention: str | None
    execution_deviation_score: float | None
    execution_deviation_fail_closed: bool | None
    execution_deviation_threshold_crossing_time_s: float | None
    execution_deviation_validity_state: Literal[
        "available", "unavailable", "invalid", "fallback", "degraded"
    ]
    execution_deviation_provenance: Literal["complete", "incomplete", "unknown"]
    execution_deviation_validity_reason: str | None
    execution_deviation_claim_boundary: str | None
    success: bool | None
    collision: bool | None
    comfort: float | None
    caveats: list[str]
    schema_version: str = REPORT_CROSSWALK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the episode diagnostic summary.
        """
        return {
            "schema_version": self.schema_version,
            "report_source": REPORT_CROSSWALK_SOURCE,
            "episode_id": self.episode_id,
            "planner_id": self.planner_id,
            "diagnosis": {
                "available": self.diagnosis_available,
                "record_count": self.diagnosis_record_count,
                "failure_type_counts": dict(self.diagnosis_failure_type_counts),
                "severity_counts": dict(self.diagnosis_severity_counts),
                "unknown_count": self.diagnosis_unknown_count,
                "validity_state": self.diagnosis_validity_state,
                "provenance": self.diagnosis_provenance,
                "validity_reason": self.diagnosis_validity_reason,
            },
            "execution_deviation": {
                "available": self.execution_deviation_available,
                "intervention": self.execution_deviation_intervention,
                "deviation_score": self.execution_deviation_score,
                "fail_closed": self.execution_deviation_fail_closed,
                "first_threshold_crossing_time_s": self.execution_deviation_threshold_crossing_time_s,
                "validity_state": self.execution_deviation_validity_state,
                "provenance": self.execution_deviation_provenance,
                "validity_reason": self.execution_deviation_validity_reason,
                "claim_boundary": self.execution_deviation_claim_boundary,
            },
            "core_metrics": {
                "success": self.success,
                "collision": self.collision,
                "comfort": self.comfort,
            },
            "caveats": list(self.caveats),
        }


def _count_by_key(
    records: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    """Count records by a string key value.

    Args:
        records: Sequence of record mappings.
        key: Key to count by.

    Returns:
        Dictionary of value counts.
    """
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _deviation_validity_reason(
    result: ExecutionDeviationResult | None,
) -> str | None:
    """Return a reason string when the execution-deviation result is not available.

    Args:
        result: The deviation result, or None.

    Returns:
        A reason string, or None when the result is available.
    """
    if result is None:
        return "execution_deviation_result_not_provided"
    if result.fail_closed:
        return "execution_deviation_fail_closed:invalid_or_stale_inputs"
    return None


def build_episode_diagnostic_summary(
    *,
    episode_id: str,
    planner_id: str,
    diagnosis_payload: Mapping[str, Any] | None = None,
    execution_deviation_result: ExecutionDeviationResult | None = None,
    success: bool | None = None,
    collision: bool | None = None,
    comfort: float | None = None,
) -> EpisodeDiagnosticSummary:
    """Build a crosswalk summary for one episode's diagnostic and monitoring fields.

    This function maps upstream ``failure_diagnosis.v1`` and
    ``execution_deviation.v1`` fields into a versioned episode-level structure
    suitable for campaign reporting artifacts.  Core benchmark metrics (success,
    collision, comfort) are passed through unchanged and kept separate from
    diagnostic quality.

    Denominators and validity rules
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``diagnosis_record_count`` is the count of records in the payload (denominator
      for per-record diagnostic coverage).
    - ``diagnosis_unknown_count`` counts records whose ``failure_type`` is ``"unknown"``
      (denominator for known-type coverage).
    - ``diagnosis_failure_type_counts`` provides per-type denominators for type-specific
      proportions.
    - Execution-deviation fields have no per-episode denominator; denominator semantics
      belong to the campaign-level ``ExecutionDeviationDiagnosticReport``.
    - Validity states (``"available"``, ``"unavailable"``, ``"invalid"``, ``"fallback"``,
      ``"degraded"``) and provenance states (``"complete"``, ``"incomplete"``,
      ``"unknown"``) are explicit and never fabricated.

    Unavailable / invalid / fallback / degraded handling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - A missing diagnosis payload produces ``diagnosis_validity_state="unavailable"``
      with a reason; record counts are all zero.
    - A missing execution-deviation result produces
      ``execution_deviation_validity_state="unavailable"`` with a reason.
    - A fail-closed deviation result produces
      ``execution_deviation_validity_state="unavailable"`` with an explicit reason
      (score is ``None``, intervention is still emitted).

    Claim boundary
    ~~~~~~~~~~~~~~
    This is a reporting and artifact-contract improvement.  It must not turn a
    diagnostic record into evidence of causality, safety, planner ranking, or
    successful intervention.

    Args:
        episode_id: Episode identifier.
        planner_id: Planner identifier.
        diagnosis_payload: Optional ``failure_diagnosis.v1`` payload.
        execution_deviation_result: Optional ``ExecutionDeviationResult``.
        success: Optional task-success metric (passed through).
        collision: Optional collision metric (passed through).
        comfort: Optional comfort metric (passed through).

    Returns:
        A versioned episode diagnostic summary.
    """
    (
        diag_validity,
        diag_provenance,
        diag_reason,
        assessed_diag_records,
    ) = _diagnosis_payload_assessment(diagnosis_payload)
    diag_available = diag_validity in {"available", "fallback", "degraded"}
    # Invalid source payloads do not contribute a denominator or label counts;
    # retaining them would make malformed provenance look like evidence.
    diag_records: list[Mapping[str, Any]] = (
        [] if diag_validity == "invalid" else list(assessed_diag_records)
    )
    diag_type_counts: dict[str, int] = {}
    diag_severity_counts: dict[str, int] = {}
    diag_unknown_count = 0

    if diag_records and diag_validity != "invalid":
        diag_type_counts = _count_by_key(diag_records, "failure_type")
        diag_severity_counts = _count_by_key(diag_records, "severity")
        diag_unknown_count = diag_type_counts.get("unknown", 0)

    dev_provenance = _deviation_result_provenance(execution_deviation_result)
    dev_available = execution_deviation_result is not None and dev_provenance == "complete"
    dev_intervention: str | None = None
    dev_score: float | None = None
    dev_fail_closed: bool | None = None
    dev_crossing: float | None = None
    dev_validity: str = "unavailable"
    dev_reason = _deviation_validity_reason(execution_deviation_result)
    dev_claim: str | None = None

    if execution_deviation_result is not None and dev_provenance != "complete":
        dev_validity = "invalid"
        dev_reason = "execution_deviation_result_provenance_incomplete"

    if execution_deviation_result is not None and dev_provenance == "complete":
        dev_intervention = execution_deviation_result.intervention
        dev_score = execution_deviation_result.deviation_score
        dev_fail_closed = execution_deviation_result.fail_closed
        dev_crossing = execution_deviation_result.first_threshold_crossing_time_s
        dev_claim = execution_deviation_result.claim_boundary
        if dev_fail_closed:
            dev_validity = "unavailable"
            dev_provenance = "complete"
            dev_reason = "execution_deviation_fail_closed:invalid_or_stale_inputs"
        else:
            dev_validity = "available"
            dev_provenance = "complete"
            dev_reason = None

    return EpisodeDiagnosticSummary(
        episode_id=episode_id,
        planner_id=planner_id,
        diagnosis_available=diag_available,
        diagnosis_record_count=len(diag_records),
        diagnosis_failure_type_counts=diag_type_counts,
        diagnosis_severity_counts=diag_severity_counts,
        diagnosis_unknown_count=diag_unknown_count,
        diagnosis_validity_state=diag_validity,
        diagnosis_provenance=diag_provenance,
        diagnosis_validity_reason=diag_reason,
        execution_deviation_available=dev_available,
        execution_deviation_intervention=dev_intervention,
        execution_deviation_score=dev_score,
        execution_deviation_fail_closed=dev_fail_closed,
        execution_deviation_threshold_crossing_time_s=dev_crossing,
        execution_deviation_validity_state=dev_validity,
        execution_deviation_provenance=dev_provenance,
        execution_deviation_validity_reason=dev_reason,
        execution_deviation_claim_boundary=dev_claim,
        success=success,
        collision=collision,
        comfort=comfort,
        caveats=list(_CROSSWALK_CAVEATS),
    )


# ---------------------------------------------------------------------------
# Campaign-level diagnostic summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CampaignDiagnosticSummary:
    """Crosswalk report mapping diagnosis and execution-deviation fields for a campaign.

    This aggregates episode-level diagnostic summaries into campaign-level
    denominators and coverage metrics.  Core benchmark metrics remain separate
    from diagnostic quality.

    Attributes:
        campaign_id: Campaign identifier.
        episode_count: Number of episodes in the campaign.
        diagnosis_available_count: Episodes with a diagnosis payload.
        diagnosis_record_total: Total diagnosis records across all episodes.
        diagnosis_failure_type_totals: Per-type totals across the campaign.
        diagnosis_severity_totals: Per-severity totals across the campaign.
        diagnosis_unknown_total: Total unknown-type records.
        diagnosis_coverage_rate: Fraction of episodes with diagnosis available.
        execution_deviation_available_count: Episodes with a deviation result.
        execution_deviation_fail_closed_count: Episodes where deviation was fail-closed.
        execution_deviation_intervention_counts: Per-intervention-label counts.
        execution_deviation_coverage_rate: Fraction of episodes with deviation available.
        execution_deviation_report: Optional campaign-level deviation diagnostic report.
        success_rate: Fraction of successful episodes (when success is known).
        collision_rate: Fraction of collision episodes (when collision is known).
        comfort_mean: Mean comfort score (when comfort is known).
        caveats: Non-causal and boundary caveats.
        schema_version: Schema identifier for this crosswalk.
    """

    campaign_id: str
    episode_count: int
    diagnosis_available_count: int
    diagnosis_record_total: int
    diagnosis_failure_type_totals: dict[str, int]
    diagnosis_severity_totals: dict[str, int]
    diagnosis_unknown_total: int
    diagnosis_coverage_rate: float | None
    execution_deviation_available_count: int
    execution_deviation_fail_closed_count: int
    execution_deviation_intervention_counts: dict[str, int]
    execution_deviation_coverage_rate: float | None
    execution_deviation_report: dict[str, Any] | None
    success_rate: float | None
    collision_rate: float | None
    comfort_mean: float | None
    caveats: list[str]
    schema_version: str = REPORT_CROSSWALK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary.

        Returns:
            Dictionary representation of the campaign diagnostic summary.
        """
        return {
            "schema_version": self.schema_version,
            "report_source": REPORT_CROSSWALK_SOURCE,
            "campaign_id": self.campaign_id,
            "episode_count": self.episode_count,
            "diagnosis": {
                "available_count": self.diagnosis_available_count,
                "record_total": self.diagnosis_record_total,
                "failure_type_totals": dict(self.diagnosis_failure_type_totals),
                "severity_totals": dict(self.diagnosis_severity_totals),
                "unknown_total": self.diagnosis_unknown_total,
                "coverage_rate": self.diagnosis_coverage_rate,
            },
            "execution_deviation": {
                "available_count": self.execution_deviation_available_count,
                "fail_closed_count": self.execution_deviation_fail_closed_count,
                "intervention_counts": dict(self.execution_deviation_intervention_counts),
                "coverage_rate": self.execution_deviation_coverage_rate,
                "diagnostic_report": self.execution_deviation_report,
            },
            "core_metrics": {
                "success_rate": self.success_rate,
                "collision_rate": self.collision_rate,
                "comfort_mean": self.comfort_mean,
            },
            "caveats": list(self.caveats),
        }


def _accumulate_diagnosis_totals(
    summary: EpisodeDiagnosticSummary,
    totals: dict[str, Any],
) -> None:
    """Accumulate diagnosis counts from one episode into running totals."""
    if not summary.diagnosis_available:
        return
    totals["available"] += 1
    totals["record_total"] += summary.diagnosis_record_count
    for ftype, count in summary.diagnosis_failure_type_counts.items():
        totals["type_totals"][ftype] = totals["type_totals"].get(ftype, 0) + count
    for sev, count in summary.diagnosis_severity_counts.items():
        totals["severity_totals"][sev] = totals["severity_totals"].get(sev, 0) + count
    totals["unknown_total"] += summary.diagnosis_unknown_count


def _accumulate_deviation_totals(
    summary: EpisodeDiagnosticSummary,
    totals: dict[str, Any],
) -> None:
    """Accumulate execution-deviation counts from one episode."""
    if not summary.execution_deviation_available:
        return
    totals["available"] += 1
    if summary.execution_deviation_fail_closed:
        totals["fail_closed"] += 1
    intervention = summary.execution_deviation_intervention
    if intervention is not None:
        totals["intervention_counts"][intervention] = (
            totals["intervention_counts"].get(intervention, 0) + 1
        )


def _accumulate_core_metric_totals(
    summary: EpisodeDiagnosticSummary,
    totals: dict[str, Any],
) -> None:
    """Accumulate core benchmark metric values from one episode."""
    if summary.success is not None:
        totals["success_known"] += 1
        if summary.success:
            totals["success_count"] += 1
    if summary.collision is not None:
        totals["collision_known"] += 1
        if summary.collision:
            totals["collision_count"] += 1
    if summary.comfort is not None:
        totals["comfort_values"].append(summary.comfort)


def _deviation_report_to_dict(
    report: ExecutionDeviationDiagnosticReport,
) -> dict[str, Any]:
    """Convert an ExecutionDeviationDiagnosticReport to a JSON-safe dict.

    Args:
        report: The execution-deviation diagnostic report.

    Returns:
        A dictionary representation of the report.
    """
    return {
        "schema_version": report.schema_version,
        "false_alarm_count": report.false_alarm_count,
        "false_alarm_denominator": report.false_alarm_denominator,
        "false_alarm_rate": report.false_alarm_rate,
        "detection_count": report.detection_count,
        "detection_denominator": report.detection_denominator,
        "detection_recall": report.detection_recall,
        "detection_delay_s": report.detection_delay_s,
        "detection_delay_denominator": report.detection_delay_denominator,
        "intervention_counts": dict(report.intervention_counts),
        "intervention_denominator": report.intervention_denominator,
        "intervention_rate": report.intervention_rate,
        "fail_closed_count": report.fail_closed_count,
        "claim_boundary": report.claim_boundary,
    }


def build_campaign_diagnostic_summary(
    *,
    campaign_id: str,
    episode_summaries: Sequence[EpisodeDiagnosticSummary],
    execution_deviation_report: ExecutionDeviationDiagnosticReport | None = None,
) -> CampaignDiagnosticSummary:
    """Aggregate episode-level diagnostic summaries into a campaign-level crosswalk.

    This function collects denominators, coverage rates, and per-type/per-label
    counts across episodes.  Core benchmark metrics (success, collision, comfort)
    are aggregated independently from diagnostic quality.

    Denominators and validity rules
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``episode_count`` is the campaign denominator for coverage rates.
    - ``diagnosis_coverage_rate`` = episodes with diagnosis / episode_count.
    - ``execution_deviation_coverage_rate`` = episodes with deviation result / episode_count.
    - ``diagnosis_record_total`` is the sum of per-episode record counts.
    - Success, collision, and comfort denominators count only episodes where
      the metric is explicitly known (``None`` values excluded).
    - ``None`` rates indicate an empty or fully-unavailable denominator.

    Claim boundary
    ~~~~~~~~~~~~~~
    This is a reporting and artifact-contract improvement.  Campaign-level
    diagnostic counts do not imply causality, safety, or ranking.

    Args:
        campaign_id: Campaign identifier.
        episode_summaries: Episode-level crosswalk summaries.
        execution_deviation_report: Optional campaign-level deviation diagnostic report.

    Returns:
        A versioned campaign diagnostic summary.
    """
    episode_count = len(episode_summaries)
    diag_totals: dict[str, Any] = {
        "available": 0,
        "record_total": 0,
        "type_totals": {},
        "severity_totals": {},
        "unknown_total": 0,
    }
    dev_totals: dict[str, Any] = {
        "available": 0,
        "fail_closed": 0,
        "intervention_counts": {},
    }
    core_totals: dict[str, Any] = {
        "success_count": 0,
        "success_known": 0,
        "collision_count": 0,
        "collision_known": 0,
        "comfort_values": [],
    }

    for summary in episode_summaries:
        _accumulate_diagnosis_totals(summary, diag_totals)
        _accumulate_deviation_totals(summary, dev_totals)
        _accumulate_core_metric_totals(summary, core_totals)

    diag_coverage = diag_totals["available"] / episode_count if episode_count else None
    dev_coverage = dev_totals["available"] / episode_count if episode_count else None
    success_rate = (
        core_totals["success_count"] / core_totals["success_known"]
        if core_totals["success_known"]
        else None
    )
    collision_rate = (
        core_totals["collision_count"] / core_totals["collision_known"]
        if core_totals["collision_known"]
        else None
    )
    comfort_vals = core_totals["comfort_values"]
    comfort_mean = sum(comfort_vals) / len(comfort_vals) if comfort_vals else None

    dev_report_dict = (
        _deviation_report_to_dict(execution_deviation_report)
        if execution_deviation_report is not None
        else None
    )

    return CampaignDiagnosticSummary(
        campaign_id=campaign_id,
        episode_count=episode_count,
        diagnosis_available_count=diag_totals["available"],
        diagnosis_record_total=diag_totals["record_total"],
        diagnosis_failure_type_totals=diag_totals["type_totals"],
        diagnosis_severity_totals=diag_totals["severity_totals"],
        diagnosis_unknown_total=diag_totals["unknown_total"],
        diagnosis_coverage_rate=diag_coverage,
        execution_deviation_available_count=dev_totals["available"],
        execution_deviation_fail_closed_count=dev_totals["fail_closed"],
        execution_deviation_intervention_counts=dev_totals["intervention_counts"],
        execution_deviation_coverage_rate=dev_coverage,
        execution_deviation_report=dev_report_dict,
        success_rate=success_rate,
        collision_rate=collision_rate,
        comfort_mean=comfort_mean,
        caveats=list(_CROSSWALK_CAVEATS),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_nonnegative_int(value: Any, field_name: str) -> None:
    """Require a count field to be an integer other than ``bool``."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_optional_finite_number(value: Any, field_name: str) -> None:
    """Require an optional numeric field to be finite when present."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ValueError(f"{field_name} must be a finite number or None")
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be a finite number or None")


def _require_optional_bool(value: Any, field_name: str) -> None:
    """Require an optional boolean field."""
    if value is not None and type(value) is not bool:
        raise ValueError(f"{field_name} must be bool or None")


def _validate_count_mapping(value: Any, field_name: str) -> None:
    """Validate a mapping of string labels to non-negative counts."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    for label, count in value.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{field_name} labels must be non-empty strings")
        _require_nonnegative_int(count, f"{field_name}[{label!r}]")


def _validate_validity_metadata(section: Mapping[str, Any], field_name: str) -> None:
    """Validate shared availability, validity, provenance, and reason fields."""
    if type(section.get("available")) is not bool:
        raise ValueError(f"{field_name}.available must be bool")
    state = section.get("validity_state")
    if state not in FIELD_VALIDITY_STATES:
        raise ValueError(f"{field_name}.validity_state is not recognized")
    provenance = section.get("provenance")
    if provenance not in FIELD_PROVENANCE_STATES:
        raise ValueError(f"{field_name}.provenance is not recognized")
    reason = section.get("validity_reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError(f"{field_name}.validity_reason must be a non-empty string or None")


def _validate_rate(value: Any, field_name: str) -> None:
    """Validate an optional rate in the closed unit interval."""
    _require_optional_finite_number(value, field_name)
    if value is not None and not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1] or None")


def _validate_diagnosis_section(diagnosis: Mapping[str, Any]) -> None:
    """Validate the nested diagnosis section of an episode report."""
    required = (
        "available",
        "record_count",
        "failure_type_counts",
        "severity_counts",
        "unknown_count",
        "validity_state",
        "provenance",
        "validity_reason",
    )
    for field_name in required:
        if field_name not in diagnosis:
            raise ValueError(f"diagnosis.{field_name} is required")
    _validate_validity_metadata(diagnosis, "diagnosis")
    _require_nonnegative_int(diagnosis["record_count"], "diagnosis.record_count")
    _validate_count_mapping(diagnosis["failure_type_counts"], "diagnosis.failure_type_counts")
    _validate_count_mapping(diagnosis["severity_counts"], "diagnosis.severity_counts")
    _require_nonnegative_int(diagnosis["unknown_count"], "diagnosis.unknown_count")


def _validate_deviation_section(deviation: Mapping[str, Any]) -> None:
    """Validate the nested execution-deviation section of an episode report."""
    required = (
        "available",
        "intervention",
        "deviation_score",
        "fail_closed",
        "first_threshold_crossing_time_s",
        "validity_state",
        "provenance",
        "validity_reason",
        "claim_boundary",
    )
    for field_name in required:
        if field_name not in deviation:
            raise ValueError(f"execution_deviation.{field_name} is required")
    _validate_validity_metadata(deviation, "execution_deviation")
    intervention = deviation["intervention"]
    if intervention is not None and intervention not in _INTERVENTION_LABELS:
        raise ValueError("execution_deviation.intervention is not recognized")
    _require_optional_finite_number(
        deviation["deviation_score"], "execution_deviation.deviation_score"
    )
    _require_optional_bool(deviation["fail_closed"], "execution_deviation.fail_closed")
    _require_optional_finite_number(
        deviation["first_threshold_crossing_time_s"],
        "execution_deviation.first_threshold_crossing_time_s",
    )
    if deviation["claim_boundary"] is not None and not isinstance(deviation["claim_boundary"], str):
        raise ValueError("execution_deviation.claim_boundary must be a string or None")


def _validate_core_metrics_section(core: Mapping[str, Any]) -> None:
    """Validate the core benchmark metrics kept separate from diagnostics."""
    for field_name in ("success", "collision", "comfort"):
        if field_name not in core:
            raise ValueError(f"core_metrics.{field_name} is required")
    _require_optional_bool(core["success"], "core_metrics.success")
    _require_optional_bool(core["collision"], "core_metrics.collision")
    _require_optional_finite_number(core["comfort"], "core_metrics.comfort")


def validate_episode_diagnostic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an episode diagnostic summary mapping.

    Args:
        summary: A summary mapping (e.g. ``EpisodeDiagnosticSummary.to_dict()``).

    Returns:
        A normalized dictionary copy.

    Raises:
        ValueError: If required fields are missing or values are out of range.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if summary.get("schema_version") != REPORT_CROSSWALK_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {REPORT_CROSSWALK_SCHEMA_VERSION!r}, "
            f"got {summary.get('schema_version')!r}"
        )
    if summary.get("report_source", REPORT_CROSSWALK_SOURCE) != REPORT_CROSSWALK_SOURCE:
        raise ValueError("report_source must identify report_crosswalk.v1")
    for field_name in ("episode_id", "planner_id"):
        value = summary.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    diagnosis = summary.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ValueError("diagnosis must be a mapping")
    _validate_diagnosis_section(diagnosis)
    dev = summary.get("execution_deviation")
    if not isinstance(dev, Mapping):
        raise ValueError("execution_deviation must be a mapping")
    _validate_deviation_section(dev)
    core = summary.get("core_metrics")
    if not isinstance(core, Mapping):
        raise ValueError("core_metrics must be a mapping")
    _validate_core_metrics_section(core)
    caveats = summary.get("caveats")
    if not isinstance(caveats, list) or not all(isinstance(caveat, str) for caveat in caveats):
        raise ValueError("caveats must be a list of strings")
    return dict(summary)


def _validate_campaign_diagnosis_section(diagnosis: Mapping[str, Any]) -> None:
    """Validate campaign-level diagnosis counts and coverage."""
    for field_name in ("available_count", "record_total", "unknown_total"):
        _require_nonnegative_int(diagnosis.get(field_name), f"diagnosis.{field_name}")
    _validate_count_mapping(diagnosis.get("failure_type_totals"), "diagnosis.failure_type_totals")
    _validate_count_mapping(diagnosis.get("severity_totals"), "diagnosis.severity_totals")
    _validate_rate(diagnosis.get("coverage_rate"), "diagnosis.coverage_rate")


def _validate_campaign_deviation_section(deviation: Mapping[str, Any]) -> None:
    """Validate campaign-level execution-deviation counts and report."""
    for field_name in ("available_count", "fail_closed_count"):
        _require_nonnegative_int(deviation.get(field_name), f"execution_deviation.{field_name}")
    _validate_count_mapping(
        deviation.get("intervention_counts"),
        "execution_deviation.intervention_counts",
    )
    _validate_rate(deviation.get("coverage_rate"), "execution_deviation.coverage_rate")
    if deviation.get("diagnostic_report") is not None and not isinstance(
        deviation.get("diagnostic_report"), Mapping
    ):
        raise ValueError("execution_deviation.diagnostic_report must be a mapping or None")


def _validate_campaign_core_section(core: Mapping[str, Any]) -> None:
    """Validate campaign-level core metric aggregates."""
    _validate_rate(core.get("success_rate"), "core_metrics.success_rate")
    _validate_rate(core.get("collision_rate"), "core_metrics.collision_rate")
    _require_optional_finite_number(core.get("comfort_mean"), "core_metrics.comfort_mean")


def validate_campaign_diagnostic_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a campaign diagnostic summary mapping.

    Args:
        summary: A summary mapping (e.g. ``CampaignDiagnosticSummary.to_dict()``).

    Returns:
        A normalized dictionary copy.

    Raises:
        ValueError: If required fields are missing or values are out of range.
    """
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be a mapping")
    if summary.get("schema_version") != REPORT_CROSSWALK_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {REPORT_CROSSWALK_SCHEMA_VERSION!r}, "
            f"got {summary.get('schema_version')!r}"
        )
    if summary.get("report_source", REPORT_CROSSWALK_SOURCE) != REPORT_CROSSWALK_SOURCE:
        raise ValueError("report_source must identify report_crosswalk.v1")
    if not isinstance(summary.get("campaign_id"), str):
        raise ValueError("campaign_id must be a string")
    _require_nonnegative_int(summary.get("episode_count"), "episode_count")
    for section in ("diagnosis", "execution_deviation", "core_metrics"):
        if not isinstance(summary.get(section), Mapping):
            raise ValueError(f"{section} must be a mapping")
    _validate_campaign_diagnosis_section(summary["diagnosis"])
    _validate_campaign_deviation_section(summary["execution_deviation"])
    _validate_campaign_core_section(summary["core_metrics"])
    caveats = summary.get("caveats")
    if not isinstance(caveats, list) or not all(isinstance(caveat, str) for caveat in caveats):
        raise ValueError("caveats must be a list of strings")
    return dict(summary)


# ---------------------------------------------------------------------------
# Deterministic fixture / example
# ---------------------------------------------------------------------------


def build_crosswalk_example_fixture() -> dict[str, Any]:
    """Build a deterministic example fixture demonstrating the crosswalk.

    The fixture includes:
    1. An episode with a known collision diagnosis (failure_type="collision").
    2. An episode with an unknown-type diagnosis.
    3. An episode with an execution-deviation case (warn intervention).
    4. An episode with a fail-closed execution-deviation result.
    5. Backward-compatible core metrics alongside diagnostic fields.

    Returns:
        A versioned fixture dictionary suitable for export and testing.
    """
    fixture = {
        "schema_version": REPORT_CROSSWALK_SCHEMA_VERSION,
        "report_source": REPORT_CROSSWALK_SOURCE,
        "fixture_id": "report_crosswalk.example.v1",
        "fixture_version": 1,
        "description": (
            "Deterministic example fixture for the report crosswalk (issue #6871). "
            "Demonstrates backward-compatible export and diagnostic-quality separation."
        ),
        "episodes": [
            {
                "episode_id": "ep_001_collision",
                "planner_id": "orca",
                "diagnosis_payload": {
                    "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
                    "diagnosis_source": DIAGNOSIS_SOURCE,
                    "generated_at_utc": "2026-01-01T00:00:00+00:00",
                    "records": [
                        {
                            "diagnosis_schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
                            "diagnosis_source": DIAGNOSIS_SOURCE,
                            "failure_level": "interaction",
                            "failure_type": "collision",
                            "onset_time_s": 1.0,
                            "onset_interval": [1.0, 1.5],
                            "severity": "critical",
                            "detection_method": "predicate",
                            "causal_evidence": [
                                {
                                    "evidence_kind": "trace_failure_predicate",
                                    "predicate_id": "collision",
                                    "time_interval_s": [1.0, 1.5],
                                    "steps": [10, 15],
                                    "involved_actors": ["robot", "ped_0"],
                                    "evidence_fields": {"min_clearance_m": 0.1},
                                    "non_causal_note": _NON_CAUSAL_CAVEAT,
                                }
                            ],
                            "contributing_factors": [],
                            "confidence": "supported_hypothesis",
                            "evidence_mode": "direct_probe",
                            "validity_status": "valid",
                            "proposed_correction": None,
                            "correction_status": DEFAULT_CORRECTION_STATUS,
                            "unknown_reason": None,
                            "caveats": [_NON_CAUSAL_CAVEAT, _DIAGNOSTIC_LABEL_CAVEAT],
                            "source_predicate": {
                                "predicate_id": "collision",
                                "time_interval_s": [1.0, 1.5],
                                "steps": [10, 15],
                                "involved_actors": ["robot", "ped_0"],
                                "scenario_family": "crosswalk",
                                "planner_id": "orca",
                                "evidence_fields": {"min_clearance_m": 0.1},
                                "severity": "critical",
                                "validity_status": "valid",
                            },
                        }
                    ],
                    "failure_type_coverage": {
                        "counts": {"collision": 1},
                        "classification_source": DIAGNOSIS_SOURCE,
                    },
                    "caveats": [
                        _PAYLOAD_NON_CLAIM_CAVEAT,
                        _NON_CAUSAL_CAVEAT,
                        _OUT_OF_SCOPE_CAVEAT,
                    ],
                },
                "execution_deviation_result": None,
                "success": False,
                "collision": True,
                "comfort": None,
            },
            {
                "episode_id": "ep_002_unknown_type",
                "planner_id": "orca",
                "diagnosis_payload": {
                    "schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
                    "diagnosis_source": DIAGNOSIS_SOURCE,
                    "generated_at_utc": "2026-01-01T00:00:00+00:00",
                    "records": [
                        {
                            "diagnosis_schema_version": FAILURE_DIAGNOSIS_SCHEMA_VERSION,
                            "diagnosis_source": DIAGNOSIS_SOURCE,
                            "failure_level": "control",
                            "failure_type": "unknown",
                            "onset_time_s": 2.0,
                            "onset_interval": [2.0, 3.0],
                            # The deterministic adapter maps a valid medium
                            # predicate to the canonical "major" token even
                            # when its failure type is unknown.
                            "severity": "major",
                            "detection_method": "predicate",
                            "causal_evidence": [
                                {
                                    "evidence_kind": "trace_failure_predicate",
                                    "predicate_id": "oscillatory_local_control",
                                    "time_interval_s": [2.0, 3.0],
                                    "steps": [20, 30],
                                    "involved_actors": ["robot"],
                                    "evidence_fields": {},
                                    "non_causal_note": _NON_CAUSAL_CAVEAT,
                                }
                            ],
                            "contributing_factors": [],
                            "confidence": "unknown",
                            "evidence_mode": "unknown",
                            "validity_status": "valid",
                            "proposed_correction": None,
                            "correction_status": DEFAULT_CORRECTION_STATUS,
                            "unknown_reason": "oscillation_not_represented_in_classifier_labels",
                            "caveats": [
                                _NON_CAUSAL_CAVEAT,
                                _UNKNOWN_CAVEAT,
                                "unknown_reason: oscillation_not_represented_in_classifier_labels",
                            ],
                            "source_predicate": {
                                "predicate_id": "oscillatory_local_control",
                                "time_interval_s": [2.0, 3.0],
                                "steps": [20, 30],
                                "involved_actors": ["robot"],
                                "scenario_family": "crosswalk",
                                "planner_id": "orca",
                                "evidence_fields": {},
                                "severity": "medium",
                                "validity_status": "valid",
                            },
                        }
                    ],
                    "failure_type_coverage": {
                        "counts": {"unknown": 1},
                        "classification_source": DIAGNOSIS_SOURCE,
                    },
                    "caveats": [
                        _PAYLOAD_NON_CLAIM_CAVEAT,
                        _NON_CAUSAL_CAVEAT,
                        _OUT_OF_SCOPE_CAVEAT,
                    ],
                },
                "execution_deviation_result": None,
                "success": True,
                "collision": False,
                "comfort": 0.8,
            },
            {
                "episode_id": "ep_003_deviation_warn",
                "planner_id": "dwa",
                "diagnosis_payload": None,
                "execution_deviation_result": ExecutionDeviationResult(
                    intervention=INTERVENTION_WARN,
                    deviation_score=0.6,
                    component_deviations=(("robot_position", 0.6),),
                    first_threshold_crossing_time_s=0.5,
                    input_age_s=0.1,
                    fail_closed=False,
                    schema_version=EXECUTION_DEVIATION_SCHEMA,
                    claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
                ),
                "success": True,
                "collision": False,
                "comfort": 0.9,
            },
            {
                "episode_id": "ep_004_deviation_fail_closed",
                "planner_id": "dwa",
                "diagnosis_payload": None,
                "execution_deviation_result": ExecutionDeviationResult(
                    intervention=INTERVENTION_WARN,
                    deviation_score=None,
                    component_deviations=(),
                    first_threshold_crossing_time_s=None,
                    input_age_s=None,
                    fail_closed=True,
                    schema_version=EXECUTION_DEVIATION_SCHEMA,
                    claim_boundary=EXECUTION_DEVIATION_CLAIM_BOUNDARY,
                ),
                "success": True,
                "collision": False,
                "comfort": None,
            },
        ],
        "caveats": list(_CROSSWALK_CAVEATS),
    }

    # Re-run each hand-readable fixture record through the public upstream
    # builder.  This keeps the example deterministic while ensuring its nested
    # schema and adapter provenance cannot drift from ``failure_diagnosis.v1``.
    for episode in fixture["episodes"]:
        payload = episode.get("diagnosis_payload")
        if isinstance(payload, Mapping):
            episode["diagnosis_payload"] = build_failure_diagnosis_payload(
                payload["records"],
                generated_at_utc=payload["generated_at_utc"],
            )
    return fixture


def export_crosswalk_example_fixture() -> dict[str, Any]:
    """Export the deterministic fixture as a versioned, JSON-safe report.

    Builds episode-level summaries for each episode in the example fixture
    and aggregates them into a campaign-level summary.

    Returns:
        A versioned crosswalk report with episode and campaign summaries.
    """
    fixture = build_crosswalk_example_fixture()
    episode_summaries: list[EpisodeDiagnosticSummary] = []

    for ep in fixture["episodes"]:
        dev_result = ep.get("execution_deviation_result")
        summary = build_episode_diagnostic_summary(
            episode_id=ep["episode_id"],
            planner_id=ep["planner_id"],
            diagnosis_payload=ep.get("diagnosis_payload"),
            execution_deviation_result=dev_result
            if isinstance(dev_result, ExecutionDeviationResult)
            else None,
            success=ep.get("success"),
            collision=ep.get("collision"),
            comfort=ep.get("comfort"),
        )
        episode_summaries.append(summary)

    campaign_summary = build_campaign_diagnostic_summary(
        campaign_id="example_campaign",
        episode_summaries=episode_summaries,
    )

    return {
        "schema_version": REPORT_CROSSWALK_SCHEMA_VERSION,
        "report_source": REPORT_CROSSWALK_SOURCE,
        "fixture_id": fixture["fixture_id"],
        "fixture_version": fixture["fixture_version"],
        "description": fixture["description"],
        "episodes": [s.to_dict() for s in episode_summaries],
        "campaign": campaign_summary.to_dict(),
        "caveats": fixture["caveats"],
    }


__all__ = [
    "FIELD_PROVENANCE_STATES",
    "FIELD_VALIDITY_STATES",
    "REPORT_CROSSWALK_SCHEMA_VERSION",
    "REPORT_CROSSWALK_SOURCE",
    "CampaignDiagnosticSummary",
    "EpisodeDiagnosticSummary",
    "build_campaign_diagnostic_summary",
    "build_crosswalk_example_fixture",
    "build_episode_diagnostic_summary",
    "export_crosswalk_example_fixture",
    "validate_campaign_diagnostic_summary",
    "validate_episode_diagnostic_summary",
]

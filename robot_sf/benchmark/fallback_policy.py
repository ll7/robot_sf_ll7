"""Canonical benchmark fallback and availability policy helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkAvailability:
    """Normalized benchmark availability and reporting status."""

    execution_mode: str
    readiness_status: str
    availability_status: str
    benchmark_success: bool
    availability_reason: str | None = None


@dataclass(frozen=True)
class CampaignOutcome:
    """Normalized camera-ready campaign outcome and exit semantics."""

    status: str
    benchmark_success: bool
    status_reason: str
    successful_runs: int
    accepted_unavailable_runs: int
    unexpected_failed_runs: int
    non_success_runs: int
    total_runs: int
    exit_code: int


@dataclass(frozen=True)
class CampaignRowStatusSummary:
    """Canonical planner-row status counters for one campaign payload."""

    successful_evidence_rows: int
    accepted_unavailable_rows: int
    unexpected_failed_rows: int
    fallback_or_degraded_rows: int


@dataclass(frozen=True)
class CampaignStatusAxes:
    """Normalized execution/evidence axes for a camera-ready campaign."""

    campaign_execution_status: str
    evidence_status: str
    row_status_summary: CampaignRowStatusSummary


_ACCEPTED_UNAVAILABLE_STATUSES = {"not_available", "excluded"}
_UNEXPECTED_FAILURE_STATUSES = {"failed", "partial-failure"}
_SUCCESS_CAMPAIGN_STATUSES = {"benchmark_success", "ok"}
_ACCEPTED_UNAVAILABLE_CAMPAIGN_STATUSES = {"accepted_unavailable_only"}
_UNEXPECTED_FAILURE_CAMPAIGN_STATUSES = {"unexpected_failure"}
_CANONICAL_EXIT_CODES: set[int] = {0, 2, 3}


def _int_field(result: dict[str, Any], key: str) -> int:
    """Read one integer-like explicit campaign field with a fail-closed default.

    Returns:
        Parsed integer value, or zero when the field is missing or malformed.
    """
    value = result.get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _status_from_row(row: Any) -> str | None:
    """Return a normalized non-empty row status, or ``None`` for malformed rows."""
    if not isinstance(row, dict):
        return None
    status = str(row.get("status", "")).strip().lower()
    return status or None


def _run_statuses_from_result(result: dict[str, Any]) -> list[str]:
    """Extract normalized planner/run statuses from a campaign payload.

    planner_rows statuses are trusted only when every row has a non-empty status.
    Otherwise the function falls back to ``runs`` entries or returns an empty list.

    Returns:
        Ordered planner/run status labels when available.
    """
    run_statuses: list[str] = []
    planner_rows = result.get("planner_rows")
    if isinstance(planner_rows, list):
        planner_statuses = [_status_from_row(row) for row in planner_rows]
        if planner_statuses and all(status is not None for status in planner_statuses):
            return [status for status in planner_statuses if status is not None]

    runs = result.get("runs")
    if isinstance(runs, list):
        for entry in runs:
            status = _status_from_row(entry)
            if status:
                run_statuses.append(status)
    return run_statuses


def _campaign_outcome_from_explicit_fields(result: dict[str, Any]) -> CampaignOutcome:
    """Resolve campaign outcome from explicit top-level fields when row statuses are absent.

    Returns:
        Campaign outcome derived from top-level status metadata.
    """
    explicit_status = (
        str(result.get("status") or result.get("campaign_status") or "").strip().lower()
    )
    explicit_reason = str(result.get("status_reason", "")).strip()
    benchmark_success = result.get("benchmark_success") is True
    (
        successful_runs,
        accepted_unavailable_runs,
        unexpected_failed_runs,
        non_success_runs,
        total_runs,
    ) = _explicit_campaign_counts(result)

    if unexpected_failed_runs > 0 or explicit_status in _UNEXPECTED_FAILURE_CAMPAIGN_STATUSES:
        resolved_unexpected_failed_runs = unexpected_failed_runs
        if resolved_unexpected_failed_runs <= 0:
            resolved_unexpected_failed_runs = max(
                non_success_runs - accepted_unavailable_runs,
                total_runs - successful_runs - accepted_unavailable_runs,
                0,
            )
        return CampaignOutcome(
            status="unexpected_failure",
            benchmark_success=False,
            status_reason=explicit_reason
            or "campaign contains unexpected failed or partially failed rows",
            successful_runs=successful_runs,
            accepted_unavailable_runs=accepted_unavailable_runs,
            unexpected_failed_runs=resolved_unexpected_failed_runs,
            non_success_runs=max(non_success_runs, resolved_unexpected_failed_runs),
            total_runs=total_runs,
            exit_code=2,
        )
    if (
        total_runs > 0
        and non_success_runs == 0
        and (not explicit_status or explicit_status in _SUCCESS_CAMPAIGN_STATUSES)
        and (
            benchmark_success
            or explicit_status in _SUCCESS_CAMPAIGN_STATUSES
            or successful_runs >= total_runs
        )
    ):
        return CampaignOutcome(
            status="benchmark_success",
            benchmark_success=True,
            status_reason=explicit_reason or "all planner rows were benchmark-success",
            successful_runs=total_runs or successful_runs,
            accepted_unavailable_runs=0,
            unexpected_failed_runs=0,
            non_success_runs=0,
            total_runs=total_runs,
            exit_code=0,
        )
    if accepted_unavailable_runs > 0 or explicit_status in _ACCEPTED_UNAVAILABLE_CAMPAIGN_STATUSES:
        resolved_non_success_runs = max(
            non_success_runs,
            accepted_unavailable_runs,
            total_runs - successful_runs,
            0,
        )
        resolved_total_runs = max(total_runs, successful_runs + resolved_non_success_runs)
        resolved_accepted_runs = max(
            accepted_unavailable_runs,
            resolved_non_success_runs - unexpected_failed_runs,
        )
        return CampaignOutcome(
            status="accepted_unavailable_only",
            benchmark_success=False,
            status_reason=explicit_reason
            or "campaign contains accepted unavailable/excluded rows and no unexpected failed rows",
            successful_runs=successful_runs,
            accepted_unavailable_runs=resolved_accepted_runs,
            unexpected_failed_runs=0,
            non_success_runs=resolved_non_success_runs,
            total_runs=resolved_total_runs,
            exit_code=3,
        )
    return CampaignOutcome(
        status="malformed",
        benchmark_success=False,
        status_reason=(
            explicit_reason
            or (
                f"campaign result payload has unknown explicit status '{explicit_status}'"
                if explicit_status
                else "campaign result payload is missing planner row status information"
            )
        ),
        successful_runs=successful_runs,
        accepted_unavailable_runs=accepted_unavailable_runs,
        unexpected_failed_runs=unexpected_failed_runs,
        non_success_runs=non_success_runs,
        total_runs=total_runs,
        exit_code=2,
    )


def _campaign_outcome_from_run_statuses(run_statuses: list[str]) -> CampaignOutcome:
    """Resolve campaign outcome from normalized planner/run statuses.

    Returns:
        Campaign outcome derived from planner/run status labels.
    """
    successful_runs = sum(1 for status in run_statuses if status == "ok")
    accepted_unavailable_runs = sum(
        1 for status in run_statuses if status in _ACCEPTED_UNAVAILABLE_STATUSES
    )
    unexpected_failed_runs = sum(
        1 for status in run_statuses if status in _UNEXPECTED_FAILURE_STATUSES
    )
    unexpected_failed_runs += sum(
        1
        for status in run_statuses
        if status not in {"ok"} | _ACCEPTED_UNAVAILABLE_STATUSES | _UNEXPECTED_FAILURE_STATUSES
    )
    total_runs = len(run_statuses)
    non_success_runs = total_runs - successful_runs
    if total_runs > 0 and non_success_runs == 0:
        return CampaignOutcome(
            status="benchmark_success",
            benchmark_success=True,
            status_reason="all planner rows were benchmark-success",
            successful_runs=successful_runs,
            accepted_unavailable_runs=0,
            unexpected_failed_runs=0,
            non_success_runs=0,
            total_runs=total_runs,
            exit_code=0,
        )
    if total_runs > 0 and accepted_unavailable_runs > 0 and unexpected_failed_runs == 0:
        return CampaignOutcome(
            status="accepted_unavailable_only",
            benchmark_success=False,
            status_reason=(
                "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
            ),
            successful_runs=successful_runs,
            accepted_unavailable_runs=accepted_unavailable_runs,
            unexpected_failed_runs=0,
            non_success_runs=non_success_runs,
            total_runs=total_runs,
            exit_code=3,
        )
    return CampaignOutcome(
        status="unexpected_failure",
        benchmark_success=False,
        status_reason=(
            "campaign contains unexpected failed or partially failed rows"
            if total_runs > 0
            else "campaign produced no planner runs"
        ),
        successful_runs=successful_runs,
        accepted_unavailable_runs=accepted_unavailable_runs,
        unexpected_failed_runs=unexpected_failed_runs,
        non_success_runs=non_success_runs,
        total_runs=total_runs,
        exit_code=2,
    )


def _explicit_campaign_counts(result: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Resolve campaign run counters from explicit fields and row-summary metadata.

    Returns:
        ``successful_runs``, ``accepted_unavailable_runs``, ``unexpected_failed_runs``,
        ``non_success_runs``, and ``total_runs`` counters.
    """
    explicit_row_status_summary = _normalized_row_status_summary(result.get("row_status_summary"))
    successful_runs = _int_field(result, "successful_runs")
    accepted_unavailable_runs = _int_field(result, "accepted_unavailable_runs")
    unexpected_failed_runs = _int_field(result, "unexpected_failed_runs")
    non_success_runs = _int_field(result, "non_success_runs")
    total_runs = _int_field(result, "total_runs")

    if explicit_row_status_summary is not None:
        successful_runs = _explicit_or_summary_count(
            result, "successful_runs", explicit_row_status_summary.successful_evidence_rows
        )
        accepted_unavailable_runs = _explicit_or_summary_count(
            result,
            "accepted_unavailable_runs",
            explicit_row_status_summary.accepted_unavailable_rows,
        )
        unexpected_failed_runs = _explicit_or_summary_count(
            result, "unexpected_failed_runs", explicit_row_status_summary.unexpected_failed_rows
        )
        non_success_runs = _explicit_or_summary_count(
            result,
            "non_success_runs",
            explicit_row_status_summary.accepted_unavailable_rows
            + explicit_row_status_summary.unexpected_failed_rows,
        )
        total_runs = _explicit_or_summary_count(
            result,
            "total_runs",
            explicit_row_status_summary.successful_evidence_rows
            + explicit_row_status_summary.accepted_unavailable_rows
            + explicit_row_status_summary.unexpected_failed_rows,
        )

    if total_runs <= 0:
        total_runs = successful_runs + accepted_unavailable_runs + unexpected_failed_runs
    if non_success_runs <= 0 and total_runs > 0:
        non_success_runs = max(
            total_runs - successful_runs,
            accepted_unavailable_runs + unexpected_failed_runs,
        )
    return (
        successful_runs,
        accepted_unavailable_runs,
        unexpected_failed_runs,
        non_success_runs,
        total_runs,
    )


def _explicit_or_summary_count(result: dict[str, Any], key: str, summary_value: int) -> int:
    """Return an explicit counter when present, otherwise the row-summary value.

    Returns:
        Non-negative integer counter.
    """
    if key in result:
        return _int_field(result, key)
    return summary_value


def _normalized_row_status_summary(raw: Any) -> CampaignRowStatusSummary | None:
    """Parse an explicit row-status summary mapping when present.

    Returns:
        Normalized row-status counters, or ``None`` when the payload is not a mapping.
    """
    if not isinstance(raw, dict):
        return None
    return CampaignRowStatusSummary(
        successful_evidence_rows=_int_field(raw, "successful_evidence_rows"),
        accepted_unavailable_rows=_int_field(raw, "accepted_unavailable_rows"),
        unexpected_failed_rows=_int_field(raw, "unexpected_failed_rows"),
        fallback_or_degraded_rows=_int_field(raw, "fallback_or_degraded_rows"),
    )


def _row_status_summary_from_run_statuses(
    run_statuses: list[str], *, fallback_or_degraded_rows: int
) -> CampaignRowStatusSummary:
    """Build row-status counters from normalized planner/run statuses.

    Returns:
        Row-status counters for success, accepted-unavailable, unexpected failure, and fallback.
    """
    return CampaignRowStatusSummary(
        successful_evidence_rows=sum(1 for status in run_statuses if status == "ok"),
        accepted_unavailable_rows=sum(
            1 for status in run_statuses if status in _ACCEPTED_UNAVAILABLE_STATUSES
        ),
        unexpected_failed_rows=sum(
            1 for status in run_statuses if status not in {"ok"} | _ACCEPTED_UNAVAILABLE_STATUSES
        ),
        fallback_or_degraded_rows=fallback_or_degraded_rows,
    )


def _row_status_summary_from_planner_rows(rows: Any) -> CampaignRowStatusSummary | None:
    """Build row-status counters from planner rows when every row is well formed.

    Returns:
        Row-status counters, or ``None`` when planner rows are absent or malformed.
    """
    if not isinstance(rows, list):
        return None
    planner_statuses = [_status_from_row(row) for row in rows]
    if not planner_statuses or any(status is None for status in planner_statuses):
        return None
    fallback_or_degraded_rows = sum(
        1
        for row in rows
        if isinstance(row, dict)
        and str(row.get("readiness_status", "")).strip().lower() in {"fallback", "degraded"}
    )
    return _row_status_summary_from_run_statuses(
        [status for status in planner_statuses if status is not None],
        fallback_or_degraded_rows=fallback_or_degraded_rows,
    )


def _row_status_summary_from_runs(runs: Any) -> CampaignRowStatusSummary | None:
    """Build row-status counters from run entries when every run supplies a status.

    Returns:
        Row-status counters, or ``None`` when run entries are absent or malformed.
    """
    if not isinstance(runs, list):
        return None
    run_statuses = [_status_from_row(entry) for entry in runs]
    if not run_statuses or any(status is None for status in run_statuses):
        return None

    fallback_or_degraded_rows = 0
    for entry in runs:
        if not isinstance(entry, dict):
            return None
        summary = entry.get("summary")
        if not isinstance(summary, dict):
            summary = entry
        if summarize_benchmark_availability(summary).readiness_status in {"fallback", "degraded"}:
            fallback_or_degraded_rows += 1
    return _row_status_summary_from_run_statuses(
        [status for status in run_statuses if status is not None],
        fallback_or_degraded_rows=fallback_or_degraded_rows,
    )


def summarize_row_status_summary(result: dict[str, Any] | None) -> CampaignRowStatusSummary:
    """Return canonical planner-row counters for a campaign payload."""
    if not isinstance(result, dict):
        return CampaignRowStatusSummary(0, 0, 0, 0)

    planner_rows_summary = _row_status_summary_from_planner_rows(result.get("planner_rows"))
    if planner_rows_summary is not None:
        return planner_rows_summary

    runs_summary = _row_status_summary_from_runs(result.get("runs"))
    if runs_summary is not None:
        return runs_summary

    explicit_summary = _normalized_row_status_summary(result.get("row_status_summary"))
    if explicit_summary is not None:
        return explicit_summary

    return CampaignRowStatusSummary(
        successful_evidence_rows=_int_field(result, "successful_runs"),
        accepted_unavailable_rows=_int_field(result, "accepted_unavailable_runs"),
        unexpected_failed_rows=_int_field(result, "unexpected_failed_runs"),
        fallback_or_degraded_rows=_int_field(result, "fallback_or_degraded_rows"),
    )


def summarize_campaign_status_axes(
    result: dict[str, Any] | None, *, expected_total_runs: int | None = None
) -> CampaignStatusAxes:
    """Return explicit execution/evidence axes for a campaign payload."""
    row_status_summary = summarize_row_status_summary(result)
    if not isinstance(result, dict):
        return CampaignStatusAxes(
            campaign_execution_status="failed",
            evidence_status="invalid",
            row_status_summary=row_status_summary,
        )

    outcome = summarize_campaign_outcome(result)
    observed_total_runs = (
        row_status_summary.successful_evidence_rows
        + row_status_summary.accepted_unavailable_rows
        + row_status_summary.unexpected_failed_rows
    )
    if expected_total_runs is None or expected_total_runs <= 0:
        expected_total_runs = None
    interrupted = (
        expected_total_runs is not None
        and observed_total_runs > 0
        and observed_total_runs < expected_total_runs
        and outcome.status in _UNEXPECTED_FAILURE_CAMPAIGN_STATUSES
    )

    if outcome.status in _UNEXPECTED_FAILURE_CAMPAIGN_STATUSES | {"malformed"}:
        campaign_execution_status = "interrupted" if interrupted else "failed"
        evidence_status = "invalid"
    elif outcome.status in _SUCCESS_CAMPAIGN_STATUSES:
        campaign_execution_status = "completed"
        evidence_status = "valid"
    else:
        campaign_execution_status = "completed"
        evidence_status = (
            "partial" if row_status_summary.successful_evidence_rows > 0 else "blocked"
        )

    return CampaignStatusAxes(
        campaign_execution_status=campaign_execution_status,
        evidence_status=evidence_status,
        row_status_summary=row_status_summary,
    )


def campaign_status_axes_payload(
    result: dict[str, Any] | None, *, expected_total_runs: int | None = None
) -> dict[str, Any]:
    """Return JSON-serializable execution/evidence axes for campaign payloads."""
    return asdict(summarize_campaign_status_axes(result, expected_total_runs=expected_total_runs))


def resolve_execution_mode(algorithm_metadata_contract: Any) -> str:
    """Resolve execution mode from algorithm metadata payload with legacy fallbacks.

    Returns:
        Canonical execution-mode label for benchmark reporting.
    """
    if not isinstance(algorithm_metadata_contract, dict):
        return "unknown"

    planner_kinematics = algorithm_metadata_contract.get("planner_kinematics")
    if isinstance(planner_kinematics, dict):
        execution_mode = planner_kinematics.get("execution_mode")
        if execution_mode is not None:
            return str(execution_mode)

    execution_mode = algorithm_metadata_contract.get("execution_mode")
    if execution_mode is not None:
        return str(execution_mode)

    adapter_impact = algorithm_metadata_contract.get("adapter_impact")
    if isinstance(adapter_impact, dict):
        execution_mode = adapter_impact.get("execution_mode")
        if execution_mode is not None:
            return str(execution_mode)

    return "unknown"


def _summary_reason(summary: dict[str, Any], preflight: dict[str, Any]) -> str | None:
    """Extract the most useful user-facing reason for a non-success planner state.

    Returns:
        Best-effort human-readable reason for a non-success benchmark outcome.
    """
    learned_contract_reason = _learned_policy_contract_reason(preflight)
    if learned_contract_reason is not None:
        return learned_contract_reason

    compatibility_reason = preflight.get("compatibility_reason")
    if compatibility_reason is not None:
        return str(compatibility_reason)

    preflight_error = preflight.get("error")
    if preflight_error is not None:
        return str(preflight_error)

    summary_error = summary.get("error")
    if summary_error is not None:
        return str(summary_error)

    failures = summary.get("failures")
    if isinstance(failures, list) and failures:
        first = failures[0]
        if isinstance(first, dict):
            failure_error = first.get("error")
            if failure_error is not None:
                return str(failure_error)
    return None


def _learned_policy_contract_reason(preflight: dict[str, Any]) -> str | None:
    """Extract mismatch/warning details from the learned-policy contract payload.

    Returns:
        Joined learned-policy mismatch/warning details when present.
    """
    learned_policy_contract = preflight.get("learned_policy_contract")
    if not isinstance(learned_policy_contract, dict):
        return None

    entries: list[str] = []
    for key in ("critical_mismatches", "warnings"):
        value = learned_policy_contract.get(key)
        if isinstance(value, list):
            entries.extend(str(item) for item in value if item is not None)
        elif value is not None:
            entries.append(str(value))
    if entries:
        return "; ".join(entries)
    return None


def _latency_stress_availability(
    summary: dict[str, Any],
    *,
    execution_mode: str,
    readiness_status: str,
    reason: str | None,
) -> BenchmarkAvailability | None:
    """Return fail-closed availability for latency-stress preflight runs."""
    latency_stress_profile = summary.get("latency_stress_profile")
    if latency_stress_profile is None:
        return None

    latency_stress_metrics = summary.get("latency_stress_metrics")
    latency_reason = "latency_stress_profile is preflight/provenance-only"
    if isinstance(latency_stress_metrics, dict):
        unavailable_latency_metrics = {
            str(value).strip().lower() for value in latency_stress_metrics.values()
        }
        if unavailable_latency_metrics == {"not_available"}:
            latency_reason = (
                "latency_stress_profile is preflight/provenance-only; "
                "runtime latency metrics are not implemented"
            )
    return BenchmarkAvailability(
        execution_mode=execution_mode,
        readiness_status=readiness_status,
        availability_status="not_available",
        benchmark_success=False,
        availability_reason=reason or latency_reason,
    )


def summarize_benchmark_availability(summary: dict[str, Any] | None) -> BenchmarkAvailability:  # noqa: C901
    """Return the canonical benchmark availability classification for one run summary."""
    if not isinstance(summary, dict):
        return BenchmarkAvailability(
            execution_mode="unknown",
            readiness_status="degraded",
            availability_status="failed",
            benchmark_success=False,
            availability_reason="malformed_summary_payload",
        )

    preflight = summary.get("preflight")
    if not isinstance(preflight, dict):
        preflight = {}
    preflight_status = str(preflight.get("status", "unknown")).strip().lower()
    status = str(summary.get("status", "unknown")).strip().lower()
    failed_jobs = int(summary.get("failed_jobs", 0) or 0)
    total_jobs = int(summary.get("total_jobs", 0) or 0)
    written = int(summary.get("written", 0) or 0)
    execution_mode = resolve_execution_mode(summary.get("algorithm_metadata_contract"))

    readiness_status = "native"
    if preflight_status == "fallback":
        readiness_status = "fallback"
    elif preflight_status == "skipped" or status == "failed":
        readiness_status = "degraded"
    elif execution_mode in {"adapter", "mixed"}:
        readiness_status = "adapter"

    reason = _summary_reason(summary, preflight)
    if preflight_status in {"fallback", "skipped"}:
        return BenchmarkAvailability(
            execution_mode=execution_mode,
            readiness_status=readiness_status,
            availability_status="not_available",
            benchmark_success=False,
            availability_reason=reason,
        )
    if status == "failed":
        return BenchmarkAvailability(
            execution_mode=execution_mode,
            readiness_status=readiness_status,
            availability_status="failed",
            benchmark_success=False,
            availability_reason=reason,
        )
    if failed_jobs > 0:
        return BenchmarkAvailability(
            execution_mode=execution_mode,
            readiness_status=readiness_status,
            availability_status="partial-failure",
            benchmark_success=False,
            availability_reason=reason,
        )
    if total_jobs > 0 and written == 0:
        return BenchmarkAvailability(
            execution_mode=execution_mode,
            readiness_status=readiness_status,
            availability_status="failed",
            benchmark_success=False,
            availability_reason=reason or "zero episodes written for scheduled jobs",
        )
    latency_availability = _latency_stress_availability(
        summary,
        execution_mode=execution_mode,
        readiness_status=readiness_status,
        reason=reason,
    )
    if latency_availability is not None:
        return latency_availability
    return BenchmarkAvailability(
        execution_mode=execution_mode,
        readiness_status=readiness_status,
        availability_status="available",
        benchmark_success=True,
        availability_reason=None,
    )


def availability_payload(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-serializable availability payload for benchmark summaries."""
    return asdict(summarize_benchmark_availability(summary))


def benchmark_run_exit_code(summary: dict[str, Any] | None) -> int:
    """Return benchmark CLI exit code for a run summary."""
    availability = summarize_benchmark_availability(summary)
    return 0 if availability.benchmark_success else 2


def summarize_campaign_outcome(result: dict[str, Any] | None) -> CampaignOutcome:
    """Return canonical campaign status for camera-ready benchmark results."""
    if not isinstance(result, dict):
        return CampaignOutcome(
            status="malformed",
            benchmark_success=False,
            status_reason="campaign result payload is missing or not a mapping",
            successful_runs=0,
            accepted_unavailable_runs=0,
            unexpected_failed_runs=0,
            non_success_runs=0,
            total_runs=0,
            exit_code=2,
        )

    run_statuses = _run_statuses_from_result(result)
    if not run_statuses:
        return _campaign_outcome_from_explicit_fields(result)
    return _campaign_outcome_from_run_statuses(run_statuses)


def campaign_exit_code(result: dict[str, Any] | None) -> int:
    """Return camera-ready campaign CLI exit code for a result payload.

    Only canonical integer (non-bool) exit codes {0, 2, 3} are trusted
    from the payload.  Bool values and unknown integers fall through to
    the summarised outcome.
    """
    if isinstance(result, dict):
        explicit_exit_code = result.get("exit_code")
        if (
            isinstance(explicit_exit_code, int)
            and not isinstance(explicit_exit_code, bool)
            and explicit_exit_code in _CANONICAL_EXIT_CODES
        ):
            return explicit_exit_code
    return summarize_campaign_outcome(result).exit_code


def classify_planner_row_status(status: str) -> str:
    """Classify a planner-row status label into a canonical outcome class.

    Returns:
        One of ``"ok"``, ``"accepted_unavailable"``, or ``"unexpected_failure"``.
    """
    normalized = str(status or "").strip().lower()
    if normalized == "ok":
        return "ok"
    if normalized in _ACCEPTED_UNAVAILABLE_STATUSES:
        return "accepted_unavailable"
    return "unexpected_failure"


__all__ = [
    "BenchmarkAvailability",
    "CampaignOutcome",
    "CampaignRowStatusSummary",
    "CampaignStatusAxes",
    "availability_payload",
    "benchmark_run_exit_code",
    "campaign_exit_code",
    "campaign_status_axes_payload",
    "classify_planner_row_status",
    "resolve_execution_mode",
    "summarize_benchmark_availability",
    "summarize_campaign_outcome",
    "summarize_campaign_status_axes",
    "summarize_row_status_summary",
]

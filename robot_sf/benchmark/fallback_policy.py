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
        for row in planner_rows:
            if isinstance(row, dict):
                status = str(row.get("status", "")).strip().lower()
                if status:
                    run_statuses.append(status)
                else:
                    run_statuses.clear()
                    break
    if run_statuses:
        return run_statuses

    runs = result.get("runs")
    if isinstance(runs, list):
        for entry in runs:
            if isinstance(entry, dict):
                status = str(entry.get("status", "")).strip().lower()
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
    successful_runs = _int_field(result, "successful_runs")
    accepted_unavailable_runs = _int_field(result, "accepted_unavailable_runs")
    unexpected_failed_runs = _int_field(result, "unexpected_failed_runs")
    non_success_runs = _int_field(result, "non_success_runs")
    total_runs = _int_field(result, "total_runs")

    if total_runs <= 0:
        total_runs = successful_runs + accepted_unavailable_runs + unexpected_failed_runs
    if non_success_runs <= 0 and total_runs > 0:
        non_success_runs = max(
            total_runs - successful_runs,
            accepted_unavailable_runs + unexpected_failed_runs,
        )

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


def summarize_benchmark_availability(summary: dict[str, Any] | None) -> BenchmarkAvailability:
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
    "availability_payload",
    "benchmark_run_exit_code",
    "campaign_exit_code",
    "classify_planner_row_status",
    "resolve_execution_mode",
    "summarize_benchmark_availability",
    "summarize_campaign_outcome",
]

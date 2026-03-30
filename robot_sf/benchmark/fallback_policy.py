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


def campaign_exit_code(result: dict[str, Any] | None) -> int:
    """Return camera-ready campaign CLI exit code for a result payload."""
    if not isinstance(result, dict):
        return 2
    benchmark_success = result.get("benchmark_success")
    return 0 if benchmark_success is True else 2


__all__ = [
    "BenchmarkAvailability",
    "availability_payload",
    "benchmark_run_exit_code",
    "campaign_exit_code",
    "resolve_execution_mode",
    "summarize_benchmark_availability",
]

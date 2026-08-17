"""Reconcile repeated analysis-trace overhead receipts without averaging drift."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf.benchmark.analysis_trace import canonical_json

RECONCILIATION_SCHEMA_VERSION = "analysis_trace_overhead_reconciliation.v1"
MEASUREMENT_SCHEMA_VERSION = "analysis_trace_overhead_measurement_receipt.v2"
ISSUE = 6987
SOURCE_ISSUE = 6972
ENVIRONMENT_KEYS = ("platform", "python", "machine")
METHOD_COMPARISON_KEYS = (
    "warmups_per_arm_per_batch",
    "samples_per_arm_per_batch",
    "batch_count",
    "arm_order",
    "stability_tolerance_fraction",
    "timer",
    "serialization",
    "compression",
)
THREAD_SETTING_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _digest(value: Any) -> str:
    """Return a stable digest for a JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a receipt file's exact bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Return *value* as a mapping or raise a readable validation error."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _execution_context(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Extract explicit execution context, preserving missing legacy fields.

    Returns:
        Normalized execution context and an explicit completeness flag.
    """

    method = _mapping(receipt.get("method"), name="method")
    context_value = receipt.get("execution_context")
    context = context_value if isinstance(context_value, Mapping) else {}
    warmup_value = context.get("warmup_state")
    cache_value = context.get("cache_state")
    thread_value = context.get("numerical_thread_settings")
    warmup_state = (
        dict(warmup_value)
        if isinstance(warmup_value, Mapping)
        else {
            "status": "derived_from_method",
            "warmups_per_arm_per_batch": method.get("warmups_per_arm_per_batch"),
        }
    )
    cache_state = (
        dict(cache_value)
        if isinstance(cache_value, Mapping)
        else {"status": "unavailable", "reason": "missing_execution_context.cache_state"}
    )
    thread_settings = (
        dict(thread_value)
        if isinstance(thread_value, Mapping)
        else {"status": "unavailable", "reason": "missing_execution_context.thread_settings"}
    )
    has_explicit_context = isinstance(context_value, Mapping)
    context_complete = (
        has_explicit_context
        and isinstance(warmup_value, Mapping)
        and isinstance(cache_value, Mapping)
        and isinstance(thread_value, Mapping)
        and all(key in thread_value for key in THREAD_SETTING_KEYS)
    )
    return {
        "execution_order": context.get("execution_order", method.get("arm_order")),
        "warmup_state": warmup_state,
        "cache_state": cache_state,
        "numerical_thread_settings": thread_settings,
        "complete": context_complete,
    }


def _validate_receipt(receipt: Mapping[str, Any]) -> list[str]:
    """Return fail-closed validation errors for one measurement receipt."""

    errors: list[str] = []
    if receipt.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        errors.append("unsupported_measurement_schema")
    if receipt.get("issue") != ISSUE:
        errors.append("issue_mismatch")
    if receipt.get("source_issue") != SOURCE_ISSUE:
        errors.append("source_issue_mismatch")
    if receipt.get("status") != "diagnostic_only":
        errors.append("status_not_diagnostic_only")
    commit = receipt.get("repository_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        errors.append("repository_commit_missing_or_invalid")
    try:
        method = _mapping(receipt.get("method"), name="method")
        for key in METHOD_COMPARISON_KEYS:
            if key not in method:
                errors.append(f"method_missing:{key}")
        _mapping(receipt.get("environment"), name="environment")
        _mapping(receipt.get("fixture"), name="fixture")
        _mapping(receipt.get("checks"), name="checks")
        _mapping(receipt.get("derived"), name="derived")
        if not isinstance(receipt.get("batches"), list) or not receipt["batches"]:
            errors.append("batches_missing_or_empty")
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def _receipt_context_signature(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the context fields that must match before comparison is meaningful."""

    method = _mapping(receipt["method"], name="method")
    environment = _mapping(receipt["environment"], name="environment")
    context = _execution_context(receipt)
    return {
        "repository_commit": receipt.get("repository_commit"),
        "environment": {key: environment.get(key) for key in ENVIRONMENT_KEYS},
        "fixture_digest": _digest(receipt.get("fixture")),
        "method": {key: method.get(key) for key in METHOD_COMPARISON_KEYS},
        "execution_context": context,
    }


def _compatibility_reasons(
    first: Mapping[str, Any], other: Mapping[str, Any], *, index: int
) -> list[str]:
    """Describe context differences that prohibit cross-receipt comparison.

    Returns:
        Stable reason codes for the receipt pair.
    """

    first_signature = _receipt_context_signature(first)
    other_signature = _receipt_context_signature(other)
    reasons: list[str] = []
    for key in ("repository_commit", "environment", "fixture_digest", "method"):
        if first_signature[key] != other_signature[key]:
            reasons.append(f"receipt_{index}_{key}_mismatch")
    first_context = first_signature["execution_context"]
    other_context = other_signature["execution_context"]
    if first_context["complete"] != other_context["complete"]:
        reasons.append(f"receipt_{index}_execution_context_presence_mismatch")
    elif first_context["complete"] and first_context != other_context:
        reasons.append(f"receipt_{index}_execution_context_mismatch")
    return reasons


def _receipt_summary(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact per-receipt summary without cross-receipt aggregation.

    Returns:
        A JSON-serializable summary containing only per-receipt values.
    """

    derived = _mapping(receipt.get("derived"), name="derived")
    checks = _mapping(receipt.get("checks"), name="checks")
    environment = _mapping(receipt.get("environment"), name="environment")
    context = _execution_context(receipt)
    integrity_keys = (
        "paired_outcomes_and_metrics_equal",
        "control_sequence_digest_stable",
        "trace_git_hash_matches_commit",
        "trace_artifact_matches_provenance",
    )
    integrity_passed = all(checks.get(key) is True for key in integrity_keys)
    return {
        "path": str(path),
        "receipt_sha256": _file_digest(path),
        "repository_commit": receipt.get("repository_commit"),
        "environment": {key: environment.get(key) for key in ENVIRONMENT_KEYS},
        "fixture_digest": _digest(receipt.get("fixture")),
        "execution_context": context,
        "batch_overhead_fractions": list(derived.get("batch_overhead_fractions", [])),
        "median_overhead_fraction": derived.get("median_overhead_fraction"),
        "target_decision": derived.get("target_decision", "unavailable"),
        "target_met": derived.get("target_met"),
        "stability_status": derived.get("stability_status", "unavailable"),
        "stability_passed": derived.get("stability_passed") is True,
        "integrity_passed": integrity_passed,
        "integrity_checks": {key: checks.get(key) for key in integrity_keys},
    }


def _error_summary(path: Path, error: str) -> dict[str, Any]:
    """Build a stable summary for an unreadable or invalid receipt.

    Returns:
        A JSON-serializable path/error record.
    """

    return {"path": str(path), "error": error}


def _load_receipts(
    paths: Sequence[Path],
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate receipt paths.

    Returns:
        Valid receipts, compact summaries, and validation/read errors.
    """

    summaries: list[dict[str, Any]] = []
    receipts: list[Mapping[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in paths:
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            receipt_mapping = _mapping(receipt, name="receipt")
            validation_errors = _validate_receipt(receipt_mapping)
            if validation_errors:
                error = ";".join(validation_errors)
                errors.append({"path": str(path), "error": error})
                summaries.append(_error_summary(path, error))
            else:
                receipts.append(receipt_mapping)
                summaries.append(_receipt_summary(path, receipt_mapping))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            summaries.append(_error_summary(path, str(exc)))
    return receipts, summaries, errors


def _compatibility_summary(
    paths: Sequence[Path],
    receipts: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, list[str]]:
    """Return compatibility, context completeness, and reason codes.

    Returns:
        A tuple of ``(compatible, context_complete, reasons)``.
    """

    reasons: list[str] = []
    if len(paths) < 2:
        reasons.append("at_least_two_receipts_required")
    if errors:
        reasons.append("receipt_validation_failed")
    if len(receipts) == len(paths) and receipts:
        for index, receipt in enumerate(receipts[1:], start=1):
            reasons.extend(_compatibility_reasons(receipts[0], receipt, index=index))
    context_complete = bool(receipts) and all(
        _execution_context(receipt)["complete"] for receipt in receipts
    )
    if not context_complete and receipts:
        reasons.append("missing_execution_context")
    return not reasons and context_complete, context_complete, sorted(set(reasons))


def _classify(
    *, compatible: bool, summaries: Sequence[Mapping[str, Any]], target_decisions: Sequence[str]
) -> tuple[str, str]:
    """Classify compatible per-receipt decisions without aggregating timings.

    Returns:
        A ``(classification, target_decision)`` tuple.
    """

    if not compatible:
        return "unavailable", "unavailable"
    has_unstable_receipt = any(
        summary["target_decision"] == "inconclusive"
        or not summary["stability_passed"]
        or not summary["integrity_passed"]
        for summary in summaries
    )
    if has_unstable_receipt or len(set(target_decisions)) != 1:
        return "measurement_unstable", "inconclusive"
    return "measurement_stable", target_decisions[0]


def reconcile_receipts(receipt_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Build a fail-closed diagnostic reconciliation packet.

    Receipts are compared only when their source commit, environment, fixture,
    method, and explicit execution context match. The packet intentionally keeps
    every overhead value per receipt; it never computes a cross-receipt average.

    Returns:
        A diagnostic-only reconciliation packet with explicit compatibility and
        classification fields.
    """

    paths = [Path(path) for path in receipt_paths]
    receipts, summaries, errors = _load_receipts(paths)
    compatible, context_complete, compatibility_reasons = _compatibility_summary(
        paths, receipts, errors
    )

    target_decisions = [
        summary.get("target_decision", "unavailable")
        for summary in summaries
        if "target_decision" in summary
    ]
    classification, decision = _classify(
        compatible=compatible,
        summaries=[summary for summary in summaries if "target_decision" in summary],
        target_decisions=target_decisions,
    )

    return {
        "schema_version": RECONCILIATION_SCHEMA_VERSION,
        "issue": ISSUE,
        "source_issue": SOURCE_ISSUE,
        "status": "diagnostic_only",
        "claim_boundary": (
            "Environment-bound local timing reconciliation only; this packet is not benchmark, "
            "paper-facing, real-world, safety, release, campaign, or optimization evidence."
        ),
        "receipts": summaries,
        "receipt_errors": errors,
        "compatibility": {
            "compatible": compatible,
            "reasons": sorted(set(compatibility_reasons)),
            "cross_receipt_aggregation": "forbidden",
        },
        "reconciliation": {
            "classification": classification,
            "target_decision": decision,
            "target_decisions": target_decisions,
            "context_complete": context_complete,
            "reason": (
                "No cross-receipt average is computed; incompatible or incomplete context is "
                "reported as unavailable."
            ),
        },
    }


__all__ = ["RECONCILIATION_SCHEMA_VERSION", "reconcile_receipts"]

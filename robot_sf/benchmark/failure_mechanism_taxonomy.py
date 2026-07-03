"""Failure-mechanism taxonomy helpers for episode-row evidence sidecars."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

MECHANISM_SCHEMA_VERSION = "failure_mechanism_taxonomy.v1"

MECHANISM_LABELS = {
    "static_deadlock_or_local_minimum",
    "route_or_topology_mismatch",
    "dynamic_phase_or_order_sensitivity",
    "proxemic_or_clearance_tradeoff",
    "guard_or_handoff_domination",
    "learned_policy_low_progress",
    "actuation_or_command_saturation",
    "seed_local_stochastic_fragility",
    "scenario_contract_blocker",
    "time_budget_artifact",
    "unknown",
}

MECHANISM_CONFIDENCES = {
    "observed_mechanism",
    "supported_hypothesis",
    "weak_hypothesis",
    "unknown",
}

TRACE_VERIFIED_EVIDENCE_MODES = {
    "paired_trace",
    "deterministic_replay",
    "direct_probe",
    "root_cause",
}

MECHANISM_EVIDENCE_MODES = TRACE_VERIFIED_EVIDENCE_MODES | {
    "aggregate_summary",
    "unknown",
}

GEOMETRY_ONLY_FIELDS = {
    "geometry_bucket",
    "scenario_geometry_bucket",
    "geometry_label",
    "scenario_family",
}

REQUIRED_MECHANISM_FIELDS = (
    "mechanism_schema_version",
    "mechanism_label",
    "mechanism_confidence",
    "mechanism_evidence_mode",
    "mechanism_evidence_uri",
    "mechanism_case_id",
    "mechanism_caveat",
)


class FailureMechanismTaxonomyError(ValueError):
    """Raised when a failure-mechanism taxonomy record violates the contract."""


def reject_geometry_only_mechanism(record: Mapping[str, Any]) -> None:
    """Reject records that try to use geometry buckets as mechanism evidence."""

    geometry_values = {field: str(record.get(field, "")).strip() for field in GEOMETRY_ONLY_FIELDS}
    mechanism_label = str(record.get("mechanism_label", "")).strip()
    for field, value in geometry_values.items():
        if value and value == mechanism_label:
            raise FailureMechanismTaxonomyError(
                f"{field} cannot substitute for trace-verified mechanism_label"
            )


def validate_failure_mechanism_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a failure-mechanism taxonomy record.

    Returns:
        A normalized copy containing the required taxonomy fields.
    """

    normalized = {field: str(record.get(field, "")).strip() for field in REQUIRED_MECHANISM_FIELDS}
    if normalized["mechanism_schema_version"] != MECHANISM_SCHEMA_VERSION:
        raise FailureMechanismTaxonomyError(
            "mechanism_schema_version must be failure_mechanism_taxonomy.v1"
        )
    if normalized["mechanism_label"] not in MECHANISM_LABELS:
        raise FailureMechanismTaxonomyError(
            f"unsupported mechanism_label: {normalized['mechanism_label']!r}"
        )
    if normalized["mechanism_confidence"] not in MECHANISM_CONFIDENCES:
        raise FailureMechanismTaxonomyError(
            f"unsupported mechanism_confidence: {normalized['mechanism_confidence']!r}"
        )
    if normalized["mechanism_evidence_mode"] not in MECHANISM_EVIDENCE_MODES:
        raise FailureMechanismTaxonomyError(
            f"unsupported mechanism_evidence_mode: {normalized['mechanism_evidence_mode']!r}"
        )
    if (
        normalized["mechanism_confidence"] == "observed_mechanism"
        and normalized["mechanism_evidence_mode"] not in TRACE_VERIFIED_EVIDENCE_MODES
    ):
        raise FailureMechanismTaxonomyError(
            "observed_mechanism requires paired_trace, deterministic_replay, "
            "direct_probe, or root_cause evidence"
        )
    if (
        normalized["mechanism_confidence"] != "observed_mechanism"
        and not normalized["mechanism_caveat"]
    ):
        raise FailureMechanismTaxonomyError(
            "mechanism_caveat is required unless confidence is observed_mechanism"
        )
    reject_geometry_only_mechanism({**record, **normalized})
    return normalized


def unknown_failure_mechanism_record(reason: str) -> dict[str, Any]:
    """Return a valid unknown mechanism record with an explicit caveat.

    Returns:
        A taxonomy record whose label, confidence, and evidence mode are unknown.
    """

    caveat = reason.strip() or "not_derivable"
    return {
        "mechanism_schema_version": MECHANISM_SCHEMA_VERSION,
        "mechanism_label": "unknown",
        "mechanism_confidence": "unknown",
        "mechanism_evidence_mode": "unknown",
        "mechanism_evidence_uri": "",
        "mechanism_case_id": "",
        "mechanism_caveat": caveat,
    }

"""Simulator-dependence validity-boundary decision checks for issue #3207.

The checker consumes already-promoted fidelity-sensitivity evidence and decides
whether the repository may make a simulator-dependence validity-boundary claim.
It does not run simulations and it fails closed: missing inputs, diagnostic-only
scope, non-identifiable ranks, rank flips, or missing no-claim boundaries all
produce a no-claim decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SIMULATOR_DEPENDENCE_DECISION_SCHEMA = "simulator_dependence_validity_boundary_decision.v1"

DECISION_SUPPORTED = "claim_ready"
DECISION_NO_CLAIM = "no_claim"
DECISION_BLOCKED = "blocked_missing_inputs"

FULL_SCOPE_CLASSIFICATIONS = frozenset({"full_fixed_scope", "fixed_scope_study"})
REQUIRED_CLAIM_BOUNDARY_PHRASES = (
    "not benchmark evidence",
    "not simulator-realism evidence",
    "not sim-to-real evidence",
    "not paper-facing evidence",
)


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from ``path``.

    Returns:
        Parsed JSON object as a mutable mapping.

    Raises:
        ValueError: If the JSON payload is not an object.
    """

    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {json_path}")
    return payload


def build_simulator_dependence_decision(
    study_summary: Mapping[str, Any] | None,
    *,
    manifest_check: Mapping[str, Any] | None = None,
    expected_axes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed validity-boundary decision packet.

    Args:
        study_summary: Promoted fidelity-sensitivity study summary.
        manifest_check: Optional dry-run manifest checker summary.
        expected_axes: Optional axis names required for a complete study packet.

    Returns:
        JSON-safe decision packet. ``decision`` is ``claim_ready`` only when all
        required evidence and rank-identifiability checks pass.
    """

    missing_inputs: list[str] = []
    no_claim_reasons: list[str] = []
    boundary_violations: list[str] = []
    supporting_inputs: dict[str, Any] = {}

    if not isinstance(study_summary, Mapping):
        missing_inputs.append("study_summary")
        study_summary = {}

    rank_stability = _mapping_or_empty(study_summary.get("rank_stability"))
    scope = _mapping_or_empty(study_summary.get("scope"))

    supporting_inputs["study_status"] = study_summary.get("status")
    supporting_inputs["scope_classification"] = scope.get("classification")
    supporting_inputs["rank_identifiable"] = rank_stability.get("rank_identifiable")
    supporting_inputs["rank_stable"] = rank_stability.get("rank_stable")
    supporting_inputs["rank_identifiability_reason"] = rank_stability.get(
        "rank_identifiability_reason"
    )

    for key in ("status", "scope", "rank_stability", "claim_boundary"):
        if key not in study_summary:
            missing_inputs.append(f"study_summary.{key}")

    scope_classification = scope.get("classification")
    if scope_classification not in FULL_SCOPE_CLASSIFICATIONS:
        no_claim_reasons.append(
            "study_scope_not_full_fixed_scope"
            if scope_classification
            else "study_scope_missing_classification"
        )

    _append_rank_reasons(rank_stability, no_claim_reasons)
    _append_axis_coverage_reasons(rank_stability, expected_axes, missing_inputs, no_claim_reasons)
    _append_manifest_reasons(manifest_check, supporting_inputs, missing_inputs, no_claim_reasons)
    _append_boundary_reasons(study_summary, boundary_violations)

    if missing_inputs:
        decision = DECISION_BLOCKED
    elif no_claim_reasons or boundary_violations:
        decision = DECISION_NO_CLAIM
    else:
        decision = DECISION_SUPPORTED

    return {
        "schema_version": SIMULATOR_DEPENDENCE_DECISION_SCHEMA,
        "issue": 3207,
        "decision": decision,
        "claim_ready": decision == DECISION_SUPPORTED,
        "evidence_status": (
            "validity_boundary_supported"
            if decision == DECISION_SUPPORTED
            else "not_benchmark_evidence"
        ),
        "claim_boundary": _decision_claim_boundary(decision),
        "missing_inputs": missing_inputs,
        "no_claim_reasons": no_claim_reasons,
        "boundary_violations": boundary_violations,
        "supporting_inputs": supporting_inputs,
    }


def write_simulator_dependence_decision(packet: Mapping[str, Any], path: str | Path) -> Path:
    """Write a deterministic simulator-dependence decision packet.

    Returns:
        Path written JSON packet.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _mapping_or_empty(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _append_rank_reasons(rank_stability: Mapping[str, Any], no_claim_reasons: list[str]) -> None:
    if rank_stability.get("rank_identifiable") is not True:
        reason = rank_stability.get("rank_identifiability_reason")
        no_claim_reasons.append(
            f"rank_non_identifiable:{reason}" if reason else "rank_non_identifiable"
        )
    if rank_stability.get("rank_stable") is not True:
        no_claim_reasons.append("rank_not_stably_supported")
    if rank_stability.get("flipping_axes"):
        no_claim_reasons.append("rank_flipping_axes_present")
    if rank_stability.get("non_identifiable_axes"):
        no_claim_reasons.append("non_identifiable_axes_present")


def _append_axis_coverage_reasons(
    rank_stability: Mapping[str, Any],
    expected_axes: Sequence[str] | None,
    missing_inputs: list[str],
    no_claim_reasons: list[str],
) -> None:
    if not expected_axes:
        return
    axes = rank_stability.get("axes")
    if not isinstance(axes, list):
        missing_inputs.append("study_summary.rank_stability.axes")
        return
    observed = {str(axis.get("axis")) for axis in axes if isinstance(axis, Mapping)}
    missing_axes = sorted(str(axis) for axis in expected_axes if str(axis) not in observed)
    if missing_axes:
        no_claim_reasons.append("missing_expected_axes:" + ",".join(missing_axes))


def _append_manifest_reasons(
    manifest_check: Mapping[str, Any] | None,
    supporting_inputs: dict[str, Any],
    missing_inputs: list[str],
    no_claim_reasons: list[str],
) -> None:
    if manifest_check is None:
        return
    if not isinstance(manifest_check, Mapping):
        missing_inputs.append("manifest_check")
        return
    supporting_inputs["manifest_check_passes"] = manifest_check.get("passes")
    supporting_inputs["manifest_axis_count"] = manifest_check.get("axis_count")
    if manifest_check.get("passes") is not True:
        no_claim_reasons.append("manifest_check_not_passing")


def _append_boundary_reasons(
    study_summary: Mapping[str, Any], boundary_violations: list[str]
) -> None:
    boundary = str(study_summary.get("claim_boundary", ""))
    for phrase in REQUIRED_CLAIM_BOUNDARY_PHRASES:
        if phrase not in boundary:
            boundary_violations.append(f"claim_boundary_missing:{phrase}")


def _decision_claim_boundary(decision: str) -> str:
    if decision == DECISION_SUPPORTED:
        return (
            "Simulator-dependence validity-boundary claim is checker-supported for the supplied "
            "fixed-scope study packet only. This does not establish simulator realism, "
            "sim-to-real transfer, or paper-facing sufficiency without separate review."
        )
    if decision == DECISION_BLOCKED:
        return (
            "No simulator-dependence validity-boundary claim: required study inputs are missing. "
            "Treat this as blocked pre-study validation, not benchmark evidence, not "
            "simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence."
        )
    return (
        "No simulator-dependence validity-boundary claim: supplied evidence is diagnostic, "
        "incomplete, rank-non-identifiable, rank-unstable, or boundary-incomplete. Treat as "
        "not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, "
        "and not paper-facing evidence."
    )

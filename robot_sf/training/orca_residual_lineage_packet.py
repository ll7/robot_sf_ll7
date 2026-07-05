"""Validation helpers for ORCA-residual behavior-cloning lineage packets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_VERSION = "orca-residual-bc-lineage-packet.v1"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DURABLE_URI_PREFIXES = ("wandb-artifact://", "artifact://", "s3://", "gs://", "https://")
_REQUIRED_OBSERVATION_KEYS = (
    "robot_state",
    "goal_state",
    "pedestrian_state",
    "occupancy_grid",
)
_FORBIDDEN_FEATURE_CLASSES = (
    "scenario_future",
    "benchmark_label",
    "oracle_trajectory_future",
    "privileged_map_solution",
)
_REQUIRED_DIAGNOSTICS = (
    "orca_command",
    "raw_residual",
    "bounded_residual",
    "final_guarded_command",
    "residual_clipping_rate",
    "guard_veto_rate",
    "fallback_degraded_status",
)
# Public alias: the canonical ORCA-residual diagnostics contract. The
# pre-submit conformance check (scripts/validation/preflight_evidence_contract.py)
# imports this so the required-field list has a single source of truth.
REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS = _REQUIRED_DIAGNOSTICS

# The four fields the smoke-evidence stage must emit non-null
# (scripts/validation/run_policy_search_candidate.py::_attach_orca_residual_smoke_evidence).
# Three derive directly from _REQUIRED_DIAGNOSTICS; ``artifact_pointer_status`` is the
# smoke-stage pointer field. Kept here (the contract owner) so neither the smoke builder
# nor the preflight check redefines the list.
_SMOKE_DERIVED_DIAGNOSTIC_FIELDS = (
    "residual_clipping_rate",
    "guard_veto_rate",
    "fallback_degraded_status",
)
assert set(_SMOKE_DERIVED_DIAGNOSTIC_FIELDS).issubset(_REQUIRED_DIAGNOSTICS), (
    "smoke-evidence required fields must be a subset of the diagnostics contract"
)
REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS = (
    *_SMOKE_DERIVED_DIAGNOSTIC_FIELDS,
    "artifact_pointer_status",
)
_SMOKE_NOMINAL_REQUIRED_SUMMARY_FIELDS = REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS + (
    "success_rate",
    "collision_rate",
    "nominal_escalation_allowed",
)
_SMOKE_ARTIFACT_POINTER_OK = {"durable", "complete"}
_SMOKE_FALLBACK_DEGRADED_OK = {"clear"}
_SMOKE_GATE_MIN_SUCCESS_RATE = 0.80
_SMOKE_GATE_MAX_COLLISION_RATE = 0.02

_REQUIRED_COMPARISONS = ("orca", "ppo_leader", "orca_residual_guarded_ppo_v0")
_REQUIRED_OUTPUTS = (
    "residual_dataset_manifest",
    "candidate_yaml",
    "checkpoint_pointer",
    "diagnostic_report_path",
    "residual_contribution_diagnostics",
    "guard_veto_fallback_status",
)
_ALLOWED_OBJECTIVE_TARGETS = (
    "bounded_policy_action_minus_orca_action",
    "progress_probe_bounded_policy_action_minus_orca_action",
)


class OrcaResidualLineagePacketError(ValueError):
    """Raised when an ORCA-residual lineage packet fails validation."""


def load_launch_packet(config_path: Path) -> dict[str, Any]:
    """Load an ORCA-residual lineage packet YAML file.

    Returns:
        Parsed packet mapping.
    """
    if not config_path.is_file():
        raise OrcaResidualLineagePacketError(f"lineage packet is not a file: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise OrcaResidualLineagePacketError("lineage packet must be a YAML mapping")
    return payload


def validate_launch_packet(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a behavior-cloning lineage packet for the first ORCA-residual policy.

    Args:
        config_path: Packet YAML path.
        repo_root: Repository root for resolving relative paths.

    Returns:
        Compact validation report.

    Raises:
        OrcaResidualLineagePacketError: If any fail-closed invariant is violated.
    """
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, root)
    packet = load_launch_packet(config_path)
    errors: list[str] = []

    if packet.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION!r}")
    _require_non_empty_string(packet, "campaign_id", errors)
    _validate_generating_commit(packet, errors)
    _require_existing_path(packet, "slurm_handoff", root, errors)

    objective = _validate_objective(packet, errors)
    observation = _validate_observation_contract(packet, errors)
    residual = _validate_residual_bounds(packet, root, errors)
    artifacts = _validate_artifact_lineage(packet, root, errors)
    diagnostics = _validate_diagnostics(packet, errors)
    comparisons = _validate_comparison_references(packet, root, errors)
    outputs = _validate_expected_outputs(packet, root, errors)
    _validate_execution_boundary(packet, errors)

    if errors:
        joined = "\n- ".join(errors)
        raise OrcaResidualLineagePacketError(
            f"ORCA-residual lineage packet failed validation:\n- {joined}"
        )

    return {
        "status": "valid",
        "schema_version": packet["schema_version"],
        "campaign_id": packet["campaign_id"],
        "objective": objective,
        "observation_contract": observation,
        "residual_bounds": residual,
        "artifact_lineage": artifacts,
        "diagnostics": diagnostics,
        "comparison_references": comparisons,
        "expected_outputs": outputs,
    }


def validate_smoke_nominal_gate(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate issue #1475 smoke evidence before nominal escalation.

    This CPU-only gate checks completed smoke summaries. It fails closed for
    fallback/degraded execution, missing telemetry, missing durable artifacts,
    and smoke metrics that do not satisfy the nominal-escalation thresholds.

    Returns:
        Compact validation report for the issue #1475 smoke-to-nominal gate.
    """
    if not isinstance(summary, dict):
        raise OrcaResidualLineagePacketError("smoke summary must be a mapping")

    errors: list[str] = []
    missing = [
        field
        for field in _SMOKE_NOMINAL_REQUIRED_SUMMARY_FIELDS
        if summary.get(field) in (None, "")
    ]
    if missing:
        errors.append(f"smoke summary missing required entries: {missing}")

    success_rate = _coerce_rate(summary.get("success_rate"), "success_rate", errors)
    collision_rate = _coerce_rate(summary.get("collision_rate"), "collision_rate", errors)
    residual_clipping_rate = _coerce_rate(
        summary.get("residual_clipping_rate"), "residual_clipping_rate", errors
    )
    guard_veto_rate = _coerce_rate(summary.get("guard_veto_rate"), "guard_veto_rate", errors)

    fallback_status = summary.get("fallback_degraded_status")
    if fallback_status not in _SMOKE_FALLBACK_DEGRADED_OK:
        errors.append(
            "fallback_degraded_status must be 'clear' before nominal escalation "
            f"(got {fallback_status!r})"
        )

    artifact_status = summary.get("artifact_pointer_status")
    if artifact_status not in _SMOKE_ARTIFACT_POINTER_OK:
        errors.append(
            "artifact_pointer_status must be durable/complete before nominal escalation "
            f"(got {artifact_status!r})"
        )

    if summary.get("nominal_escalation_allowed") is not True:
        errors.append("nominal_escalation_allowed must be true")

    if success_rate is not None and success_rate < _SMOKE_GATE_MIN_SUCCESS_RATE:
        errors.append(
            "success_rate below nominal smoke gate: "
            f"{success_rate:.4f} < {_SMOKE_GATE_MIN_SUCCESS_RATE:.4f}"
        )
    if collision_rate is not None and collision_rate > _SMOKE_GATE_MAX_COLLISION_RATE:
        errors.append(
            "collision_rate above nominal smoke gate: "
            f"{collision_rate:.4f} > {_SMOKE_GATE_MAX_COLLISION_RATE:.4f}"
        )

    if errors:
        joined = "\n- ".join(errors)
        raise OrcaResidualLineagePacketError(
            f"ORCA-residual smoke nominal gate failed:\n- {joined}"
        )

    return {
        "status": "valid",
        "gate": "issue_1475_smoke_to_nominal",
        "min_success_rate": _SMOKE_GATE_MIN_SUCCESS_RATE,
        "max_collision_rate": _SMOKE_GATE_MAX_COLLISION_RATE,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "residual_clipping_rate": residual_clipping_rate,
        "guard_veto_rate": guard_veto_rate,
        "fallback_degraded_status": fallback_status,
        "artifact_pointer_status": artifact_status,
        "nominal_escalation_allowed": True,
    }


def _coerce_rate(value: Any, label: str, errors: list[str]) -> float | None:
    """Return a finite rate in [0, 1], appending an error when invalid."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} must be numeric rate")
        return None
    if not 0.0 <= rate <= 1.0:
        errors.append(f"{label} must be between 0 and 1")
        return None
    return rate


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _require_non_empty_string(mapping: dict[str, Any], key: str, errors: list[str]) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")


def _require_existing_path(
    mapping: dict[str, Any],
    key: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty path string")
        return
    if not _resolve_path(value, repo_root).exists():
        errors.append(f"{key} does not exist: {value}")


def _validate_generating_commit(packet: dict[str, Any], errors: list[str]) -> None:
    commit = packet.get("generating_commit")
    if not isinstance(commit, str) or not _GIT_SHA_RE.match(commit.strip()):
        errors.append("generating_commit must be a 40-character git SHA")


def _validate_objective(packet: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    objective = packet.get("objective")
    if not isinstance(objective, dict):
        errors.append("objective must be a mapping")
        return {}
    if objective.get("method") != "behavior_cloning_residual":
        errors.append("objective.method must be 'behavior_cloning_residual'")
    if objective.get("target") not in _ALLOWED_OBJECTIVE_TARGETS:
        errors.append(
            "objective.target must be one of "
            f"{', '.join(repr(target) for target in _ALLOWED_OBJECTIVE_TARGETS)}"
        )
    if objective.get("teacher_policy") != "ppo_leader":
        errors.append("objective.teacher_policy must be 'ppo_leader'")
    if objective.get("base_policy") != "orca":
        errors.append("objective.base_policy must be 'orca'")
    if objective.get("full_training_in_packet") is not False:
        errors.append("objective.full_training_in_packet must be false")
    return {
        "method": objective.get("method"),
        "target": objective.get("target"),
        "teacher_policy": objective.get("teacher_policy"),
        "base_policy": objective.get("base_policy"),
        "revision_id": objective.get("revision_id"),
    }


def _validate_observation_contract(
    packet: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    contract = packet.get("observation_contract")
    if not isinstance(contract, dict):
        errors.append("observation_contract must be a mapping")
        return {}
    if contract.get("source") != "runtime_socnav_struct":
        errors.append("observation_contract.source must be 'runtime_socnav_struct'")
    if contract.get("runtime_available_only") is not True:
        errors.append("observation_contract.runtime_available_only must be true")
    if contract.get("privileged_features_allowed") is not False:
        errors.append("observation_contract.privileged_features_allowed must be false")
    _validate_required_list(contract, "required_keys", _REQUIRED_OBSERVATION_KEYS, errors)
    _validate_required_list(
        contract,
        "forbidden_feature_classes",
        _FORBIDDEN_FEATURE_CLASSES,
        errors,
    )
    return {
        "source": contract.get("source"),
        "required_keys": list(contract.get("required_keys", [])),
        "forbidden_feature_classes": list(contract.get("forbidden_feature_classes", [])),
    }


def _validate_residual_bounds(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    bounds = packet.get("residual_bounds")
    if not isinstance(bounds, dict):
        errors.append("residual_bounds must be a mapping")
        return {}
    for key in ("linear_delta", "angular_delta"):
        value = bounds.get(key)
        if not isinstance(value, int | float) or float(value) <= 0.0:
            errors.append(f"residual_bounds.{key} must be a positive number")
    if bounds.get("applied_before_guard") is not True:
        errors.append("residual_bounds.applied_before_guard must be true")
    if bounds.get("hard_guard_authoritative") is not True:
        errors.append("residual_bounds.hard_guard_authoritative must be true")

    candidate_path_text = bounds.get("candidate_config")
    if not isinstance(candidate_path_text, str) or not candidate_path_text.strip():
        errors.append("residual_bounds.candidate_config must be a path string")
        return {}
    candidate_path = _resolve_path(candidate_path_text, repo_root)
    if not candidate_path.exists():
        errors.append(f"residual_bounds.candidate_config does not exist: {candidate_path_text}")
        return {}
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    params = candidate.get("params") if isinstance(candidate, dict) else None
    if not isinstance(params, dict):
        errors.append("residual_bounds.candidate_config must contain params")
        return {}
    if params.get("prior_policy") != "orca" or params.get("prior_residual_mode") is not True:
        errors.append("candidate config must use ORCA prior_residual_mode")
    _compare_bound(
        bounds,
        params,
        "linear_delta",
        "prior_residual_max_linear_delta",
        errors,
    )
    _compare_bound(
        bounds,
        params,
        "angular_delta",
        "prior_residual_max_angular_delta",
        errors,
    )
    return {
        "linear_delta": bounds.get("linear_delta"),
        "angular_delta": bounds.get("angular_delta"),
        "candidate_config": candidate_path_text,
    }


def _compare_bound(
    bounds: dict[str, Any],
    params: dict[str, Any],
    packet_key: str,
    candidate_key: str,
    errors: list[str],
) -> None:
    try:
        packet_value = float(bounds[packet_key])
        candidate_value = float(params[candidate_key])
    except (KeyError, TypeError, ValueError):
        errors.append(f"could not compare {packet_key} against candidate {candidate_key}")
        return
    if abs(packet_value - candidate_value) > 1e-9:
        errors.append(f"residual_bounds.{packet_key} must match candidate {candidate_key}")


def _validate_artifact_lineage(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    lineage = packet.get("artifact_lineage")
    if not isinstance(lineage, dict):
        errors.append("artifact_lineage must be a mapping")
        return {}
    if lineage.get("missing_artifact_behavior") != "fail_closed":
        errors.append("artifact_lineage.missing_artifact_behavior must be 'fail_closed'")
    _validate_artifact_list(lineage, "input_artifacts", repo_root, errors)
    _validate_artifact_list(lineage, "planned_output_artifacts", repo_root, errors)
    return {
        "missing_artifact_behavior": lineage.get("missing_artifact_behavior"),
        "input_artifacts": len(lineage.get("input_artifacts", [])),
        "planned_output_artifacts": len(lineage.get("planned_output_artifacts", [])),
    }


def _validate_diagnostics(packet: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    diagnostics = packet.get("diagnostics")
    if not isinstance(diagnostics, dict):
        errors.append("diagnostics must be a mapping")
        return {}
    _validate_required_list(diagnostics, "required_fields", _REQUIRED_DIAGNOSTICS, errors)
    if diagnostics.get("aggregate_rates_required") is not True:
        errors.append("diagnostics.aggregate_rates_required must be true")
    if diagnostics.get("fallback_degraded_rows_count_as_success") is not False:
        errors.append("diagnostics.fallback_degraded_rows_count_as_success must be false")
    return {"required_fields": list(diagnostics.get("required_fields", []))}


def _validate_comparison_references(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    references = packet.get("comparison_references")
    if not isinstance(references, dict):
        errors.append("comparison_references must be a mapping")
        return {}
    missing = sorted(set(_REQUIRED_COMPARISONS) - set(references))
    if missing:
        errors.append(f"comparison_references missing required references: {missing}")
    for name, reference in references.items():
        if not isinstance(reference, dict):
            errors.append(f"comparison_references.{name} must be a mapping")
            continue
        _require_non_empty_string(reference, "candidate_id", errors)
        _require_existing_path(reference, "source", repo_root, errors)
    return {"reference_ids": sorted(references)}


def _validate_expected_outputs(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    outputs = packet.get("expected_outputs")
    if not isinstance(outputs, dict):
        errors.append("expected_outputs must be a mapping")
        return {}
    missing = sorted(set(_REQUIRED_OUTPUTS) - set(outputs))
    if missing:
        errors.append(f"expected_outputs missing required entries: {missing}")
    for key, value in outputs.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"expected_outputs.{key} must be a non-empty string")
            continue
        _validate_artifact_pointer(value.strip(), f"expected_outputs.{key}", repo_root, errors)
    return {"entries": sorted(outputs)}


def _validate_execution_boundary(packet: dict[str, Any], errors: list[str]) -> None:
    execution = packet.get("execution_boundary")
    if not isinstance(execution, dict):
        errors.append("execution_boundary must be a mapping")
        return
    if execution.get("submit_slurm_from_this_issue") is not False:
        errors.append("execution_boundary.submit_slurm_from_this_issue must be false")
    if execution.get("bounded_slurm_followup_required") is not True:
        errors.append("execution_boundary.bounded_slurm_followup_required must be true")
    if execution.get("nominal_gate_before_stress") is not True:
        errors.append("execution_boundary.nominal_gate_before_stress must be true")
    _require_non_empty_string(execution, "local_preflight_command", errors)
    _require_non_empty_string(execution, "dataset_command_shape", errors)
    _require_non_empty_string(execution, "bc_training_command_shape", errors)
    _require_non_empty_string(execution, "smoke_candidate_command_shape", errors)
    _require_non_empty_string(execution, "slurm_command_shape", errors)


def _validate_required_list(
    mapping: dict[str, Any],
    key: str,
    required_values: tuple[str, ...],
    errors: list[str],
) -> None:
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        errors.append(f"{key} must be a non-empty list")
        return
    normalized = {str(value).strip() for value in values if str(value).strip()}
    missing = sorted(set(required_values) - normalized)
    if missing:
        errors.append(f"{key} is missing required values: {missing}")


def _validate_artifact_list(
    mapping: dict[str, Any],
    key: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    artifacts = mapping.get(key)
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{key} must be a non-empty list")
        return
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            errors.append(f"{key}[{index}] must be a mapping")
            continue
        _require_non_empty_string(artifact, "role", errors)
        pointer = artifact.get("uri") or artifact.get("path")
        if not isinstance(pointer, str) or not pointer.strip():
            errors.append(f"{key}[{index}] must define uri or path")
            continue
        _validate_artifact_pointer(pointer.strip(), f"{key}[{index}]", repo_root, errors)
        checksum = artifact.get("sha256")
        if checksum is not None and not _sha256_matches(pointer.strip(), str(checksum), repo_root):
            errors.append(f"{key}[{index}] sha256 mismatch for {pointer}")


def _validate_artifact_pointer(
    pointer: str,
    label: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    if pointer.startswith(_DURABLE_URI_PREFIXES):
        return
    resolved = _resolve_path(pointer, repo_root)
    try:
        resolved.relative_to(repo_root / "output")
    except ValueError:
        pass
    else:
        errors.append(f"{label} must not depend on worktree-local output: {pointer}")
        return
    if not resolved.is_file():
        errors.append(f"{label} path does not exist or is a directory: {pointer}")


def _sha256_matches(pointer: str, expected: str, repo_root: Path) -> bool:
    if pointer.startswith(_DURABLE_URI_PREFIXES):
        return True
    path = _resolve_path(pointer, repo_root)
    if not path.is_file():
        return False
    digest_builder = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    return digest == expected


__all__ = [
    "REQUIRED_ORCA_RESIDUAL_DIAGNOSTICS",
    "REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS",
    "OrcaResidualLineagePacketError",
    "load_launch_packet",
    "validate_launch_packet",
]

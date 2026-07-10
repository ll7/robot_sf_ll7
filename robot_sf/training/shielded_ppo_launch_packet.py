"""Validation helpers for shielded-PPO repair launch packets."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from robot_sf.errors import RobotSfError

_SCHEMA_VERSION = "shielded-ppo-repair-launch-packet.v1"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DURABLE_URI_PREFIXES = ("wandb-artifact://", "artifact://", "s3://", "gs://", "https://")
_REQUIRED_DIAGNOSTICS = (
    "guard_veto_count",
    "guard_fallback_count",
    "raw_ppo_action",
    "guarded_action",
    "fallback_action_source",
)
_REQUIRED_STOP_STAGES = ("smoke", "nominal_sanity")
_REQUIRED_REFERENCES = ("ppo_baseline", "risk_guarded_ppo_v1")


class ShieldedPPOLaunchPacketError(RobotSfError, ValueError):
    """Raised when a shielded-PPO launch packet fails validation."""


def load_launch_packet(config_path: Path) -> dict[str, Any]:
    """Load a shielded-PPO repair launch packet.

    Returns:
        Parsed launch-packet mapping.
    """
    if not config_path.is_file():
        raise ShieldedPPOLaunchPacketError(f"launch packet is not a file: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ShieldedPPOLaunchPacketError("launch packet must be a YAML mapping")
    return payload


def validate_launch_packet(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a shielded-PPO repair launch packet.

    Returns:
        Compact validation report.
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
    hypothesis = _validate_repair_hypothesis(packet, errors)
    starting_points = _validate_starting_points(packet, root, errors)
    guard = _validate_runtime_guard(packet, root, errors)
    references = _validate_comparison_references(packet, root, errors)
    stop_gates = _validate_stop_gates(packet, errors)
    _validate_execution_boundary(packet, errors)

    if errors:
        joined = "\n- ".join(errors)
        raise ShieldedPPOLaunchPacketError(
            f"shielded-PPO launch packet failed validation:\n- {joined}"
        )

    return {
        "status": "valid",
        "schema_version": packet["schema_version"],
        "campaign_id": packet["campaign_id"],
        "repair_hypothesis": hypothesis,
        "starting_points": starting_points,
        "runtime_guard": guard,
        "comparison_references": references,
        "stop_gates": stop_gates,
    }


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


def _validate_repair_hypothesis(
    packet: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    hypothesis = packet.get("repair_hypothesis")
    if not isinstance(hypothesis, dict):
        errors.append("repair_hypothesis must be a mapping")
        return {}
    _require_non_empty_string(hypothesis, "id", errors)
    _require_non_empty_string(hypothesis, "statement", errors)
    deltas = hypothesis.get("enabled_deltas")
    if not isinstance(deltas, list) or len(deltas) != 1:
        errors.append("repair_hypothesis.enabled_deltas must contain exactly one delta")
        return {"enabled_deltas": deltas if isinstance(deltas, list) else []}
    delta = deltas[0]
    if not isinstance(delta, dict):
        errors.append("repair_hypothesis.enabled_deltas[0] must be a mapping")
        return {"enabled_deltas": []}
    _require_non_empty_string(delta, "type", errors)
    _require_non_empty_string(delta, "parameter", errors)
    if "value" not in delta:
        errors.append("repair_hypothesis.enabled_deltas[0].value is required")
    return {"id": hypothesis.get("id"), "enabled_deltas": [delta]}


def _validate_starting_points(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    points = packet.get("training_starting_points")
    if not isinstance(points, dict):
        errors.append("training_starting_points must be a mapping")
        return {}
    _require_existing_path(points, "base_ppo_training_config", repo_root, errors)
    _require_existing_path(points, "guarded_candidate_config", repo_root, errors)
    _require_existing_path(points, "guarded_algo_config", repo_root, errors)
    _validate_artifact_checksums(points, repo_root, errors)
    return {
        "base_ppo_training_config": points.get("base_ppo_training_config"),
        "guarded_candidate_config": points.get("guarded_candidate_config"),
    }


def _validate_runtime_guard(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    guard = packet.get("runtime_guard")
    if not isinstance(guard, dict):
        errors.append("runtime_guard must be a mapping")
        return {}
    if guard.get("active_in_all_evaluations") is not True:
        errors.append("runtime_guard.active_in_all_evaluations must be true")
    _require_existing_path(guard, "evaluation_candidate_config", repo_root, errors)
    _validate_required_list(guard, "required_diagnostics", _REQUIRED_DIAGNOSTICS, errors)
    return {
        "active_in_all_evaluations": guard.get("active_in_all_evaluations"),
        "required_diagnostics": list(guard.get("required_diagnostics", [])),
    }


def _validate_comparison_references(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    references = packet.get("comparison_references")
    if not isinstance(references, dict):
        errors.append("comparison_references must be a mapping")
        return {}
    missing = sorted(set(_REQUIRED_REFERENCES) - set(references))
    if missing:
        errors.append(f"comparison_references missing required references: {missing}")
    for name, reference in references.items():
        if not isinstance(reference, dict):
            errors.append(f"comparison_references.{name} must be a mapping")
            continue
        _require_non_empty_string(reference, "candidate_id", errors)
        _require_existing_path(reference, "config", repo_root, errors)
        _require_existing_path(reference, "source_report", repo_root, errors)
        _validate_seed_list(reference, f"comparison_references.{name}.seeds", errors)
        _validate_artifact_list(reference, "summary_artifacts", repo_root, errors)
    return {"reference_ids": sorted(references)}


def _validate_stop_gates(packet: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    gates = packet.get("stop_gates")
    if not isinstance(gates, dict):
        errors.append("stop_gates must be a mapping")
        return {}
    missing = sorted(set(_REQUIRED_STOP_STAGES) - set(gates))
    if missing:
        errors.append(f"stop_gates missing required stages: {missing}")
    for stage, gate in gates.items():
        if not isinstance(gate, dict):
            errors.append(f"stop_gates.{stage} must be a mapping")
            continue
        for key in ("min_success_rate", "max_collision_rate", "max_guard_fallback_rate"):
            if key not in gate:
                errors.append(f"stop_gates.{stage}.{key} is required")
        if stage == "nominal_sanity" and gate.get("allows_stress_escalation") is not True:
            errors.append("stop_gates.nominal_sanity.allows_stress_escalation must be true")
    return {"stages": sorted(gates)}


def _validate_execution_boundary(packet: dict[str, Any], errors: list[str]) -> None:
    execution = packet.get("execution_boundary")
    if not isinstance(execution, dict):
        errors.append("execution_boundary must be a mapping")
        return
    if execution.get("submit_slurm_from_this_issue") is not False:
        errors.append("execution_boundary.submit_slurm_from_this_issue must be false")
    if execution.get("full_training_in_this_issue") is not False:
        errors.append("execution_boundary.full_training_in_this_issue must be false")
    _require_non_empty_string(execution, "local_preflight_command", errors)
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


def _validate_seed_list(mapping: dict[str, Any], label: str, errors: list[str]) -> None:
    seeds = mapping.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append(f"{label} must be a non-empty list")
        return
    try:
        [int(seed) for seed in seeds]
    except (TypeError, ValueError):
        errors.append(f"{label} must contain integer seeds")


def _validate_artifact_checksums(
    mapping: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    checksums = mapping.get("checksums", {})
    if not isinstance(checksums, dict):
        errors.append("checksums must be a mapping")
        return
    for key in ("base_ppo_training_config", "guarded_candidate_config"):
        path_text = mapping.get(key)
        if isinstance(path_text, str):
            _validate_checksum(path_text, _resolve_path(path_text, repo_root), checksums, errors)


def _validate_artifact_list(
    mapping: dict[str, Any],
    key: str,
    repo_root: Path,
    errors: list[str],
) -> None:
    paths = mapping.get(key)
    checksums = mapping.get("checksums", {})
    if not isinstance(paths, list) or not paths:
        errors.append(f"{key} must be a non-empty list")
        return
    if not isinstance(checksums, dict):
        errors.append("checksums must be a mapping when local artifacts are listed")
        checksums = {}
    durable_count = 0
    for raw in paths:
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"{key} entries must be non-empty strings")
            continue
        path_text = raw.strip()
        if path_text.startswith(_DURABLE_URI_PREFIXES):
            durable_count += 1
            continue
        local_path = _resolve_path(path_text, repo_root)
        output_dir = (repo_root / "output").resolve()
        if local_path.resolve() == output_dir or local_path.resolve().is_relative_to(output_dir):
            errors.append(f"{key} must not depend on worktree-local output: {path_text}")
        _validate_checksum(path_text, local_path, checksums, errors)
    if durable_count == 0:
        errors.append(f"{key} must include at least one durable artifact URI")


def _validate_checksum(
    path_text: str,
    local_path: Path,
    checksums: dict[str, Any],
    errors: list[str],
) -> None:
    if not local_path.is_file():
        errors.append(f"local artifact is missing: {path_text}")
        return
    expected = checksums.get(path_text)
    if not isinstance(expected, str) or not expected.strip():
        errors.append(f"checksums missing SHA-256 entry for {path_text}")
        return
    sha256 = hashlib.sha256()
    with local_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if actual != expected.strip().lower():
        errors.append(f"checksum mismatch for {path_text}: expected {expected}, got {actual}")


__all__ = [
    "ShieldedPPOLaunchPacketError",
    "load_launch_packet",
    "validate_launch_packet",
]

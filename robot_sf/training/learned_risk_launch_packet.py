"""Validation helpers for learned-risk-model launch packets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_VERSION = "learned-risk-launch-packet.v1"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DURABLE_URI_PREFIXES = ("wandb-artifact://", "artifact://", "s3://", "gs://", "https://")
_REQUIRED_SPLITS = ("stress_slice", "full_matrix")
_REQUIRED_LABELS = ("collision", "near_miss", "low_progress")
_REQUIRED_DIAGNOSTICS = (
    "learned_risk_score",
    "hard_guard_decision",
    "auxiliary_cost_weight",
)


class LearnedRiskLaunchPacketError(ValueError):
    """Raised when a learned-risk launch packet fails validation."""


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file streamed in chunks.

    Shared by the launch-packet and trace-manifest validators so both compute
    artifact checksums identically.

    Args:
        path: File to digest.

    Returns:
        Lowercase hex SHA-256 string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_launch_packet(config_path: Path) -> dict[str, Any]:
    """Load a learned-risk launch packet YAML file.

    Args:
        config_path: Path to the YAML packet.

    Returns:
        Parsed packet mapping.

    Raises:
        LearnedRiskLaunchPacketError: If the file is missing or malformed.
    """
    if not config_path.is_file():
        raise LearnedRiskLaunchPacketError(f"launch packet is not a file: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LearnedRiskLaunchPacketError("launch packet must be a YAML mapping")
    return payload


def validate_launch_packet(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate a learned-risk-model launch packet.

    Args:
        config_path: Packet YAML path.
        repo_root: Repository root for resolving relative paths.

    Returns:
        Compact validation report.

    Raises:
        LearnedRiskLaunchPacketError: If any fail-closed invariant is violated.
    """
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, root)
    packet = load_launch_packet(config_path)
    errors: list[str] = []

    if packet.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION!r}")
    if packet.get("candidate_id") != "learned_risk_model_v1":
        errors.append("candidate_id must be 'learned_risk_model_v1'")
    _require_existing_path(packet, "slurm_handoff", root, errors)
    _validate_generating_commit(packet, errors)
    trace_report = _validate_trace_contract(packet, root, errors)
    baseline_report = _validate_baseline_packet(packet, root, errors)
    safety_report = _validate_safety_policy(packet, errors)
    _validate_execution_boundary(packet, errors)

    if errors:
        joined = "\n- ".join(errors)
        raise LearnedRiskLaunchPacketError(
            f"learned-risk launch packet failed validation:\n- {joined}"
        )

    return {
        "status": "valid",
        "schema_version": packet["schema_version"],
        "candidate_id": packet["candidate_id"],
        "trace_contract": trace_report,
        "baseline_comparison": baseline_report,
        "safety_policy": safety_report,
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


def _validate_trace_contract(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    contract = packet.get("trace_input_contract")
    if not isinstance(contract, dict):
        errors.append("trace_input_contract must be a mapping")
        return {}

    _validate_required_list(
        contract,
        "required_episode_fields",
        ("scenario_id", "seed", "termination_reason", "metrics"),
        errors,
    )
    _validate_required_list(contract, "feature_inputs", (), errors)
    _validate_required_list(contract, "label_targets", _REQUIRED_LABELS, errors)
    if contract.get("missing_required_fields_behavior") != "fail_closed":
        errors.append("trace_input_contract.missing_required_fields_behavior must be 'fail_closed'")
    local_fixtures = _validate_artifacts(contract, "trace_fixture_paths", repo_root, errors)
    required_fields = contract.get("required_episode_fields", [])
    if isinstance(required_fields, list):
        _validate_trace_fixtures(local_fixtures, required_fields, errors)
    return {
        "label_targets": list(contract.get("label_targets", [])),
        "trace_fixture_count": len(local_fixtures),
    }


def _validate_baseline_packet(
    packet: dict[str, Any],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Any]:
    baseline = packet.get("baseline_comparison")
    if not isinstance(baseline, dict):
        errors.append("baseline_comparison must be a mapping")
        return {}
    _require_non_empty_string(baseline, "candidate_id", errors)
    _require_existing_path(baseline, "candidate_config", repo_root, errors)
    _require_existing_path(baseline, "source_report", repo_root, errors)
    _validate_required_list(baseline, "scenario_slices", _REQUIRED_SPLITS, errors)
    _validate_seed_list(baseline, errors)
    _validate_artifacts(baseline, "summary_artifacts", repo_root, errors)
    return {
        "candidate_id": baseline.get("candidate_id"),
        "scenario_slices": list(baseline.get("scenario_slices", [])),
        "seeds": list(baseline.get("seeds", [])),
    }


def _validate_safety_policy(packet: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    safety = packet.get("safety_policy")
    if not isinstance(safety, dict):
        errors.append("safety_policy must be a mapping")
        return {}
    if safety.get("hard_guards_authoritative") is not True:
        errors.append("safety_policy.hard_guards_authoritative must be true")
    if safety.get("learned_output_role") != "auxiliary_cost_only":
        errors.append("safety_policy.learned_output_role must be 'auxiliary_cost_only'")
    _validate_required_list(safety, "required_diagnostics", _REQUIRED_DIAGNOSTICS, errors)
    return {
        "learned_output_role": safety.get("learned_output_role"),
        "required_diagnostics": list(safety.get("required_diagnostics", [])),
    }


def _validate_execution_boundary(packet: dict[str, Any], errors: list[str]) -> None:
    execution = packet.get("execution_boundary")
    if not isinstance(execution, dict):
        errors.append("execution_boundary must be a mapping")
        return
    if execution.get("submit_slurm_from_this_issue") is not False:
        errors.append("execution_boundary.submit_slurm_from_this_issue must be false")
    if execution.get("full_training_in_this_issue") is not False:
        errors.append("execution_boundary.full_training_in_this_issue must be false")
    _require_non_empty_string(execution, "slurm_command_shape", errors)
    _require_non_empty_string(execution, "local_preflight_command", errors)


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


def _validate_seed_list(baseline: dict[str, Any], errors: list[str]) -> None:
    seeds = baseline.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("baseline_comparison.seeds must be a non-empty list")
        return
    try:
        [int(seed) for seed in seeds]
    except (TypeError, ValueError):
        errors.append("baseline_comparison.seeds must contain integer seeds")


def _validate_artifacts(
    mapping: dict[str, Any],
    key: str,
    repo_root: Path,
    errors: list[str],
) -> list[Path]:
    raw_paths = mapping.get(key)
    checksums = mapping.get("checksums", {})
    if not isinstance(raw_paths, list) or not raw_paths:
        errors.append(f"{key} must be a non-empty list")
        return []
    if not isinstance(checksums, dict):
        errors.append("checksums must be a mapping when local artifacts are listed")
        checksums = {}

    local_paths: list[Path] = []
    durable_count = 0
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"{key} entries must be non-empty strings")
            continue
        path_text = raw.strip()
        if "output" in Path(path_text).parts:
            errors.append(f"{key} must not depend on worktree-local output: {path_text}")
        if path_text.startswith(_DURABLE_URI_PREFIXES):
            durable_count += 1
            continue
        local_path = _resolve_path(path_text, repo_root)
        local_paths.append(local_path)
        _validate_checksum(path_text, local_path, checksums, errors)
    if durable_count == 0 and key == "summary_artifacts":
        errors.append("baseline_comparison.summary_artifacts must include a durable artifact URI")
    return local_paths


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
    actual = sha256_file(local_path)
    if actual != expected.strip().lower():
        errors.append(f"checksum mismatch for {path_text}: expected {expected}, got {actual}")


def _validate_trace_fixtures(
    fixture_paths: list[Path],
    required_fields: list[Any],
    errors: list[str],
) -> None:
    required = [str(field) for field in required_fields]
    for fixture_path in fixture_paths:
        with open(fixture_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{fixture_path}:{line_number}: invalid JSON: {exc.msg}")
                if not isinstance(record, dict):
                    errors.append(f"{fixture_path}:{line_number} must be a JSON object")
                    continue
                missing = [field for field in required if field not in record]
                if missing:
                    errors.append(
                        f"{fixture_path}:{line_number} missing required fields: {missing}"
                    )


__all__ = [
    "LearnedRiskLaunchPacketError",
    "load_launch_packet",
    "sha256_file",
    "validate_launch_packet",
]

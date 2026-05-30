"""Validate learned-policy transfer benchmark metadata.

The v1 object is intended to live at ``algorithm_metadata.transfer_benchmark`` in episode or
campaign rows. Validation is metadata-only: it does not hydrate checkpoints, import external
policies, or run benchmark episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_VERSION = "learned_policy_transfer_benchmark.v1"
_SCHEMA_FILE = "learned-policy-transfer-metadata.schema.v1.json"
_NON_SUCCESS_READINESS = {"fallback", "degraded"}
_NON_SUCCESS_AVAILABILITY = {"partial-failure", "failed", "not_available"}
_SUCCESS_EXECUTION_MODES = {"native", "adapter"}
_SUCCESS_READINESS = {"native", "adapter"}
_SUCCESS_STAGES = {"robot_sf_smoke", "transfer_benchmark"}
_SUCCESS_ARTIFACT_STATUSES = {"complete", "not_required"}
_UNRESOLVED_MARKERS = {"missing", "not_declared", "unknown", "unresolved"}


def _schema_path() -> Path:
    """Return the packaged v1 transfer metadata schema path."""
    return Path(__file__).resolve().parent / "schemas" / _SCHEMA_FILE


def load_transfer_metadata_schema() -> dict[str, Any]:
    """Load the learned-policy transfer metadata JSON Schema.

    Returns:
        Parsed v1 transfer metadata schema.
    """
    return json.loads(_schema_path().read_text(encoding="utf-8"))


def _format_error_path(error: jsonschema.ValidationError) -> str:
    """Return a stable dotted path for a JSON Schema validation error."""
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "required" and isinstance(error.instance, dict):
        required_fields = list(error.validator_value or ())
        missing_fields = [field for field in required_fields if field not in error.instance]
        if missing_fields:
            parts.append(str(missing_fields[0]))
    if not parts:
        return "<root>"
    return ".".join(parts)


def _raise_first_schema_error(payload: dict[str, Any]) -> None:
    """Validate against JSON Schema and raise a concise ValueError for the first issue."""
    validator = jsonschema.Draft202012Validator(load_transfer_metadata_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda err: list(err.absolute_path))
    if not errors:
        return

    error = errors[0]
    path = _format_error_path(error)
    raise ValueError(f"{path}: {error.message}") from error


def _validate_success_semantics(payload: dict[str, Any]) -> None:
    """Enforce fail-closed benchmark-success semantics beyond structural schema checks."""
    benchmark_success = payload["benchmark_success"]
    readiness_status = payload["readiness_status"]
    availability_status = payload["availability_status"]

    if benchmark_success and readiness_status in _NON_SUCCESS_READINESS:
        raise ValueError(
            "benchmark_success=true is incompatible with "
            f"readiness_status={readiness_status!r}"
        )
    if benchmark_success and availability_status in _NON_SUCCESS_AVAILABILITY:
        raise ValueError(
            "benchmark_success=true is incompatible with "
            f"availability_status={availability_status!r}"
        )
    if benchmark_success and payload["execution_mode"] not in _SUCCESS_EXECUTION_MODES:
        raise ValueError(
            "benchmark_success=true requires execution_mode to be 'native' or 'adapter'"
        )
    if benchmark_success and readiness_status not in _SUCCESS_READINESS:
        raise ValueError(
            "benchmark_success=true requires readiness_status to be 'native' or 'adapter'"
        )
    if benchmark_success and availability_status != "available":
        raise ValueError("benchmark_success=true requires availability_status='available'")
    if benchmark_success and payload["transfer_stage"] not in _SUCCESS_STAGES:
        raise ValueError(
            "benchmark_success=true requires transfer_stage to be 'robot_sf_smoke' "
            "or 'transfer_benchmark'"
        )

    artifact_status = payload["artifact_provenance"]["artifact_manifest_status"]
    if benchmark_success and artifact_status not in _SUCCESS_ARTIFACT_STATUSES:
        raise ValueError(
            "benchmark_success=true requires artifact_provenance.artifact_manifest_status "
            "to be 'complete' or 'not_required'"
        )

    if benchmark_success:
        _validate_success_contract_placeholders(payload)


def _validate_success_contract_placeholders(payload: dict[str, Any]) -> None:
    """Reject placeholder contract values in success-capable metadata."""
    contract_paths = {
        "observation_contract.observation_level": payload["observation_contract"][
            "observation_level"
        ],
        "observation_contract.planner_observation_mode": payload["observation_contract"][
            "planner_observation_mode"
        ],
        "action_contract.raw_action_shape": payload["action_contract"]["raw_action_shape"],
        "action_contract.raw_action_frame": payload["action_contract"]["raw_action_frame"],
        "action_contract.adapted_action_frame": payload["action_contract"]["adapted_action_frame"],
        "action_contract.action_bounds": payload["action_contract"]["action_bounds"],
        "action_contract.kinematics_compatibility": payload["action_contract"][
            "kinematics_compatibility"
        ],
        "action_contract.projection_policy": payload["action_contract"]["projection_policy"],
    }
    for path, value in contract_paths.items():
        normalized = str(value).strip().lower()
        if normalized in _UNRESOLVED_MARKERS:
            raise ValueError(
                f"benchmark_success=true requires resolved contract value for {path}"
            )


def validate_transfer_metadata(payload: dict[str, Any]) -> None:
    """Validate one ``algorithm_metadata.transfer_benchmark`` object.

    Args:
        payload: Transfer metadata mapping to validate.

    Raises:
        ValueError: If the metadata violates the v1 schema or fail-closed success semantics.
    """
    if not isinstance(payload, dict):
        raise ValueError("transfer metadata must be a mapping")

    _raise_first_schema_error(payload)
    _validate_success_semantics(payload)

"""Validation helpers for diagnostic social preference label configs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION = "social-preference-labels.v1"
REQUIRED_LABEL_IDS = frozenset(
    {
        "clearance",
        "ttc_margin",
        "pedestrian_displacement",
        "path_blocking",
        "oscillation",
        "detour_burden",
        "recovery_smoothness",
    }
)
REQUIRED_LABEL_FIELDS = frozenset(
    {
        "id",
        "description",
        "metric_family",
        "unit",
        "preferred_direction",
        "diagnostic_thresholds",
        "required_trace_fields",
        "candidate_metric_keys",
        "notes",
    }
)


class SocialPreferenceLabelConfigError(ValueError):
    """Raised when a social preference label config violates the v1 contract."""


def load_social_preference_label_config(path: Path) -> dict[str, Any]:
    """Load and validate a social preference label YAML config.

    Returns:
        The validated YAML payload as a dictionary.
    """

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return validate_social_preference_label_config(payload)


def validate_social_preference_label_config(payload: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Validate the diagnostic social preference label config contract.

    Returns:
        The validated payload as a dictionary.
    """

    if not isinstance(payload, Mapping):
        raise SocialPreferenceLabelConfigError("config payload must be a mapping")
    _validate_top_level_contract(payload)
    labels = _validate_labels(payload.get("labels"), _allowed_direction_set(payload))
    _validate_required_label_ids(labels)
    return dict(payload)


def label_availability(label: Mapping[str, Any], available_fields: set[str]) -> str:
    """Return diagnostic availability for one label and a set of available trace fields."""

    if label.get("computation_status") == "not_available":
        return "not_available"
    required_fields = label.get("required_trace_fields")
    if not isinstance(required_fields, list):
        raise SocialPreferenceLabelConfigError("label required_trace_fields must be a list")
    missing_fields = [field for field in required_fields if field not in available_fields]
    return "not_available" if missing_fields else "diagnostic_available"


def _validate_top_level_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION:
        raise SocialPreferenceLabelConfigError(
            f"schema_version must be {SOCIAL_PREFERENCE_LABEL_SCHEMA_VERSION}"
        )

    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise SocialPreferenceLabelConfigError("claim_boundary is required")
    _require_boundary_language(claim_boundary)

    _validate_source_literature(payload.get("source_literature"))


def _allowed_direction_set(payload: Mapping[str, Any]) -> set[str]:
    allowed_directions = payload.get("allowed_preferred_directions")
    if not isinstance(allowed_directions, list) or not allowed_directions:
        raise SocialPreferenceLabelConfigError(
            "allowed_preferred_directions must be a non-empty list"
        )

    allowed_direction_set = set()
    for direction in allowed_directions:
        if not isinstance(direction, str) or not direction:
            raise SocialPreferenceLabelConfigError("preferred directions must be non-empty strings")
        allowed_direction_set.add(direction)
    return allowed_direction_set


def _validate_labels(labels: Any, allowed_directions: set[str]) -> list[Mapping[str, Any]]:
    if not isinstance(labels, list) or not labels:
        raise SocialPreferenceLabelConfigError("labels must be a non-empty list")

    validated_labels: list[Mapping[str, Any]] = []
    for index, label in enumerate(labels):
        if not isinstance(label, Mapping):
            raise SocialPreferenceLabelConfigError(f"label {index} must be a mapping")
        _validate_label_fields(index, label)
        _validate_label_direction(label, allowed_directions)
        _validate_label_thresholds(label)
        _validate_label_trace_fields(label)
        _validate_label_candidate_metrics(label)
        validated_labels.append(label)
    return validated_labels


def _validate_required_label_ids(labels: list[Mapping[str, Any]]) -> None:
    label_ids = [label["id"] for label in labels]
    duplicate_ids = sorted({label_id for label_id in label_ids if label_ids.count(label_id) > 1})
    if duplicate_ids:
        raise SocialPreferenceLabelConfigError(f"duplicate label ids: {', '.join(duplicate_ids)}")

    missing_required = sorted(REQUIRED_LABEL_IDS.difference(label_ids))
    if missing_required:
        raise SocialPreferenceLabelConfigError(
            f"missing required label ids: {', '.join(missing_required)}"
        )


def _require_boundary_language(claim_boundary: str) -> None:
    normalized = claim_boundary.casefold()
    required_phrases = ("diagnostic", "not a reward", "not calibrated")
    missing = [phrase for phrase in required_phrases if phrase not in normalized]
    if missing:
        raise SocialPreferenceLabelConfigError(
            "claim_boundary must explicitly state diagnostic, not-reward, and not-calibrated status"
        )


def _validate_source_literature(source_literature: Any) -> None:
    if not isinstance(source_literature, list) or not source_literature:
        raise SocialPreferenceLabelConfigError("source_literature must be a non-empty list")
    for entry in source_literature:
        if not isinstance(entry, Mapping):
            raise SocialPreferenceLabelConfigError("source_literature entries must be mappings")
        if entry.get("role") != "motivation_only":
            raise SocialPreferenceLabelConfigError(
                "source_literature entries must use role motivation_only"
            )
        if not entry.get("url"):
            raise SocialPreferenceLabelConfigError("source_literature entries require url")


def _validate_label_fields(index: int, label: Mapping[str, Any]) -> None:
    missing_fields = sorted(REQUIRED_LABEL_FIELDS.difference(label))
    if missing_fields:
        raise SocialPreferenceLabelConfigError(
            f"label {index} missing required fields: {', '.join(missing_fields)}"
        )

    label_id = label["id"]
    if not isinstance(label_id, str) or not _is_lowercase_snake_case(label_id):
        raise SocialPreferenceLabelConfigError(f"label {index} id must be lowercase snake_case")

    for field_name in ("description", "metric_family", "unit", "notes"):
        value = label[field_name]
        if not isinstance(value, str) or not value.strip():
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} {field_name} must be a non-empty string"
            )


def _validate_label_direction(label: Mapping[str, Any], allowed_directions: set[str]) -> None:
    label_id = str(label["id"])
    preferred_direction = label["preferred_direction"]
    if preferred_direction not in allowed_directions:
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} preferred_direction {preferred_direction!r} is not allowed"
        )


def _validate_label_thresholds(label: Mapping[str, Any]) -> None:
    label_id = str(label["id"])
    diagnostic_thresholds = label["diagnostic_thresholds"]
    if not isinstance(diagnostic_thresholds, Mapping):
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} diagnostic_thresholds must be a mapping"
        )

    status = diagnostic_thresholds.get("status")
    if status != "placeholder_default_not_human_calibrated":
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} threshold status must be placeholder_default_not_human_calibrated"
        )


def _validate_label_trace_fields(label: Mapping[str, Any]) -> None:
    _require_string_list(str(label["id"]), label["required_trace_fields"], "required_trace_fields")


def _validate_label_candidate_metrics(label: Mapping[str, Any]) -> None:
    label_id = str(label["id"])
    candidate_metric_keys = label["candidate_metric_keys"]
    if not isinstance(candidate_metric_keys, list):
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} candidate_metric_keys must be a list"
        )
    for metric_key in candidate_metric_keys:
        if not isinstance(metric_key, str) or not metric_key:
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} candidate_metric_keys must contain non-empty strings"
            )
    if not candidate_metric_keys and label.get("computation_status") != "not_available":
        raise SocialPreferenceLabelConfigError(
            f"label {label_id} without candidate_metric_keys must be not_available"
        )


def _require_string_list(label_id: str, value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise SocialPreferenceLabelConfigError(f"label {label_id} {field_name} must be a list")
    for item in value:
        if not isinstance(item, str) or not item:
            raise SocialPreferenceLabelConfigError(
                f"label {label_id} {field_name} must contain non-empty strings"
            )


def _is_lowercase_snake_case(value: str) -> bool:
    return value[0].islower() and all(
        char.islower() or char.isdigit() or char == "_" for char in value
    )

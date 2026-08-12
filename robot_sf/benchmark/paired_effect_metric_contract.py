"""Fail-closed retained-row contract for paired safety-wrapper effects.

The issue #4598 report builder consumes ``metric_values`` rows.  This module owns
the versioned field manifest and the producer-side validation used by the #6970
camera-ready runner gate.  It deliberately does not derive values from similarly
named legacy metrics: a missing or non-finite field is a hard contract failure.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CONTRACT_SCHEMA_VERSION = "paired_effect_metric_contract.v1"
REPORT_BUILDER_ISSUE = 4598
REQUIRED_METRIC_NAMES: tuple[str, ...] = (
    "exact_collision_probability",
    "near_miss_probability",
    "min_predicted_separation_m",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "stop_yield_latency_s",
    "wrapper_intervention_rate",
)
REQUIRED_RETAINED_ROW_PATH = "metric_values"
MAX_INVALID_ROW_SAMPLES = 10
_REQUIRED_FIELD_KEYS = {
    "name",
    "path",
    "unit",
    "definition",
    "emitting_component",
    "representation",
    "value_type",
}
_REPRESENTATIONS = {"raw", "normalized"}
_VALUE_TYPES = {"finite_scalar"}
_PATH_PREFIX = f"{REQUIRED_RETAINED_ROW_PATH}."
_BOUNDED_METRIC_NAMES = {
    "exact_collision_probability",
    "near_miss_probability",
    "completion_probability",
    "progress_at_timeout",
    "false_positive_stop_rate",
    "wrapper_intervention_rate",
}


class PairedEffectMetricContractError(ValueError):
    """Raised when a retained-row contract or its episode records is invalid."""


def load_paired_effect_metric_contract(path: str | Path) -> dict[str, Any]:
    """Load and validate one versioned paired-effect metric contract.

    Returns:
        Normalized contract mapping.
    """

    contract_path = Path(path)
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PairedEffectMetricContractError(
            f"paired-effect metric contract cannot be read: {contract_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PairedEffectMetricContractError(
            f"paired-effect metric contract must be a mapping: {contract_path}"
        )
    return validate_paired_effect_metric_contract(payload, source=contract_path)


def validate_paired_effect_metric_contract(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and return a normalized retained-row field manifest.

    Validation is intentionally strict: the report-builder outcome roster, field
    paths, representations, and producer metadata are all fixed before a run.

    Returns:
        Normalized contract mapping.
    """

    location = f" in {source}" if source is not None else ""
    normalized = dict(payload)
    _validate_contract_header(normalized, location=location)

    raw_fields = normalized.get("fields")
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        raise PairedEffectMetricContractError(f"fields must be a list{location}")
    if len(raw_fields) != len(REQUIRED_METRIC_NAMES):
        raise PairedEffectMetricContractError(
            f"fields must contain exactly {len(REQUIRED_METRIC_NAMES)} entries{location}"
        )

    fields = [
        _validate_contract_field(field, index=index, location=location)
        for index, field in enumerate(raw_fields)
    ]

    if tuple(field["name"] for field in fields) != REQUIRED_METRIC_NAMES:
        raise PairedEffectMetricContractError(
            "fields must use the exact #4598 outcome order "
            f"{list(REQUIRED_METRIC_NAMES)!r}{location}"
        )
    normalized["fields"] = fields
    normalized["required_metric_names"] = list(REQUIRED_METRIC_NAMES)
    return normalized


def _validate_contract_header(payload: Mapping[str, Any], *, location: str) -> None:
    """Validate fixed contract-level identifiers."""
    expected_values = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "issue": 6970,
        "report_builder_issue": REPORT_BUILDER_ISSUE,
        "retained_row_path": REQUIRED_RETAINED_ROW_PATH,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise PairedEffectMetricContractError(f"{key} must be {expected!r}{location}")
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        raise PairedEffectMetricContractError(f"claim_boundary must be non-empty{location}")


def _validate_contract_field(
    raw_field: Any,
    *,
    index: int,
    location: str,
) -> dict[str, Any]:
    """Validate and normalize one retained metric field declaration.

    Returns:
        Normalized field mapping.
    """
    if not isinstance(raw_field, Mapping):
        raise PairedEffectMetricContractError(f"fields[{index}] must be a mapping{location}")
    missing = sorted(_REQUIRED_FIELD_KEYS - set(raw_field))
    if missing:
        raise PairedEffectMetricContractError(
            f"fields[{index}] missing required keys {missing}{location}"
        )
    field = dict(raw_field)
    name = field.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PairedEffectMetricContractError(f"fields[{index}].name must be non-empty{location}")
    name = name.strip()
    path = field.get("path")
    if not isinstance(path, str) or path != f"{_PATH_PREFIX}{name}":
        raise PairedEffectMetricContractError(
            f"fields[{index}].path must be {_PATH_PREFIX}{name}{location}"
        )
    for key in ("unit", "definition", "emitting_component"):
        value = field.get(key)
        if not isinstance(value, str) or not value.strip() or "<" in value or ">" in value:
            raise PairedEffectMetricContractError(
                f"fields[{index}].{key} must be a concrete non-placeholder string{location}"
            )
        field[key] = value.strip()
    if field.get("representation") not in _REPRESENTATIONS:
        raise PairedEffectMetricContractError(
            f"fields[{index}].representation must be raw or normalized{location}"
        )
    if field.get("value_type") not in _VALUE_TYPES:
        raise PairedEffectMetricContractError(
            f"fields[{index}].value_type must be finite_scalar{location}"
        )
    if name in _BOUNDED_METRIC_NAMES and field.get("bounds") != {
        "lower": 0.0,
        "upper": 1.0,
    }:
        raise PairedEffectMetricContractError(
            f"fields[{index}].bounds must be {{'lower': 0.0, 'upper': 1.0}}{location}"
        )
    field["name"] = name
    return field


def validate_paired_effect_metric_record(
    record: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Validate one retained episode row against a validated contract.

    ``None``, booleans, non-numeric values, non-finite values, and out-of-range
    normalized values are all rejected.  Similar legacy fields are never used as
    aliases or fallbacks.

    Returns:
        JSON-safe row validation report.
    """

    validated = validate_paired_effect_metric_contract(contract)
    prefix = f"row {row_index}: " if row_index is not None else ""
    missing: list[str] = []
    invalid: list[dict[str, Any]] = []
    metric_values = record.get(REQUIRED_RETAINED_ROW_PATH)
    if not isinstance(metric_values, Mapping):
        missing.extend(REQUIRED_METRIC_NAMES)
        return {
            "status": "blocked",
            "row_index": row_index,
            "missing_fields": missing,
            "invalid_fields": invalid,
            "message": f"{prefix}missing mapping {REQUIRED_RETAINED_ROW_PATH!r}",
        }

    for field in validated["fields"]:
        name = str(field["name"])
        value = metric_values.get(name)
        if value is None:
            missing.append(str(field["path"]))
            continue
        if isinstance(value, bool):
            invalid.append({"field": str(field["path"]), "reason": "boolean_is_not_scalar"})
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append({"field": str(field["path"]), "reason": "not_numeric"})
            continue
        if not math.isfinite(numeric):
            invalid.append({"field": str(field["path"]), "reason": "non_finite"})
            continue
        bounds = field.get("bounds")
        if isinstance(bounds, Mapping):
            lower = float(bounds["lower"])
            upper = float(bounds["upper"])
            if numeric < lower or numeric > upper:
                invalid.append(
                    {
                        "field": str(field["path"]),
                        "reason": "out_of_bounds",
                        "value": numeric,
                        "bounds": {"lower": lower, "upper": upper},
                    }
                )

    status = "ok" if not missing and not invalid else "blocked"
    return {
        "status": status,
        "row_index": row_index,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "message": None if status == "ok" else f"{prefix}retained metric contract failed",
    }


def validate_paired_effect_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    include_row_reports: bool = False,
) -> dict[str, Any]:
    """Validate all retained episode rows and return a JSON-safe gate report.

    Returns:
        Aggregate validation report.
    """

    validated = validate_paired_effect_metric_contract(contract)
    missing_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    row_reports: list[dict[str, Any]] = []
    invalid_row_samples: list[dict[str, Any]] = []
    valid_row_count = 0
    for index, row in enumerate(rows):
        report = validate_paired_effect_metric_record(row, validated, row_index=index)
        missing_counts.update(report["missing_fields"])
        invalid_counts.update(item["field"] for item in report["invalid_fields"])
        if report["status"] == "ok":
            valid_row_count += 1
        elif len(invalid_row_samples) < MAX_INVALID_ROW_SAMPLES:
            invalid_row_samples.append(report)
        if include_row_reports:
            row_reports.append(report)
    complete = bool(rows) and valid_row_count == len(rows)
    result: dict[str, Any] = {
        "schema_version": "paired_effect_metric_validation.v1",
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "ok" if complete else "blocked",
        "complete": complete,
        "row_count": len(rows),
        "valid_row_count": valid_row_count,
        "required_metric_names": list(REQUIRED_METRIC_NAMES),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "invalid_field_counts": dict(sorted(invalid_counts.items())),
        "invalid_row_samples": invalid_row_samples,
        "claim_boundary": (
            "Instrumentation contract gate only. A passing retained-row check does not make a "
            "campaign result benchmark or paper evidence; it only establishes that the declared "
            "paired metrics were retained without alias substitution."
        ),
    }
    if include_row_reports:
        result["row_reports"] = row_reports
    return result


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON-list or JSONL episode rows for the validation CLI.

    Returns:
        Parsed object rows.
    """

    row_path = Path(path)
    if row_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            row_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise PairedEffectMetricContractError(
                    f"{row_path}:{line_number} must contain a JSON object"
                )
            rows.append(value)
        return rows
    value = json.loads(row_path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise PairedEffectMetricContractError(
            f"{row_path} must contain a JSON list or JSONL object rows"
        )
    return value


def enforce_paired_effect_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    include_row_reports: bool = False,
) -> dict[str, Any]:
    """Raise on a retained-row mismatch and return the successful gate report.

    Returns:
        Successful aggregate validation report.
    """

    report = validate_paired_effect_metric_rows(
        rows,
        contract,
        include_row_reports=include_row_reports,
    )
    if not report["complete"]:
        raise PairedEffectMetricContractError(
            f"paired-effect retained-row contract failed: {json.dumps(report, sort_keys=True)}"
        )
    return report


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "REQUIRED_METRIC_NAMES",
    "PairedEffectMetricContractError",
    "enforce_paired_effect_metric_rows",
    "load_json_rows",
    "load_paired_effect_metric_contract",
    "validate_paired_effect_metric_contract",
    "validate_paired_effect_metric_record",
    "validate_paired_effect_metric_rows",
]

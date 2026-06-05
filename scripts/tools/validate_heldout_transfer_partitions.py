#!/usr/bin/env python3
"""Validate held-out scenario-family transfer partition manifests."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

EXPECTED_SCHEMA_VERSION = "robot_sf.heldout_transfer_partitions.v1"
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "issue",
    "status",
    "claim_boundary",
    "training_family_pool",
    "validation_family_pool",
    "benchmark_set_evaluation",
    "heldout_family_evaluation",
    "planner_inclusion",
    "metrics",
    "leakage_audit_required",
    "planned_outputs",
)
REQUIRED_EXECUTION_MODES = {
    "native",
    "adapter",
    "fallback",
    "degraded",
    "failed",
    "not_available",
}
REQUIRED_OUTPUT_ROLES = {
    "benchmark_set_table",
    "heldout_family_table",
    "family_breakdown_table",
    "comparable_delta_table",
    "leakage_audit_checklist",
    "transfer_delta_figure",
    "artifact_catalog",
}


def validate_partition_manifest(path: Path) -> list[str]:
    """Return validation errors for a held-out transfer partition manifest."""

    errors: list[str] = []
    payload = _load_yaml_mapping(path, errors)
    if payload is None:
        return errors

    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append(
            f"{path}: schema_version must be {EXPECTED_SCHEMA_VERSION!r}, "
            f"found {payload.get('schema_version')!r}"
        )

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if _is_missing(payload.get(field)):
            errors.append(f"{path}: missing required field {field!r}")

    _validate_pool(path, payload.get("training_family_pool"), "training_family_pool", errors)
    _validate_pool(path, payload.get("validation_family_pool"), "validation_family_pool", errors)
    _validate_evaluation(
        path,
        payload.get("benchmark_set_evaluation"),
        "benchmark_set_evaluation",
        errors,
        require_benchmark_config=True,
    )
    _validate_evaluation(
        path,
        payload.get("heldout_family_evaluation"),
        "heldout_family_evaluation",
        errors,
        require_benchmark_config=True,
    )
    _validate_planner_inclusion(path, payload.get("planner_inclusion"), errors)
    _validate_planned_outputs(path, payload.get("planned_outputs"), errors)

    return errors


def _validate_pool(path: Path, value: Any, field_name: str, errors: list[str]) -> None:
    """Validate a training or validation pool block."""

    if not isinstance(value, Mapping):
        errors.append(f"{path}: {field_name} must be a mapping")
        return
    _validate_repo_path(path, value.get("scenario_matrix"), f"{field_name}.scenario_matrix", errors)
    if not _is_nonempty_str_list(value.get("scenario_families")):
        errors.append(f"{path}: {field_name}.scenario_families must be a non-empty string list")
    if _is_missing(value.get("leakage_rule")):
        errors.append(f"{path}: {field_name}.leakage_rule is required")


def _validate_evaluation(
    path: Path,
    value: Any,
    field_name: str,
    errors: list[str],
    *,
    require_benchmark_config: bool,
) -> None:
    """Validate an evaluation partition block."""

    if not isinstance(value, Mapping):
        errors.append(f"{path}: {field_name} must be a mapping")
        return
    _validate_repo_path(path, value.get("scenario_matrix"), f"{field_name}.scenario_matrix", errors)
    if require_benchmark_config:
        _validate_repo_path(
            path,
            value.get("benchmark_config"),
            f"{field_name}.benchmark_config",
            errors,
        )
    if not _is_nonempty_int_list(value.get("seeds")):
        errors.append(f"{path}: {field_name}.seeds must be a non-empty integer list")
    if _is_missing(value.get("claim_label")):
        errors.append(f"{path}: {field_name}.claim_label is required")


def _validate_planner_inclusion(path: Path, value: Any, errors: list[str]) -> None:
    """Validate planner inclusion and execution-mode semantics."""

    if not isinstance(value, Mapping):
        errors.append(f"{path}: planner_inclusion must be a mapping")
        return
    if not _is_nonempty_str_list(value.get("include")):
        errors.append(f"{path}: planner_inclusion.include must be a non-empty string list")
    modes = value.get("execution_modes")
    if not isinstance(modes, Mapping):
        errors.append(f"{path}: planner_inclusion.execution_modes must be a mapping")
        return
    missing_modes = sorted(REQUIRED_EXECUTION_MODES - set(modes))
    if missing_modes:
        errors.append(f"{path}: planner_inclusion.execution_modes missing {missing_modes}")


def _validate_planned_outputs(path: Path, value: Any, errors: list[str]) -> None:
    """Validate planned output roles without requiring local output files to exist."""

    if not isinstance(value, list) or not value:
        errors.append(f"{path}: planned_outputs must be a non-empty list")
        return
    roles: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"{path}: planned_outputs[{index}] must be a mapping")
            continue
        output_path = item.get("path")
        if not isinstance(output_path, str) or not output_path:
            errors.append(f"{path}: planned_outputs[{index}].path must be a non-empty string")
        role = item.get("evidence_role")
        if not isinstance(role, str) or not role:
            errors.append(
                f"{path}: planned_outputs[{index}].evidence_role must be a non-empty string"
            )
            continue
        roles.add(role)
    missing_roles = sorted(REQUIRED_OUTPUT_ROLES - roles)
    if missing_roles:
        errors.append(f"{path}: planned_outputs missing evidence roles {missing_roles}")


def _validate_repo_path(path: Path, value: Any, field_name: str, errors: list[str]) -> None:
    """Validate that a referenced repository path exists."""

    if not isinstance(value, str) or not value:
        errors.append(f"{path}: {field_name} must be a non-empty string path")
        return
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{path}: {field_name} must be a repository-relative path")
        return
    if not candidate.exists():
        errors.append(f"{path}: {field_name} does not exist: {value}")


def _load_yaml_mapping(path: Path, errors: list[str]) -> Mapping[str, Any] | None:
    """Load a YAML mapping while reporting parse/read errors."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"{path}: cannot read YAML: {exc}")
        return None
    except yaml.YAMLError as exc:
        errors.append(f"{path}: invalid YAML: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{path}: YAML document must be a mapping")
        return None
    return payload


def _is_missing(value: Any) -> bool:
    """Return whether a required manifest value is absent or empty."""

    return value in (None, "", [], {})


def _is_nonempty_str_list(value: Any) -> bool:
    """Return whether value is a non-empty list of non-empty strings."""

    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _is_nonempty_int_list(value: Any) -> bool:
    """Return whether value is a non-empty list of integers."""

    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def main(argv: list[str] | None = None) -> int:
    """Run the held-out transfer partition manifest validator CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Partition manifest YAML path.")
    args = parser.parse_args(argv)

    errors = validate_partition_manifest(args.manifest)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated held-out transfer partition manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate the question-first experiment registry."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

REGISTRY_SCHEMA_VERSION = "experiment-registry.v1"
RECORD_SCHEMA_VERSION = "experiment-record.v1"
REQUIRED_RECORD_FIELDS = (
    "experiment_id",
    "issue",
    "issue_url",
    "question",
    "hypothesis",
    "config",
    "command",
    "inputs",
    "outputs",
    "expected_artifacts",
    "evidence_grade",
    "paper_relevance",
    "status",
)
VALID_EVIDENCE_GRADES = {"proposal", "inferred", "observed"}
VALID_PAPER_RELEVANCE = {"none", "exploratory", "paper_candidate", "paper_facing"}
LOCAL_OUTPUT_PREFIXES = ("output/", "./output/")


def validate_registry(registry_path: Path) -> list[str]:
    """Return validation errors for an experiment registry."""
    errors: list[str] = []
    registry = _load_yaml_mapping(registry_path, errors, display=str(registry_path))
    if registry is None:
        return errors

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        errors.append(
            f"{registry_path}: expected schema_version {REGISTRY_SCHEMA_VERSION!r}, "
            f"found {registry.get('schema_version')!r}"
        )

    records = registry.get("records")
    if not isinstance(records, list) or not records:
        errors.append(f"{registry_path}: records must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    for record_entry in records:
        if not isinstance(record_entry, str) or not record_entry:
            errors.append(f"{registry_path}: record entries must be non-empty paths")
            continue
        record_path = (registry_path.parent / record_entry).resolve()
        record = _load_yaml_mapping(record_path, errors, display=record_entry)
        if record is None:
            continue
        _validate_record(record, record_entry, errors, seen_ids=seen_ids)

    return errors


def _validate_record(
    record: Mapping[str, Any],
    display_path: str,
    errors: list[str],
    *,
    seen_ids: set[str],
) -> None:
    """Append validation errors for one experiment record."""
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        errors.append(
            f"{display_path}: expected schema_version {RECORD_SCHEMA_VERSION!r}, "
            f"found {record.get('schema_version')!r}"
        )

    for field in REQUIRED_RECORD_FIELDS:
        if _is_missing(record.get(field)):
            errors.append(f"{display_path}: missing required field {field!r}")

    experiment_id = record.get("experiment_id")
    if isinstance(experiment_id, str):
        if experiment_id in seen_ids:
            errors.append(f"{display_path}: duplicate experiment_id {experiment_id!r}")
        seen_ids.add(experiment_id)

    _validate_vocab(
        record.get("evidence_grade"),
        "evidence_grade",
        VALID_EVIDENCE_GRADES,
        display_path,
        errors,
    )
    paper_relevance = record.get("paper_relevance")
    _validate_vocab(
        paper_relevance,
        "paper_relevance",
        VALID_PAPER_RELEVANCE,
        display_path,
        errors,
    )

    if paper_relevance == "paper_facing":
        for artifact in _iter_artifact_items(record.get("outputs")):
            _validate_paper_facing_artifact(artifact, display_path, errors)
        for artifact in _iter_artifact_items(record.get("expected_artifacts")):
            _validate_paper_facing_artifact(artifact, display_path, errors)


def _validate_paper_facing_artifact(
    artifact: Mapping[str, Any],
    display_path: str,
    errors: list[str],
) -> None:
    """Reject paper-facing output artifacts that only point at local output paths."""
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not _is_local_output_path(artifact_path):
        return
    durable_reference = artifact.get("durable_reference")
    if _is_missing(durable_reference):
        errors.append(
            f"{display_path}: paper-facing record references local-only output/ artifact "
            f"without durable_reference: {artifact_path}"
        )


def _validate_vocab(
    value: Any,
    field_name: str,
    allowed_values: set[str],
    display_path: str,
    errors: list[str],
) -> None:
    """Append an error when a controlled-vocabulary value is unknown."""
    if value is not None and value not in allowed_values:
        errors.append(f"{display_path}: {field_name} must be one of {sorted(allowed_values)}")


def _iter_artifact_items(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield artifact mappings from string or mapping lists."""
    if not isinstance(value, list):
        return
    for item in value:
        if isinstance(item, str):
            yield {"path": item}
        elif isinstance(item, Mapping):
            yield item


def _is_local_output_path(path: str) -> bool:
    """Return whether a path points at the disposable worktree output root."""
    normalized = path.replace("\\", "/")
    return normalized.startswith(LOCAL_OUTPUT_PREFIXES)


def _is_missing(value: Any) -> bool:
    """Return whether a required registry value is absent or empty."""
    if value is None:
        return True
    return value in ("", [], {})


def _load_yaml_mapping(path: Path, errors: list[str], *, display: str) -> Mapping[str, Any] | None:
    """Load a YAML file and return a mapping, appending errors instead of raising."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        errors.append(f"{display}: cannot read YAML: {exc}")
        return None
    except yaml.YAMLError as exc:
        errors.append(f"{display}: invalid YAML: {exc}")
        return None
    if not isinstance(payload, Mapping):
        errors.append(f"{display}: YAML document must be a mapping")
        return None
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the experiment registry validator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        default="experiments/registry.yaml",
        type=Path,
        help="Path to the experiment registry index.",
    )
    args = parser.parse_args(argv)

    errors = validate_registry(args.registry)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated experiment registry: {args.registry}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

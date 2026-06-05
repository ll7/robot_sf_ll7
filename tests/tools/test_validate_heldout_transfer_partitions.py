"""Tests for held-out transfer partition manifest validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools.validate_heldout_transfer_partitions import validate_partition_manifest

FIXTURE = Path("configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml")


def _write_manifest(tmp_path: Path, updates: dict) -> Path:
    """Copy the repository fixture, apply top-level updates, and return the path."""

    payload = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    payload.update(updates)
    path = tmp_path / "partitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_repository_partition_manifest_validates() -> None:
    """The issue #2128 partition manifest should satisfy the local schema."""

    assert validate_partition_manifest(FIXTURE) == []


def test_wrong_schema_version_reports_actionable_error(tmp_path: Path) -> None:
    """A custom partition manifest must explicitly use the supported schema version."""

    path = _write_manifest(tmp_path, {"schema_version": "robot_sf.heldout_transfer_partitions.v0"})

    errors = validate_partition_manifest(path)

    assert any("schema_version" in error for error in errors)


def test_missing_required_output_role_reports_actionable_error(tmp_path: Path) -> None:
    """The output contract should stay aligned across protocol, manifest, and experiment card."""

    payload = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    payload["planned_outputs"] = [
        item
        for item in payload["planned_outputs"]
        if item.get("evidence_role") != "leakage_audit_checklist"
    ]
    path = tmp_path / "partitions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    errors = validate_partition_manifest(path)

    assert any("leakage_audit_checklist" in error for error in errors)

"""Tests for stale artifact freshness classification."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools import stale_artifact_detector as detector

if TYPE_CHECKING:
    from pathlib import Path


def _write_artifact(tmp_path: Path, name: str, content: bytes = b"artifact") -> tuple[Path, str]:
    path = tmp_path / "durable" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def test_classifies_current_artifact_with_matching_checksum(tmp_path: Path) -> None:
    """Current artifacts need durable path, matching checksum, and current schema."""
    artifact_path, checksum = _write_artifact(tmp_path, "current.md")
    relative_artifact = artifact_path.relative_to(tmp_path)

    result = detector.classify_artifact(
        {
            "artifact_id": "current-report",
            "schema_version": "artifact_catalog.v1",
            "checksum": checksum,
            "outputs": {"md": {"path": str(relative_artifact)}},
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.CURRENT
    assert result.checksum_actual == checksum
    assert result.reasons == ["output md checksum matches"]


def test_classifies_historical_valid_when_checksum_matches_old_schema(tmp_path: Path) -> None:
    """Older schema with matching durable artifact remains citable only historically."""
    artifact_path, checksum = _write_artifact(tmp_path, "historical.csv", b"old")
    relative_artifact = artifact_path.relative_to(tmp_path)

    result = detector.classify_artifact(
        {
            "artifact_id": "historical-table",
            "schema_version": "artifact_catalog.v0",
            "sha256": checksum,
            "outputs": {"csv": {"path": str(relative_artifact)}},
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.HISTORICAL_VALID
    assert "schema is not current" in result.reasons[-1]


def test_classifies_stale_when_checksum_mismatches(tmp_path: Path) -> None:
    """Checksum drift means the artifact must be refreshed before manuscript use."""
    artifact_path, _checksum = _write_artifact(tmp_path, "stale.json", b"new")
    relative_artifact = artifact_path.relative_to(tmp_path)

    result = detector.classify_artifact(
        {
            "artifact_id": "stale-json",
            "schema_version": "artifact_catalog.v1",
            "checksum": "0" * 64,
            "outputs": {"json": {"path": str(relative_artifact)}},
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert "output json checksum mismatch" in result.reasons


def test_classifies_blocked_for_manuscript_use_from_readiness_marker() -> None:
    """Explicit diagnostic or blocked markers should dominate checksum freshness."""
    result = detector.classify_artifact(
        {
            "artifact_id": "diagnostic-only",
            "schema_version": "artifact_catalog.v1",
            "readiness": "diagnostic",
        }
    )

    assert result.state == detector.ArtifactState.BLOCKED_FOR_MANUSCRIPT_USE
    assert result.reasons == ["explicit manuscript blocker: readiness=diagnostic"]


def test_classifies_blocked_for_manuscript_use_from_blocked_reason() -> None:
    """A provenance blocker marks the artifact as blocked for manuscript use."""
    result = detector.classify_artifact(
        {
            "artifact_id": "missing-source",
            "schema_version": "artifact_catalog.v1",
            "blocked_reason": "missing source provenance",
        }
    )

    assert result.state == detector.ArtifactState.BLOCKED_FOR_MANUSCRIPT_USE
    assert "missing source provenance" in result.reasons[0]


def test_local_only_output_paths_are_stale_even_with_matching_checksum(tmp_path: Path) -> None:
    """Ignored local output paths are not durable manuscript proof."""
    output_path = tmp_path / "output" / "local.md"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("local", encoding="utf-8")
    checksum = hashlib.sha256(b"local").hexdigest()

    result = detector.classify_artifact(
        {
            "artifact_id": "local-output",
            "schema_version": "artifact_catalog.v1",
            "checksum": checksum,
            "outputs": {"md": {"path": "output/local.md"}},
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert any("local-only" in warning for warning in result.warnings)
    assert "artifact is only represented by disposable local output" in result.reasons


def test_missing_metadata_is_stale_not_crashing() -> None:
    """Malformed entries should fail closed instead of crashing."""
    result = detector.classify_artifact({"artifact_id": ""})

    assert result.artifact_id == "unnamed"
    assert result.state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert result.reasons == ["output artifact missing output path"]


def test_local_only_path_traversal_is_stale_even_when_checksum_matches(tmp_path: Path) -> None:
    """Traversal through a disposable root should not evade local-only detection."""
    artifact_path = tmp_path / "durable" / "escaped.md"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("escaped", encoding="utf-8")
    checksum = hashlib.sha256(b"escaped").hexdigest()

    result = detector.classify_artifact(
        {
            "artifact_id": "escaped-output",
            "schema_version": "artifact_catalog.v1",
            "outputs": {"md": {"path": "output/../durable/escaped.md", "sha256": checksum}},
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert any("local-only" in warning for warning in result.warnings)


def test_multi_output_requires_every_output_to_match(tmp_path: Path) -> None:
    """A single stale output makes the whole artifact stale."""
    current_path, current_checksum = _write_artifact(tmp_path, "multi-current.md", b"current")
    stale_path, _stale_checksum = _write_artifact(tmp_path, "multi-stale.json", b"changed")

    result = detector.classify_artifact(
        {
            "artifact_id": "multi-output",
            "schema_version": "artifact_catalog.v1",
            "outputs": {
                "md": {
                    "path": str(current_path.relative_to(tmp_path)),
                    "sha256": current_checksum,
                },
                "json": {
                    "path": str(stale_path.relative_to(tmp_path)),
                    "sha256": "0" * 64,
                },
            },
        },
        manifest_dir=tmp_path,
    )

    assert result.state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert "output md checksum matches" in result.reasons
    assert "output json checksum mismatch" in result.reasons


def test_scan_manifest_handles_non_mapping_entries(tmp_path: Path) -> None:
    """Non-mapping manifest rows become stale entries in the report."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(["not-a-mapping"]), encoding="utf-8")

    report = detector.scan_manifest(manifest)

    assert report.exit_code == 1
    assert report.results[0].state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert report.results[0].reasons == ["manifest entry is not a mapping"]


def test_scan_manifest_summarizes_all_four_states(tmp_path: Path) -> None:
    """Reports expose compact counts by classification state."""
    current_path, current_checksum = _write_artifact(tmp_path, "current.md", b"current")
    historical_path, historical_checksum = _write_artifact(tmp_path, "historical.md", b"history")
    stale_path, _stale_checksum = _write_artifact(tmp_path, "stale.md", b"changed")
    current_relative = current_path.relative_to(tmp_path)
    historical_relative = historical_path.relative_to(tmp_path)
    stale_relative = stale_path.relative_to(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            [
                {
                    "artifact_id": "current",
                    "schema_version": "artifact_catalog.v1",
                    "checksum": current_checksum,
                    "outputs": {"md": {"path": str(current_relative)}},
                },
                {
                    "artifact_id": "historical",
                    "schema_version": "artifact_catalog.v0",
                    "checksum": historical_checksum,
                    "outputs": {"md": {"path": str(historical_relative)}},
                },
                {
                    "artifact_id": "stale",
                    "schema_version": "artifact_catalog.v1",
                    "checksum": "0" * 64,
                    "outputs": {"md": {"path": str(stale_relative)}},
                },
                {"artifact_id": "blocked", "readiness": "blocked"},
            ]
        ),
        encoding="utf-8",
    )

    report = detector.scan_manifest(manifest)
    data = report.to_dict()

    assert report.exit_code == 1
    assert data["summary"] == {
        "blocked-for-manuscript-use": 1,
        "current": 1,
        "historical-valid": 1,
        "stale-needs-refresh": 1,
    }


def test_load_manifest_supports_json_and_single_mapping(tmp_path: Path) -> None:
    """JSON and single mapping manifests are normalized to entry lists."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"artifact_id": "one"}), encoding="utf-8")

    entries = detector.load_manifest(manifest)

    assert entries == [{"artifact_id": "one"}]


def test_scan_manifest_reports_parse_failure_as_stale(tmp_path: Path) -> None:
    """Manifest load failures produce a non-clean stale report."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not-json", encoding="utf-8")

    report = detector.scan_manifest(manifest)

    assert report.exit_code == 1
    assert report.results[0].artifact_id == "manifest.json"
    assert report.results[0].state == detector.ArtifactState.STALE_NEEDS_REFRESH
    assert "cannot load manifest" in report.results[0].reasons[0]


def test_cli_writes_json_report(tmp_path: Path) -> None:
    """CLI writes a machine-readable JSON report and returns report status."""
    artifact_path, checksum = _write_artifact(tmp_path, "current.md", b"cli")
    relative_artifact = artifact_path.relative_to(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    report_path = tmp_path / "report.json"
    manifest.write_text(
        yaml.safe_dump(
            [
                {
                    "artifact_id": "cli-current",
                    "schema_version": "artifact_catalog.v1",
                    "checksum": checksum,
                    "outputs": {"md": {"path": str(relative_artifact)}},
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = detector.main([str(manifest), "--json-out", str(report_path)])

    assert exit_code == 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["summary"]["current"] == 1

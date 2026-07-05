"""Tests AMV calibration and command-response source status."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.tools.manage_external_data import (
    build_amv_calibration_status,
    build_amv_command_response_source_status,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_2415_MANIFEST = (
    REPO_ROOT / "configs" / "research" / "amv_command_response_trace_manifest_issue_2415.yaml"
)


def _manifest_payload() -> dict:
    return yaml.safe_load(ISSUE_2415_MANIFEST.read_text(encoding="utf-8"))


def _write_manifest_with_status(tmp_path: Path, staging_status: str) -> Path:
    payload = _manifest_payload()
    payload["traces"][0]["staging_status"] = staging_status
    if staging_status == "staged":
        payload["traces"][0]["blocker_issues"] = []
    manifest_path = tmp_path / "amv_trace_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_source_ready_manifest(tmp_path: Path) -> Path:
    payload = _manifest_payload()
    payload["traces"][0]["staging_status"] = "staged"
    payload["traces"][0]["blocker_issues"] = []
    payload["traces"][0]["provenance"]["license_status"] = "local_restricted_accepted"
    manifest_path = tmp_path / "amv_trace_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _stage_amv_calibration_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external_root = tmp_path / "external"
    asset_dir = external_root / "amv_calibration"
    asset_dir.mkdir(parents=True)
    (asset_dir / "trace_metadata.json").write_text("{}", encoding="utf-8")
    (asset_dir / "README.md").write_text(
        "Local restricted AMV command-response source accepted by maintainer.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(external_root))


def test_amv_calibration_status_defaults_to_blocked_external_input() -> None:
    """Issue #2415 currently has no real AMV trace bundle and must fail closed."""
    report = build_amv_calibration_status()

    assert report["schema"] == "amv_calibration_status.v1"
    assert report["issue"] == 2415
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert "amv_calibration_asset_missing" in report["blockers"]
    assert "command_response_trace_manifest_not_ready" in report["blockers"]
    assert report["asset_status"]["status"] == "missing"
    assert report["trace_manifest_report"]["manifest_status"] == "blocked-external-input"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is False


def test_amv_calibration_status_does_not_admit_asset_without_staged_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present source asset is insufficient while the trace manifest remains blocked."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    blocked_manifest = _write_manifest_with_status(tmp_path, "blocked-external-input")

    report = build_amv_calibration_status(blocked_manifest)

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert report["blockers"] == ["command_response_trace_manifest_not_ready"]
    assert report["trace_manifest_report"]["manifest_status"] == "blocked-external-input"


def test_amv_calibration_status_admits_staged_manifest_and_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged bundle plus staged manifest is the calibration-admission condition."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")

    report = build_amv_calibration_status(staged_manifest)

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["blockers"] == []
    assert report["trace_manifest_report"]["manifest_status"] == "ready"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is True


def test_amv_command_response_source_status_defaults_to_blocked_external_input() -> None:
    """Issue #2000 has no maintainer-accepted real command-response source staged."""
    report = build_amv_command_response_source_status()

    assert report["schema"] == "amv_command_response_source_status.v1"
    assert report["issue"] == 2000
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert "amv_command_response_asset_missing" in report["blockers"]
    assert "command_response_trace_not_staged" in report["blockers"]
    assert "source_blocker_issues_open" in report["blockers"]
    assert "source_license_or_access_terms_unknown" in report["blockers"]
    assert report["missing_signal_inventory"] == {}
    assert report["evidence_boundary"] == (
        "source_acquisition_contract_only_no_trace_ingest_no_calibration_run"
    )


def test_amv_command_response_source_status_accepts_local_restricted_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged local source with access terms satisfies the #2000 acquisition gate."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    manifest_path = _write_source_ready_manifest(tmp_path)

    report = build_amv_command_response_source_status(manifest_path)

    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["blockers"] == []
    assert report["asset_status"]["status"] == "available"
    assert report["license_status"] == "local_restricted_accepted"
    assert report["trace_blocker_issues"] == []

"""Tests for AMV calibration admission status in manage_external_data."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools.manage_external_data import build_amv_calibration_status

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_2415_MANIFEST = (
    REPO_ROOT / "configs" / "research" / "amv_command_response_trace_manifest_issue_2415.yaml"
)


def _write_manifest_with_status(tmp_path: Path, staging_status: str) -> Path:
    payload = yaml.safe_load(ISSUE_2415_MANIFEST.read_text(encoding="utf-8"))
    payload["traces"][0]["staging_status"] = staging_status
    if staging_status == "staged":
        payload["traces"][0]["blocker_issues"] = []
    manifest_path = tmp_path / "amv_trace_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


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
    tmp_path: Path, monkeypatch
) -> None:
    """A present asset is insufficient while the issue #2415 trace manifest remains blocked."""
    external_root = tmp_path / "external"
    amv_bundle = external_root / "amv_calibration"
    amv_bundle.mkdir(parents=True)
    (amv_bundle / "source.yaml").write_text("source: maintainer-accepted\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(external_root))

    report = build_amv_calibration_status()

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is False
    assert report["blockers"] == ["command_response_trace_manifest_not_ready"]
    assert report["trace_manifest_report"]["manifest_status"] == "blocked-external-input"


def test_amv_calibration_status_admits_staged_manifest_and_asset(
    tmp_path: Path, monkeypatch
) -> None:
    """A staged bundle plus staged manifest is the future calibration-admission condition."""
    external_root = tmp_path / "external"
    amv_bundle = external_root / "amv_calibration"
    amv_bundle.mkdir(parents=True)
    (amv_bundle / "source.yaml").write_text("source: maintainer-accepted\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(external_root))
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")

    report = build_amv_calibration_status(staged_manifest)

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["blockers"] == []
    assert report["trace_manifest_report"]["manifest_status"] == "ready"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is True

"""Tests AMV calibration admission and command-response source status."""

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


def _write_manifest_with_status(tmp_path: Path, staging_status: str) -> Path:
    payload = yaml.safe_load(ISSUE_2415_MANIFEST.read_text(encoding="utf-8"))
    payload["traces"][0]["staging_status"] = staging_status
    if staging_status == "staged":
        payload["traces"][0]["blocker_issues"] = []
    manifest_path = tmp_path / "amv_trace_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _write_source_manifest(tmp_path: Path, *, source_type: str = "hardware_trace") -> Path:
    payload = {
        "schema_version": "amv_calibration_source_manifest.v1",
        "manifest_id": f"test_{source_type}_source",
        "issue": 1585,
        "source_status": "ready",
        "source_type": source_type,
        "source_uri": "https://example.com/amv/source-bundle",
        "license": "test-fixture-redistribution-approved",
        "license_status": "accepted",
        "claim_boundary": (
            "platform_class_proxy"
            if source_type == "platform_class_proxy"
            else "hardware-calibrated"
        ),
        "calibration_fields": [
            {
                "name": "max_linear_accel_m_s2",
                "support_status": "supported",
                "units": "m/s^2",
            }
        ],
    }
    manifest_path = tmp_path / f"amv_{source_type}_source_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest_path


def _stage_amv_calibration_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external_root = tmp_path / "external"
    amv_bundle = external_root / "amv_calibration"
    amv_bundle.mkdir(parents=True)
    (amv_bundle / "source.yaml").write_text("source: maintainer-accepted\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(external_root))


def test_amv_calibration_status_defaults_to_blocked_external_input() -> None:
    """Issue #2415 currently has no real AMV trace bundle and fails closed."""
    report = build_amv_calibration_status()

    assert report["schema"] == "amv_calibration_status.v1"
    assert report["issue"] == 2415
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert "amv_calibration_asset_missing" in report["blockers"]
    assert "command_response_trace_manifest_not_ready" in report["blockers"]
    assert "amv_calibration_source_manifest_not_ready" in report["blockers"]
    assert report["asset_status"]["status"] == "missing"
    assert report["trace_manifest_report"]["manifest_status"] == "blocked-external-input"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is False
    assert report["source_manifest_report"]["source_status"] == "blocked-external-input"
    assert report["source_manifest_report"]["hardware_calibration_claim_allowed"] is False


def test_amv_calibration_status_does_not_admit_asset_without_staged_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present asset is insufficient while issue #2415 trace manifest remains blocked."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)

    report = build_amv_calibration_status()

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is False
    assert report["blockers"] == [
        "command_response_trace_manifest_not_ready",
        "amv_calibration_source_manifest_not_ready",
    ]
    assert report["trace_manifest_report"]["manifest_status"] == "blocked-external-input"


def test_amv_calibration_status_rejects_staged_manifest_without_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged bundle and trace manifest still need issue #1585 source provenance."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")

    report = build_amv_calibration_status(staged_manifest)

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is False
    assert report["blockers"] == ["amv_calibration_source_manifest_not_ready"]
    assert report["trace_manifest_report"]["manifest_status"] == "ready"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is True


def test_amv_calibration_status_rejects_platform_class_proxy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A proxy source can be ready for proxy wording but not hardware-calibrated admission."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")
    source_manifest = _write_source_manifest(tmp_path, source_type="platform_class_proxy")

    report = build_amv_calibration_status(staged_manifest, source_manifest)

    assert report["ready"] is False
    assert report["blockers"] == ["amv_calibration_source_not_hardware_calibrated"]
    assert report["source_manifest_report"]["source_status"] == "ready"
    assert report["source_manifest_report"]["hardware_calibration_claim_allowed"] is False


def test_amv_calibration_status_admits_staged_manifest_asset_and_hardware_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Future admission requires staged trace, asset presence, and hardware source provenance."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")
    source_manifest = _write_source_manifest(tmp_path)

    report = build_amv_calibration_status(staged_manifest, source_manifest)

    assert report["asset_status"]["status"] == "available"
    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["blockers"] == []
    assert report["trace_manifest_report"]["manifest_status"] == "ready"
    assert report["trace_manifest_report"]["calibration_ingest_allowed"] is True
    assert report["source_manifest_report"]["source_status"] == "ready"
    assert report["source_manifest_report"]["hardware_calibration_claim_allowed"] is True


def test_amv_command_response_source_status_defaults_to_blocked_external_input() -> None:
    """Issue #2000 has no maintainer-accepted real command-response source staged."""
    report = build_amv_command_response_source_status()

    assert report["schema"] == "amv_command_response_source_status.v1"
    assert report["issue"] == 2000
    assert report["ready"] is False
    assert report["blocked_external_input"] is True
    assert "amv_command_response_asset_missing" in report["blockers"]
    assert "command_response_trace_not_staged" in report["blockers"]
    assert "trace_blocker_issues_open" in report["blockers"]
    assert "amv_calibration_source_manifest_not_ready" in report["blockers"]
    assert report["missing_signal_inventory"] == {}
    assert report["evidence_boundary"] == (
        "source_acquisition_contract_only_no_trace_ingest_no_calibration_run"
    )


def test_amv_command_response_source_status_accepts_hardware_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged trace plus hardware source provenance satisfies the #2000 source gate."""
    _stage_amv_calibration_asset(tmp_path, monkeypatch)
    staged_manifest = _write_manifest_with_status(tmp_path, "staged")
    source_manifest = _write_source_manifest(tmp_path)

    report = build_amv_command_response_source_status(staged_manifest, source_manifest)

    assert report["ready"] is True
    assert report["blocked_external_input"] is False
    assert report["blockers"] == []
    assert report["asset_status"]["status"] == "available"
    assert report["trace_blocker_issues"] == []
    assert report["source_manifest_report"]["hardware_calibration_claim_allowed"] is True

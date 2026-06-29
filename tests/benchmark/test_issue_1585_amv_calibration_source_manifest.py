"""Tests for the AMV calibration source-identification manifest (#1585)."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.amv_calibration_source_manifest import (
    AMV_CALIBRATION_SOURCE_EVIDENCE_BOUNDARY,
    SOURCE_STATUS_BLOCKED_EXTERNAL,
    SOURCE_STATUS_MISSING,
    SOURCE_STATUS_READY,
    check_amv_calibration_source_manifest,
    load_amv_calibration_source_manifest,
)

EXAMPLE_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_1585_amv_calibration_source_manifest.yaml"
)


def _ready_manifest() -> dict:
    """Return a synthetic, metadata-only accepted hardware-source manifest."""
    return {
        "schema_version": "amv_calibration_source_manifest.v1",
        "manifest_id": "synthetic_ready_source_contract",
        "issue": 1585,
        "source_status": "ready",
        "source_type": "official_spec",
        "source_uri": "https://example.com/amv/specification-v1.pdf",
        "license": "redistribution-approved-fixture",
        "license_status": "accepted",
        "claim_boundary": "hardware-calibrated",
        "calibration_fields": [
            {
                "name": "max_linear_accel_m_s2",
                "support_status": "supported",
                "units": "m/s^2",
            },
            {
                "name": "max_yaw_rate_rad_s",
                "support_status": "missing",
                "units": "rad/s",
                "reason": "Spec excerpt does not identify yaw-rate limits.",
            },
        ],
    }


def test_tracked_issue_manifest_is_blocked_external_input() -> None:
    """The checked-in #1585 manifest records the current blocked source state."""
    manifest = load_amv_calibration_source_manifest(EXAMPLE_MANIFEST_PATH)
    report = check_amv_calibration_source_manifest(manifest)

    assert report.source_status == SOURCE_STATUS_BLOCKED_EXTERNAL
    assert report.evidence_boundary == AMV_CALIBRATION_SOURCE_EVIDENCE_BOUNDARY
    assert not report.is_ready
    assert not report.hardware_calibration_claim_allowed
    assert report.source_uri == "https://github.com/ll7/robot_sf_ll7/issues/1585"


def test_ready_manifest_requires_source_uri_license_and_supported_field() -> None:
    """A complete metadata manifest can be ready without ingesting data."""
    report = check_amv_calibration_source_manifest(_ready_manifest())

    assert report.source_status == SOURCE_STATUS_READY
    assert report.is_ready
    assert report.hardware_calibration_claim_allowed
    assert report.blockers == ()


def test_missing_manifest_state_is_not_ready() -> None:
    """Missing source identification remains explicit and fail-closed."""
    manifest = _ready_manifest()
    manifest.update(
        {
            "source_status": "missing",
            "source_type": "unknown",
            "source_uri": None,
            "license": None,
            "license_status": "unknown",
        }
    )

    report = check_amv_calibration_source_manifest(manifest)

    assert report.source_status == SOURCE_STATUS_MISSING
    assert not report.is_ready
    assert not report.hardware_calibration_claim_allowed


def test_ready_source_cannot_use_tracking_issue_uri() -> None:
    """A tracking issue URL is a blocker, not a source artifact URI."""
    manifest = _ready_manifest()
    manifest["source_uri"] = "https://github.com/ll7/robot_sf_ll7/issues/1585"

    report = check_amv_calibration_source_manifest(manifest)

    assert report.source_status == SOURCE_STATUS_BLOCKED_EXTERNAL
    assert any("tracking issue" in blocker for blocker in report.blockers)


def test_proxy_source_ready_does_not_allow_hardware_calibration_claim() -> None:
    """Accepted platform-class proxy source stays outside hardware-calibrated claims."""
    manifest = _ready_manifest()
    manifest["source_type"] = "platform_class_proxy"
    manifest["claim_boundary"] = "platform_class_proxy"

    report = check_amv_calibration_source_manifest(manifest)

    assert report.source_status == SOURCE_STATUS_READY
    assert report.is_ready
    assert not report.hardware_calibration_claim_allowed


def test_unknown_calibration_field_blocks_ready_manifest() -> None:
    """Calibration fields must stay aligned with synthetic-actuation envelope names."""
    manifest = _ready_manifest()
    manifest["calibration_fields"][0]["name"] = "not_a_real_actuation_field"

    report = check_amv_calibration_source_manifest(manifest)

    assert report.source_status == SOURCE_STATUS_BLOCKED_EXTERNAL
    assert any("canonical synthetic-actuation" in blocker for blocker in report.blockers)

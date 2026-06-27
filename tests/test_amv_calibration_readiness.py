"""Focused synthetic tests for the AMV actuation calibration readiness/preflight checker (#1559).

These tests verify fail-closed behavior on placeholder/blocked external artifacts. They use only
synthetic in-test mappings plus the tracked #1586 skeleton config; no real calibration data, envelope
tuning, or campaign execution is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.amv_calibration_readiness import (
    assess_amv_calibration_readiness,
    assess_amv_calibration_readiness_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SKELETON_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_1586_calibrated_actuation_profile_skeleton_v0.yaml"
)


def _real_proxy_provenance() -> dict:
    """A fully-populated, non-placeholder proxy-source provenance block."""
    return {
        "source_id": "trl-escooter-performance-2020",
        "source_uri": "https://www.trl.co.uk/uploads/trl/documents/ACA104.pdf",
        "source_type": "platform-class-proxy",
        "profile_version": "v1",
        "measurement_date": "2026-06-01",
        "supported_actuation_fields": ["max_linear_accel_m_s2", "max_linear_decel_m_s2"],
        "units": {"max_linear_accel_m_s2": "m/s^2"},
        "claim_boundary": "platform-class-proxy-longitudinal-only",
    }


def _calibrated_profile(**overrides) -> dict:
    """A calibrated-labeled actuation profile with a real (non-placeholder) proxy provenance."""
    profile = {
        "name": "amv-actuation-proxy-v1",
        "claim_scope": "hardware-calibrated",
        "claim_boundary": "calibrated-amv-actuation",
        "max_linear_accel_m_s2": 1.5,
        "provenance": _real_proxy_provenance(),
    }
    profile.update(overrides)
    return profile


def test_skeleton_config_is_blocked_with_placeholder_provenance() -> None:
    """The tracked #1586 skeleton must fail closed: placeholder/pending provenance is not ready."""
    readiness = assess_amv_calibration_readiness_from_config(SKELETON_CONFIG)
    assert readiness.status == "blocked"
    assert not readiness.is_ready
    assert not readiness.paper_facing_allowed
    # Skeleton uses "pending-#1585", measurement_date "pending", "external-trace-collection-pending".
    assert "source_id" in readiness.placeholder_fields
    assert "measurement_date" in readiness.placeholder_fields
    assert readiness.blocking_reasons


def test_real_proxy_profile_is_ready_but_not_paper_facing() -> None:
    """A populated proxy source unblocks calibrated exploratory use but never paper-facing claims."""
    readiness = assess_amv_calibration_readiness(_calibrated_profile())
    assert readiness.status == "ready"
    assert readiness.is_ready
    assert readiness.source_class == "proxy"
    assert not readiness.paper_facing_allowed
    assert readiness.placeholder_fields == ()
    assert readiness.missing_provenance_fields == ()


def test_hardware_source_allows_paper_facing() -> None:
    """A hardware-trace source class is the only path to paper_facing_allowed=True."""
    provenance = _real_proxy_provenance()
    provenance["source_type"] = "hardware-measured-trace"
    provenance["source_id"] = "amv-bench-rig-2026"
    provenance["claim_boundary"] = "hardware-calibrated-amv"
    readiness = assess_amv_calibration_readiness(_calibrated_profile(provenance=provenance))
    assert readiness.status == "ready"
    assert readiness.source_class == "hardware"
    assert readiness.paper_facing_allowed


def test_synthetic_only_profile_is_blocked() -> None:
    """A synthetic diagnostic profile is not a calibration candidate and must be blocked."""
    synthetic = {
        "name": "amv-actuation-stress-v0",
        "claim_scope": "synthetic-only",
        "claim_boundary": "diagnostic-only",
    }
    readiness = assess_amv_calibration_readiness(synthetic)
    assert readiness.status == "blocked"
    assert not readiness.looks_calibrated
    assert not readiness.paper_facing_allowed
    assert any("synthetic-only" in reason for reason in readiness.blocking_reasons)


def test_conflation_is_blocked_via_structural_validation() -> None:
    """A synthetic-scope profile carrying calibrated markers must fail closed (conflation)."""
    conflated = {
        "name": "amv-actuation-calibrated-sneaky",
        "claim_scope": "synthetic-only",
        "claim_boundary": "diagnostic-only",
    }
    readiness = assess_amv_calibration_readiness(conflated)
    assert readiness.status == "blocked"
    assert any("structural validation failed" in r for r in readiness.blocking_reasons)


def test_missing_provenance_fields_are_blocked() -> None:
    """Dropping required provenance fields blocks readiness and lists the gaps."""
    profile = _calibrated_profile()
    del profile["provenance"]["measurement_date"]
    del profile["provenance"]["units"]
    readiness = assess_amv_calibration_readiness(profile)
    assert readiness.status == "blocked"
    assert "measurement_date" in readiness.missing_provenance_fields
    assert "units" in readiness.missing_provenance_fields


def test_source_uri_tracking_issue_is_blocked() -> None:
    """A source_uri pointing at a GitHub issue is a blocked external artifact, not a source."""
    provenance = _real_proxy_provenance()
    provenance["source_uri"] = "https://github.com/ll7/robot_sf_ll7/issues/1585"
    readiness = assess_amv_calibration_readiness(_calibrated_profile(provenance=provenance))
    assert readiness.status == "blocked"
    assert any("tracking issue" in reason for reason in readiness.blocking_reasons)


@pytest.mark.parametrize("bad_input", [None, "not-a-mapping", 42, []])
def test_non_mapping_input_is_blocked(bad_input) -> None:
    """Any non-mapping profile fails closed instead of raising."""
    readiness = assess_amv_calibration_readiness(bad_input)
    assert readiness.status == "blocked"
    assert not readiness.is_ready


def test_missing_config_file_is_blocked(tmp_path) -> None:
    """A missing config path fails closed rather than raising."""
    readiness = assess_amv_calibration_readiness_from_config(tmp_path / "does_not_exist.yaml")
    assert readiness.status == "blocked"
    assert any("not found" in reason for reason in readiness.blocking_reasons)


def test_malformed_yaml_config_is_blocked(tmp_path) -> None:
    """A config that is not valid YAML fails closed rather than raising."""
    bad = tmp_path / "broken.yaml"
    bad.write_text("synthetic_actuation_profile: [unterminated\n", encoding="utf-8")
    readiness = assess_amv_calibration_readiness_from_config(bad)
    assert readiness.status == "blocked"
    assert any("not valid YAML" in reason for reason in readiness.blocking_reasons)


def test_nested_placeholder_in_provenance_is_blocked() -> None:
    """A placeholder hidden in a nested provenance value (e.g. units) still fails closed."""
    provenance = _real_proxy_provenance()
    provenance["units"] = {"max_linear_accel_m_s2": "pending"}
    readiness = assess_amv_calibration_readiness(_calibrated_profile(provenance=provenance))
    assert readiness.status == "blocked"
    assert "units" in readiness.placeholder_fields


def test_to_dict_is_json_serializable() -> None:
    """The readiness outcome serializes cleanly for diagnostic reports."""
    import json

    readiness = assess_amv_calibration_readiness(_calibrated_profile())
    payload = json.dumps(readiness.to_dict())
    assert "status" in payload

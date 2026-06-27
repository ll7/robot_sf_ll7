"""Tests for the AMV command-response trace staging-manifest preflight (issue #2415)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.synthetic_actuation import actuation_variability_fields
from robot_sf.research.amv_command_response_trace_manifest import (
    AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY,
    AMV_TRACE_MANIFEST_SCHEMA_VERSION,
    MANIFEST_STATUS_BLOCKED_EXTERNAL,
    MANIFEST_STATUS_INVALID,
    MANIFEST_STATUS_READY,
    AmvTraceManifestError,
    check_amv_trace_manifest,
    load_amv_trace_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "research" / "amv_command_response_trace_manifest_issue_2415.yaml"
)

# Canonical synthetic-actuation envelope fields used as the allowed calibration-target vocabulary.
ALLOWED_TARGETS = set(actuation_variability_fields())


def _trace(staging_status: str = "blocked-external-input", **overrides: object) -> dict:
    """Return one valid trace staging entry, overridable per-test."""
    trace = {
        "trace_id": "amv_command_response_primary",
        "asset_id": "amv-calibration",
        "title": "Real AMV command-response actuation trace bundle",
        "staging_status": staging_status,
        "redistribution": "none",
        "blocker_issues": [2000, 1585],
        "provenance": {
            "source_url": "https://github.com/ll7/robot_sf_ll7/issues/2000",
            "license": "Access depends on source; private traces stay local",
            "license_status": "unknown",
            "citation": "Real AMV command-response trace source to be identified via #2000/#1585.",
        },
        "command_channels": ["linear_velocity_command_mps", "yaw_rate_command_rad_s"],
        "response_channels": ["measured_linear_velocity_mps", "measured_yaw_rate_rad_s"],
        "timing_fields": ["timestamp_s", "control_latency_s", "sample_rate_hz"],
        "calibration_targets": ["max_linear_accel_m_s2", "latency_mode"],
    }
    trace.update(overrides)
    return trace


def _manifest(traces: list[dict] | None = None) -> dict:
    """Return a minimal valid staging manifest."""
    return {
        "schema_version": AMV_TRACE_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_manifest",
        "issue": 2415,
        "claim_boundary": "staging manifest metadata only",
        "calibration_asset_id": "amv-calibration",
        "synthetic_envelope_module": "robot_sf/benchmark/synthetic_actuation.py",
        "traces": traces if traces is not None else [_trace()],
    }


def test_blocked_external_input_manifest_is_blocked_not_ready() -> None:
    """A well-formed manifest with no staged trace reports blocked-external-input."""
    report = check_amv_trace_manifest(_manifest(), allowed_calibration_targets=ALLOWED_TARGETS)
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL
    assert report.calibration_ingest_allowed is False
    assert report.calibration_ready_traces == []
    assert report.evidence_boundary == AMV_TRACE_MANIFEST_EVIDENCE_BOUNDARY
    assert report.blockers == []


def test_staged_clean_trace_unlocks_calibration() -> None:
    """A staged, blocker-free trace makes the manifest ready and calibration-ingest-allowed."""
    report = check_amv_trace_manifest(
        _manifest([_trace(staging_status="staged")]),
        allowed_calibration_targets=ALLOWED_TARGETS,
    )
    assert report.manifest_status == MANIFEST_STATUS_READY
    assert report.calibration_ingest_allowed is True
    assert report.calibration_ready_traces == ["amv_command_response_primary"]


def test_missing_trace_is_not_calibration_ready() -> None:
    """A 'missing' trace is well-formed but not calibration-ready."""
    report = check_amv_trace_manifest(
        _manifest([_trace(staging_status="missing")]),
        allowed_calibration_targets=ALLOWED_TARGETS,
    )
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL
    assert report.calibration_ready_traces == []
    trace = report.traces[0]
    assert trace.effective_staged is False
    assert trace.blockers == []


def test_unknown_calibration_target_fails_closed() -> None:
    """A calibration target outside the envelope vocabulary is an invalid-manifest blocker."""
    report = check_amv_trace_manifest(
        _manifest([_trace(calibration_targets=["latency_mode", "not_a_real_field"])]),
        allowed_calibration_targets=ALLOWED_TARGETS,
    )
    assert report.manifest_status == MANIFEST_STATUS_INVALID
    assert report.traces[0].unknown_calibration_targets == ["not_a_real_field"]
    assert any("not_a_real_field" in blocker for blocker in report.blockers)


def test_calibration_target_check_skipped_without_vocabulary() -> None:
    """Without a vocabulary, unknown calibration targets are tolerated (no drift check)."""
    report = check_amv_trace_manifest(
        _manifest([_trace(calibration_targets=["anything_goes"])]),
        allowed_calibration_targets=None,
    )
    assert report.traces[0].unknown_calibration_targets == []
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL


def test_blocked_trace_without_blocker_issue_is_invalid() -> None:
    """A blocked-external-input trace must name at least one blocker issue."""
    report = check_amv_trace_manifest(
        _manifest([_trace(blocker_issues=[])]),
        allowed_calibration_targets=ALLOWED_TARGETS,
    )
    assert report.manifest_status == MANIFEST_STATUS_INVALID
    assert any("blocker_issues" in blocker for blocker in report.blockers)


def test_declared_staged_but_live_missing_fails_closed() -> None:
    """A trace declared 'staged' whose live probe is missing fails closed."""
    report = check_amv_trace_manifest(
        _manifest([_trace(staging_status="staged")]),
        allowed_calibration_targets=ALLOWED_TARGETS,
        live_staging_status={"amv-calibration": "missing"},
    )
    assert report.manifest_status == MANIFEST_STATUS_INVALID
    trace = report.traces[0]
    assert trace.effective_staged is False
    assert trace.calibration_ready is False
    assert any("live probe" in blocker for blocker in report.blockers)


def test_declared_staged_and_live_available_is_ready() -> None:
    """A trace declared 'staged' and live-available reconciles to calibration-ready."""
    report = check_amv_trace_manifest(
        _manifest([_trace(staging_status="staged")]),
        allowed_calibration_targets=ALLOWED_TARGETS,
        live_staging_status={"amv-calibration": "available"},
    )
    assert report.manifest_status == MANIFEST_STATUS_READY
    assert report.traces[0].live_staging_status == "available"
    assert report.calibration_ready_traces == ["amv_command_response_primary"]


def test_schema_violation_raises() -> None:
    """An invalid payload (bad enum) raises a schema error."""
    bad = _manifest([_trace(staging_status="totally-invalid")])
    with pytest.raises(AmvTraceManifestError):
        check_amv_trace_manifest(bad, allowed_calibration_targets=ALLOWED_TARGETS)


def test_missing_command_channels_raises() -> None:
    """A trace with no command channels violates the schema (minItems)."""
    bad = _manifest([_trace(command_channels=[])])
    with pytest.raises(AmvTraceManifestError):
        check_amv_trace_manifest(bad, allowed_calibration_targets=ALLOWED_TARGETS)


def test_missing_response_channels_raises() -> None:
    """A trace with no response channels violates the schema (minItems)."""
    bad = _manifest([_trace(response_channels=[])])
    with pytest.raises(AmvTraceManifestError):
        check_amv_trace_manifest(bad, allowed_calibration_targets=ALLOWED_TARGETS)


def test_missing_timing_fields_raises() -> None:
    """A trace with no timing fields violates the schema (minItems)."""
    bad = _manifest([_trace(timing_fields=[])])
    with pytest.raises(AmvTraceManifestError):
        check_amv_trace_manifest(bad, allowed_calibration_targets=ALLOWED_TARGETS)


def test_missing_required_provenance_field_raises() -> None:
    """A trace missing a required provenance field raises a schema error."""
    bad = _manifest()
    del bad["traces"][0]["provenance"]["citation"]
    with pytest.raises(AmvTraceManifestError):
        check_amv_trace_manifest(bad, allowed_calibration_targets=ALLOWED_TARGETS)


def test_report_to_dict_is_json_serializable() -> None:
    """The report serializes to JSON (CLI contract)."""
    report = check_amv_trace_manifest(_manifest(), allowed_calibration_targets=ALLOWED_TARGETS)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["schema_version"] == AMV_TRACE_MANIFEST_SCHEMA_VERSION
    assert payload["traces"][0]["trace_id"] == "amv_command_response_primary"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent manifest path raises an actionable error."""
    with pytest.raises(AmvTraceManifestError):
        load_amv_trace_manifest(tmp_path / "nope.yaml")


def test_load_roundtrip(tmp_path: Path) -> None:
    """A manifest written to disk loads and validates."""
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")
    loaded = load_amv_trace_manifest(path)
    assert loaded["manifest_id"] == "test_manifest"


def test_repo_example_manifest_is_valid_and_blocked() -> None:
    """The shipped #2415 example manifest validates and is blocked-external-input today."""
    manifest = load_amv_trace_manifest(EXAMPLE_MANIFEST_PATH)
    report = check_amv_trace_manifest(manifest, allowed_calibration_targets=ALLOWED_TARGETS)
    assert report.manifest_status == MANIFEST_STATUS_BLOCKED_EXTERNAL
    assert report.calibration_ingest_allowed is False
    assert report.blockers == []
    assert {t.trace_id for t in report.traces} == {"amv_command_response_primary"}


def test_repo_example_calibration_targets_match_live_envelope_vocabulary() -> None:
    """Every example calibration target is a real synthetic-actuation envelope field.

    This guards against the example manifest drifting from the synthetic actuation envelope it would
    calibrate; it imports the live vocabulary rather than the test constant.
    """
    manifest = load_amv_trace_manifest(EXAMPLE_MANIFEST_PATH)
    report = check_amv_trace_manifest(
        manifest, allowed_calibration_targets=set(actuation_variability_fields())
    )
    for trace in report.traces:
        assert trace.unknown_calibration_targets == []


def test_example_manifest_matches_repo_path() -> None:
    """The example manifest ships at the path the CLI defaults to."""
    assert EXAMPLE_MANIFEST_PATH.is_file()
    payload = yaml.safe_load(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == AMV_TRACE_MANIFEST_SCHEMA_VERSION
    assert payload["calibration_asset_id"] == "amv-calibration"


def test_multiple_traces_one_staged_is_ready() -> None:
    """With several traces, one staged-and-clean entry unlocks the calibration."""
    traces = [
        _trace(trace_id="amv_primary", staging_status="staged"),
        _trace(
            trace_id="amv_secondary",
            staging_status="blocked-external-input",
            blocker_issues=[2000],
        ),
    ]
    report = check_amv_trace_manifest(
        _manifest(traces), allowed_calibration_targets=ALLOWED_TARGETS
    )
    assert report.manifest_status == MANIFEST_STATUS_READY
    assert report.calibration_ready_traces == ["amv_primary"]


def test_copy_independence() -> None:
    """The checker does not mutate the input manifest mapping."""
    manifest = _manifest()
    snapshot = copy.deepcopy(manifest)
    check_amv_trace_manifest(manifest, allowed_calibration_targets=ALLOWED_TARGETS)
    assert manifest == snapshot

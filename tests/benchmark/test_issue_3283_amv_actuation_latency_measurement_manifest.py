"""Tests for the AMV actuation-latency measurement-manifest checker (issue #3283)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.actuation_latency_measurement_manifest import (
    AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_VERSION,
    CONTRACT_STATUS_BLOCKED,
    CONTRACT_STATUS_READY,
    CONTRACT_STATUS_SYNTHETIC_ONLY,
    MEASUREMENT_INTAKE_EVIDENCE_BOUNDARY,
    PROPOSED_LATENCY_PROFILE_FIELDS,
    PROPOSED_RIDER_COUPLING_PROFILE_FIELDS,
    REQUIRED_MEASUREMENT_QUANTITIES,
    AmvActuationLatencyManifestError,
    check_amv_actuation_latency_measurement_manifest,
    load_amv_actuation_latency_measurement_manifest,
)

EXAMPLE_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmarks"
    / "issue_3283_amv_actuation_latency_measurement_manifest_example.yaml"
)


def _sensor_channels() -> list[dict]:
    """Return one channel per canonical measurement quantity."""
    units = {
        "command_issuance": "timestamp_s",
        "mechanical_response": "rad/s",
        "yaw_response": "rad/s",
        "braking_latency": "bar",
        "acceleration_latency": "m/s^2",
        "rider_load_condition": "kg",
        "rider_response": "N",
    }
    return [
        {
            "quantity": quantity,
            "sensor": f"{quantity}_sensor",
            "units": units[quantity],
            "sampling_rate_hz": 100.0,
        }
        for quantity in REQUIRED_MEASUREMENT_QUANTITIES
    ]


def _proposed_fields(value_status: str) -> list[dict]:
    """Return the canonical proposed profile fields at a given value_status."""
    fields = []
    for name, unit in PROPOSED_LATENCY_PROFILE_FIELDS.items():
        fields.append(
            {
                "name": name,
                "parameter_class": "latency",
                "units": unit,
                "value_status": value_status,
            }
        )
    for name, unit in PROPOSED_RIDER_COUPLING_PROFILE_FIELDS.items():
        fields.append(
            {
                "name": name,
                "parameter_class": "rider_coupling",
                "units": unit,
                "value_status": value_status,
            }
        )
    return fields


def _manifest(measurement_status: str, value_status: str) -> dict:
    """Return a structurally complete manifest for the given lifecycle state."""
    return {
        "schema_version": AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"intake_{measurement_status}",
        "measurement_status": measurement_status,
        "description": "Planned AMV actuation latency and rider-coupling measurement.",
        "synchronization": {
            "method": "shared_hardware_trigger_with_ptp",
            "reference_clock": "ptp_grandmaster",
            "max_skew_ms": 2.0,
        },
        "sensor_channels": _sensor_channels(),
        "proposed_profile_fields": _proposed_fields(value_status),
        "synthetic_separation": {"separation": "enforced"},
    }


def _measured_manifest() -> dict:
    """Return a complete measured manifest with accepted provenance."""
    manifest = _manifest("measured", "measured")
    manifest["provenance"] = {
        "source_id": "amv-bench-2026-001",
        "source_uri": "wandb://robot-sf/amv-actuation/amv-bench-2026-001",
        "source_type": "on-vehicle-bench-measurement",
        "measurement_date": "2026-06-27",
        "units": {"command_to_motion_latency_s": "s"},
    }
    return manifest


def test_complete_measured_manifest_is_ready_and_claim_allowed() -> None:
    """A complete measured manifest with provenance is ready and may assert measured values."""
    report = check_amv_actuation_latency_measurement_manifest(_measured_manifest())

    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.measured_value_claim_allowed is True
    assert report.evidence_boundary == MEASUREMENT_INTAKE_EVIDENCE_BOUNDARY
    assert report.blockers == []
    assert report.missing_latency_quantities == []
    assert report.missing_rider_coupling_quantities == []


def test_blocked_external_input_plan_is_blocked_without_measured_claim() -> None:
    """A complete blocked-external-input plan is a valid plan but stays blocked, no claim."""
    report = check_amv_actuation_latency_measurement_manifest(
        _manifest("blocked-external-input", "pending")
    )

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.measured_value_claim_allowed is False
    # The plan itself is complete: no channel/sync/separation/field blockers.
    assert report.blockers == []
    assert report.provenance_blockers == []


def test_synthetic_only_manifest_is_terminal_without_measured_claim() -> None:
    """A synthetic-only manifest is its own terminal state and never allows a measured claim."""
    report = check_amv_actuation_latency_measurement_manifest(
        _manifest("synthetic-only", "synthetic-placeholder")
    )

    assert report.contract_status == CONTRACT_STATUS_SYNTHETIC_ONLY
    assert report.measured_value_claim_allowed is False
    assert report.blockers == []


def test_measured_manifest_without_provenance_is_blocked() -> None:
    """A measured manifest missing provenance is blocked and cannot claim a value."""
    manifest = _manifest("measured", "measured")  # no provenance block

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.measured_value_claim_allowed is False
    assert any("provenance" in blocker for blocker in report.provenance_blockers)


def test_synthetic_only_with_measured_source_is_rejected_as_conflation() -> None:
    """A synthetic-only manifest declaring a measured source conflates the boundary -> blocked."""
    manifest = _manifest("synthetic-only", "synthetic-placeholder")
    manifest["provenance"] = {"source_uri": "wandb://robot-sf/amv/real-trace-001"}

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("separate from measured values" in b for b in report.provenance_blockers)


def test_value_status_must_match_measurement_status() -> None:
    """Measured value_status under a blocked plan is a proposed-field blocker."""
    manifest = _manifest("blocked-external-input", "measured")

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("requires 'pending'" in b for b in report.proposed_field_blockers)


def test_duplicate_proposed_field_name_is_blocker() -> None:
    """A duplicate proposed field name (case-insensitive) fails closed as a blocker."""
    manifest = _measured_manifest()
    duplicate = dict(manifest["proposed_profile_fields"][0])
    duplicate["name"] = duplicate["name"].upper()  # case variant of an existing field
    manifest["proposed_profile_fields"] = manifest["proposed_profile_fields"] + [duplicate]

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("duplicate name" in b for b in report.proposed_field_blockers)


def test_proposed_field_parameter_class_must_match_canonical() -> None:
    """A proposed field whose parameter_class disagrees with the canonical map is blocked."""
    manifest = _measured_manifest()
    # Mislabel a latency field as rider_coupling.
    fields = [dict(field) for field in manifest["proposed_profile_fields"]]
    latency_name = next(iter(PROPOSED_LATENCY_PROFILE_FIELDS))
    for field in fields:
        if field["name"] == latency_name:
            field["parameter_class"] = "rider_coupling"
    manifest["proposed_profile_fields"] = fields

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("parameter_class" in b for b in report.proposed_field_blockers)


def test_missing_latency_channel_is_blocker() -> None:
    """Dropping a required latency channel surfaces a missing-latency blocker."""
    manifest = _measured_manifest()
    manifest["sensor_channels"] = [
        ch for ch in manifest["sensor_channels"] if ch["quantity"] != "yaw_response"
    ]

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("yaw_response" in b for b in report.missing_latency_quantities)


def test_separation_not_enforced_is_blocker() -> None:
    """A manifest that does not enforce synthetic separation is blocked."""
    manifest = _measured_manifest()
    manifest["synthetic_separation"] = {"separation": "not-enforced"}

    report = check_amv_actuation_latency_measurement_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.separation_blockers


def test_schema_violation_raises() -> None:
    """An unknown measurement_status is rejected at the schema layer."""
    manifest = _manifest("blocked-external-input", "pending")
    manifest["measurement_status"] = "not-a-status"

    with pytest.raises(AmvActuationLatencyManifestError):
        check_amv_actuation_latency_measurement_manifest(manifest)


def test_example_manifest_loads_and_is_blocked_external_input() -> None:
    """The shipped example manifest is a valid, blocked-external-input intake plan."""
    manifest = load_amv_actuation_latency_measurement_manifest(EXAMPLE_MANIFEST_PATH)
    report = check_amv_actuation_latency_measurement_manifest(
        manifest, source=EXAMPLE_MANIFEST_PATH
    )

    assert report.measurement_status == "blocked-external-input"
    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.measured_value_claim_allowed is False
    # Example ships a complete plan: only the external data is missing.
    assert report.blockers == []


def test_example_manifest_yaml_is_well_formed() -> None:
    """The example YAML parses to a mapping (guards against accidental corruption)."""
    payload = yaml.safe_load(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["schema_version"] == AMV_ACTUATION_LATENCY_MEASUREMENT_MANIFEST_SCHEMA_VERSION


def test_to_dict_is_json_safe_and_does_not_mutate_input() -> None:
    """The report serializes to plain types and the checker does not mutate its input."""
    manifest = _measured_manifest()
    before = copy.deepcopy(manifest)

    report = check_amv_actuation_latency_measurement_manifest(manifest)
    payload = report.to_dict()

    assert manifest == before
    assert payload["contract_status"] == CONTRACT_STATUS_READY
    assert isinstance(payload["declared_quantities"], list)

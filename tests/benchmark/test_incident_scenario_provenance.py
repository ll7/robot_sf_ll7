"""Tests for the incident-to-scenario provenance contract (issue #7888)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.collision.incident_scenario_provenance import (
    INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE,
    IncidentScenarioProvenanceError,
    reconcile_incident_scenario_provenance,
    validate_incident_scenario_provenance,
)

FACT = {
    "fact_id": "fact-1",
    "statement": "A pedestrian crossed the shared space from the right side.",
}
HYPOTHESIS = {
    "hypothesis_id": "hyp-1",
    "statement": "The pedestrian initiated the crossing after the robot entered the zone.",
}
ASSUMPTION = {
    "assumption_id": "asm-1",
    "statement": "Pedestrian motion follows the repository Social Force model.",
}
ACTOR = {"actor_id": "actor-1", "role": "pedestrian_initiator"}
PARAMETER_MAPPING = {
    "parameter": "crossing_approach_speed_mps",
    "source_field": "incident.speed_estimate",
    "transformation": "speed_to_crossing_approach",
    "unit": "m/s",
    "status": "mapped",
    "confidence": "medium",
}


def _valid_record(**overrides: object) -> dict:
    record: dict = {
        "schema_version": "incident_scenario_provenance.v1",
        "record_id": "record-1",
        "source": {
            "identity": "incident-report-2026-0001",
            "digest_sha256": "0123456789abcdef" * 4,
            "observed_facts": [FACT],
        },
        "extraction": {
            "status": "human_corrected",
            "verification_record": {
                "reviewer": "reviewer-1",
                "method": "human review of extracted trace",
                "at": "2026-08-25T00:00:00Z",
            },
            "actors": [ACTOR],
            "hypotheses": [HYPOTHESIS],
            "simulator_assumptions": [ASSUMPTION],
        },
        "scenario_parameters": [PARAMETER_MAPPING],
        "execution": {
            "claimed": True,
            "scenario_config_digest_sha256": "fedcba9876543210" * 4,
            "seed": 42,
            "software_commit": "0123456789abcdef0123456789abcdef01234567",
            "replay_identity": "replay-1",
            "observed_outcome": {"kind": "near_miss", "detail": "Minimum distance 0.4 m."},
        },
        "normative_fault": "not_assessed",
        "admission": "admitted",
    }
    record.update(overrides)
    return record


def test_schema_file_exists_and_is_versioned() -> None:
    assert INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE.exists()
    assert INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE.name == ("incident_scenario_provenance.v1.json")


def test_valid_record_round_trips() -> None:
    record = _valid_record()
    validated = validate_incident_scenario_provenance(record)
    assert validated["record_id"] == "record-1"
    assert validated["normative_fault"] == "not_assessed"
    assert reconcile_incident_scenario_provenance(record) == []


def test_every_transformation_step_is_traceable() -> None:
    record = _valid_record()
    assert record["source"]["observed_facts"] == [FACT]
    assert record["extraction"]["hypotheses"] == [HYPOTHESIS]
    assert record["extraction"]["simulator_assumptions"] == [ASSUMPTION]
    mapping = record["scenario_parameters"][0]
    assert mapping["source_field"] == "incident.speed_estimate"
    assert mapping["transformation"] == "speed_to_crossing_approach"
    assert mapping["unit"] == "m/s"
    assert mapping["status"] == "mapped"
    # The fixture resolves to an immutable configuration digest and seed.
    assert record["execution"]["scenario_config_digest_sha256"] == "fedcba9876543210" * 4
    assert record["execution"]["seed"] == 42
    # A synthetic fixture does not need to run a campaign.
    assert record["execution"]["claimed"] is True


def test_unknown_top_level_field_fails_closed() -> None:
    record = _valid_record(unexpected_field="nope")
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert "schema" in violations[0]
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)


def test_missing_actor_identity_fails_closed() -> None:
    record = _valid_record()
    record["extraction"]["actors"] = [{"actor_id": "", "role": "unknown"}]
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert "schema" in violations[0]


def test_unsupported_parameter_mapping_fails_closed() -> None:
    record = _valid_record()
    record["scenario_parameters"] = [
        {**PARAMETER_MAPPING, "status": "unsupported", "transformation": ""}
    ]
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert "schema" in violations[0]


def test_unverified_generated_content_cannot_be_verified_without_record() -> None:
    record = _valid_record()
    record["extraction"] = {
        "status": "verified",
        "actors": [ACTOR],
        "hypotheses": [HYPOTHESIS],
        "simulator_assumptions": [ASSUMPTION],
    }
    violations = reconcile_incident_scenario_provenance(record)
    assert any("verification_record" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)


def test_unsupported_parameter_requires_unavailable_confidence() -> None:
    record = _valid_record(
        scenario_parameters=[{**PARAMETER_MAPPING, "status": "unsupported", "confidence": "low"}],
        admission="unsupported",
    )
    violations = reconcile_incident_scenario_provenance(record)
    assert any("confidence='unavailable'" in violation for violation in violations)


def test_admitted_record_rejects_unsupported_mapping() -> None:
    record = _valid_record(
        scenario_parameters=[
            {**PARAMETER_MAPPING, "status": "unsupported", "confidence": "unavailable"}
        ],
    )
    violations = reconcile_incident_scenario_provenance(record)
    assert any("admission='admitted'" in violation for violation in violations)


def test_admitted_record_rejects_rejected_extraction() -> None:
    record = _valid_record(
        extraction={
            "status": "rejected",
            "actors": [ACTOR],
            "hypotheses": [HYPOTHESIS],
            "simulator_assumptions": [ASSUMPTION],
        },
    )
    violations = reconcile_incident_scenario_provenance(record)
    assert any("extraction.status='rejected'" in violation for violation in violations)


def test_unverified_status_is_accepted_without_record() -> None:
    record = _valid_record()
    record["extraction"] = {
        "status": "unverified",
        "actors": [ACTOR],
        "hypotheses": [HYPOTHESIS],
        "simulator_assumptions": [ASSUMPTION],
    }
    assert reconcile_incident_scenario_provenance(record) == []


def test_missing_scenario_digest_fails_closed_when_execution_claimed() -> None:
    record = _valid_record()
    record["execution"]["scenario_config_digest_sha256"] = None
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    # The schema rejects the missing digest before the semantic rule runs.
    assert any("schema" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)


def test_normative_fault_cannot_be_anything_but_not_assessed() -> None:
    record = _valid_record(normative_fault="assessed")
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    # The schema const rejects the value; the semantic guard is a second net.
    assert any("not_assessed" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)


def test_rejected_record_stays_outside_admitted_denominator() -> None:
    record = _valid_record(admission="rejected")
    assert reconcile_incident_scenario_provenance(record) == []
    assert record["admission"] == "rejected"


def test_semantic_execution_rule_requires_replay_identity() -> None:
    # The schema minLength rejects an empty replay_identity before the
    # semantic rule runs; the rule is a second net for the claimed contract.
    record = _valid_record()
    record["execution"]["replay_identity"] = ""
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert any("schema" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)

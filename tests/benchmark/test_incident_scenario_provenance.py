"""Tests for the incident-to-scenario provenance contract (issue #7888)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robot_sf.benchmark.collision.incident_scenario_provenance import (
    INCIDENT_SCENARIO_PROVENANCE_SCHEMA_FILE,
    IncidentScenarioProvenanceError,
    reconcile_incident_scenario_provenance,
    validate_incident_scenario_provenance,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

SCENARIO_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures/incident_scenario_provenance/scenario.yaml"
)
SCENARIO_FIXTURE_SHA256 = "355d6b60b355a5c5eb902120a1f6dff82a54742e4d5aeaf0dc94a9fd9b822cdd"
SOURCE_DESCRIPTION = "A pedestrian crossed the shared space from the right side at 0.8 m/s."

FACT = {
    "fact_id": "fact-1",
    "statement": SOURCE_DESCRIPTION,
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
    "parameter": "single_pedestrians[0].speed_m_s",
    "source_kind": "observed_fact",
    "source_id": "fact-1",
    "source_field": "statement",
    "transformation": "copy stated crossing speed into the pedestrian speed field",
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
            "digest_sha256": hashlib.sha256(SOURCE_DESCRIPTION.encode()).hexdigest(),
            "observed_facts": [dict(FACT)],
        },
        "extraction": {
            "status": "human_corrected",
            "verification_record": {
                "reviewer": "reviewer-1",
                "method": "human review of extracted trace",
                "at": "2026-08-25T00:00:00Z",
            },
            "actors": [dict(ACTOR)],
            "hypotheses": [dict(HYPOTHESIS)],
            "simulator_assumptions": [dict(ASSUMPTION)],
        },
        "scenario_parameters": [dict(PARAMETER_MAPPING)],
        "scenario_config": {
            "identity": "tests/fixtures/incident_scenario_provenance/scenario.yaml",
            "digest_sha256": SCENARIO_FIXTURE_SHA256,
            "seed": 42,
        },
        "execution": {
            "claimed": False,
        },
        "normative_fault": "not_assessed",
        "admission": "admitted",
    }
    record.update(overrides)
    return record


def _claim_execution(record: dict) -> None:
    record["execution"] = {
        "claimed": True,
        "software_commit": "0123456789abcdef0123456789abcdef01234567",
        "replay_identity": "synthetic-replay-1",
    }


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
    assert mapping["source_kind"] == "observed_fact"
    assert mapping["source_id"] == "fact-1"
    assert mapping["source_field"] == "statement"
    assert mapping["transformation"] == (
        "copy stated crossing speed into the pedestrian speed field"
    )
    assert mapping["unit"] == "m/s"
    assert mapping["status"] == "mapped"
    # The fixture resolves to an immutable configuration digest and seed.
    assert record["scenario_config"]["digest_sha256"] == SCENARIO_FIXTURE_SHA256
    assert record["scenario_config"]["seed"] == 42
    # A synthetic fixture does not need to run a campaign.
    assert record["execution"]["claimed"] is False


def test_fixture_binds_source_and_loadable_scenario_bytes() -> None:
    record = _valid_record()
    assert (
        hashlib.sha256(SOURCE_DESCRIPTION.encode()).hexdigest() == record["source"]["digest_sha256"]
    )
    assert (
        hashlib.sha256(SCENARIO_FIXTURE.read_bytes()).hexdigest()
        == record["scenario_config"]["digest_sha256"]
    )
    scenarios = load_scenarios(SCENARIO_FIXTURE)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["name"] == "incident_provenance_synthetic_crossing"
    assert scenario["single_pedestrians"][0]["speed_m_s"] == 0.8
    assert scenario["seeds"] == [record["scenario_config"]["seed"]]
    config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_FIXTURE)
    assert config.map_pool is not None


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
    assert any("requires extraction.status" in violation for violation in violations)


def test_unverified_status_is_accepted_without_record() -> None:
    record = _valid_record()
    record["extraction"] = {
        "status": "unverified",
        "actors": [ACTOR],
        "hypotheses": [HYPOTHESIS],
        "simulator_assumptions": [ASSUMPTION],
    }
    record["admission"] = "ambiguous"
    assert reconcile_incident_scenario_provenance(record) == []


def test_unverified_record_cannot_be_admitted() -> None:
    record = _valid_record()
    record["extraction"] = {
        "status": "unverified",
        "actors": [ACTOR],
        "hypotheses": [HYPOTHESIS],
        "simulator_assumptions": [ASSUMPTION],
    }
    violations = reconcile_incident_scenario_provenance(record)
    assert any("requires extraction.status" in violation for violation in violations)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("observed_facts", [], "source.observed_facts"),
        ("actors", [], "extraction.actors"),
        ("scenario_parameters", [], "scenario_parameters"),
    ],
)
def test_admitted_record_requires_nonempty_provenance(
    field: str, value: list, expected: str
) -> None:
    record = _valid_record()
    if field == "observed_facts":
        record["source"][field] = value
    elif field == "actors":
        record["extraction"][field] = value
    else:
        record[field] = value
    violations = reconcile_incident_scenario_provenance(record)
    assert any(expected in violation for violation in violations)


def test_parameter_mapping_source_reference_must_resolve() -> None:
    record = _valid_record()
    record["scenario_parameters"][0]["source_id"] = "missing-fact"
    violations = reconcile_incident_scenario_provenance(record)
    assert any("must resolve to exactly one" in violation for violation in violations)


def test_parameter_mapping_source_reference_must_be_unambiguous() -> None:
    record = _valid_record()
    record["source"]["observed_facts"].append(dict(FACT))
    violations = reconcile_incident_scenario_provenance(record)
    assert any("must resolve to exactly one" in violation for violation in violations)


def test_parameter_mapping_source_field_must_exist() -> None:
    record = _valid_record()
    record["scenario_parameters"][0]["source_field"] = "missing_field"
    violations = reconcile_incident_scenario_provenance(record)
    assert any("does not exist" in violation for violation in violations)


def test_admission_disposition_is_required() -> None:
    record = _valid_record()
    del record["admission"]
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert "schema" in violations[0]


def test_missing_scenario_digest_fails_closed_when_execution_claimed() -> None:
    record = _valid_record()
    _claim_execution(record)
    record["scenario_config"]["digest_sha256"] = None
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    # The schema rejects the missing digest before the semantic rule runs.
    assert any("schema" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)


def test_zero_seed_is_valid_when_execution_is_claimed() -> None:
    record = _valid_record()
    _claim_execution(record)
    record["scenario_config"]["seed"] = 0
    assert reconcile_incident_scenario_provenance(record) == []


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
    _claim_execution(record)
    record["execution"]["replay_identity"] = ""
    violations = reconcile_incident_scenario_provenance(record)
    assert violations
    assert any("schema" in violation for violation in violations)
    with pytest.raises(IncidentScenarioProvenanceError):
        validate_incident_scenario_provenance(record)

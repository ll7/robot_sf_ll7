"""Contract-level tests for the Robot SF ecosystem producer declaration."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.tools import build_robot_sf_ecosystem_contract as builder

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/ecosystem_contract/v1"
CONTRACT_PATH = ROOT / builder.DEFAULT_CONTRACT_PATH
CONTRACT_SCHEMA_PATH = ROOT / builder.DEFAULT_CONTRACT_SCHEMA_PATH
DIGEST_PATH = ROOT / builder.DEFAULT_DIGEST_PATH

EXPECTED_CAPABILITY_IDS = {
    "robot_sf.artifact.sha256sums.v1",
    "robot_sf.cli.envs.describe.v1",
    "robot_sf.cli.envs.list.v1",
    "robot_sf.cli.validate_config.v1",
    "robot_sf.protocol.benchmark_result_provenance.v1",
    "robot_sf.protocol.campaign_result_store.v1",
    "robot_sf.protocol.publication_bundle.v2",
    "robot_sf.protocol.release_manifest.v0",
    "robot_sf.schema.aggregate.v1",
    "robot_sf.schema.episode.v1",
    "robot_sf.schema.evidence_bundle.v1",
    "robot_sf.schema.release_assurance_case.v1",
    "robot_sf.schema.report_metadata.v1",
    "robot_sf.schema.result_job_durability.v1",
    "robot_sf.schema.scenario_matrix.v1",
}


def _load_fixture(name: str) -> dict[str, object]:
    """Load one strict compatibility fixture."""
    return builder._mapping(builder.load_json(FIXTURES / name), name)


def _contract_schema() -> dict[str, object]:
    """Load the meta-validated producer contract schema."""
    return builder._load_schema(CONTRACT_SCHEMA_PATH, "test contract schema")


def _requirements() -> dict[str, object]:
    """Return explicit downstream requirements for the synthetic fixture."""
    return {
        "schema_version": builder.REQUIREMENTS_SCHEMA_VERSION,
        "supported_contract_schema_majors": [1],
        "supported_consumer_features": sorted(builder.MINIMUM_CONSUMER_FEATURES),
        "required_capabilities": [
            {
                "capability_id": "robot_sf.schema.example.v1",
                "interface_major": 1,
                "accepted_statuses": ["beta", "deprecated"],
                "semantics_id": "robot_sf.schema.example.semantics.v1",
            }
        ],
    }


def test_committed_contract_is_canonical_digest_bound_and_source_current() -> None:
    """The checked-in producer contract must match every current source selector."""
    contract = builder.validate_contract_path(
        CONTRACT_PATH,
        schema_path=CONTRACT_SCHEMA_PATH,
        root=ROOT,
        digest_path=DIGEST_PATH,
        verify_sources=True,
    )

    assert CONTRACT_PATH.read_bytes() == builder.canonical_bytes(contract)
    assert not CONTRACT_PATH.read_bytes().endswith(b"\n")
    assert contract["canonicalization"]["profile"] == "RFC8785"


def test_committed_contract_has_the_reviewed_capability_inventory() -> None:
    """Generation must not add prose-derived or unreviewed public capabilities."""
    contract = builder._mapping(builder.load_json(CONTRACT_PATH), "committed contract")
    capability_ids = {item["capability_id"] for item in contract["capabilities"]}

    assert capability_ids == EXPECTED_CAPABILITY_IDS
    assert {item["kind"] for item in contract["capabilities"]} == {
        "artifact_identity",
        "cli",
        "protocol",
        "schema",
    }
    assert all(item["status"] != "stable" for item in contract["capabilities"])


def test_contract_keeps_revision_data_out_of_the_invariant_payload() -> None:
    """An unrelated commit must not change the revision-invariant contract bytes."""
    contract = builder._mapping(builder.load_json(CONTRACT_PATH), "committed contract")

    assert {"source_commit", "release_tag", "lock_digest"}.isdisjoint(contract)
    assert contract["compatibility_policy"]["whole_contract_digest_role"] == "provenance_only"
    assert contract["compatibility_policy"]["revision_fields_role"] == "provenance_only"


def test_contract_does_not_invent_public_fixtures_before_issue_6711() -> None:
    """The v1 registry must leave public fixture declaration to the fixture task."""
    contract = builder._mapping(builder.load_json(CONTRACT_PATH), "committed contract")

    assert contract["canonical_public_fixtures"] == []


@pytest.mark.parametrize(
    ("fixture_name", "detected_change"),
    [
        ("additive.json", "additive"),
        ("deprecated.json", "deprecated"),
        ("breaking.json", "breaking"),
    ],
)
def test_declared_change_fixtures_are_accepted(fixture_name: str, detected_change: str) -> None:
    """Correct additive, deprecation, and breaking records must validate."""
    baseline = _load_fixture("valid_initial.json")
    candidate = _load_fixture(fixture_name)

    report = builder.check_declared_change(baseline, candidate, schema=_contract_schema())

    assert report["valid"] is True
    assert report["detected_change"] == detected_change


def test_breaking_change_mislabeled_as_additive_fails_closed() -> None:
    """A changed semantics ID cannot pass as a same-major additive change."""
    report = builder.check_declared_change(
        _load_fixture("valid_initial.json"),
        _load_fixture("breaking_mislabeled_additive.json"),
        schema=_contract_schema(),
    )

    assert report["valid"] is False
    assert report["detected_change"] == "breaking"
    assert any("declares 'additive'" in error for error in report["errors"])
    assert any("new contract major" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    [
        ("stale_digest.json", "stale contract_digest"),
        ("duplicate_capability_ids.json", "capability IDs must be unique and sorted"),
    ],
)
def test_invalid_contract_fixtures_fail_closed(fixture_name: str, message: str) -> None:
    """Stale digests and ambiguous capability IDs must never reach consumers."""
    with pytest.raises(builder.ContractError, match=message):
        builder.validate_contract_document(_load_fixture(fixture_name), schema=_contract_schema())


def test_malformed_contract_fixture_fails_before_schema_validation() -> None:
    """Malformed JSON must produce a typed input error, not partial data."""
    with pytest.raises(builder.ContractError, match="malformed JSON document"):
        builder.load_json(FIXTURES / "malformed.fixture")


def test_compatibility_matches_capability_identity_not_whole_digest() -> None:
    """An additive producer digest change must preserve compatible requirements."""
    report = builder.check_compatibility(
        _load_fixture("additive.json"), _requirements(), schema=_contract_schema()
    )

    assert report == {
        "schema_version": builder.COMPATIBILITY_REPORT_SCHEMA_VERSION,
        "compatible": True,
        "errors": [],
        "warnings": [],
        "matched_capability_ids": ["robot_sf.schema.example.v1"],
    }


def test_compatibility_rejects_unsupported_contract_major() -> None:
    """A consumer that does not implement contract schema v1 must fail closed."""
    requirements = _requirements()
    requirements["supported_contract_schema_majors"] = [2]

    report = builder.check_compatibility(
        _load_fixture("valid_initial.json"),
        requirements,
        schema=_contract_schema(),
    )

    assert report["compatible"] is False
    assert report["errors"] == ["consumer does not support ecosystem contract schema major 1"]


def test_compatibility_rejects_missing_validator_feature() -> None:
    """Consumers must declare every validator feature required by the producer."""
    requirements = _requirements()
    requirements["supported_consumer_features"] = ["sha256.v1"]

    report = builder.check_compatibility(
        _load_fixture("valid_initial.json"),
        requirements,
        schema=_contract_schema(),
    )

    assert report["compatible"] is False
    assert len(report["errors"]) == 3
    assert all("consumer lacks required validator feature" in error for error in report["errors"])

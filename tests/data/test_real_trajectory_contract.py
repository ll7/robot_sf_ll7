"""Tests for the real-trajectory ingestion/artifact-staging contract (GitHub issue #3065).

These tests use only synthetic in-memory metadata plus the tracked example manifest template. They
never download, stage, or read real external data.
"""

from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

from robot_sf.data_ingestion import (
    MANIFEST_SCHEMA_ID,
    ContractError,
    load_manifest,
    load_schema,
    run_preflight,
    validate_manifest_structure,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs" / "data" / "real_trajectory_manifest.example.yaml"


@pytest.fixture
def valid_manifest() -> dict:
    """Return a minimal, structurally and semantically valid synthetic manifest."""
    return {
        "schema": MANIFEST_SCHEMA_ID,
        "dataset_id": "synthetic_byo",
        "title": "Synthetic BYO trajectories",
        "source": {
            "url": "https://example.org/ds",
            "version": "v1",
            "citation": "Synthetic citation",
            "access_date": "2026-06-27",
        },
        "license": {
            "name": "synthetic-license",
            "url": None,
            "posture": "bring-your-own",
            "supplier_acknowledgment": True,
            "redistribution": False,
        },
        "retrieval": {
            "instructions": "Obtain locally under license terms.",
            "download_url": None,
            "fail_closed": True,
        },
        "checksums": {
            "algorithm": "SHA-256",
            "tree_sha256": None,
            "expected_tree_sha256": None,
        },
        "conversion": {
            "frame_rate_hz": 2.5,
            "length_unit": "meters",
            "coordinate_frame": "world_meters_xy",
            "timestamp_field": "frame",
            "agent_id_field": "ped_id",
            "position_fields": ["x", "y"],
            "map_context_field": "scene",
            "missing_data_behavior": "drop",
        },
        "splits": {"naming": "train_val_test", "members": ["train", "val", "test"]},
        "staging": {
            "staging_dir": "output/external_data/synthetic_byo",
            "local_only_raw": True,
            "durable_storage_target": "local-only-byo",
        },
        "privacy": {"pii_reviewed": True, "notes": "anonymized"},
        "availability": "missing",
        "benchmark_eligibility": "diagnostic_only",
        "related_issues": [3065],
    }


def test_schema_loads_and_has_expected_id() -> None:
    """The bundled schema loads and pins the expected schema id."""
    schema = load_schema()
    assert schema["properties"]["schema"]["const"] == MANIFEST_SCHEMA_ID


def test_valid_manifest_passes_structure_and_preflight(valid_manifest: dict) -> None:
    """A well-formed synthetic manifest passes structure and preflight cleanly."""
    validate_manifest_structure(valid_manifest)
    result = run_preflight(valid_manifest)
    assert result.ok
    assert result.errors == []
    assert result.dataset_id == "synthetic_byo"


def test_example_template_is_valid_and_passes_preflight() -> None:
    """The tracked example manifest must always satisfy its own contract."""
    manifest = load_manifest(EXAMPLE_MANIFEST)
    result = run_preflight(manifest)
    assert result.ok, [f"{i.code}: {i.message}" for i in result.errors]
    # The template ships unstaged, so it must stay below benchmark-claim grade.
    assert manifest["availability"] == "missing"
    assert manifest["benchmark_eligibility"] != "benchmark_candidate"


def test_missing_required_field_fails_structure(valid_manifest: dict) -> None:
    """Dropping a required top-level block fails JSON Schema validation."""
    del valid_manifest["conversion"]
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_structure(valid_manifest)


def test_unknown_field_rejected(valid_manifest: dict) -> None:
    """Unexpected top-level keys are rejected (additionalProperties: false)."""
    valid_manifest["surprise"] = "nope"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_structure(valid_manifest)


def test_redistribution_must_be_false(valid_manifest: dict) -> None:
    """The no-redistribution invariant is encoded as a schema const."""
    valid_manifest["license"]["redistribution"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_structure(valid_manifest)


def test_byo_requires_supplier_acknowledgment(valid_manifest: dict) -> None:
    """Bring-your-own data without a supplier acknowledgment fails preflight."""
    valid_manifest["license"]["supplier_acknowledgment"] = False
    result = run_preflight(valid_manifest)
    assert not result.ok
    assert any(i.code == "license.acknowledgment_missing" for i in result.errors)


def test_staging_dir_must_be_gitignored(valid_manifest: dict) -> None:
    """A staging dir outside the git-ignored roots fails preflight."""
    valid_manifest["staging"]["staging_dir"] = "robot_sf/committed_data"
    result = run_preflight(valid_manifest)
    assert not result.ok
    assert any(i.code == "staging.not_gitignored" for i in result.errors)


def test_external_data_root_staging_dir_is_allowed(valid_manifest: dict) -> None:
    """A staging dir under the external-data-root env var is accepted."""
    valid_manifest["staging"]["staging_dir"] = "${ROBOT_SF_EXTERNAL_DATA_ROOT}/synthetic_byo"
    result = run_preflight(valid_manifest)
    assert result.ok


def test_durable_target_cannot_be_output(valid_manifest: dict) -> None:
    """The disposable output/ root is rejected as a durable storage target."""
    valid_manifest["staging"]["durable_storage_target"] = "output/derived/synthetic"
    result = run_preflight(valid_manifest)
    assert not result.ok
    assert any(i.code == "staging.durable_target_is_output" for i in result.errors)


def test_benchmark_candidate_requires_validated_availability(valid_manifest: dict) -> None:
    """benchmark_candidate eligibility requires validated availability (fail-closed)."""
    valid_manifest["benchmark_eligibility"] = "benchmark_candidate"
    valid_manifest["availability"] = "staged"  # not yet validated
    result = run_preflight(valid_manifest)
    assert not result.ok
    assert any(i.code == "eligibility.not_validated" for i in result.errors)


def test_validated_availability_requires_checksum(valid_manifest: dict) -> None:
    """A validated availability without a tree checksum fails preflight."""
    valid_manifest["availability"] = "validated"
    valid_manifest["checksums"]["tree_sha256"] = None
    result = run_preflight(valid_manifest)
    assert not result.ok
    assert any(i.code == "checksums.missing_for_validated" for i in result.errors)


def test_fully_validated_benchmark_candidate_passes(valid_manifest: dict) -> None:
    """A staged, checksum-validated benchmark candidate passes preflight."""
    valid_manifest["availability"] = "validated"
    valid_manifest["benchmark_eligibility"] = "benchmark_candidate"
    valid_manifest["checksums"]["tree_sha256"] = "a" * 64
    valid_manifest["checksums"]["expected_tree_sha256"] = "a" * 64
    result = run_preflight(valid_manifest)
    assert result.ok, [f"{i.code}: {i.message}" for i in result.errors]


def test_project_hosted_posture_warns(valid_manifest: dict) -> None:
    """The project-hosted posture produces an advisory warning, not a hard error."""
    valid_manifest["license"]["posture"] = "project-hosted"
    result = run_preflight(valid_manifest)
    assert result.ok
    assert any(i.code == "license.project_hosted_requires_decision" for i in result.warnings)


def test_load_manifest_rejects_non_mapping(tmp_path: Path) -> None:
    """Loading a manifest that is not a mapping raises ContractError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ContractError):
        load_manifest(bad)


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    """Loading a missing manifest path raises ContractError."""
    with pytest.raises(ContractError):
        load_manifest(tmp_path / "does_not_exist.yaml")


def test_invalid_checksum_pattern_rejected(valid_manifest: dict) -> None:
    """A malformed tree checksum is rejected by the schema pattern."""
    valid_manifest["checksums"]["tree_sha256"] = "not-a-hash"
    with pytest.raises(jsonschema.ValidationError):
        validate_manifest_structure(valid_manifest)

"""Tests for the external pedestrian-prior extraction preflight checker (issue #2918)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.pedestrian_prior_extraction_manifest import (
    ALLOWED_SOURCE_TYPE_ASSET_IDS,
    CONTRACT_STATUS_BLOCKED,
    CONTRACT_STATUS_PROXY_ONLY,
    CONTRACT_STATUS_READY,
    DATASET_BACKED_REQUIRED_PROVENANCE_FIELDS,
    PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_VERSION,
    PRIOR_EXTRACTION_EVIDENCE_BOUNDARY,
    REQUIRED_PRIOR_PARAMETERS,
    PedestrianPriorExtractionManifestError,
    check_pedestrian_prior_extraction_manifest,
    load_pedestrian_prior_extraction_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST_PATH = (
    _REPO_ROOT
    / "configs"
    / "research"
    / "pedestrian_prior_extraction_manifest_issue_2918_example.yaml"
)

_PRIOR_PARAMETER_UNITS = {
    "walking_speed": "m/s",
    "crossing_angle": "deg",
    "density": "ped/m^2",
    "interaction_distance": "m",
    "stop_yield_timing": "s",
}


def _prior_parameters(value_status: str) -> list[dict]:
    """Return the canonical prior parameters at a given value_status."""
    return [
        {
            "name": name,
            "units": _PRIOR_PARAMETER_UNITS[name],
            "value_status": value_status,
        }
        for name in REQUIRED_PRIOR_PARAMETERS
    ]


def _manifest(extraction_status: str, value_status: str, *, source_type: str = "sdd") -> dict:
    """Return a structurally complete manifest for the given lifecycle state."""
    return {
        "schema_version": PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"extraction_{extraction_status}",
        "extraction_status": extraction_status,
        "description": "Planned pedestrian-prior extraction from a staged trajectory dataset.",
        "source": {"type": source_type},
        "prior_parameters": _prior_parameters(value_status),
        "authored_separation": {"separation": "enforced"},
    }


def _dataset_backed_manifest(*, source_type: str = "sdd") -> dict:
    """Return a complete dataset-backed manifest with accepted provenance."""
    manifest = _manifest("dataset-backed", "dataset-backed", source_type=source_type)
    manifest["provenance"] = {
        "source_id": "sdd-2026-001",
        "source_uri": "file:///local/external_data/sdd",
        "license": "CC BY-NC-SA 3.0",
        "citation": "Robicquet et al. 2016",
        "access_date": "2026-06-27",
        "checksum": "sha256:deadbeef",
    }
    return manifest


def test_complete_dataset_backed_manifest_is_ready_and_claim_allowed() -> None:
    """A complete dataset-backed manifest with provenance is ready and may assert a prior."""
    report = check_pedestrian_prior_extraction_manifest(_dataset_backed_manifest())

    assert report.contract_status == CONTRACT_STATUS_READY
    assert report.dataset_backed_prior_claim_allowed is True
    assert report.evidence_boundary == PRIOR_EXTRACTION_EVIDENCE_BOUNDARY
    assert report.blockers == []
    assert report.missing_prior_parameters == []
    assert report.external_data_asset_id == "sdd"


def test_blocked_external_input_plan_is_blocked_without_claim() -> None:
    """A complete blocked-external-input plan is valid but stays blocked, no claim."""
    report = check_pedestrian_prior_extraction_manifest(
        _manifest("blocked-external-input", "pending")
    )

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.dataset_backed_prior_claim_allowed is False
    # The plan itself is complete: only the external data is missing.
    assert report.blockers == []
    assert report.provenance_blockers == []


def test_proxy_only_manifest_is_terminal_without_claim() -> None:
    """A proxy-only manifest is its own terminal state and never allows a dataset-backed claim."""
    report = check_pedestrian_prior_extraction_manifest(
        _manifest("proxy-only", "proxy-placeholder")
    )

    assert report.contract_status == CONTRACT_STATUS_PROXY_ONLY
    assert report.dataset_backed_prior_claim_allowed is False
    assert report.blockers == []


def test_dataset_backed_manifest_without_provenance_is_blocked() -> None:
    """A dataset-backed manifest missing provenance is blocked and cannot claim a value."""
    manifest = _manifest("dataset-backed", "dataset-backed")  # no provenance block

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.dataset_backed_prior_claim_allowed is False
    assert any("provenance" in blocker for blocker in report.provenance_blockers)


def test_dataset_backed_manifest_missing_one_provenance_field_is_blocked() -> None:
    """Dropping a single required provenance field fails closed."""
    manifest = _dataset_backed_manifest()
    del manifest["provenance"]["checksum"]

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("provenance.checksum" in b for b in report.provenance_blockers)


def test_proxy_only_with_dataset_source_is_rejected_as_conflation() -> None:
    """A proxy-only manifest declaring a dataset-backed source conflates the boundary -> blocked."""
    manifest = _manifest("proxy-only", "proxy-placeholder")
    manifest["provenance"] = {"source_uri": "file:///local/external_data/sdd"}

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("separate from dataset-backed priors" in b for b in report.provenance_blockers)


def test_byo_only_source_cannot_back_a_dataset_backed_claim() -> None:
    """A source family with no registered staging contract (eth_ucy) fails closed."""
    manifest = _dataset_backed_manifest(source_type="eth_ucy")

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.dataset_backed_prior_claim_allowed is False
    assert report.external_data_asset_id is None
    assert any(
        "no registered external-data staging contract" in b for b in report.provenance_blockers
    )


def test_value_status_must_match_extraction_status() -> None:
    """A dataset-backed value_status under a blocked plan is a prior-parameter blocker."""
    manifest = _manifest("blocked-external-input", "dataset-backed")

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("requires 'pending'" in b for b in report.prior_parameter_blockers)


def test_missing_prior_parameter_is_blocker() -> None:
    """Dropping a required prior parameter surfaces a missing-parameter blocker."""
    manifest = _dataset_backed_manifest()
    manifest["prior_parameters"] = [
        p for p in manifest["prior_parameters"] if p["name"] != "stop_yield_timing"
    ]

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("stop_yield_timing" in b for b in report.missing_prior_parameters)


def test_duplicate_prior_parameter_name_is_blocker() -> None:
    """A duplicate prior-parameter name (case-insensitive) fails closed as a blocker."""
    manifest = _dataset_backed_manifest()
    duplicate = dict(manifest["prior_parameters"][0])
    duplicate["name"] = duplicate["name"].upper()  # case variant of an existing parameter
    manifest["prior_parameters"] = manifest["prior_parameters"] + [duplicate]

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert any("duplicate name" in b for b in report.prior_parameter_blockers)


def test_separation_not_enforced_is_blocker() -> None:
    """A manifest that does not enforce authored separation is blocked."""
    manifest = _dataset_backed_manifest()
    manifest["authored_separation"] = {"separation": "not-enforced"}

    report = check_pedestrian_prior_extraction_manifest(manifest)

    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.separation_blockers


def test_schema_violation_raises() -> None:
    """An unknown extraction_status is rejected at the schema layer."""
    manifest = _manifest("blocked-external-input", "pending")
    manifest["extraction_status"] = "not-a-status"

    with pytest.raises(PedestrianPriorExtractionManifestError):
        check_pedestrian_prior_extraction_manifest(manifest)


def test_disallowed_source_type_raises_at_schema() -> None:
    """A source type outside the allowed enum is rejected at the schema layer."""
    manifest = _manifest("blocked-external-input", "pending")
    manifest["source"] = {"type": "waymo"}

    with pytest.raises(PedestrianPriorExtractionManifestError):
        check_pedestrian_prior_extraction_manifest(manifest)


def test_example_manifest_loads_and_is_blocked_external_input() -> None:
    """The shipped example manifest is a valid, blocked-external-input extraction plan."""
    manifest = load_pedestrian_prior_extraction_manifest(EXAMPLE_MANIFEST_PATH)
    report = check_pedestrian_prior_extraction_manifest(manifest, source=EXAMPLE_MANIFEST_PATH)

    assert report.extraction_status == "blocked-external-input"
    assert report.contract_status == CONTRACT_STATUS_BLOCKED
    assert report.dataset_backed_prior_claim_allowed is False
    # Example ships a complete plan: only the external data is missing.
    assert report.blockers == []


def test_example_manifest_yaml_is_well_formed() -> None:
    """The example YAML parses to a mapping (guards against accidental corruption)."""
    payload = yaml.safe_load(EXAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["schema_version"] == PEDESTRIAN_PRIOR_EXTRACTION_MANIFEST_SCHEMA_VERSION


def test_to_dict_is_json_safe_and_does_not_mutate_input() -> None:
    """The report serializes to plain types and the checker does not mutate its input."""
    manifest = _dataset_backed_manifest()
    before = copy.deepcopy(manifest)

    report = check_pedestrian_prior_extraction_manifest(manifest)
    payload = report.to_dict()

    assert manifest == before
    assert payload["contract_status"] == CONTRACT_STATUS_READY
    assert isinstance(payload["declared_prior_parameters"], list)


def test_source_asset_ids_match_external_data_registry() -> None:
    """Cross-check: every non-None source->asset id mapping exists in manage_external_data.

    This keeps ``ALLOWED_SOURCE_TYPE_ASSET_IDS`` from silently drifting away from the
    canonical external-data asset registry (the source of truth for staging).
    """
    tool_path = _REPO_ROOT / "scripts" / "tools" / "manage_external_data.py"
    spec = importlib.util.spec_from_file_location("manage_external_data", tool_path)
    assert spec and spec.loader
    med = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses resolve their own module.
    sys.modules[spec.name] = med
    try:
        spec.loader.exec_module(med)
    finally:
        sys.modules.pop(spec.name, None)

    registered_ids = {asset.asset_id for asset in med.list_assets()}
    mapped_ids = {asset_id for asset_id in ALLOWED_SOURCE_TYPE_ASSET_IDS.values() if asset_id}

    assert mapped_ids, "expected at least one source type to map to a registered asset id"
    assert mapped_ids <= registered_ids, (
        f"mapped asset ids {sorted(mapped_ids)} not all in external-data registry "
        f"{sorted(registered_ids)}"
    )
    # The dataset-backed provenance contract must require an immutable pin (checksum).
    assert "checksum" in DATASET_BACKED_REQUIRED_PROVENANCE_FIELDS

"""Tests for the issue #2479 real-trajectory scenario-prior manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools import import_sdd_scenarios, manage_external_data

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs/research/scenario_distribution_prior_issue_2479.yaml"


def _manifest() -> dict[str, object]:
    """Load the #2479 scenario-prior manifest."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _asset_by_id(asset_id: str) -> manage_external_data.AssetSpec:
    """Return one external-data asset spec by id."""
    for asset in manage_external_data.list_assets():
        if asset.asset_id == asset_id:
            return asset
    raise AssertionError(f"missing external-data asset: {asset_id}")


def test_manifest_references_existing_repo_surfaces() -> None:
    """Every contract surface in the manifest should exist in the repository."""
    manifest = _manifest()
    surfaces = manifest["contract_surfaces"]
    assert isinstance(surfaces, dict)

    for rel_path in surfaces.values():
        assert (REPO_ROOT / str(rel_path)).exists(), rel_path


def test_sdd_candidate_matches_importer_and_external_data_registry() -> None:
    """The first candidate should reuse the existing SDD importer and staging contract."""
    manifest = _manifest()
    candidates = {
        str(candidate["key"]): candidate
        for candidate in manifest["candidate_datasets"]
        if isinstance(candidate, dict)
    }
    sdd = candidates["sdd"]
    asset = _asset_by_id("sdd")

    assert sdd["repo_status"] == "supported_local_asset"
    assert sdd["license_note"] == import_sdd_scenarios.SDD_LICENSE
    assert sdd["source_url"] == import_sdd_scenarios.SDD_SOURCE_URL
    assert sdd["source_url"] == asset.source_url
    assert sdd["license_note"] == asset.license_note.rstrip(".")
    assert {1497, 1126}.issubset(set(asset.related_issues))
    assert {1497, 1126, 1091}.issubset(set(sdd["related_issues"]))


def test_unverified_dataset_candidates_stay_behind_provenance_gates() -> None:
    """Non-SDD candidates should not look ready without source/license verification."""
    manifest = _manifest()
    candidates = manifest["candidate_datasets"]
    assert isinstance(candidates, list)

    unverified = [candidate for candidate in candidates if candidate["key"] != "sdd"]
    assert unverified
    for candidate in unverified:
        assert candidate["license_status"] == "not_verified_in_repo"
        assert "verification" in candidate["repo_status"]
        assert "provenance" in candidate["provenance_gate"].lower()
        assert candidate["first_use"] == "follow_up_only"


def test_scenario_distribution_representation_has_required_feature_groups() -> None:
    """The scenario-prior representation should cover identity, normalization, and dynamics."""
    manifest = _manifest()
    representation = manifest["scenario_distribution_representation"]
    assert isinstance(representation, dict)

    unit_contract = representation["unit_contract"]
    required_groups = representation["required_feature_groups"]
    assert isinstance(unit_contract, dict)
    assert isinstance(required_groups, dict)

    assert unit_contract["position"] == "m"
    assert unit_contract["velocity"] == "m/s"
    assert unit_contract["time"] == "s"
    for group in (
        "scene_identity",
        "normalization",
        "agent_population",
        "kinematics",
        "interaction_geometry",
        "scenario_prior",
    ):
        assert group in required_groups
        assert required_groups[group], group


def test_first_spike_and_claim_boundary_stop_before_training_or_benchmark_claims() -> None:
    """The manifest should recommend a manifest-only spike, not a realism claim."""
    manifest = _manifest()
    first_spike = manifest["first_non_training_spike"]
    blocked_claims = set(manifest["blocked_claims"])
    claim_boundary = str(manifest["claim_boundary"]).lower()
    assert isinstance(first_spike, dict)

    assert manifest["benchmark_evidence"] is False
    assert first_spike["dataset_key"] == "sdd"
    assert "Stop before training" in first_spike["stop_rule"]
    assert "does not prove" in claim_boundary
    assert "learned_prior_realism" in blocked_claims
    assert "license_safe_redistribution_of_raw_data" in blocked_claims

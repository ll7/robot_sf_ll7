"""Tests for the issue #2523 scenario-prior smoke artifact."""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.manifest_lineage import validate_lineage_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_2523_scenario_prior_smoke"
ARTIFACT_PATH = EVIDENCE_DIR / "scenario_prior.v1.json"
SUMMARY_PATH = EVIDENCE_DIR / "summary.json"


def test_scenario_prior_proxy_artifact_preserves_claim_boundary() -> None:
    """The smoke artifact must not look like real-data or benchmark evidence."""
    artifact = _load_json(ARTIFACT_PATH)
    source = artifact["source"]
    claim_boundary = artifact["claim_boundary"]
    adequacy = artifact["adequacy"]
    assert isinstance(source, dict)
    assert isinstance(claim_boundary, dict)
    assert isinstance(adequacy, dict)

    assert artifact["schema_version"] == "scenario_prior.v1"
    assert artifact["generator_id"] == "issue_2523_proxy_scenario_prior_fixture"
    assert artifact["validator_version"] == "scenario_prior_proxy_fixture_validator.v1"
    assert artifact["evidence_tier"] == "smoke evidence"
    assert artifact["denominator_policy"] == "proxy_fixture_not_benchmark_denominator"
    assert artifact["execution_gate"] == "not_executable_until_real_data_staged"
    assert validate_lineage_contract(artifact) == []
    assert source["input_type"] == "proxy_synthetic"
    assert source["real_data_status"] == "missing"
    assert claim_boundary["benchmark_evidence"] is False
    assert claim_boundary["realism_claim"] is False
    assert claim_boundary["training_input"] is False
    assert claim_boundary["classification"] == "proxy_schema_adequate_for_smoke"
    assert adequacy["schema_coverage"] == "adequate_for_proxy_smoke"
    assert adequacy["real_data_readiness"] == "blocked_by_missing_staged_sdd"


def test_scenario_prior_proxy_artifact_covers_required_feature_groups() -> None:
    """The proxy artifact should exercise the #2479 representation shape."""
    artifact = _load_json(ARTIFACT_PATH)
    unit_contract = artifact["unit_contract"]
    scene_summaries = artifact["scene_summaries"]
    scenario_prior = artifact["scenario_prior"]
    assert isinstance(unit_contract, dict)
    assert isinstance(scene_summaries, list)
    assert isinstance(scenario_prior, dict)

    assert unit_contract == {
        "position": "m",
        "velocity": "m/s",
        "time": "s",
        "density": "agents_per_m2",
        "coordinate_frame": "local_scene_frame",
    }
    assert len(scene_summaries) == 2
    required_scene_fields = {
        "scene_id",
        "source_file_or_manifest",
        "source_license",
        "source_checksum_or_artifact_pointer",
        "coordinate_frame",
        "frame_rate_hz",
        "map_extent_m",
        "agent_count_distribution",
        "pedestrian_label_filter",
        "track_length_distribution_s",
        "missing_or_occluded_fraction",
        "speed_distribution_mps",
        "acceleration_distribution_mps2",
        "heading_change_distribution_rad",
        "stop_fraction",
        "nearest_neighbor_distance_distribution_m",
        "crossing_angle_distribution_rad",
        "time_to_closest_approach_distribution_s",
        "group_co_motion_fraction",
    }
    for scene in scene_summaries:
        assert isinstance(scene, dict)
        assert required_scene_fields.issubset(scene)
        assert scene["source_license"] == "not_applicable_proxy_fixture"

    for weight_group in (
        "encounter_type_weights",
        "density_bin_weights",
        "route_endpoint_distribution",
        "temporal_window_distribution_s",
    ):
        weights = scenario_prior[weight_group]
        assert isinstance(weights, dict)
        assert sum(weights.values()) == 1.0


def test_scenario_prior_summary_links_artifact_and_validation() -> None:
    """The summary should make provenance and validation discoverable."""
    summary = _load_json(SUMMARY_PATH)

    assert summary["artifact"] == ARTIFACT_PATH.name
    assert summary["artifact_schema"] == "scenario_prior.v1"
    assert summary["input_type"] == "proxy_synthetic"
    assert summary["real_data_status"] == "missing"
    assert summary["benchmark_evidence"] is False
    assert summary["realism_claim"] is False
    assert summary["training_input"] is False
    assert (
        "docs/context/issue_2479_real_trajectory_scenario_prior.md" in summary["related_surfaces"]
    )
    assert any("test_issue_2523_scenario_prior_smoke.py" in cmd for cmd in summary["validation"])

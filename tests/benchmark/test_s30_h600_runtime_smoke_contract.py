"""Contract tests for the bounded S30/H600 14-arm runtime smoke."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready._preflight import _load_campaign_scenarios
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
)
FULL_MANIFEST_PATH = REPO_ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
SMOKE_CONFIG_PATH = (
    REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml"
)
SMOKE_MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
EXPECTED_PLANNER_KEYS = [
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_smoke_is_one_scenario_one_seed_at_h600() -> None:
    """The smoke keeps production horizon/kinematics while bounding cardinality."""
    config = _load_yaml(SMOKE_CONFIG_PATH)
    cfg = load_campaign_config(SMOKE_CONFIG_PATH)
    scenarios = _load_campaign_scenarios(cfg)

    assert config["derived_from"]["config"] == (
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml"
    )
    assert config["derived_from"]["config_sha256"] == _sha256(SOURCE_CONFIG_PATH)
    source = _load_yaml(SOURCE_CONFIG_PATH)
    assert config["comparability_mapping"] == source["comparability_mapping"]
    assert config["route_clearance_certifications"] == source["route_clearance_certifications"]
    assert cfg.horizon == 600
    assert cfg.dt == 0.1
    assert cfg.workers == 1
    assert cfg.kinematics_matrix == ("differential_drive",)
    assert cfg.resume is False
    assert cfg.stop_on_failure is True
    assert len(scenarios) == 1
    assert scenarios[0]["name"] == "francis2023_blind_corner"
    assert list(scenarios[0]["seeds"]) == [111]


def test_runtime_smoke_preserves_all_fourteen_source_arms_without_fallback() -> None:
    """Every source arm is present, and prerequisite gaps fail closed in smoke mode."""
    source = _load_yaml(SOURCE_CONFIG_PATH)
    smoke = _load_yaml(SMOKE_CONFIG_PATH)
    source_planners = {row["key"]: row for row in source["planners"]}
    smoke_planners = {row["key"]: row for row in smoke["planners"]}

    assert list(smoke_planners) == EXPECTED_PLANNER_KEYS
    assert list(source_planners) == EXPECTED_PLANNER_KEYS
    assert len(smoke_planners) == 14
    for key in EXPECTED_PLANNER_KEYS:
        source_row = dict(source_planners[key])
        smoke_row = dict(smoke_planners[key])
        assert smoke_row == source_row
        if smoke_row.get("algo_config"):
            assert (REPO_ROOT / smoke_row["algo_config"]).is_file()
        assert smoke_row.get("socnav_missing_prereq_policy") != "fallback"
        assert source_row.get("socnav_missing_prereq_policy", "fail-fast") == "fail-fast"


def test_full_and_smoke_release_configs_are_strict_and_use_new_identity() -> None:
    """Canonical benchmark-data execution cannot use fallback or predecessor metadata."""
    source = _load_yaml(SOURCE_CONFIG_PATH)
    smoke = _load_yaml(SMOKE_CONFIG_PATH)
    full_manifest = _load_yaml(FULL_MANIFEST_PATH)

    for payload in (source, smoke):
        assert payload["checkpoint_provenance_enforcement"] == "error"
        assert all(
            row.get("socnav_missing_prereq_policy", "fail-fast") == "fail-fast"
            for row in payload["planners"]
        )

    assert source["release_tag"] == full_manifest["release_tag"]
    assert source["doi"] == full_manifest["provenance"]["doi"]
    serialized = yaml.safe_dump(source)
    assert "0.0.3.post1" not in serialized
    assert "19482025" not in serialized
    assert "19563812" not in serialized


def test_release_protocol_rejects_relaxed_or_predecessor_execution_contract() -> None:
    """Manifest validation blocks fallback, audit-only checkpoints, and old identity drift."""
    manifest = load_release_manifest(FULL_MANIFEST_PATH)
    cfg = load_campaign_config(SOURCE_CONFIG_PATH)
    planners = tuple(
        replace(
            planner,
            socnav_missing_prereq_policy=("fallback" if planner.key == "orca" else "fail-fast"),
        )
        for planner in cfg.planners
    )
    drifted_cfg = replace(
        cfg,
        checkpoint_provenance_enforcement="warn",
        planners=planners,
        release_tag="0.0.3.post1",
        doi="10.5281/zenodo.19482025",
    )

    validation = validate_release_manifest(manifest, campaign_config=drifted_cfg)

    assert validation["status"] == "invalid"
    assert any(
        "checkpoint_provenance_enforcement=error" in problem for problem in validation["problems"]
    )
    assert any(
        "fail-fast missing-prerequisite policy" in problem for problem in validation["problems"]
    )
    assert "campaign config release_tag does not match release manifest" in validation["problems"]
    assert "campaign config doi does not match release manifest" in validation["problems"]


def test_runtime_smoke_claim_boundary_and_fresh_zenodo_requirement() -> None:
    """Smoke metadata separates benchmark-data evidence from software and ranking claims."""
    config = _load_yaml(SMOKE_CONFIG_PATH)
    manifest = _load_yaml(SMOKE_MANIFEST_PATH)

    for payload in (config, manifest):
        assert payload["release_kind"] == "benchmark-data"
        assert payload["claim_boundary"]["evidence_class"] == "runtime-smoke"
        assert payload["claim_boundary"]["snqi"] == "advisory-no-ranking"
        assert payload["claim_boundary"]["software_release"] is False
        assert payload["zenodo"]["concept"] == "fresh-required"
        assert payload["zenodo"]["doi_status"] == "pending-assignment"
        assert payload["zenodo"]["reuse_existing_concept"] is False
        assert payload["artifact_provenance"]["raw_episode_outputs"] == (
            "worktree-local-ignored-cache"
        )
        assert payload["artifact_provenance"]["compact_manifest"] == "tracked-compact-evidence"
        assert payload["artifact_provenance"]["source_commit"] == "record-at-run"
        assert payload["artifact_provenance"]["campaign_root"].startswith("output/")
        serialized = yaml.safe_dump(payload)
        assert "19482025" not in serialized
        assert "19563812" not in serialized


def test_runtime_smoke_manifest_validates_against_config_and_assets() -> None:
    """The manifest remains consumable by the release protocol without publishing."""
    manifest = load_release_manifest(SMOKE_MANIFEST_PATH)
    validation = validate_release_manifest(manifest)

    assert validation == {
        "manifest_path": "configs/benchmarks/releases/"
        "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml",
        "status": "valid",
        "problem_count": 0,
        "problems": [],
    }
    assert manifest.planner_keys == tuple(EXPECTED_PLANNER_KEYS)
    assert manifest.seed_policy["mode"] == "fixed-list"
    assert manifest.seed_policy["seeds"] == [111]
    assert manifest.expected_kinematics_matrix == ("differential_drive",)

"""Tests for the issue #4400 release-gate campaign pre-registration config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

CONFIG_PATH = Path("configs/benchmarks/release_gate_current_roster_social_proxemic_issue_4400.yaml")
BASELINE_CONFIG_PATH = Path("configs/benchmarks/camera_ready_all_planners.yaml")
EXPECTED_RELEASE_GATE_FIELDS = {"min_clearance_m", "proxemic_intrusion_rate"}
FORBIDDEN_TRANSIENT_FIELDS = {
    "target_host",
    "submit_host",
    "queue_host",
    "packet_lineage",
    "packet_lineage_pointer",
    "private_ops_state",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_issue_4400_campaign_config_loads_and_matches_current_roster() -> None:
    """Config loader accepts the pre-registration and keeps the #4313 roster."""
    cfg = load_campaign_config(CONFIG_PATH)
    baseline = load_campaign_config(BASELINE_CONFIG_PATH)

    assert cfg.name == "release_gate_current_roster_social_proxemic_issue_4400"
    assert cfg.scenario_matrix_path == baseline.scenario_matrix_path
    assert cfg.seed_policy.mode == baseline.seed_policy.mode
    assert cfg.seed_policy.seed_set == baseline.seed_policy.seed_set
    assert [planner.key for planner in cfg.planners] == [
        "prediction_planner",
        "goal",
        "social_force",
        "orca",
        "ppo",
        "socnav_sampling",
        "sacadrl",
        "socnav_bench",
    ]
    assert [planner.key for planner in cfg.planners] == [
        planner.key for planner in baseline.planners
    ]


def test_issue_4400_config_declares_social_proxemic_retained_summary_contract() -> None:
    """The release-gate footgun stays explicit instead of relying on defaults."""
    payload = _load_yaml(CONFIG_PATH)
    prereg = payload["release_gate_preregistration"]
    social_proxemic = prereg["metric_groups"]["social_proxemic"]

    assert social_proxemic["enabled"] is True
    assert set(prereg["retained_summary_contract"]["required_fields"]) == (
        EXPECTED_RELEASE_GATE_FIELDS
    )
    assert set(social_proxemic["retained_summary_fields"]) == EXPECTED_RELEASE_GATE_FIELDS
    assert set(social_proxemic["required_episode_metrics"]) == {
        "min_clearance",
        "social_proxemic_intrusion_frac",
    }
    assert prereg["release_gate_spec"] == (
        "configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml"
    )
    assert "certification" in prereg["claim_boundary"]


def test_issue_4400_config_preserves_no_submit_boundary_without_transient_state() -> None:
    """The tracked config is a reusable contract, not a private queue packet."""
    payload = _load_yaml(CONFIG_PATH)
    prereg = payload["release_gate_preregistration"]
    boundary = prereg["no_submit_boundary"]

    assert boundary == {
        "submit_in_this_pr": False,
        "slurm_or_gpu_submission": False,
        "campaign_execution": False,
        "release_approval_claim": False,
    }
    serialized = yaml.safe_dump(payload)
    for field in FORBIDDEN_TRANSIENT_FIELDS:
        assert field not in serialized

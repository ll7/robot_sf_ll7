"""Contract tests for issue #1586 calibrated AMV actuation profile skeleton config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_1586_calibrated_actuation_profile_skeleton_v0.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a benchmark YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_issue_1586_config_declares_calibrated_actuation_placeholder() -> None:
    """The config should declare a calibrated-actuation profile with provenance placeholders."""
    payload = _load_yaml(CONFIG_PATH)

    assert payload["paper_facing"] is False
    assert payload["kinematics_matrix"] == ["differential_drive"]
    assert payload["paper_interpretation_profile"] == "issue-1586-calibrated-actuation-skeleton"

    profile = payload["synthetic_actuation_profile"]
    assert profile["claim_scope"] == "hardware-calibrated"
    assert profile["claim_boundary"] == "calibrated-amv-actuation"
    assert "calibrated" in profile["name"]

    provenance = profile.get("provenance", {})
    assert provenance["source_id"] == "pending-#1585"
    assert provenance["source_type"] == "external-trace-collection-pending"
    assert provenance["profile_version"] == "v0-placeholder"
    assert "max_linear_accel_m_s2" in provenance["supported_actuation_fields"]
    assert "m/s^2" in provenance["units"].get("max_linear_accel_m_s2", "")

    assert profile["max_linear_accel_m_s2"] > 0
    assert profile["max_linear_decel_m_s2"] > 0


def test_issue_1586_config_planner_set_is_compact() -> None:
    """The skeleton should use a minimal planner set."""
    payload = _load_yaml(CONFIG_PATH)
    assert len(payload["planners"]) == 2
    assert payload["planners"][0]["key"] == "goal"
    assert payload["planners"][1]["key"] == "orca"


def test_issue_1586_config_scenario_set_is_minimal() -> None:
    """The skeleton should use a single scenario for smoke validation."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["scenario_candidates"] == ["classic_overtaking_medium"]


def test_issue_1586_config_seed_policy_uses_eval_seed_set() -> None:
    """The config should use the eval seed set."""
    payload = _load_yaml(CONFIG_PATH)
    assert payload["seed_policy"]["mode"] == "seed-set"
    assert payload["seed_policy"]["seed_set"] == "eval"


def test_issue_1586_config_loads_via_campaign_loader() -> None:
    """The calibrated profile config should load through the campaign config loader."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg = load_campaign_config(CONFIG_PATH)
    assert cfg.synthetic_actuation_profile is not None
    assert cfg.synthetic_actuation_profile.claim_scope == "hardware-calibrated"
    assert cfg.synthetic_actuation_profile.claim_boundary == "calibrated-amv-actuation"
    assert cfg.paper_facing is False

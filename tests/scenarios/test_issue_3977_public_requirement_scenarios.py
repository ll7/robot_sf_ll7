"""Scenario-loader coverage for issue #3977 public-requirement fixtures."""

from __future__ import annotations

from pathlib import Path

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3977_public_requirements.yaml"
EXPECTED_CATEGORIES = {
    "safe_braking",
    "visibility_and_intent",
    "emergency_reaction",
    "speed_limit",
}


def test_issue_3977_public_requirement_manifest_loads_four_families() -> None:
    """Manifest enumerates all four public-requirement scenario categories."""
    scenarios = load_scenarios(SCENARIO_SET)

    assert len(scenarios) == 4
    categories = {scenario["metadata"]["public_requirement"]["category"] for scenario in scenarios}
    assert categories == EXPECTED_CATEGORIES

    for scenario in scenarios:
        metadata = scenario["metadata"]
        contract = metadata["public_requirement"]
        assert contract["schema_version"] == "public-requirement-scenario.v1"
        assert contract["claim_boundary"] == "authored_scenario_proxy_not_human_subject_evidence"
        assert isinstance(contract["event_contract"]["type"], str)
        for descriptive_key in ("archetype", "flow", "behavior", "purpose"):
            assert isinstance(metadata[descriptive_key], str)
            assert descriptive_key not in contract
        build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_SET)


def test_issue_3977_speed_limit_exposes_public_cap() -> None:
    """Speed-limit family exposes authored cap without making it a benchmark gate."""
    scenario = _scenario_by_category("speed_limit")
    contract = scenario["metadata"]["public_requirement"]

    assert scenario["scenario_family"] == "public_requirement_speed_limit"
    assert contract["event_contract"]["type"] == "speed_limit_monitor"
    assert contract["event_contract"]["speed_limit_m_s"] == 0.8
    assert contract["event_contract"]["violation_margin_m_s"] == 0.05
    assert (
        scenario["robot_config"]["max_linear_speed"] > contract["event_contract"]["speed_limit_m_s"]
    )


def test_issue_3977_runtime_event_scenarios_keep_named_actors() -> None:
    """Safe-braking and emergency-reaction contracts point at configured actors."""
    for category, actor_key in [
        ("safe_braking", "pedestrian_id"),
        ("emergency_reaction", "actor_id"),
    ]:
        scenario = _scenario_by_category(category)
        contract = scenario["metadata"]["public_requirement"]["event_contract"]
        pedestrian_ids = {ped["id"] for ped in scenario["single_pedestrians"]}
        assert contract[actor_key] in pedestrian_ids
        assert "conflict_point" in contract


def _scenario_by_category(category: str) -> dict:
    scenarios = load_scenarios(SCENARIO_SET)
    for scenario in scenarios:
        if scenario["metadata"]["public_requirement"]["category"] == category:
            return dict(scenario)
    raise AssertionError(f"Missing issue #3977 scenario category: {category}")

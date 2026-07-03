"""Tests issue #3977 emergency-reaction public-requirement scenario slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3977_public_requirements.yaml"
SINGLE_SCENARIO = REPO_ROOT / "configs/scenarios/single/issue_3977_emergency_reaction.yaml"


def _load_emergency_reaction_scenario() -> dict:
    scenarios = load_scenarios(SCENARIO_SET)
    matches = [
        scenario
        for scenario in scenarios
        if scenario["name"] == "issue_3977_emergency_reaction_sudden_obstacle_proxy"
    ]
    assert len(matches) == 1
    return dict(matches[0])


def _single_pedestrian(config):
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.single_pedestrians) == 1
    return map_def.single_pedestrians[0]


def test_emergency_reaction_contract_loads_through_scenario_set() -> None:
    """Scenario set now includes one bounded emergency-reaction family."""
    scenarios = load_scenarios(SCENARIO_SET)
    categories = {
        scenario["metadata"]["public_requirement"]["category"]
        for scenario in scenarios
        if "public_requirement" in scenario.get("metadata", {})
    }
    assert {"safe_braking", "visibility_and_intent", "emergency_reaction"} <= categories

    scenario = _load_emergency_reaction_scenario()
    public_requirement = scenario["metadata"]["public_requirement"]
    contract = public_requirement["event_contract"]

    assert scenario["scenario_family"] == "public_requirement_emergency_reaction"
    assert public_requirement["schema_version"] == "public-requirement-scenario.v1"
    assert public_requirement["category"] == "emergency_reaction"
    assert (
        public_requirement["claim_boundary"] == "authored_scenario_proxy_not_human_subject_evidence"
    )
    assert contract["type"] == "sudden_obstacle_proxy"
    assert contract["actor_id"] == "h1"
    assert contract["release_condition"] == "robot_proximity_hold"
    assert contract["conflict_point"] == [14.0, 14.0]
    assert contract["trigger_radius_m"] == pytest.approx(0.75)
    assert contract["robot_trigger_radius_m"] == pytest.approx(5.5)
    assert contract["validated_robot_speed_m_s"] == [1.0, 2.0, 3.0]


def test_emergency_reaction_builds_deterministic_proxy_geometry() -> None:
    """Scenario generation preserves the authored sudden-obstacle proxy geometry."""
    scenario = _load_emergency_reaction_scenario()
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)
    ped = _single_pedestrian(config)

    assert ped.id == "h1"
    assert ped.start == (14.0, 26.0)
    assert ped.trajectory == [(14.0, 22.0), (14.0, 17.5), (14.0, 14.0), (14.0, 6.0)]
    assert ped.speed_m_s == pytest.approx(2.0)
    assert ped.start_delay_s == pytest.approx(0.0)
    assert ped.hold_until_robot_within_m == pytest.approx(5.5)
    assert ped.hold_ref_point == (14.0, 14.0)
    assert ped.hold_timeout_s == pytest.approx(6.0)

    map_def = next(iter(config.map_pool.map_defs.values()))
    robot_route = map_def.robot_routes[0]
    assert robot_route.waypoints == [(7.0, 14.0), (25.5, 14.0)]
    assert tuple(
        scenario["metadata"]["public_requirement"]["event_contract"]["conflict_point"]
    ) in (ped.trajectory or [])


def test_emergency_reaction_short_headless_smoke_episode(monkeypatch) -> None:
    """Emergency-reaction scenario steps headlessly without benchmark campaigns."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = _load_emergency_reaction_scenario()
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)
    env = RobotEnv(config, debug=False)

    try:
        env.reset(seed=3977)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(5):
            env.step(action)
        ped_positions = np.asarray(env.simulator.ped_pos, dtype=float)
        assert ped_positions.shape[0] == 1
    finally:
        env.close()


@pytest.mark.parametrize("robot_speed_m_s", [1.0, 2.0, 3.0])
def test_emergency_reaction_releases_by_robot_proximity_at_required_speeds(
    monkeypatch, robot_speed_m_s: float
) -> None:
    """Runtime proof: 1-3 m/s robot approaches engage the emergency proxy contract."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = _load_emergency_reaction_scenario()
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)
    env = RobotEnv(config, debug=False)

    try:
        env.reset(seed=3977)
        behavior = next(
            b for b in env.simulator.peds_behaviors if isinstance(b, SinglePedestrianBehavior)
        )
        runtime = behavior._runtimes[0]
        dt = float(config.sim_config.time_per_step_in_secs)
        step_index = 0

        def robot_pose_provider():
            x = min(14.0, 7.0 + robot_speed_m_s * step_index * dt)
            return [((x, 14.0), 0.0)]

        behavior.set_robot_pose_provider(robot_pose_provider)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        for step in range(180):
            step_index = step
            env.step(action)
            if runtime.hold_released_by is not None:
                break

        assert runtime.hold_released_by == "robot_proximity"
        assert float(env.simulator.ped_pos[0][1]) < 17.5
    finally:
        env.close()

"""Tests issue #3977 visibility-and-intent public-requirement scenario slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.ped_npc.ped_behavior import SinglePedestrianBehavior
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3977_public_requirements.yaml"
SINGLE_SCENARIO = REPO_ROOT / "configs/scenarios/single/issue_3977_visibility_and_intent.yaml"


def _load_visibility_and_intent_scenario() -> dict:
    scenarios = load_scenarios(SCENARIO_SET)
    matches = [
        scenario
        for scenario in scenarios
        if scenario["name"] == "issue_3977_visibility_and_intent_turn_near_ped"
    ]
    assert len(matches) == 1
    return dict(matches[0])


def _single_pedestrian(config):
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.single_pedestrians) == 1
    return map_def.single_pedestrians[0]


def test_visibility_and_intent_contract_loads_through_scenario_set() -> None:
    """Scenario set now includes the bounded visibility-and-intent family."""
    scenarios = load_scenarios(SCENARIO_SET)
    names = {scenario["name"] for scenario in scenarios}

    assert "issue_3977_safe_braking_ped_steps_in_front" in names
    assert "issue_3977_visibility_and_intent_turn_near_ped" in names

    scenario = _load_visibility_and_intent_scenario()
    public_requirement = scenario["metadata"]["public_requirement"]
    assert public_requirement["schema_version"] == "public-requirement-scenario.v1"
    assert public_requirement["category"] == "visibility_and_intent"
    assert (
        public_requirement["claim_boundary"] == "authored_scenario_proxy_not_human_subject_evidence"
    )

    event_contract = public_requirement["event_contract"]
    assert event_contract == {
        "type": "turn_or_start_stop_near_pedestrian",
        "pedestrian_id": "h1",
        "turn_point": [14.0, 14.0],
        "conflict_point": [15.5, 14.0],
        "trigger_radius_m": 1.8,
        "trigger": "runtime_robot_proximity_release",
        "hold_release_radius_m": 6.0,
    }


def test_visibility_and_intent_uses_validated_proximity_release_fields() -> None:
    """Scenario plumbing preserves the proximity-release contract and turn route."""
    scenario = _load_visibility_and_intent_scenario()
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)

    ped = _single_pedestrian(config)
    assert ped.id == "h1"
    assert ped.trajectory == [(15.5, 22.0), (15.5, 17.0), (15.5, 14.0), (15.5, 10.0)]
    assert ped.speed_m_s == pytest.approx(2.0)
    assert ped.hold_until_robot_within_m == pytest.approx(6.0)
    assert ped.hold_ref_point == (15.5, 14.0)
    assert ped.hold_timeout_s == pytest.approx(7.0)
    assert ped.metadata["intent_conditioned_behavior"]["intent_label"] == (
        "robot_turn_or_start_stop_near_pedestrian"
    )

    map_def = next(iter(config.map_pool.map_defs.values()))
    robot_route = map_def.robot_routes[0]
    assert robot_route.waypoints == [(7.0, 14.0), (14.0, 14.0), (14.0, 22.0)]


def test_visibility_and_intent_short_headless_smoke_episode(monkeypatch) -> None:
    """The scenario constructs and steps headlessly without relying on benchmark campaigns."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = _load_visibility_and_intent_scenario()
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
def test_visibility_and_intent_releases_by_robot_proximity_at_required_speeds(
    monkeypatch, robot_speed_m_s: float
) -> None:
    """Runtime proof: 1-3 m/s robot approaches engage the proximity-release contract."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = _load_visibility_and_intent_scenario()
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)
    env = RobotEnv(config, debug=False)

    try:
        env.reset(seed=3977)
        behavior = next(
            b for b in env.simulator.peds_behaviors if isinstance(b, SinglePedestrianBehavior)
        )
        runtime = behavior._runtimes[0]
        dt = float(config.sim_config.time_per_step_in_secs)
        step_index = {"value": 0}

        def robot_pose_provider():
            x = min(14.0, 7.0 + robot_speed_m_s * step_index["value"] * dt)
            return [((x, 14.0), 0.0)]

        behavior.set_robot_pose_provider(robot_pose_provider)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        for step in range(140):
            step_index["value"] = step
            env.step(action)
            if runtime.hold_released_by is not None:
                break

        assert runtime.hold_released_by == "robot_proximity"
        assert float(env.simulator.ped_pos[0][1]) < 17.0
    finally:
        env.close()

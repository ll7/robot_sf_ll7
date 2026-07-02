"""Tests for issue #3977 public-requirement safe-braking scenario slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3977_public_requirements.yaml"
SINGLE_SCENARIO = REPO_ROOT / "configs/scenarios/single/issue_3977_safe_braking.yaml"


def _load_safe_braking_scenario() -> dict:
    scenarios = load_scenarios(SCENARIO_SET)
    assert len(scenarios) == 1
    return dict(scenarios[0])


def _single_pedestrian(config):
    map_def = next(iter(config.map_pool.map_defs.values()))
    assert len(map_def.single_pedestrians) == 1
    return map_def.single_pedestrians[0]


def test_safe_braking_scenario_manifest_loads_with_public_requirement_contract() -> None:
    """The first #3977 slice exposes a machine-readable safe-braking contract."""
    scenario = _load_safe_braking_scenario()

    assert scenario["name"] == "issue_3977_safe_braking_ped_steps_in_front"
    public_requirement = scenario["metadata"]["public_requirement"]
    assert public_requirement["schema_version"] == "public-requirement-scenario.v1"
    assert public_requirement["category"] == "safe_braking"
    assert (
        public_requirement["claim_boundary"] == "authored_scenario_proxy_not_human_subject_evidence"
    )

    event_contract = public_requirement["event_contract"]
    assert event_contract == {
        "type": "pedestrian_steps_in_front",
        "pedestrian_id": "h1",
        "conflict_point": [14.0, 14.0],
        "trigger_radius_m": 0.75,
    }


def test_safe_braking_scenario_generation_is_deterministic() -> None:
    """Scenario loading and config generation preserve the same crossing geometry."""
    scenario_a = _load_safe_braking_scenario()
    scenario_b = _load_safe_braking_scenario()
    assert scenario_a == scenario_b

    config_a = build_robot_config_from_scenario(scenario_a, scenario_path=SINGLE_SCENARIO)
    config_b = build_robot_config_from_scenario(scenario_b, scenario_path=SINGLE_SCENARIO)
    ped_a = _single_pedestrian(config_a)
    ped_b = _single_pedestrian(config_b)

    assert ped_a.id == ped_b.id == "h1"
    assert ped_a.start == ped_b.start == (14.0, 26.0)
    assert (
        ped_a.trajectory
        == ped_b.trajectory
        == [
            (14.0, 22.0),
            (14.0, 17.5),
            (14.0, 14.0),
            (14.0, 4.0),
        ]
    )
    assert ped_a.speed_m_s == ped_b.speed_m_s == pytest.approx(2.0)
    assert ped_a.start_delay_s == ped_b.start_delay_s == pytest.approx(0.2)

    map_a = next(iter(config_a.map_pool.map_defs.values()))
    robot_route = map_a.robot_routes[0]
    assert robot_route.waypoints == [(7.0, 14.0), (25.5, 14.0)]
    assert tuple(
        scenario_a["metadata"]["public_requirement"]["event_contract"]["conflict_point"]
    ) in (ped_a.trajectory or [])


def test_safe_braking_short_headless_smoke_episode(monkeypatch) -> None:
    """The authored safe-braking scenario constructs and steps headlessly."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    scenario = _load_safe_braking_scenario()
    contract = scenario["metadata"]["public_requirement"]["event_contract"]
    conflict_point = np.asarray(contract["conflict_point"], dtype=float)
    trigger_radius = float(contract["trigger_radius_m"])
    config = build_robot_config_from_scenario(scenario, scenario_path=SINGLE_SCENARIO)
    env = RobotEnv(config, debug=False)

    try:
        env.reset(seed=3977)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(5):
            _obs, _reward, _terminated, _truncated, _info = env.step(action)
            ped_positions = np.asarray(env.simulator.ped_pos, dtype=float)
            assert ped_positions.shape[0] == 1
    finally:
        env.close()

    ped = _single_pedestrian(config)
    assert tuple(conflict_point.tolist()) in (ped.trajectory or [])
    assert trigger_radius == pytest.approx(0.75)

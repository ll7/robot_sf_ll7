"""ScenarioBelief MVP adapter and projection tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.unified_config import ObservationVisibilitySettings, RobotSimulationConfig
from robot_sf.representation import (
    VisibilityState,
    scenario_belief_from_simulator_oracle,
    scenario_belief_from_visibility_limited_simulator,
)


def _simulator_fixture() -> SimpleNamespace:
    """Return one deterministic simulator-like step for representation tests."""
    return SimpleNamespace(
        ped_pos=np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ped_vel=np.array([[0.5, 0.0], [0.0, -0.25]], dtype=np.float32),
        robots=[
            SimpleNamespace(
                pose=((0.0, 0.0), 0.0),
                current_speed=np.array([0.1, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=0.4),
            )
        ],
        goal_pos=[np.array([5.0, 0.0], dtype=np.float32)],
        next_goal_pos=[None],
        map_def=SimpleNamespace(width=10.0, height=8.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )


def test_oracle_and_partial_adapters_share_schema_and_projection_keys() -> None:
    """Different input paths should keep the same belief and policy-observation contracts."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()

    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    assert oracle.schema_version == partial.schema_version == "scenario-belief.v1"
    assert oracle.policy_projection_keys() == partial.policy_projection_keys()
    assert set(oracle.to_socnav_struct().keys()) == set(partial.to_socnav_struct().keys())
    assert set(oracle.to_socnav_struct()["pedestrians"].keys()) == set(
        partial.to_socnav_struct()["pedestrians"].keys()
    )


def test_partial_adapter_marks_unseen_agents_without_changing_projection_schema() -> None:
    """Partial perception should differ via visibility/uncertainty, not key drift."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(
        enabled=True,
        fov_degrees=90.0,
    )
    simulator = _simulator_fixture()

    oracle = scenario_belief_from_simulator_oracle(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )
    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    oracle_obs = oracle.to_socnav_struct()
    partial_obs = partial.to_socnav_struct()
    np.testing.assert_allclose(oracle_obs["robot"]["speed"], np.array([0.1, 0.0], dtype=np.float32))
    assert oracle_obs["pedestrians"]["count"][0] == pytest.approx(2.0)
    assert partial_obs["pedestrians"]["count"][0] == pytest.approx(1.0)
    assert (
        partial_obs["pedestrians"]["positions"].shape
        == oracle_obs["pedestrians"]["positions"].shape
    )

    unseen = [
        agent for agent in partial.agents if agent.visibility_state is VisibilityState.OUTSIDE_FOV
    ]
    assert len(unseen) == 1
    assert unseen[0].source.adapter == "visibility_limited_simulator"
    assert unseen[0].position.confidence < oracle.agents[1].position.confidence
    assert unseen[0].missing_fields == ("policy_position", "policy_velocity")


def test_debug_projection_is_deterministic_and_exposes_uncertainty() -> None:
    """Debug output should explain source, confidence, visibility, and missing-data differences."""
    env_config = RobotSimulationConfig()
    env_config.observation_visibility = ObservationVisibilitySettings(enabled=True, max_range_m=2.5)
    simulator = _simulator_fixture()

    partial = scenario_belief_from_visibility_limited_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=4,
    )

    debug_a = partial.to_debug_dict()
    debug_b = partial.to_debug_dict()
    assert debug_a == debug_b
    assert debug_a["design_parent_issue"] == 1966
    assert debug_a["source_summary"]["adapter"] == "visibility_limited_simulator"
    assert debug_a["agents"][1]["visibility_state"] == "out_of_range"
    assert (
        debug_a["agents"][1]["position"]["confidence"]
        < debug_a["agents"][0]["position"]["confidence"]
    )
    assert "policy_position" in debug_a["agents"][1]["missing_fields"]

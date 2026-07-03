"""Tests for issue #4164 planner goal-posterior input wiring."""

from __future__ import annotations

import numpy as np

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import _build_goal_posterior_planner_input
from robot_sf.prediction.goal_intention import planner_goal_posterior_channel_from_state


class _FakePysfState:
    def __init__(self, states: np.ndarray) -> None:
        self._states = states

    def pysf_states(self) -> np.ndarray:
        return self._states


class _FakeSimulator:
    def __init__(self, states: np.ndarray) -> None:
        self.pysf_state = _FakePysfState(states)


def test_planner_goal_posterior_state_channel_absent_when_disabled() -> None:
    """Disabled planner-input channel carries no posterior summaries."""

    channel = planner_goal_posterior_channel_from_state(
        enabled=False,
        positions=[(0.0, 0.0)],
        velocities=[(1.0, 0.0)],
        goals=[(5.0, 0.0)],
    )

    assert channel == {"enabled": False, "pedestrian_goal_posteriors": {}}


def test_planner_goal_posterior_state_channel_present_when_enabled() -> None:
    """Enabled planner-input channel exposes top-goal metadata."""

    channel = planner_goal_posterior_channel_from_state(
        enabled=True,
        positions=[(0.0, 0.0)],
        velocities=[(1.0, 0.0)],
        goals=[(5.0, 0.0)],
        pedestrian_ids=["crossing_ped"],
    )

    summary = channel["pedestrian_goal_posteriors"]["crossing_ped"]
    assert channel["enabled"] is True
    assert summary["candidate_source"] == "scenario_route_endpoints"
    assert summary["top_goal_id"] == "crossing_ped_route_goal"
    assert summary["top_goal_confidence"] == 1.0
    assert isinstance(summary["config_hash"], str)


def test_planner_goal_posterior_state_channel_stationary_blocker() -> None:
    """Stationary pedestrians report blocker provenance instead of NaNs."""

    channel = planner_goal_posterior_channel_from_state(
        enabled=True,
        positions=[(0.0, 0.0)],
        velocities=[(0.0, 0.0)],
        goals=[(5.0, 0.0)],
        pedestrian_ids=["waiting_ped"],
    )

    summary = channel["pedestrian_goal_posteriors"]["waiting_ped"]
    assert summary["blocker"] == "stationary_below_velocity_min_mps"
    assert summary["top_goal_id"] == "waiting_ped_route_goal"


def test_robot_env_goal_posterior_planner_input_is_opt_in_metadata() -> None:
    """RobotEnv helper builds opt-in metadata from PySocialForce state columns."""

    config = EnvSettings(goal_posterior_planner_input_enabled=True)
    simulator = _FakeSimulator(
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 5.0, 0.0],
                [1.0, 1.0, 0.0, 1.0, 1.0, 5.0],
            ],
            dtype=float,
        )
    )

    channel = _build_goal_posterior_planner_input(config, simulator)

    assert channel["enabled"] is True
    assert sorted(channel["pedestrian_goal_posteriors"]) == ["ped_0", "ped_1"]
    assert channel["pedestrian_goal_posteriors"]["ped_0"]["top_goal_id"] == ("ped_0_route_goal")

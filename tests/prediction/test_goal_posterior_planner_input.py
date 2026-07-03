"""Tests for issue #4164 planner goal-posterior input wiring."""

from __future__ import annotations

import numpy as np

from robot_sf.gym_env.env_config import EnvSettings
from robot_sf.gym_env.robot_env import (
    _attach_goal_posterior_planner_input,
    _build_goal_posterior_planner_input,
    _build_step_info,
)
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


def _enabled_config_and_sim() -> tuple[EnvSettings, _FakeSimulator]:
    config = EnvSettings(goal_posterior_planner_input_enabled=True)
    simulator = _FakeSimulator(np.array([[0.0, 0.0, 1.0, 0.0, 5.0, 0.0]], dtype=float))
    return config, simulator


def test_step_and_reset_expose_channel_at_top_level_info() -> None:
    """Step and reset must both place the channel at info['planner_goal_posterior_channel'].

    Regression guard: the step path builds info via ``_build_step_info`` (which nests
    the meta dict under ``info['meta']``); the attach must run on the built info so the
    channel lands at the top level, matching the reset path and the documented contract.
    """

    config, simulator = _enabled_config_and_sim()

    reset_info: dict[str, object] = {"meta": {}}
    _attach_goal_posterior_planner_input(reset_info, config, simulator)
    assert reset_info["planner_goal_posterior_channel"]["enabled"] is True

    step_info = _build_step_info({"step": 1})
    _attach_goal_posterior_planner_input(step_info, config, simulator)
    assert step_info["planner_goal_posterior_channel"]["enabled"] is True
    # Not buried under info['meta'] (the pre-fix location).
    assert "planner_goal_posterior_channel" not in step_info["meta"]


def test_disabled_config_attaches_no_channel() -> None:
    """Default-disabled config must not attach the channel to step or reset info."""

    config = EnvSettings()
    simulator = _FakeSimulator(np.array([[0.0, 0.0, 1.0, 0.0, 5.0, 0.0]], dtype=float))

    info = _build_step_info({"step": 1})
    _attach_goal_posterior_planner_input(info, config, simulator)
    assert "planner_goal_posterior_channel" not in info

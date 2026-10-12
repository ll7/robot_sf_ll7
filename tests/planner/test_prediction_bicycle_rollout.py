"""Compare prediction candidates with the executed bicycle adapter and plant."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings


@pytest.mark.parametrize(
    "initial_speed,command", [(0.0, (1.34, 1.0)), (1.0, (0.2, -0.5)), (0.0, (-1.0, 0.5))]
)
@pytest.mark.parametrize("search", ["action", "sequence"])
def test_bicycle_candidate_matches_executed_plant(initial_speed, command, search):
    """Acceleration, braking and capped reverse use the same clock and yaw cone."""
    settings = BicycleDriveSettings(
        radius=0.64,
        wheelbase=0.9,
        max_steer=0.52,
        max_velocity=1.34,
        max_accel=0.7,
        max_decel=1.3,
        limited_reverse=True,
        max_reverse_speed=0.3,
    )
    robot = BicycleDriveRobot(settings)
    robot.state.velocity = initial_speed
    planner = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.2))
    planner.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[robot])))
    observation = {
        "robot": {"speed": [initial_speed], "angular_velocity": [0.0]},
        "sim": {"timestep": [0.1]},
    }
    actions = [command, (0.0, 0.0), (-0.3, -0.2)] if search == "sequence" else [command] * 3
    if search == "sequence":
        positions, headings = planner._rollout_robot_sequence(
            sequence=actions,
            segment_steps=1,
            dt=0.2,
            observation=observation,
        )
    else:
        positions = planner._rollout_robot(
            v=command[0], w=command[1], dt=0.2, steps=3, observation=observation
        )
        headings = None
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    expected_positions, expected_headings = [], []
    for action in actions:
        for _ in range(2):
            robot.apply_action(tuple(adapter.from_velocity_command(action)), 0.1)
        expected_positions.append(robot.pos)
        expected_headings.append(robot.pose[1])
    np.testing.assert_allclose(positions, expected_positions, atol=1e-8)
    if headings is not None:
        np.testing.assert_allclose(headings, expected_headings, atol=1e-8)


@pytest.mark.parametrize("search", ["action", "sequence"])
def test_bicycle_scoring_rejects_contact_during_braking(search):
    """A stop candidate must score the moving body until the plant can brake."""
    robot = BicycleDriveRobot(BicycleDriveSettings(radius=0.1, max_decel=1.0))
    planner = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.2))
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                robots=[robot],
                iter_obstacle_segments=lambda: [((0.25, -1.0), (0.25, 1.0))],
            )
        )
    )
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [1.0]},
        "goal": {"current": [3.0, 0.0]},
        "sim": {"timestep": [0.1]},
    }
    kwargs = {
        "observation": observation,
        "future_peds": np.zeros((0, 2, 2)),
        "mask": np.zeros(0),
        "steps": 2,
    }
    if search == "action":
        cost = planner._score_action(**kwargs, v=0.0, w=0.0)
    else:
        cost = planner._score_action_sequence(**kwargs, sequence=[(0.0, 0.0)])
    assert cost == float("inf")

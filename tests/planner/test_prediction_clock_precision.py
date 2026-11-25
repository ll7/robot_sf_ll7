"""Production float32 clocks must retain bound prediction scoring and fixed ticks."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def scoring_inputs(drive_type, forecast_dt, control_dt, flattened):
    """Bind a real plant and preserve the observation producer's float32 clock."""
    robot = (
        BicycleDriveRobot(BicycleDriveSettings(radius=0.2))
        if drive_type == "bicycle"
        else DifferentialDriveRobot(DifferentialDriveSettings(radius=0.2))
    )
    planner = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=forecast_dt))
    planner.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[robot])))
    observation = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "speed": [0.0],
            "angular_velocity": [0.0],
        },
        "goal": {"current": [3.0, 0.0]},
        "sim": {"timestep": [control_dt]},
    }
    observation = {
        block: {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
        for block, fields in observation.items()
    }
    if flattened:
        observation = {
            f"{block}_{key}": value
            for block, fields in observation.items()
            for key, value in fields.items()
        }
    return planner, robot, observation


def score(planner, observation, search):
    """Exercise the actual candidate scorer with a finite, unobstructed goal."""
    kwargs = {
        "observation": observation,
        "future_peds": np.zeros((0, 3, 2)),
        "mask": np.zeros(0),
        "steps": 3,
    }
    if search == "action":
        return planner._score_action(**kwargs, v=1.0, w=0.2)
    return planner._score_action_sequence(**kwargs, sequence=[(1.0, 0.2)])


@pytest.mark.parametrize("forecast_dt,control_ticks", [(0.1, 1), (0.2, 2)])
@pytest.mark.parametrize("drive_type", ["bicycle", "differential"])
@pytest.mark.parametrize("search", ["action", "sequence"])
@pytest.mark.parametrize("flattened", [False, True])
def test_float32_clock_preserves_shipped_scoring_and_motion(
    forecast_dt, control_ticks, drive_type, search, flattened
):
    """All sixteen production clock/search/plant combinations remain executable."""
    planner, robot, observation = scoring_inputs(drive_type, forecast_dt, 0.1, flattened)
    actual_score = score(planner, observation, search)
    assert np.isfinite(actual_score)
    reference = dict(observation)
    if flattened:
        reference["sim_timestep"] = np.array([0.1], dtype=np.float64)
    else:
        reference["sim"] = {"timestep": np.array([0.1], dtype=np.float64)}
    assert actual_score == pytest.approx(score(planner, reference, search), rel=1e-6, abs=1e-6)

    if search == "action":
        positions = planner._rollout_robot(
            v=1.0, w=0.2, dt=forecast_dt, steps=3, observation=observation
        )
        headings = None
    else:
        positions, headings = planner._rollout_robot_sequence(
            sequence=[(1.0, 0.2)], segment_steps=3, dt=forecast_dt, observation=observation
        )
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    expected_positions, expected_headings = [], []
    for _ in range(3):
        for _ in range(control_ticks):
            robot.apply_action(tuple(adapter.from_velocity_command((1.0, 0.2))), 0.1)
        expected_positions.append(robot.pos)
        expected_headings.append(robot.pose[1])
    np.testing.assert_allclose(positions, expected_positions, rtol=1e-6, atol=1e-8)
    if headings is not None:
        np.testing.assert_allclose(headings, expected_headings, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("control_dt", [0.15, 0.3])
@pytest.mark.parametrize("drive_type", ["bicycle", "differential"])
@pytest.mark.parametrize("search", ["action", "sequence"])
@pytest.mark.parametrize("flattened", [False, True])
def test_float32_clock_still_rejects_nondivisible_scoring(
    control_dt, drive_type, search, flattened
):
    """Float32 precision must not admit genuinely fractional plant ticks."""
    planner, _, observation = scoring_inputs(drive_type, 0.2, control_dt, flattened)
    with pytest.raises(
        ValueError, match="rollout timestep must be an integer multiple of the control timestep"
    ):
        score(planner, observation, search)

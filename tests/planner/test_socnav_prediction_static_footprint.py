"""Physical static-footprint witnesses for native prediction scoring."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.differential_drive import DifferentialDriveSettings


@pytest.mark.parametrize("search", ["action", "sequence"])
@pytest.mark.parametrize("hazard", ["wall", "graze", "between_samples", "braking"])
@pytest.mark.parametrize("heading", [0.0, np.pi / 2.0])
def test_native_prediction_rejects_swept_static_contact(search, hazard, heading):
    """A clear centre/endpoint or zero command cannot waive physical footprint contact."""
    settings = DifferentialDriveSettings(radius=0.2, max_linear_accel=10.0)
    if hazard == "braking":
        settings = DifferentialDriveSettings(radius=0.2, max_linear_accel=1.0)
    lines = {
        "wall": [((0.6, -1.0), (0.6, 1.0))],
        "graze": [((0.0, 0.15), (2.0, 0.15))],
        "between_samples": [((0.25, -1.0), (0.25, 1.0))],
        "braking": [((0.4, -1.0), (0.4, 1.0))],
    }[hazard]
    position = np.array([10.0, 5.0])
    rotation = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    lines = (np.asarray(lines) @ rotation + position).tolist()
    adapter = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.5))
    adapter.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                robots=[SimpleNamespace(config=settings)], iter_obstacle_segments=lambda: lines
            )
        )
    )
    observation = {
        "robot": {
            "position": position,
            "heading": [heading],
            "speed": [1.0 if hazard == "braking" else 0.0],
            "angular_velocity": [0.0],
        },
        "goal": {"current": np.array([3.0, 0.0]) @ rotation + position},
        "pedestrians": {},
    }
    # Use the actual structured observation contract, not scoring mocks.
    observation["sim"] = {"timestep": [0.1]}
    kwargs = {
        "observation": observation,
        "future_peds": np.zeros((1, 2, 2)),
        "mask": np.zeros(1),
        "steps": 2,
    }
    v = 0.0 if hazard == "braking" else 1.0
    if search == "action":
        cost = adapter._score_action(**kwargs, v=v, w=0.0)
    else:
        cost = adapter._score_action_sequence(**kwargs, sequence=[(v, 0.0)])
    assert cost == float("inf")

    clear_lines = np.array([[[5.0, -1.0], [5.0, 1.0]]]) @ rotation + position
    adapter.bind_obstacle_lines(clear_lines)
    if search == "action":
        control = adapter._score_action(**kwargs, v=v, w=0.0)
    else:
        control = adapter._score_action_sequence(**kwargs, sequence=[(v, 0.0)])
    assert np.isfinite(control), "Rebinding clear geometry must remove the veto"

"""Prediction and trajectory metrics must use the same physical disc envelope."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData, min_clearance
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings


@pytest.mark.parametrize("drive_type", [BicycleDriveSettings, DifferentialDriveSettings])
@pytest.mark.parametrize("radius", [0.2, 0.64, 1.0])
def test_prediction_clearance_uses_live_robot_radius(drive_type, radius):
    """A stale planner radius cannot disagree with recorded footprint clearance."""
    planner = PredictionPlannerAdapter(
        SocNavPlannerConfig(
            predictive_clearance_model="surface_v2",
            predictive_robot_radius=0.1,
            predictive_pedestrian_radius=0.4,
        )
    )
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(robots=[SimpleNamespace(config=drive_type(radius=radius))])
        )
    )
    future = np.array([[[1.5, 0.0]]])
    data = EpisodeData(
        robot_pos=np.zeros((1, 2)),
        peds_pos=future.transpose(1, 0, 2),
        robot_vel=np.zeros((1, 2)),
        robot_acc=np.zeros((1, 2)),
        ped_forces=np.zeros((1, 1, 2)),
        goal=np.array([3.0, 0.0]),
        dt=0.1,
        robot_radius=radius,
        ped_radius=0.4,
    )
    actual = planner._min_predicted_distance(future_peds=future, mask=np.ones(1))
    assert actual == pytest.approx(min_clearance(data))
    assert planner.config.predictive_robot_radius == 0.1, (
        "Binding must not mutate reusable configuration"
    )


@pytest.mark.parametrize("model", ["center_v1", "surface_v2"])
@pytest.mark.parametrize("search", ["action", "sequence"])
def test_prediction_scores_use_bound_radius_without_mutating_config(model, search):
    """Both scorers resolve physical size independently of stale config and rebind."""
    planners = [
        PredictionPlannerAdapter(
            SocNavPlannerConfig(
                predictive_clearance_model=model,
                # Keep this finite-score witness outside the new admission veto.
                predictive_safe_distance=0.35,
                predictive_robot_radius=radius,
                predictive_pedestrian_radius=0.4,
            )
        )
        for radius in (0.1, 0.64)
    ]
    env = SimpleNamespace(
        simulator=SimpleNamespace(
            robots=[SimpleNamespace(config=DifferentialDriveSettings(radius=0.64))]
        )
    )
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
        "goal": {"current": [3.0, 0.0]},
        "sim": {"timestep": [0.1]},
    }
    kwargs = {
        "observation": observation,
        "future_peds": np.array([[[1.5, 0.0]]]),
        "mask": np.ones(1),
        "steps": 1,
    }
    costs = []
    for planner in planners:
        planner.bind_env(env)
        if search == "action":
            costs.append(planner._score_action(**kwargs, v=0.0, w=0.0))
        else:
            costs.append(planner._score_action_sequence(**kwargs, sequence=[(0.0, 0.0)]))
    assert np.isfinite(costs).all()
    assert costs[0] == pytest.approx(costs[1])
    planners[0].bind_env(SimpleNamespace())
    assert planners[0]._min_predicted_distance(
        future_peds=kwargs["future_peds"], mask=np.ones(1)
    ) == pytest.approx(1.5 if model == "center_v1" else 1.0)

"""Regression oracles for prediction hunt A1-A4 using production inputs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import _build_socnav_config
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from scripts.training import collect_predictive_hardcase_data, collect_predictive_planner_data

ROOT = Path(__file__).resolve().parents[2]


def release_config(planner="prediction_planner"):
    """Read the actual release planner config through the production builder."""
    params = yaml.safe_load((ROOT / f"configs/algos/{planner}_release_v0_0_8.yaml").read_text())
    params["predictive_model_id"] = None
    return _build_socnav_config(params)


def observation(heading=0.0):
    """Build SOCNAV_STRUCT bytes; north-facing robot sees east velocity as -y."""
    return {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([heading]),
            "speed": np.zeros(2),
        },
        "goal": {"current": np.array([5.0, 0.0]), "next": np.array([5.0, 0.0])},
        "pedestrians": {
            "positions": np.array([[2.0, 0.0]]),
            "velocities": np.array([[0.0, -1.0]]),
            "count": np.array([1]),
        },
    }


@pytest.mark.parametrize(
    "collector", [collect_predictive_planner_data, collect_predictive_hardcase_data]
)
@pytest.mark.parametrize("flat", [False, True])
def test_collector_preserves_already_ego_velocity_at_north_heading(collector, flat):
    """The observation producer already rotates east world velocity to [0,-1]."""
    obs = observation(np.pi / 2)
    if flat:
        obs = {
            f"{block}_{key}": value
            for block, values in obs.items()
            for key, value in values.items()
        }
    frame = collector._extract_frame(obs, max_agents=2)
    state, _, mask, _ = collector._frames_to_samples(
        [frame, frame], max_agents=2, horizon_steps=1, ego_conditioning=False
    )
    assert mask[0, 0] == 1
    np.testing.assert_allclose(state[0, 0, 2:4], [0.0, -1.0], atol=1e-6)


@pytest.mark.parametrize("sequence", [False, True])
def test_wall_occupancy_increases_prediction_cost(sequence):
    """Occupied obstacles/combined bytes must raise cost with no pedestrians."""
    planner = PredictionPlannerAdapter(release_config(), allow_fallback=True)
    obs = observation()
    obs.update(
        occupancy_grid=np.zeros((4, 20, 20)),
        occupancy_grid_meta_origin=np.array([-2.0, -2.0]),
        occupancy_grid_meta_size=np.array([4.0, 4.0]),
        occupancy_grid_meta_resolution=np.array([0.2]),
        occupancy_grid_meta_channel_indices=np.array([0, 1, 2, 3]),
        occupancy_grid_meta_use_ego_frame=np.array([0]),
    )
    kwargs = {
        "observation": obs,
        "future_peds": np.zeros((0, 8, 2)),
        "mask": np.zeros(0),
        "steps": 8,
    }

    def score():
        if sequence:
            return planner._score_action_sequence(**kwargs, sequence=[(1.0, 0.0)])
        return planner._score_action(**kwargs, v=1.0, w=0.0)

    free = score()
    obs["occupancy_grid"][[0, 3], :, :] = 1.0
    wall = score()
    assert wall - free == pytest.approx(planner.config.occupancy_weight)


@pytest.mark.parametrize("near", [False, True])
@pytest.mark.parametrize("planner_name", ["prediction_planner", "predictive_mppi"])
def test_release_heading_lattice_keeps_every_heading_distinct(near, planner_name):
    """Both release planners retain distinct configured deltas within the turn limit."""
    planner = PredictionPlannerAdapter(release_config(planner_name), allow_fallback=True)
    future = np.full((1, 8, 2), 10.0 if not near else 0.1)
    mask = np.ones(1)
    candidates = planner._candidate_set(future_peds=future, mask=mask)
    rates = {round(w, 10) for _, w in candidates}
    deltas = set(planner.config.predictive_candidate_heading_deltas)
    if near:
        deltas.update(planner.config.predictive_near_field_heading_deltas)
    assert len(rates) == len(deltas)
    assert len(rates) == (11 if near else 7)
    assert max(abs(w) for w in rates) <= 1.0
    if not near:
        assert sorted(rates)[1] == pytest.approx(-0.523599 / 0.8, abs=1e-7)


def test_mppi_rejects_24_steps_from_eight_step_forecast():
    """Configured 2.4s horizon cannot silently become 0.8s."""
    planner = PredictiveMPPIAdapter(
        build_predictive_mppi_config({"horizon_steps": 24}), allow_fallback=True
    )
    with pytest.raises(ValueError, match="horizon_steps=24.*8"):
        planner._predict_future(observation())


def test_mppi_preserves_requested_shorter_horizon():
    """A supported 4-step request is not expanded to all 8 model outputs."""
    planner = PredictiveMPPIAdapter(
        build_predictive_mppi_config({"horizon_steps": 4}), allow_fallback=True
    )
    assert planner._predict_future(observation())[2] == 4

"""Seed-free checks for early passing and soft pedestrian clearance."""

import numpy as np

from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.velocity_sampler_r4 import PassingVelocitySamplerAdapter as Adapter
from robot_sf.planner.velocity_sampler_r4 import PassingWeights as Weights


def test_prefers_clear_forecast_without_vetoing_all_motion() -> None:
    """Prefer a clear path when available, but select a path when both are close."""
    adapter = Adapter(
        config=SocNavPlannerConfig(sampling_pedestrian_prediction=True),
        weights=Weights(),
    )
    adapter._continuation_index = None
    points = np.array([[[0.0, 0.0], [1.7, 0.0]], [[0.0, 0.0], [1.1, 0.0]]])
    costs = np.array([0.0, 0.3])
    safe = np.array([True, True])
    speeds = np.array([1.0, 0.4])
    brake_clear = np.ones((2, 1))
    times = np.array([0.0, 1.0])
    peds = np.array([[3.0, 0.0]])
    velocity = np.zeros((1, 2))
    direction = np.array([1.0, 0.0])
    chosen = adapter._select_candidate(
        costs,
        safe,
        speeds,
        brake_clear,
        np.array([[-0.1], [0.5]]),
        points,
        times,
        peds,
        velocity,
        direction,
    )
    assert chosen == 1
    chosen = adapter._select_candidate(
        costs,
        safe,
        speeds,
        brake_clear,
        np.array([[-0.1], [-0.2]]),
        points,
        times,
        peds,
        velocity,
        direction,
    )
    assert chosen in (0, 1)
    assert speeds[chosen] > 0


def test_steers_early_for_an_oncoming_pedestrian() -> None:
    """Choose a passing direction before an oncoming pedestrian is in braking range."""
    from types import SimpleNamespace

    from robot_sf.robot.differential_drive import DifferentialDriveSettings

    adapter = Adapter(
        config=SocNavPlannerConfig(
            socnav_sampling_version="bounded_v2",
            sampling_pedestrian_prediction=True,
            sampling_path_distance=False,
            max_linear_speed=2.0,
            max_angular_speed=1.0,
        ),
        weights=Weights(
            horizon_s=1.5,
            pedestrian_weight=3.0,
            obstacle_weight=2.0,
            clearance_scale=0.3,
            stage_s=0.4,
            continuation_s=0.2,
        ),
    )
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: []),
        )
    )
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [2.0, 0.0], "radius": [1.0]},
        "goal": {"current": [30.0, 0.0]},
        "pedestrians": {
            "positions": [[9.0, 0.1]],
            "velocities": [[-0.65, 0.0]],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    assert adapter.plan(obs)[1] < 0


def test_multiple_nearby_people_keep_the_existing_selector() -> None:
    """Do not apply a one-person clearance preference to a crowd."""
    adapter = Adapter(
        config=SocNavPlannerConfig(sampling_pedestrian_prediction=True),
        weights=Weights(),
    )
    adapter._continuation_index = None
    points = np.array([[[0.0, 0.0], [1.7, 0.0]], [[0.0, 0.0], [1.1, 0.0]]])
    chosen = adapter._select_candidate(
        np.array([0.0, 0.3]),
        np.array([True, True]),
        np.array([1.0, 0.4]),
        np.ones((2, 1)),
        np.array([[-0.1], [0.5]]),
        points,
        np.array([0.0, 1.0]),
        np.array([[3.0, 0.0], [3.0, 2.0]]),
        np.zeros((2, 2)),
        np.array([1.0, 0.0]),
    )
    assert chosen == 0

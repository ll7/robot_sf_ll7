"""Seed-free behavior checks for recovery after sustained lack of progress."""

from types import SimpleNamespace

import pytest

from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.velocity_sampler_r3 import (
    StallRecoveryVelocitySamplerAdapter as RecoveryAdapter,
)
from robot_sf.planner.velocity_sampler_r3 import (
    StallRecoveryWeights as RecoveryWeights,
)
from robot_sf.robot.differential_drive import DifferentialDriveSettings


def _adapter(walls=()):
    config = SocNavPlannerConfig(
        socnav_sampling_version="bounded_v2",
        sampling_pedestrian_prediction=True,
        sampling_path_distance=False,
    )
    weights = RecoveryWeights(
        horizon_s=1.5,
        pedestrian_weight=3.0,
        obstacle_weight=2.0,
        clearance_scale=0.3,
        stage_s=0.4,
        continuation_s=0.2,
    )
    adapter = RecoveryAdapter(config=config, weights=weights)
    adapter.bind_env(
        SimpleNamespace(
            env_config=SimpleNamespace(robot_config=DifferentialDriveSettings()),
            simulator=SimpleNamespace(iter_obstacle_segments=lambda: walls),
        )
    )
    return adapter


def test_stalled_pedestrian_cost_relief_ends_after_progress() -> None:
    """Relief starts after sustained stalling and ends when the robot moves away."""
    adapter = _adapter()
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0, 0.0], "radius": [1.0]},
        "goal": {"current": [20.0, 0.0]},
        "pedestrians": {
            "positions": [[1.8, 1.2]],
            "velocities": [[0.0, 0.0]],
            "count": [1],
            "radius": [0.4],
        },
        "sim": {"timestep": [0.1]},
    }
    speeds = [adapter.plan(obs)[0] for _ in range(45)]
    assert speeds[:35] == pytest.approx([0.0] * 35)
    assert speeds[-1] == pytest.approx(0.1)
    before_progress = adapter.diagnostics()["recovery_steps"]
    obs["robot"]["position"] = [0.8, 0.0]
    obs["pedestrians"]["positions"] = [[2.6, 1.2]]
    adapter.plan(obs)
    assert adapter.diagnostics()["recovery_steps"] == before_progress
    adapter.reset()
    assert adapter.diagnostics()["recovery_steps"] == 0


def test_stall_recovery_preserves_wall_braking() -> None:
    """The recovery mode still rejects a target whose braking path meets a wall."""
    adapter = _adapter(walls=[((1.6, -5.0), (1.6, 5.0))])
    obs = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [1.0, 0.0], "radius": [1.0]},
        "goal": {"current": [10.0, 0.0]},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.4]},
        "sim": {"timestep": [0.1]},
    }
    speeds = [adapter.plan(obs)[0] for _ in range(45)]
    assert adapter.diagnostics()["recovery_steps"] > 0
    assert speeds[-1] == pytest.approx(0.9)

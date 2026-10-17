"""Regression evidence for real constant-velocity pedestrian TTC rewards."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env.reward import route_completion_v3_reward
from robot_sf.gym_env.snqi_proxy import StepSNQIProxy


def _metrics(position, velocity, *, robot_speed=1.0):
    simulator = SimpleNamespace(
        robot_poses=[((0.0, 0.0), 0.0)],
        robots=[SimpleNamespace(current_speed=(robot_speed, 0.0), radius=0.2)],
        robot_radius=0.2,
        ped_radius=0.3,
        ped_pos=np.array([position]),
        ped_vel=np.array([velocity]),
        last_ped_forces=[],
    )
    return StepSNQIProxy().compute_step_metrics(simulator, dt=0.1)


def test_head_on_ttc_reaches_reward_metadata_without_copying_clearance():
    """Five metres at 2 m/s relative speed contacts 0.5 m discs after 2.25 s."""
    meta = _metrics((5.0, 0.0), (-1.0, 0.0))
    assert meta.get("time_to_collision", float("inf")) == pytest.approx(2.25)
    route_completion_v3_reward(meta, weights={"ttc_risk": -1.0, "near_miss": -1.0})
    assert meta["reward_terms"]["ttc_risk"] == pytest.approx(-1.0 / 2.25)
    assert meta["reward_terms"]["near_miss"] == 0.0


@pytest.mark.parametrize(
    ("position", "velocity", "robot_speed", "expected"),
    [
        ((1.6, 0.0), (2.0, 0.0), 1.0, float("inf")),
        ((5.0, 2.0), (-1.0, 0.0), 1.0, float("inf")),
        ((1.6, 0.0), (0.0, 0.0), 0.0, float("inf")),
        ((0.4, 0.0), (0.0, 0.0), 0.0, 0.0),
        ((1.6, 0.0), (float("nan"), 0.0), 1.0, float("inf")),
    ],
)
def test_ttc_uses_contact_geometry_and_relative_velocity(position, velocity, robot_speed, expected):
    meta = _metrics(position, velocity, robot_speed=robot_speed)
    assert meta.get("time_to_collision", -1.0) == expected
    route_completion_v3_reward(meta, weights={"ttc_risk": -1.0})
    assert meta["reward_terms"]["ttc_risk"] == (-1.0 if expected == 0.0 else 0.0)


def test_missing_ttc_does_not_duplicate_near_miss():
    meta = {"near_misses": 1.0}
    route_completion_v3_reward(meta, weights={"ttc_risk": -1.0})
    assert meta["reward_terms"]["ttc_risk"] == 0.0

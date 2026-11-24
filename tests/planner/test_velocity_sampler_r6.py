"""Development checks for the crowd clearance and narrow passing rules."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.planner.velocity_sampler_r4 import PassingWeights
from robot_sf.planner.velocity_sampler_r6 import ClearanceVelocitySamplerAdapter


def _sampler():
    return ClearanceVelocitySamplerAdapter(
        config=SocNavPlannerConfig(sampling_pedestrian_prediction=True),
        weights=PassingWeights(),
    )


@pytest.mark.parametrize(("forward_distance", "expected"), [(1.7, 1), (0.5, 0)])
def test_crowd_clearance_requires_comparable_forward_progress(forward_distance, expected) -> None:
    """Pay for clearance when a clear forecast can advance at a comparable rate."""
    sampler = _sampler()
    sampler.reset(seed=1163)
    sampler._continuation_index = None
    points = np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [forward_distance, 0.0]]])

    chosen = sampler._select_candidate(
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

    assert chosen == expected


@pytest.mark.parametrize(
    ("half_width", "brake_safe", "expected"),
    [(1.8, True, 1), (4.0, True, 0), (1.8, False, 0)],
)
def test_yields_before_continuing_through_a_narrow_pass(half_width, brake_safe, expected) -> None:
    """Choose a wall-clear braking profile if the available pass is too narrow."""
    sampler = _sampler()
    sampler.reset(seed=1163)
    sampler._passing_offset = 1.7
    sampler._velocity_drive = SimpleNamespace(radius=1.0)
    sampler._walls = np.array([[-1.0, side, 10.0, side] for side in (-half_width, half_width)])
    sampler._profiles = [(2.0, 0.0, 2.0, 0.0, 2, 0, 1), (0.0, 0.0, 1.3, 0.0, 4, 2, 2)]
    sampler._continuation_index = 0

    chosen = sampler._select_candidate(
        np.array([0.0, 0.3]),
        np.array([True, brake_safe]),
        np.array([1.0, 0.4]),
        np.ones((2, 1)),
        np.array([[0.2], [0.25]]),
        np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [1.1, 0.0]]]),
        np.array([0.0, 1.0]),
        np.array([[3.0, 0.0]]),
        np.array([[-0.7, 0.0]]),
        np.array([1.0, 0.0]),
    )

    assert chosen == expected

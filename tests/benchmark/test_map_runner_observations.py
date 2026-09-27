"""Tests for the map-runner observation bridge frame transparency (issue #9752)."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.map_runner.map_runner_observations import normalize_map_observation


def test_normalize_map_observation_preserves_velocity_values_unchanged() -> None:
    """The bridge must mirror flat fields without frame conversion (issue #9752).

    Pedestrian velocities stay ego-frame through normalization; every consumer
    converts to its own working frame. Any future conversion here must be an
    explicit, reviewed contract change.
    """
    flat = {
        "robot_position": [0.0, 0.0],
        "robot_heading": [1.57079632679],
        "robot_speed": [0.0],
        "robot_radius": [0.25],
        "goal_current": [-4.0, 0.0],
        "goal_next": [-4.0, 0.0],
        "pedestrians_positions": [[-2.0, 0.0]],
        "pedestrians_velocities": [[-0.65, -0.04]],
        "pedestrians_count": [1],
        "pedestrians_radius": [0.25],
        "sim_timestep": 0.1,
    }
    nested = normalize_map_observation(flat)
    assert nested["pedestrians"]["velocities"] == [[-0.65, -0.04]]
    assert nested["pedestrians"]["positions"] == [[-2.0, 0.0]]
    np.testing.assert_allclose(
        np.asarray(nested["pedestrians"]["velocities"]),
        np.asarray(flat["pedestrians_velocities"]),
    )

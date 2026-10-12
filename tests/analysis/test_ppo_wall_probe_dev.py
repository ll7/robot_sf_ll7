"""Independent geometry and claim-boundary oracles for the dev wall probe."""

import numpy as np
import pytest

from scripts.analysis.ppo_wall_probe_dev import classify, forward_wall_distance


def test_cell_centers_channel_order_and_world_rotation():
    """A wall one meter ahead stays there in ego and rotated world frames."""
    grid = np.zeros((2, 2, 3))
    grid[1, 0, 1] = 1
    obs = {
        "occupancy_grid": grid,
        "occupancy_grid_meta_channel_indices": np.array([1, 0, -1, -1]),
        "occupancy_grid_meta_resolution": np.array([1.0]),
        "occupancy_grid_meta_origin": np.array([-0.5, -0.5]),
        "occupancy_grid_meta_use_ego_frame": np.array([1]),
        "robot_radius": np.array([0.2]),
    }
    assert forward_wall_distance(obs) == pytest.approx(1.0)
    obs["occupancy_grid_meta_use_ego_frame"] = np.array([0])
    obs["occupancy_grid_meta_robot_pose"] = np.array([1.0, -1.0, np.pi / 2])
    assert forward_wall_distance(obs) == pytest.approx(1.0)
    obs["occupancy_grid_meta_channel_indices"][0] = -1
    with pytest.raises(ValueError, match="Static obstacle channel missing"):
        forward_wall_distance(obs)


def test_probe_does_not_promote_missing_or_slow_states():
    """The all-offset signature rejects absent wall cells and one slowing action."""
    rows = [{"forward_wall_cell_m": d, "adapter_command": [2.0, 0]} for d in (4, 3, 2, 1)]
    assert classify(rows) == "holds"
    rows[-1]["adapter_command"][0] = 1.0
    assert classify(rows) == "does_not_hold"
    rows[-1]["forward_wall_cell_m"] = None
    assert classify(rows) == "not_identifiable"
    assert classify(rows[:2]) == "not_identifiable"


def test_velocity_delta_is_not_the_absolute_target_speed():
    """A small positive delta can maintain a 2 m/s target near an approaching wall."""
    rows = [
        {
            "forward_wall_cell_m": d,
            "raw_actor_velocity_delta_mean": [0.2, 0.0],
            "adapter_command": [2.0, 0.0],
        }
        for d in (4.0, 3.0, 2.0, 1.0)
    ]
    assert classify(rows) == "holds"
    rows[-1]["adapter_command"][0] = 1.0
    assert classify(rows) == "does_not_hold"

"""Contract tests for robot-to-pedestrian force helpers."""

import numpy as np

from robot_sf.ped_npc.ped_robot_force import (
    der_euclid_dist,
    ped_robot_force,
    potential_field_force,
)


def test_ped_robot_force_applies_inverse_cubic_force_inside_threshold() -> None:
    """Apply forces only to pedestrians within the configured activation distance."""
    out_forces = np.zeros((2, 2), dtype=float)
    ped_positions = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=float,
    )

    ped_robot_force.py_func(out_forces, ped_positions, (0.0, 0.0), 2.0)

    assert np.allclose(out_forces[0], [1.0, 0.0])
    assert np.allclose(out_forces[1], [0.0, 0.0])


def test_force_helper_math_contracts() -> None:
    """Keep derivative and potential helpers aligned with the documented force field."""
    dx_dist, dy_dist = der_euclid_dist.py_func((3.0, 4.0), (0.0, 0.0), 5.0)
    force_x, force_y = potential_field_force.py_func(2.0, dx_dist, dy_dist)

    assert np.allclose([dx_dist, dy_dist], [0.6, 0.8])
    assert np.allclose([force_x, force_y], [0.075, 0.1])

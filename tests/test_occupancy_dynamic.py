"""Tests for occupancy dynamic collision helpers."""

from __future__ import annotations

import numpy as np

from robot_sf.nav.occupancy import (
    ContinuousOccupancy,
    circle_collides_any,
    circle_collides_any_lines,
)


def test_circle_collides_any_detects_collision() -> None:
    """Detect circle collisions so occupancy checks stay correct after refactors."""
    circle = ((0.0, 0.0), 1.0)
    others = [((2.0, 0.0), 1.0)]
    assert circle_collides_any(circle, others) is True


def test_circle_collides_any_lines_detects_collision() -> None:
    """Detect circle-line collisions to prevent false negatives in obstacle checks."""
    circle = ((0.0, 0.0), 1.0)
    segments = np.array(
        [
            [2.0, 0.0, 3.0, 0.0],
            [0.5, -2.0, 0.5, 2.0],
        ]
    )
    assert circle_collides_any_lines(circle, segments) is True


def test_dynamic_collision_uses_dynamic_objects() -> None:
    """Honor dynamic object providers so runtime collision checks include moving objects."""
    occ = ContinuousOccupancy(
        width=10.0,
        height=10.0,
        get_agent_coords=lambda: (0.0, 0.0),
        get_goal_coords=lambda: (5.0, 5.0),
        get_obstacle_coords=lambda: np.zeros((0, 4)),
        get_pedestrian_coords=lambda: np.zeros((0, 2)),
        get_dynamic_objects=lambda: [((0.5, 0.0), 0.6)],
        agent_radius=0.5,
    )
    assert occ.is_dynamic_collision is True

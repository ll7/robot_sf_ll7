"""Tests for pedestrian population sampling safeguards."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.ped_npc.ped_population import sample_route


def test_sample_route_fails_when_obstacles_block_sampling():
    """Ensure obstacle constraints are enforced to prevent invalid pedestrian spawns."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.0, 0.0), (1.0, 0.0)],
        spawn_zone=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        goal_zone=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
    )
    obstacles = [[(-1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-1.0, 1.0)]]

    with pytest.raises(RuntimeError, match="without violating obstacle constraints"):
        sample_route(
            route,
            num_samples=3,
            sidewalk_width=1.0,
            obstacle_polygons=obstacles,
            rng=np.random.default_rng(0),
        )

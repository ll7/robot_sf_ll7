"""
Tests for pedestrian route sampling edge cases.

These tests ensure obstacle-aware sampling fails fast instead of silently
dropping obstacle constraints when a route is fully blocked.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.ped_npc.ped_population import sample_route


def test_sample_route_raises_when_obstacles_block_sampling() -> None:
    """Verify obstacle-aware route sampling raises when no valid points exist."""
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.0, 0.0), (1.0, 0.0)],
        spawn_zone=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        goal_zone=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
    )
    obstacle = [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)]

    with pytest.raises(RuntimeError, match="without violating obstacle constraints"):
        sample_route(
            route,
            num_samples=2,
            sidewalk_width=1.0,
            obstacle_polygons=[obstacle],
            rng=np.random.default_rng(0),
        )

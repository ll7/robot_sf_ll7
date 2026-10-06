"""The combined reset policy keeps both scalar and per-pedestrian clearances."""

from math import dist

import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.spawn_clearance import relocate_overlapping_pedestrians


def test_combined_relocation_keeps_larger_margin_independently_per_row() -> None:
    """A short per-row reaction distance cannot shrink the scalar start buffer."""
    map_def = MapDefinition(
        width=40.0,
        height=20.0,
        obstacles=[],
        bounds=[
            (0.0, 40.0, 0.0, 0.0),
            (40.0, 40.0, 0.0, 20.0),
            (40.0, 0.0, 20.0, 20.0),
            (0.0, 0.0, 20.0, 0.0),
        ],
        robot_spawn_zones=[((1.0, 1.0), (3.0, 1.0), (3.0, 3.0))],
        robot_goal_zones=[((36.0, 16.0), (38.0, 16.0), (38.0, 18.0))],
        robot_routes=[
            GlobalRoute(
                spawn_id=0,
                goal_id=0,
                waypoints=[(2.0, 2.0), (37.0, 17.0)],
                spawn_zone=((1.0, 1.0), (3.0, 1.0), (3.0, 3.0)),
                goal_zone=((36.0, 16.0), (38.0, 16.0), (38.0, 18.0)),
            )
        ],
        ped_spawn_zones=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
    )
    robots = [((10.8, 10.0), 1.0), ((30.8, 10.0), 1.0)]
    report = relocate_overlapping_pedestrians(
        [(10.0, 10.0), (30.0, 10.0)],
        0.4,
        robots,
        map_def,
        robot_margin=0.75,
        reaction_clearance_m=[0.2, 1.0],
    )
    assert report.unresolved == []
    assert set(report.relocated) == {0, 1}
    # row 0: max(0.75, 0.1 + 0.2); row 1: max(0.75, 0.1 + 1.0).
    for row, expected in enumerate([0.75, 1.1]):
        gap = dist(report.relocated[row][1], robots[row][0]) - 1.0 - 0.4
        assert gap == pytest.approx(expected + 1e-6)

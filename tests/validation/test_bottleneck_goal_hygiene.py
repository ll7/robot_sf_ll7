"""Successor bottleneck goals are feasible geometry, through the real loader."""

import os
from pathlib import Path

import pytest
from shapely.geometry import LineString, Point, box

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation.check_scenario_archetype_geometry import _rect_polygon

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_goal_hygiene_0_1_0_v1.yaml")
WALL_LAW_MATRIX = Path("configs/scenarios/issue_10304_wall_law_bottleneck_successor_v1.yaml")
BODY_RADIUS_M = 0.40


def test_bottleneck_goal_is_in_bounds_clear_and_contains_route_endpoint():
    # Select the pre-fix source explicitly when proving this test on base.
    matrix = Path(os.environ.get("SCENARIO_MATRIX", str(MATRIX)))
    scenario = next(s for s in load_scenarios(matrix) if s["name"] == "classic_bottleneck_low")
    config = build_robot_config_from_scenario(scenario, scenario_path=matrix.resolve())
    definition = next(iter(config.map_pool.map_defs.values()))
    goal = _rect_polygon(definition.ped_goal_zones[0])
    assert goal.difference(box(0, 0, definition.width, definition.height)).area == pytest.approx(0)
    for obstacle in definition.obstacles:
        for polygon in obstacle.iter_polygons():
            assert goal.intersection(polygon).area == pytest.approx(0)
    assert goal.covers(Point(definition.ped_routes[0].waypoints[-1]))


def test_wall_law_bottleneck_successor_single_pedestrians_are_legal_and_routed():
    scenario = next(
        s
        for s in load_scenarios(WALL_LAW_MATRIX)
        if s["name"] == "classic_realworld_double_bottleneck_high"
    )
    config = build_robot_config_from_scenario(scenario, scenario_path=WALL_LAW_MATRIX.resolve())
    definition = next(iter(config.map_pool.map_defs.values()))
    obstacles = [
        polygon for obstacle in definition.obstacles for polygon in obstacle.iter_polygons()
    ]

    assert len(definition.single_pedestrians) == 8
    for ped in definition.single_pedestrians:
        assert ped.trajectory is not None and len(ped.trajectory) >= 2, ped.id
        route = [ped.start, *ped.trajectory]
        direct = LineString([ped.start, route[-1]])
        swept_route = LineString(route).buffer(BODY_RADIUS_M, cap_style="round")
        start_body = Point(ped.start).buffer(BODY_RADIUS_M, quad_segs=32)

        for obstacle in obstacles:
            assert start_body.intersection(obstacle).area == pytest.approx(0), ped.id
            assert direct.intersection(obstacle).area == pytest.approx(0), ped.id
            assert swept_route.intersection(obstacle).area == pytest.approx(0), ped.id

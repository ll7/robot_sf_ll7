"""Successor bottleneck goals are feasible geometry, through the real loader."""

import os
from pathlib import Path

import pytest
from shapely.geometry import Point, box

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation.check_scenario_archetype_geometry import _rect_polygon

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_goal_hygiene_0_1_0_v1.yaml")


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

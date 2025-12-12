"""Lightweight performance sanity checks for planner (US4)."""
# ruff: noqa: D103

import time

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig


def test_cached_planning_is_not_slower_than_first_call():
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")
    config = PlannerConfig(cache_graphs=True)
    planner = GlobalPlanner(map_def, config=config)

    start = (1.2, 4.5)
    goal = (18.0, 4.5)

    t0 = time.perf_counter()
    planner.plan(start, goal)
    first = time.perf_counter() - t0

    t1 = time.perf_counter()
    planner.plan(start, goal)
    second = time.perf_counter() - t1

    assert second <= first * 5  # ensure caching doesn't regress badly

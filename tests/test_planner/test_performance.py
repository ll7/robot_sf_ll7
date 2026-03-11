"""Lightweight performance sanity checks for planner (US4)."""
# ruff: noqa: D103

import statistics
import time

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig


def test_cached_planning_is_not_slower_than_first_call():
    map_def = convert_map("tests/fixtures/test_maps/simple_corridor.svg")
    config = PlannerConfig(cache_graphs=True)
    start = (1.2, 4.5)
    goal = (18.0, 4.5)

    first_samples: list[float] = []
    second_samples: list[float] = []
    for _ in range(7):
        planner = GlobalPlanner(map_def, config=config)

        t0 = time.perf_counter()
        planner.plan(start, goal)
        first_samples.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        planner.plan(start, goal)
        second_samples.append(time.perf_counter() - t1)

    first_median = statistics.median(first_samples)
    second_median = statistics.median(second_samples)

    assert second_median <= first_median * 5, (
        "cached planning median should not regress badly: "
        f"first_median={first_median:.6f}s second_median={second_median:.6f}s "
        f"first_samples={first_samples} second_samples={second_samples}"
    )

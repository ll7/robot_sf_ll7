"""Simple benchmark for global planner build/query times."""

import time

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig


def main() -> None:
    """Benchmark global planner build and cached query times."""
    map_def = convert_map("tests/fixtures/test_maps/complex_warehouse.svg")
    planner = GlobalPlanner(map_def, PlannerConfig(cache_graphs=True))

    start = (1.0, 7.0)
    goal = (22.0, 12.0)

    t0 = time.perf_counter()
    planner.plan(start, goal)
    cold = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    planner.plan(start, goal)
    warm = (time.perf_counter() - t1) * 1000

    print(f"Cold build+plan: {cold:.1f} ms")
    print(f"Warm cached plan: {warm:.1f} ms")


if __name__ == "__main__":
    main()

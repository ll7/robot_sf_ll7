"""Global planner demo: load a map, plan a path, and print waypoints."""

from pathlib import Path

from robot_sf.common.types import Vec2D
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig


def main() -> None:
    """Run a simple waypoint planning demo on the classic bottleneck map."""
    map_path = Path("maps/svg_maps/classic_bottleneck.svg")
    map_def = convert_map(str(map_path))
    planner = GlobalPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.4,
            min_safe_clearance=0.3,
            enable_smoothing=True,
            smoothing_epsilon=0.2,
        ),
    )

    start: Vec2D = (5.0, 20.0)
    goal: Vec2D = (35.0, 20.0)
    path = planner.plan(start, goal, via_pois=["poi_entry", "poi_exit"])

    print(f"Planned path on {map_path.name} with {len(path)} waypoints:")
    for idx, wp in enumerate(path):
        print(f"  {idx}: ({wp[0]:.2f}, {wp[1]:.2f})")


if __name__ == "__main__":
    main()

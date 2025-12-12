"""Test script to verify planner paths don't collide with obstacles."""

from pathlib import Path

from shapely.geometry import LineString

from robot_sf.common.logging import configure_logging
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, plot_global_plan


def main() -> None:
    """Test planner paths for collisions."""
    configure_logging()

    map_path = Path("maps/svg_maps/MIT_corridor.svg")
    map_def = convert_map(str(map_path))

    planner = GlobalPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.25,
            min_safe_clearance=0.5,
            enable_smoothing=False,
        ),
    )

    start = (5.0, 5.0)
    goal = (45.0, 25.0)
    path = planner.plan(start, goal)

    print(f"\nPath has {len(path)} waypoints:")
    for i, wp in enumerate(path):
        print(f"  {i}: {wp}")

    if len(path) == 0:
        print("\nNo path found; skipping collision check and plotting.")
        return
    if len(path) == 1:
        print("\nDegenerate path (single waypoint); skipping collision check and plotting.")
        return

    try:
        # Check if path intersects any actual obstacles
        collision_obstacles = planner._inflate_obstacles_for_collision()
        path_line = LineString(path)

        print(f"\nChecking path against {len(collision_obstacles)} collision obstacles...")
        collisions = 0
        for i, obs in enumerate(collision_obstacles):
            if obs.intersects(path_line) and not obs.touches(path_line):
                collisions += 1
                print(f"  ✗ COLLISION with obstacle {i}!")

        if collisions == 0:
            print("  ✓ No collisions detected")
        else:
            print(f"\n⚠️  FOUND {collisions} COLLISIONS!")
    except Exception as exc:
        print(f"\nError during collision check: {exc}")
        return

    # Save visualization
    try:
        plot_global_plan(
            planner,
            path,
            title="Planner Test - Collision Check",
            save_path=Path("output/plots/planner_collision_test.png"),
            show=False,
            flip_y=True,
        )
    except Exception as exc:
        print(f"\nError during plotting: {exc}")
    else:
        print("\nVisualization saved to output/plots/planner_collision_test.png")


if __name__ == "__main__":
    main()

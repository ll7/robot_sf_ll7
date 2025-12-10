"""POI routing demo using GlobalPlanner and POISampler."""

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, POISampler


def main() -> None:
    """Demonstrate routing through randomly sampled POIs."""
    map_path = Path("maps/svg_maps/MIT_corridor.svg")
    map_def = convert_map(str(map_path))

    planner = GlobalPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.4,
            min_safe_clearance=0.3,
            enable_smoothing=True,
            smoothing_epsilon=0.15,
        ),
    )

    sampler = POISampler(map_def, seed=7)
    via_points = sampler.sample(count=2, strategy="random")

    start = (5.0, 5.0)
    goal = (45.0, 25.0)

    path = planner.plan(start, goal, via_pois=list(map_def.poi_labels.keys())[: len(via_points)])

    print(f"POI routing on {map_path.name} with {len(path)} waypoints:")
    for idx, wp in enumerate(path):
        print(f"  {idx}: ({wp[0]:.2f}, {wp[1]:.2f})")


if __name__ == "__main__":
    main()

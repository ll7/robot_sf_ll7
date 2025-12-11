"""POI routing demo using GlobalPlanner and POISampler with live visualization."""

from pathlib import Path

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, POISampler, plot_global_plan


def main() -> None:
    """Demonstrate routing through randomly sampled POIs with a Matplotlib plot."""
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
    start = (5.0, 5.0)
    goal = (45.0, 25.0)

    via_ids = sampler.sample_ids(count=2, strategy="random", start=start)
    poi_ids = list(map_def.poi_labels.keys())
    via_points = [map_def.poi_positions[poi_ids.index(poi_id)] for poi_id in via_ids]

    path = planner.plan(start, goal, via_pois=via_ids)

    print(f"POI routing on {map_path.name} with {len(path)} waypoints:")
    for idx, wp in enumerate(path):
        print(f"  {idx}: ({wp[0]:.2f}, {wp[1]:.2f})")

    plot_global_plan(
        planner=planner,
        path=path,
        via_points=via_points,
        title=f"Global planner route on {map_path.name}",
        save_path=Path("output/plots/poi_routing_demo.png"),
        show=True,
    )


if __name__ == "__main__":
    main()

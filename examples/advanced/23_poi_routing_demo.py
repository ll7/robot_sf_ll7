"""POI routing demo using GlobalPlanner and POISampler with live visualization."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib

from robot_sf.common.logging import configure_logging
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner import GlobalPlanner, PlannerConfig, POISampler, plot_global_plan


def _ensure_interactive_backend() -> None:
    """Switch away from headless Agg when possible to show the plot interactively."""
    backend = matplotlib.get_backend().lower()
    if backend != "agg":
        return
    for candidate in ("MacOSX", "TkAgg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


def _visualize_planning_debug(  # noqa: C901
    planner: GlobalPlanner, path: list, via_points: list, map_path: Path
) -> None:
    """Create detailed debug visualization showing inflated obstacles and path."""
    try:
        import matplotlib.pyplot as plt
        from shapely.geometry import Polygon
    except ImportError:
        return

    _fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Get inflated obstacles for display
    inflated = planner._inflate_obstacles()

    # Analyze clearance along path
    from shapely.geometry import LineString

    min_clearance = float("inf")
    clearance_violations = []

    for i, (p1, p2) in enumerate(itertools.pairwise(path)):
        segment = LineString([p1, p2])
        for j, poly in enumerate(inflated):
            dist = segment.distance(poly.exterior)
            min_clearance = min(min_clearance, dist)
            if dist < 0.01:  # Nearly touching
                clearance_violations.append((i, j, dist))

    # Left: Inflated obstacle zones
    ax = axes[0]
    title = f"Inflated Obstacle Zones (radius={planner.config.robot_radius}m + clearance={planner.config.min_safe_clearance}m)"
    if clearance_violations:
        title += f"\n⚠️ {len(clearance_violations)} segments violate clearance!"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    # Show inflated obstacles as keep-out zones
    for i, poly in enumerate(inflated):
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, color="red", edgecolor="darkred", linewidth=1.5)
            # Label obstacles
            centroid = poly.centroid
            ax.text(
                centroid.x,
                centroid.y,
                f"O{i}",
                fontsize=8,
                ha="center",
                color="darkred",
                fontweight="bold",
            )

    # Plot start, goal, and via points
    ax.plot(path[0][0], path[0][1], "go", markersize=12, label="Start", zorder=5)
    ax.plot(path[-1][0], path[-1][1], "r*", markersize=18, label="Goal", zorder=5)
    for vp in via_points:
        ax.plot(vp[0], vp[1], "y^", markersize=10, label="Via" if vp == via_points[0] else "")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, planner.map_def.width)
    ax.set_ylim(0, planner.map_def.height)

    # Right: Computed path with segment analysis
    ax = axes[1]
    violation_note = f"Path min clearance: {min_clearance:.3f}m" if min_clearance < 0.1 else ""
    ax.set_title(f"Planned Path Analysis\n{violation_note}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")

    # Show inflated obstacles
    for poly in inflated:
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.2, color="red", edgecolor="darkred", linewidth=1)

    # Plot path segments with clearance analysis
    for i, (p1, p2) in enumerate(itertools.pairwise(path)):
        # Check if this segment has a violation
        is_violation = any(seg_idx == i for seg_idx, _, _ in clearance_violations)
        color = "red" if is_violation else "blue"
        linewidth = 4 if is_violation else 3

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, alpha=0.8, zorder=4
        )
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b--", linewidth=1, alpha=0.3)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(
            mid[0],
            mid[1],
            f"seg{i}",
            fontsize=9,
            ha="center",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # Plot path nodes
    for i, pt in enumerate(path):
        ax.plot(pt[0], pt[1], "bo", markersize=6, zorder=5)

    ax.plot(path[0][0], path[0][1], "go", markersize=12, label="Start", zorder=6)
    ax.plot(path[-1][0], path[-1][1], "r*", markersize=18, label="Goal", zorder=6)
    for vp in via_points:
        ax.plot(vp[0], vp[1], "y^", markersize=10, zorder=6)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, planner.map_def.width)
    ax.set_ylim(0, planner.map_def.height)

    plt.tight_layout()
    save_path = Path("output/plots/poi_routing_debug.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[DEBUG] Saved debug visualization to {save_path}")
    print("[DEBUG] Note: Red zones = inflated obstacles (keep-out regions)")
    print("[DEBUG] Note: Blue path goes through narrow passages = potential issue")
    plt.close()


def main() -> None:
    """Demonstrate routing through randomly sampled POIs with comprehensive debugging."""
    configure_logging(verbose=True)
    _ensure_interactive_backend()

    map_path = Path("maps/svg_maps/MIT_corridor.svg")
    map_def = convert_map(str(map_path))

    # Configure planner with realistic clearance values.
    # Note: robot_radius (0.25m) + min_safe_clearance (0.5m) = 0.75m total inflation.
    # Narrow corridor environments may produce "narrow passage detected" validation warnings,
    # which are informational and indicate the path uses constrained passages.
    # The path is still valid; warnings are triggered when path segments touch inflated boundaries.
    planner = GlobalPlanner(
        map_def,
        PlannerConfig(
            robot_radius=0.25,
            min_safe_clearance=0.5,
            enable_smoothing=False,  # Disable smoothing; can cause narrow passages
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

    # Generate both standard and debug visualizations
    plot_global_plan(
        planner=planner,
        path=path,
        via_points=via_points,
        title=f"Global planner route on {map_path.name}",
        save_path=Path("output/plots/poi_routing_demo.png"),
        show=True,
        flip_y=True,
    )

    # Generate debug visualization showing obstacle inflation
    _visualize_planning_debug(planner, path, via_points, map_path)


if __name__ == "__main__":
    main()

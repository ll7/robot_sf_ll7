#!/usr/bin/env python
"""Diagnose global planner inflation and visibility graph construction.

This script visualizes the intermediate steps of path planning to identify
why paths violate clearance constraints.

Usage:
    uv run python examples/advanced/25_planner_diagnostic.py

Generated outputs:
    - output/plots/planner_diagnostic_inflation.png - Inflation analysis
    - output/plots/planner_diagnostic_visibility_graph.png - Graph structure
"""

from pathlib import Path

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.global_planner import GlobalPlanner, PlannerConfig


def main() -> None:  # noqa: C901
    """Run comprehensive planner diagnostics."""

    # Ensure output directory
    Path("output/plots").mkdir(parents=True, exist_ok=True)

    # Load the test map
    map_path = Path("maps/svg_maps/planner_test_corridor.svg")
    if not map_path.exists():
        print(f"[ERROR] Map not found: {map_path}")
        return

    print(f"\n{'=' * 70}")
    print("Global Planner Diagnostic Analysis")
    print(f"{'=' * 70}\n")

    map_def = convert_map(map_path)
    print(f"[INFO] Map: {map_def.width}x{map_def.height}m, {len(map_def.obstacles)} obstacles")

    # Test with realistic clearance
    config = PlannerConfig(
        robot_radius=0.25,
        min_safe_clearance=0.5,
        enable_smoothing=False,
    )
    total_inflation = config.robot_radius + config.min_safe_clearance
    print(
        f"[INFO] Config: robot_radius={config.robot_radius}m, clearance={config.min_safe_clearance}m"
    )
    print(f"[INFO] Total inflation margin: {total_inflation}m")
    print(
        "[NOTE] Collision envelope = robot radius (validation); planning keep-out = radius + clearance (advisory)."
    )

    planner = GlobalPlanner(map_definition=map_def, config=config)

    # Analyze inflation
    print("\n[STEP 1] Analyzing obstacle inflation:")
    inflated_obstacles = planner._inflate_obstacles()
    collision_obstacles = planner._inflate_obstacles_for_collision()
    graph_obstacles = planner._prepare_graph_obstacles(inflated_obstacles)
    print(f"  Original obstacles: {len(map_def.obstacles)}")
    print(f"  Inflated obstacles: {len(inflated_obstacles)}")

    for i, (orig, inflated) in enumerate(zip(map_def.obstacles, inflated_obstacles, strict=True)):
        orig_poly = Polygon(orig.vertices)
        print(f"  Obstacle {i}:")
        print(f"    Original area: {orig_poly.area:.2f}m²")
        print(f"    Inflated area: {inflated.area:.2f}m²")
        print(f"    Inflation ratio: {inflated.area / orig_poly.area:.2f}x")

    # Analyze graph obstacles
    print("\n[STEP 2] Analyzing visibility graph obstacles:")
    print(f"  Graph obstacles: {len(graph_obstacles)}")

    for i, (inflated, graph_obs) in enumerate(
        zip(inflated_obstacles, graph_obstacles, strict=True)
    ):
        print(f"  Obstacle {i}:")
        print(f"    Inflated area: {inflated.area:.2f}m²")
        print(f"    Graph obstacle area: {graph_obs.area:.2f}m²")
        print(f"    Extra buffer: {(graph_obs.area - inflated.area) / inflated.area * 100:.4f}%")

    # Visualize the layers
    _, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Original obstacles only
    ax = axes[0, 0]
    ax.set_title("Original Map Obstacles", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    for i, obs in enumerate(map_def.obstacles):
        poly = Polygon(obs.vertices)
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.6, color="black", edgecolor="black", linewidth=2)
        centroid = poly.centroid
        ax.text(
            centroid.x,
            centroid.y,
            f"O{i}",
            fontsize=10,
            ha="center",
            color="white",
            fontweight="bold",
        )
    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    # Panel 2: Inflated obstacles (keep-out zones)
    ax = axes[0, 1]
    ax.set_title(
        f"Inflated (planning) margin={total_inflation:.2f}m\nCollision margin={config.robot_radius:.2f}m",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_aspect("equal")

    # Show original in background
    for obs in map_def.obstacles:
        poly = Polygon(obs.vertices)
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=0.3, color="black", edgecolor="black", linewidth=1)

    # Show inflated zones
    for i, inflated in enumerate(inflated_obstacles):
        x, y = inflated.exterior.xy
        ax.fill(x, y, alpha=0.4, color="red", edgecolor="darkred", linewidth=2)
        ax.plot(x, y, "r--", linewidth=1.5, label=f"Inflated {i}" if i == 0 else "")

    # Show collision-only zones (robot radius)
    for i, coll in enumerate(collision_obstacles):
        x, y = coll.exterior.xy
        ax.plot(
            x,
            y,
            color="orange",
            linestyle="--",
            linewidth=1.3,
            label="Collision (radius)" if i == 0 else "",
        )

    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right")

    # Panel 3: Graph obstacles (no extra buffer)
    ax = axes[1, 0]
    ax.set_title("Graph Obstacles (no extra inflation)", fontsize=13, fontweight="bold")
    ax.set_aspect("equal")

    # Show inflated in background
    for inflated in inflated_obstacles:
        x, y = inflated.exterior.xy
        ax.fill(x, y, alpha=0.2, color="red", edgecolor="darkred", linewidth=1)

    # Show graph obstacles
    for i, graph_obs in enumerate(graph_obstacles):
        x, y = graph_obs.exterior.xy
        ax.plot(x, y, "b-", linewidth=2, alpha=0.8, label=f"Graph obstacle {i}" if i == 0 else "")
        # Mark vertices where visibility graph nodes will be placed
        for vx, vy in zip(x, y, strict=False):
            ax.plot(vx, vy, "bo", markersize=4, alpha=0.6)

    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right")

    # Panel 4: Problem illustration
    ax = axes[1, 1]
    ax.set_title(
        "Problem: Paths Can Touch Inflated Boundaries", fontsize=13, fontweight="bold", color="red"
    )
    ax.set_aspect("equal")

    # Show inflated zones
    for inflated in inflated_obstacles:
        x, y = inflated.exterior.xy
        ax.fill(x, y, alpha=0.3, color="red", edgecolor="darkred", linewidth=2)

    # Illustrate the issue: visibility graph allows paths along boundaries
    # Draw a sample path that hugs the boundary
    if graph_obstacles:
        obs = graph_obstacles[0]
        coords = list(obs.exterior.coords)
        if len(coords) >= 3:
            # Pick a few boundary points
            p1 = coords[0]
            p2 = coords[len(coords) // 4]
            p3 = coords[len(coords) // 2]

            ax.plot(
                [p1[0], p2[0], p3[0]],
                [p1[1], p2[1], p3[1]],
                "b-",
                linewidth=3,
                alpha=0.8,
                label="Boundary-hugging path",
            )
            ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], "bo", markersize=8)

            # Annotate the problem
            ax.annotate(
                "Path touches\ninflated boundary\n(0m clearance!)",
                xy=p2,
                xytext=(p2[0] + 2, p2[1] + 2),
                arrowprops={"arrowstyle": "->", "color": "red", "lw": 2},
                fontsize=10,
                color="red",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "yellow", "alpha": 0.8},
            )

    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right")

    plt.tight_layout()
    save_path = Path("output/plots/planner_diagnostic_inflation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Saved diagnostic visualization to {save_path}")
    plt.close()

    # Summary and recommendations
    print(f"\n{'=' * 70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'=' * 70}")
    print("\n[ISSUE IDENTIFIED]")
    print("  The visibility graph places vertices ON inflated obstacle boundaries.")
    print("  Shortest paths naturally run along these boundaries → 0m clearance.")
    print("\n[ROOT CAUSE]")
    print(f"  1. Obstacles inflated by {total_inflation}m (correct)")
    print("  2. Graph obstacles reuse planning inflation (no extra buffer)")
    print("  3. Visibility graph vertices placed on graph obstacle boundaries")
    print("  4. Shortest path algorithm connects these boundary vertices")
    print("  → Result: Path touches inflated boundary with ~0m clearance")
    print("\n[RECOMMENDED FIXES]")
    print("  Option A: Keep zero extra graph buffer and rely on collision validation")
    print("  Option B: Add post-processing to push paths away from boundaries")
    print("  Option C: Increase graph buffer if future scenarios demand extra clearance")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()

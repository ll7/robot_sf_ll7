#!/usr/bin/env python
"""Test global planner with POI routing on a simple bottleneck map.

This script tests the global planner's ability to route through a narrow
bottleneck while maintaining clearance constraints. It's designed to help
debug planner behavior and visualize path solutions.

Usage:
    uv run python examples/advanced/24_planner_bottleneck_test.py

Generated outputs:
    - output/plots/bottleneck_full_route.png - Complete path visualization
    - output/plots/bottleneck_clearance_analysis.png - Clearance analysis
"""

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger
from shapely.geometry import LineString, Polygon

from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.global_planner import GlobalPlanner, PlannerConfig


def main() -> None:  # noqa: C901
    """Run planner test on bottleneck map."""

    # Ensure output directory
    Path("output/plots").mkdir(parents=True, exist_ok=True)

    # Load the simple corridor test map
    map_path = Path("maps/svg_maps/planner_test_corridor.svg")
    if not map_path.exists():
        logger.error(f"Map not found: {map_path}")
        return

    logger.info("Global Planner Bottleneck Test")

    logger.info(f"Loading map from {map_path}")

    # Convert SVG to map definition
    map_def = convert_map(map_path)
    logger.info(
        f"Map loaded: {map_def.width}x{map_def.height}m, {len(map_def.obstacles)} obstacles"
    )

    # Create planner with clearance constraints
    config = PlannerConfig(
        robot_radius=0.25,
        min_safe_clearance=0.5,
        enable_smoothing=False,
    )
    logger.info(
        f"Planner config: robot_radius={config.robot_radius}m, clearance={config.min_safe_clearance}m"
    )

    planner = GlobalPlanner(map_definition=map_def, config=config)

    # Define test waypoints
    start = (3.0, 10.0)
    goal = (47.0, 10.0)
    via_points = []

    logger.info("Simple start-to-goal planning test:")

    # Attempt planning - simple start-to-goal first
    try:
        path = planner.plan(start=start, goal=goal)
        logger.success(f"Path found with {len(path)} waypoints")
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return

    # Analyze clearance (collision contract vs planning advisory)
    inflated = planner._inflate_obstacles()
    collision_obs = planner._inflate_obstacles_for_collision()
    min_clearance_collision = float("inf")
    min_clearance_planning = float("inf")
    collision_violations: list[tuple[int, int, float]] = []
    planning_contacts: list[tuple[int, int, float]] = []

    logger.info("Clearance analysis:")
    logger.debug("Path segments:")
    from itertools import pairwise

    for i, (p1, p2) in enumerate(pairwise(path)):
        segment = LineString([p1, p2])
        logger.debug(f"Segment {i}: ({p1[0]:.2f},{p1[1]:.2f}) → ({p2[0]:.2f},{p2[1]:.2f})")

        # Collision envelope (robot radius only) — authoritative validity check
        for j, poly in enumerate(collision_obs):
            dist_exterior = segment.distance(poly.exterior)
            intersects = segment.intersects(poly)
            logger.debug(
                f"vs Collision Obstacle {j}: dist_exterior={dist_exterior:.4f}m, intersects={intersects}"
            )
            min_clearance_collision = min(min_clearance_collision, dist_exterior)
            if dist_exterior < 0.01 or intersects:
                collision_violations.append((i, j, dist_exterior))
                logger.warning("COLLISION violation (radius check)!")

        # Planning keep-out (radius + clearance) — advisory; can touch boundaries
        for j, poly in enumerate(inflated):
            dist_exterior = segment.distance(poly.exterior)
            intersects = segment.intersects(poly)
            min_clearance_planning = min(min_clearance_planning, dist_exterior)
            if dist_exterior < 0.01 or intersects:
                planning_contacts.append((i, j, dist_exterior))
                logger.debug("Touches planning keep-out (radius+clearance)")

    logger.info(f"Minimum collision clearance: {min_clearance_collision:.4f}m")
    if collision_violations:
        logger.warning(f"{len(collision_violations)} collision violations detected!")
    else:
        logger.info("Collision-safe w.r.t radius-only envelope")

    logger.info(f"Minimum planning clearance (advisory): {min_clearance_planning:.4f}m")
    if planning_contacts:
        logger.debug(
            f"{len(planning_contacts)} segments touch the planning keep-out (expected with boundary-hugging graph paths)"
        )
    else:
        logger.info("No contacts with planning keep-out (radius+clearance)")

    # Create visualization
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Full route with waypoints
    ax = axes[0]
    ax.set_title(
        "Bottleneck Test: Path vs Obstacles\n"
        f"robot_radius={planner.config.robot_radius}m, clearance={planner.config.min_safe_clearance}m",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_aspect("equal")

    # Draw original obstacles
    for i, obs in enumerate(map_def.obstacles):
        poly = Polygon(obs.vertices)
        x, y = poly.exterior.xy
        ax.fill(
            x,
            y,
            alpha=0.4,
            color="black",
            edgecolor="black",
            linewidth=1.5,
            label="Obstacle" if i == 0 else "",
        )

    # Draw planning inflated obstacles (radius + clearance)
    for i, poly in enumerate(inflated):
        if isinstance(poly, Polygon):
            x, y = poly.exterior.xy
            ax.fill(
                x,
                y,
                alpha=0.18,
                color="red",
                edgecolor="darkred",
                linewidth=1,
                label="Planning inflated" if i == 0 else "",
            )

    # Draw collision envelope (robot radius only) around obstacles for reference
    collision_obs = planner._inflate_obstacles_for_collision()
    for i, poly in enumerate(collision_obs):
        x, y = poly.exterior.xy
        ax.plot(
            x,
            y,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="Collision (radius only)" if i == 0 else "",
        )

    # Draw path footprint (robot radius buffer around centerline)
    path_corridor = LineString(path).buffer(planner.config.robot_radius)
    cx, cy = path_corridor.exterior.xy
    ax.fill(
        cx,
        cy,
        alpha=0.15,
        color="blue",
        edgecolor="blue",
        linewidth=1,
        label="Path footprint (robot radius)",
    )

    # Draw path centerline
    for i, (p1, p2) in enumerate(pairwise(path)):
        in_collision = any(s == i for s, _, _ in collision_violations)
        color = "red" if in_collision else "blue"
        linewidth = 3.5 if in_collision else 2.5
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, alpha=0.9, zorder=5
        )

    # Draw waypoints
    ax.plot(start[0], start[1], "go", markersize=12, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=18, label="Goal", zorder=5)
    for i, vp in enumerate(via_points):
        ax.plot(vp[0], vp[1], "y^", markersize=10, zorder=5)
        ax.text(vp[0], vp[1] - 1, f"WP{i + 1}", fontsize=9, ha="center")

    ax.set_xlabel("X [m]", fontsize=11)
    ax.set_ylabel("Y [m]", fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 21)

    # Right: Clearance along path
    ax = axes[1]
    ax.set_title("Clearance Analysis Along Path", fontsize=13, fontweight="bold")

    segment_clearances_collision = []
    segment_clearances_planning = []
    segment_labels = []
    for i, (p1, p2) in enumerate(pairwise(path)):
        segment = LineString([p1, p2])
        min_dist_coll = min(segment.distance(poly.exterior) for poly in collision_obs)
        min_dist_plan = min(segment.distance(poly.exterior) for poly in inflated)
        segment_clearances_collision.append(min_dist_coll)
        segment_clearances_planning.append(min_dist_plan)
        segment_labels.append(f"Seg {i}")

    indices = list(range(len(segment_clearances_collision)))
    bar_width = 0.4
    colors = ["red" if c < 0.01 else "green" for c in segment_clearances_collision]
    ax.bar(
        [i - bar_width / 2 for i in indices],
        segment_clearances_collision,
        width=bar_width,
        color=colors,
        alpha=0.75,
        label="Collision clearance (radius)",
    )
    ax.bar(
        [i + bar_width / 2 for i in indices],
        segment_clearances_planning,
        width=bar_width,
        color="orange",
        alpha=0.55,
        label="Planning clearance (radius+clearance)",
    )
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Collision boundary")
    ax.axhline(
        y=planner.config.min_safe_clearance,
        color="gray",
        linestyle=":",
        linewidth=1.3,
        label="Desired clearance",
    )
    ax.set_xticks(indices)
    ax.set_xticklabels(segment_labels, rotation=45, ha="right")
    ax.set_ylabel("Clearance Distance [m]", fontsize=11)
    ax.set_xlabel("Path Segment", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    plt.tight_layout()
    save_path = Path("output/plots/bottleneck_full_route.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved full route visualization to {save_path}")
    plt.close()

    # Summary
    logger.info("=" * 70)
    logger.info("Test Complete")
    logger.info("=" * 70)
    path_len = sum(
        (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5) for p1, p2 in pairwise(path)
    )
    logger.info(f"Path length: {path_len:.2f}m")
    logger.info(f"Min collision clearance: {min_clearance_collision:.4f}m")
    logger.info(f"Min planning clearance (advisory): {min_clearance_planning:.4f}m")
    status = "PASS" if not collision_violations else "FAIL - Collision violations!"
    logger.info(f"Status: {status}")


if __name__ == "__main__":
    main()

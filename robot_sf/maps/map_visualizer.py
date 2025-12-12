"""Matplotlib visualizer for MapDefinition objects.

This module renders map elements (obstacles, spawn/goal zones, routes, crowded zones,
POIs) using the color suggestions from docs/SVG_MAP_EDITOR.md so JSON- and SVG-based
maps can be compared side-by-side.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    from collections.abc import Iterable

    from robot_sf.nav.global_route import GlobalRoute
    from robot_sf.nav.map_config import MapDefinition


# Colors follow docs/SVG_MAP_EDITOR.md suggestions.
OBSTACLE_COLOR = "#000000"
ROBOT_SPAWN_COLOR = "#ffdf00"
ROBOT_GOAL_COLOR = "#ff6c00"
ROBOT_ROUTE_COLOR = "#0078d5"
PED_SPAWN_COLOR = "#23ff00"
PED_GOAL_COLOR = "#107400"
PED_ROUTE_COLOR = "#c40202"
PED_CROWDED_COLOR = "#b3b3b3"  # not specified in doc; neutral gray for crowd areas
BOUNDARY_COLOR = "#5b5b5b"
POI_COLOR = "#8c4bff"


def _plot_zones(ax, zones: Iterable[tuple], color: str, label: str, alpha: float = 0.5) -> None:
    for idx, zone in enumerate(zones):
        patch = Polygon(zone, closed=True, facecolor=color, edgecolor="black", alpha=alpha)
        ax.add_patch(patch)
        center_x = sum(pt[0] for pt in zone) / len(zone)
        center_y = sum(pt[1] for pt in zone) / len(zone)
        ax.text(center_x, center_y, f"{label}{idx}", ha="center", va="center", fontsize=8)


def _plot_routes(ax, routes: Iterable[GlobalRoute], color: str, prefix: str) -> None:
    for route in routes:
        if not route.waypoints:
            continue
        xs, ys = zip(*route.waypoints, strict=False)
        ax.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            linewidth=1.5,
            label=f"{prefix}{route.spawn_id}->{route.goal_id}",
        )


def _plot_pois(ax, map_def) -> None:
    if not getattr(map_def, "poi_positions", None):
        return
    poi_labels = list(getattr(map_def, "poi_labels", {}).values())
    for idx, poi in enumerate(map_def.poi_positions):
        ax.scatter([poi[0]], [poi[1]], color=POI_COLOR, marker="x", s=50, linewidths=2)
        if idx < len(poi_labels):
            ax.text(poi[0], poi[1], poi_labels[idx], color=POI_COLOR, fontsize=8)


def visualize_map_definition(
    map_def: MapDefinition,
    output_path: str | Path | None = None,
    *,
    title: str | None = None,
    equal_aspect: bool = True,
    show: bool = False,
) -> None:
    """Render a MapDefinition with consistent colors for quick inspection.

    Args:
        map_def: Parsed map definition (from SVG or JSON).
        output_path: Optional path to save the figure (PNG). When None, only shows if ``show`` is True.
        title: Optional plot title.
        equal_aspect: Whether to enforce equal axis scaling.
        show: Whether to display the plot interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Obstacles
    for obstacle in map_def.obstacles:
        patch = Polygon(
            obstacle.vertices, closed=True, facecolor=OBSTACLE_COLOR, edgecolor="black", alpha=0.8
        )
        ax.add_patch(patch)

    # Boundaries
    for x1, x2, y1, y2 in map_def.bounds:
        ax.plot([x1, x2], [y1, y2], color=BOUNDARY_COLOR, linestyle="--", linewidth=1)

    # Zones
    _plot_zones(ax, map_def.robot_spawn_zones, ROBOT_SPAWN_COLOR, "RS ")
    _plot_zones(ax, map_def.robot_goal_zones, ROBOT_GOAL_COLOR, "RG ")
    _plot_zones(ax, map_def.ped_spawn_zones, PED_SPAWN_COLOR, "PS ")
    _plot_zones(ax, map_def.ped_goal_zones, PED_GOAL_COLOR, "PG ")
    _plot_zones(ax, map_def.ped_crowded_zones, PED_CROWDED_COLOR, "CZ ", alpha=0.3)

    # Routes
    _plot_routes(ax, map_def.robot_routes, ROBOT_ROUTE_COLOR, "R ")
    _plot_routes(ax, map_def.ped_routes, PED_ROUTE_COLOR, "P ")

    # POIs
    _plot_pois(ax, map_def)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    ax.set_xlim(0, map_def.width)
    ax.set_ylim(0, map_def.height)

    if title:
        ax.set_title(title)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles, strict=False))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if show and not output_path:
        plt.show()
    else:
        plt.close(fig)


__all__ = ["visualize_map_definition"]

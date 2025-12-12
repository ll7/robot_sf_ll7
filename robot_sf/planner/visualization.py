"""Visualization utilities for global planner outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from loguru import logger
from shapely.geometry import LineString, Point, Polygon
from shapely.plotting import plot_polygon

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from robot_sf.common.types import Vec2D
    from robot_sf.planner.global_planner import GlobalPlanner

ObstacleList = Iterable[Polygon]


def plot_global_plan(
    planner: GlobalPlanner,
    path: list[Vec2D],
    *,
    via_points: list[Vec2D] | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax: Axes | None = None,
    show: bool = True,
    flip_y: bool = True,
) -> plt.Figure:
    """Plot obstacles, POIs, and a planned waypoint path.

    Args:
        planner: Planner used to generate the path (supplies map + inflation config).
        path: Ordered waypoint list returned by ``GlobalPlanner.plan``.
        via_points: Optional via POI coordinates for highlighting.
        title: Optional plot title.
        save_path: When set, write the figure to this location (directories are created).
        ax: Optional Matplotlib axes to draw on; a new figure is created otherwise.
        show: When True, call ``plt.show()`` after rendering.
        flip_y: Invert the y-axis to match SVG coordinate origin (top-left).

    Returns:
        Matplotlib Figure containing the rendered plot.
    """
    if not path:
        raise ValueError("path must not be empty for plotting")

    via_points = via_points or []
    figure, axes = _init_axes(planner, ax=ax, title=title, flip_y=flip_y)

    inflated_obstacles = planner.build_inflated_obstacles()
    _plot_obstacles(inflated_obstacles, axes)
    _plot_pois(planner, axes)
    _plot_path(path, via_points, axes)

    axes.legend(loc="upper right", frameon=True)
    axes.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()

    if save_path:
        _save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _init_axes(
    planner: GlobalPlanner,
    *,
    ax: Axes | None,
    title: str | None,
    flip_y: bool,
) -> tuple[plt.Figure, Axes]:
    """Create or reuse axes configured to the map bounds.

    Returns:
        Figure and axes configured with map extents and labels.
    """
    if ax is None:
        figure, axes = plt.subplots(figsize=(10, 6))
    else:
        figure, axes = ax.figure, ax

    axes.set_xlim(0, planner.map_def.width)
    axes.set_ylim(0, planner.map_def.height)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_title(title or "Global Planner Route")
    if flip_y:
        axes.invert_yaxis()
    return figure, axes


def _plot_obstacles(obstacles: ObstacleList, ax: Axes) -> None:
    """Render inflated obstacles using Shapely's plotting helper."""
    for poly in obstacles:
        if poly.is_empty:
            continue
        plot_polygon(
            poly,
            ax=ax,
            add_points=False,
            facecolor="#cbd5e1",
            edgecolor="#475569",
            alpha=0.45,
            linewidth=1.2,
            zorder=1,
        )


def _plot_pois(planner: GlobalPlanner, ax: Axes) -> None:
    """Plot POI markers and labels from the map definition."""
    poi_positions = planner.map_def.poi_positions
    if not poi_positions:
        return

    xs = [pos[0] for pos in poi_positions]
    ys = [pos[1] for pos in poi_positions]
    ax.scatter(xs, ys, marker="*", color="#ca8a04", s=90, label="POIs", zorder=3)

    for (poi_id, poi_label), pos in zip(
        planner.map_def.poi_labels.items(),
        poi_positions,
        strict=True,
    ):
        ax.annotate(
            poi_label,
            xy=pos,
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#1f2937",
            zorder=4,
        )
        logger.debug(
            "Plotted POI {poi_id} ({label}) at {pos}",
            poi_id=poi_id,
            label=poi_label,
            pos=pos,
        )


def _plot_path(path: list[Vec2D], via_points: list[Vec2D], ax: Axes) -> None:
    """Plot the planner path, start/goal markers, and via points."""
    line = LineString(path)
    ax.plot(*line.xy, color="#2563eb", linewidth=2.5, label="planned path", zorder=2)
    xs, ys = zip(*path, strict=True)
    ax.scatter(xs, ys, color="#1d4ed8", s=28, zorder=3)

    start = Point(path[0])
    goal = Point(path[-1])
    ax.scatter(
        start.x,
        start.y,
        color="#16a34a",
        s=70,
        marker="o",
        label="start",
        zorder=4,
    )
    ax.scatter(
        goal.x,
        goal.y,
        color="#dc2626",
        s=70,
        marker="X",
        label="goal",
        zorder=4,
    )

    if via_points:
        via_xs, via_ys = zip(*via_points, strict=True)
        ax.scatter(
            via_xs,
            via_ys,
            color="#f59e0b",
            s=70,
            marker="D",
            label="via POIs",
            zorder=4,
        )


def plot_visibility_graph(
    planner: GlobalPlanner,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax: Axes | None = None,
    show: bool = True,
    flip_y: bool = True,
) -> plt.Figure:
    """Plot the visibility graph structure, obstacles, and POIs.

    Shows how the planner constructs its graph vertices (at obstacle corners)
    and edges (straight-line visibility paths between non-colliding points).

    Args:
        planner: Planner instance to extract graph and obstacles from.
        title: Optional plot title.
        save_path: When set, write the figure to this location (directories are created).
        ax: Optional Matplotlib axes to draw on; a new figure is created otherwise.
        show: When True, call ``plt.show()`` after rendering.
        flip_y: Invert the y-axis to match SVG coordinate origin (top-left).

    Returns:
        Matplotlib Figure containing the rendered visibility graph.
    """
    figure, axes = _init_axes(planner, ax=ax, title=title, flip_y=flip_y)

    inflated_obstacles = planner.build_inflated_obstacles()
    _plot_obstacles(inflated_obstacles, axes)

    # Plot the visibility graph if available
    if planner._graph is not None and planner._graph.networkx_graph is not None:
        _plot_visibility_graph_edges(planner._graph.networkx_graph, axes)
        _plot_visibility_graph_vertices(planner._graph.networkx_graph, axes)

    _plot_pois(planner, axes)

    axes.legend(loc="upper right", frameon=True)
    axes.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()

    if save_path:
        _save_figure(figure, save_path)
    if show:
        plt.show()
    return figure


def _plot_visibility_graph_edges(graph, ax: Axes) -> None:
    """Plot visibility graph edges (straight-line paths between nodes)."""
    if not graph or not graph.edges():
        return
    for u, v in graph.edges():
        x = [u[0], v[0]]
        y = [u[1], v[1]]
        ax.plot(x, y, color="#9ca3af", linewidth=0.8, alpha=0.6, zorder=1)
    logger.debug("Plotted {count} visibility graph edges", count=len(list(graph.edges())))


def _plot_visibility_graph_vertices(graph, ax: Axes) -> None:
    """Plot visibility graph vertices (obstacle corners and key points)."""
    if not graph or not graph.nodes():
        return
    nodes = list(graph.nodes())
    xs = [n[0] for n in nodes]
    ys = [n[1] for n in nodes]
    ax.scatter(
        xs,
        ys,
        color="#6366f1",
        s=12,
        marker=".",
        alpha=0.7,
        label=f"visibility graph ({len(nodes)} nodes)",
        zorder=2,
    )
    logger.debug("Plotted {count} visibility graph vertices", count=len(nodes))


def _save_figure(fig: plt.Figure, target: str | Path) -> None:
    """Persist the figure to disk, creating parent directories as needed."""
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=200, bbox_inches="tight")
    logger.info("Saved planner plot to {path}", path=target_path)

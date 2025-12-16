"""Adapter to convert SVG map definitions into python_motion_planning Grid maps.

This module provides utilities for converting robot_sf MapDefinition objects into
python_motion_planning Grid representations. It handles coordinate system conversions,
obstacle rasterization, and provides visualization helpers.

Key Features:
    - Converts vector-based SVG obstacles to rasterized grid cells
    - Handles Y-axis coordinate flipping (world Y-up vs. grid Y-down)
    - Supports configurable resolution and obstacle inflation
    - Provides visualization and analysis utilities

Example:
    >>> from robot_sf.nav.svg_map_parser import convert_map
    >>> from robot_sf.nav.motion_planning_adapter import (
    ...     MotionPlanningGridConfig,
    ...     map_definition_to_motion_planning_grid,
    ...     count_obstacle_cells,
    ... )
    >>> map_def = convert_map("maps/svg_maps/example.svg")
    >>> config = MotionPlanningGridConfig(cells_per_meter=2.0, inflate_radius_cells=2)
    >>> grid = map_definition_to_motion_planning_grid(map_def, config)
    >>> obstacle_count = count_obstacle_cells(grid)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
from python_motion_planning.common import TYPES, Grid, Visualizer
from shapely.affinity import scale
from shapely.geometry import Polygon, box

from robot_sf.common import ensure_interactive_backend

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


class ClassicPlanVisualizer(Visualizer):
    """Visualizer that can show grid axes in world coordinates."""

    _DEFAULT_ALPHA_3D = {
        TYPES.FREE: 0.0,
        TYPES.OBSTACLE: 0.5,
        TYPES.START: 0.5,
        TYPES.GOAL: 0.5,
        TYPES.INFLATION: 0.0,
        TYPES.EXPAND: 0.1,
        TYPES.CUSTOM: 0.5,
    }

    def __init__(
        self,
        figname: str = "",
        figsize: tuple[int, int] = (10, 8),
        meters_per_cell: float | None = None,
        cells_per_meter: float | None = None,
    ) -> None:
        """Initialize the classic planner visualizer with optional scaling metadata.

        Args:
            figname: Title for the figure window.
            figsize: Figure dimensions as (width, height) in inches.
            meters_per_cell: Scale factor for axis labels (meters per cell).
            cells_per_meter: Alternative scale factor (cells per meter).
        """
        super().__init__(figname, figsize)
        if cells_per_meter == 0:
            raise ValueError("cells_per_meter must be non-zero when provided")
        if meters_per_cell is not None and cells_per_meter is not None:
            derived = 1.0 / cells_per_meter
            if not np.isclose(meters_per_cell, derived):
                raise ValueError(
                    "meters_per_cell and cells_per_meter are inconsistent "
                    f"({meters_per_cell} vs {derived})"
                )
        self._meters_per_cell = (
            meters_per_cell
            if meters_per_cell is not None
            else (1.0 / cells_per_meter if cells_per_meter is not None else None)
        )
        self.cmap_dict[TYPES.EXPAND] = "#dddddd"
        self.cmap = mcolors.ListedColormap(list(self.cmap_dict.values()))
        self.norm = mcolors.BoundaryNorm(list(range(self.cmap.N + 1)), self.cmap.N)

    def _resolve_meters_per_cell(
        self, grid_map: Grid, meters_per_cell: float | None
    ) -> float | None:
        """Derive meters-per-cell from explicit overrides or grid metadata.

        Returns:
            Meters-per-cell value if discoverable, otherwise None.
        """
        if meters_per_cell is not None:
            return meters_per_cell
        if self._meters_per_cell is not None:
            return self._meters_per_cell
        if hasattr(grid_map, "meters_per_cell"):
            value = grid_map.meters_per_cell
            if value:
                return float(value)
        if hasattr(grid_map, "cells_per_meter"):
            cells_per_meter = grid_map.cells_per_meter
            if cells_per_meter:
                return 1.0 / float(cells_per_meter)
        return None

    def _set_world_axis_formatters(self, grid_map: Grid, meters_per_cell: float | None) -> None:
        """Label axes in meters when scale information is available."""
        scale_factor = self._resolve_meters_per_cell(grid_map, meters_per_cell)
        if scale_factor is None or grid_map.dim != 2:
            return

        formatter = FuncFormatter(lambda value, _: f"{value * scale_factor:.2f}")
        self.ax.xaxis.set_major_formatter(formatter)
        self.ax.yaxis.set_major_formatter(formatter)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")

    def plot_grid_map(  # type: ignore[override]
        self,
        grid_map: Grid,
        equal: bool = True,
        alpha_3d: dict | None = None,
        show_esdf: bool = False,
        alpha_esdf: float = 0.5,
        meters_per_cell: float | None = None,
    ) -> None:
        """Plot grid cells and relabel axes in meters when scaling is known."""
        resolved_alpha = alpha_3d if alpha_3d is not None else self._DEFAULT_ALPHA_3D

        super().plot_grid_map(
            grid_map,
            equal=equal,
            alpha_3d=resolved_alpha,
            show_esdf=show_esdf,
            alpha_esdf=alpha_esdf,
        )
        self._set_world_axis_formatters(grid_map, meters_per_cell)


@dataclass(slots=True)
class MotionPlanningGridConfig:
    """Configuration for converting a map into a motion-planning Grid."""

    cells_per_meter: float = 1.0
    add_boundary_obstacles: bool = True
    inflate_radius_cells: int | None = None

    @property
    def meters_per_cell(self) -> float:
        """Inverse scaling factor (meters represented by a single grid cell)."""
        return 1.0 / self.cells_per_meter


def _world_to_grid(value: float, cells_per_meter: float) -> int:
    """Scale a world coordinate in meters to a grid cell index.

    Returns:
        Grid cell index (floored).
    """
    return math.floor(value * cells_per_meter)


def _mark_obstacle_cells(
    grid: Grid,  # type: ignore[name-defined]
    polygon: Polygon,
    cells_per_meter: float,
    types: object,
) -> None:
    """Rasterize a single obstacle polygon into the grid's type_map.

    This function handles coordinate system conversion between world space (Y-up)
    and grid space (Y-down for visualization). The python_motion_planning Grid
    type_map is indexed [x][y] where shape=(width, height).

    Args:
        grid: Target grid to mark obstacles in.
        polygon: Shapely polygon representing the obstacle in world coordinates.
        cells_per_meter: Scaling factor for coordinate conversion.
        types: TYPES enum from python_motion_planning for cell marking.

    Note:
        - World coordinates: Y=0 at bottom, Y increases upward
        - Grid coordinates: row 0 at top, Y increases downward
        - Conversion: grid_y = height_cells - 1 - y_world
    """
    if polygon.is_empty:
        return

    type_map = grid.type_map
    width_cells = type_map.shape[0]  # X dimension
    height_cells = type_map.shape[1]  # Y dimension

    # Scale polygon coordinates to grid space
    scaled = scale(polygon, xfact=cells_per_meter, yfact=cells_per_meter, origin=(0, 0))
    minx, miny, maxx, maxy = scaled.bounds

    x_start = max(0, _world_to_grid(minx, 1.0))
    x_end = min(width_cells, math.ceil(maxx))
    y_start = max(0, _world_to_grid(miny, 1.0))
    y_end = min(height_cells, math.ceil(maxy))

    for y_world in range(y_start, y_end):
        for x_world in range(x_start, x_end):
            # Create cell polygon in world coordinates
            cell_poly = box(x_world, y_world, x_world + 1, y_world + 1)
            if scaled.intersects(cell_poly):
                # Convert world Y to grid: Y-axis is flipped
                grid_y = height_cells - 1 - y_world
                if 0 <= x_world < width_cells and 0 <= grid_y < height_cells:
                    type_map[x_world][grid_y] = types.OBSTACLE


def map_definition_to_motion_planning_grid(
    map_def: MapDefinition, config: MotionPlanningGridConfig | None = None
) -> Grid:  # type: ignore[name-defined]
    """Convert a MapDefinition into a python_motion_planning Grid.

    This is the main entry point for converting robot_sf map definitions into
    grid-based representations for motion planning algorithms.

    Args:
        map_def: Parsed SVG map definition containing obstacles and dimensions.
        config: Optional configuration for grid resolution, inflation, and boundaries.
                If None, uses default MotionPlanningGridConfig.

    Returns:
        Grid object with obstacles marked and optionally inflated.

    Example:
        >>> from robot_sf.nav.svg_map_parser import convert_map
        >>> map_def = convert_map("maps/svg_maps/example.svg")
        >>> config = MotionPlanningGridConfig(cells_per_meter=2.0)
        >>> grid = map_definition_to_motion_planning_grid(map_def, config)
    """
    cfg = config or MotionPlanningGridConfig()

    width_cells = math.ceil(map_def.width * cfg.cells_per_meter)
    height_cells = math.ceil(map_def.height * cfg.cells_per_meter)
    grid = Grid(bounds=[[0, width_cells], [0, height_cells]])
    grid.cells_per_meter = cfg.cells_per_meter
    grid.meters_per_cell = cfg.meters_per_cell

    if cfg.add_boundary_obstacles:
        grid.fill_boundary_with_obstacles()

    # Rasterize all obstacles
    for obstacle in map_def.obstacles:
        poly = Polygon(obstacle.vertices)
        if not poly.is_valid or poly.is_empty:
            logger.warning("Skipping invalid obstacle during grid rasterization.")
            continue
        _mark_obstacle_cells(grid, poly, cfg.cells_per_meter, TYPES)

    # Apply inflation if configured
    if cfg.inflate_radius_cells is not None:
        grid.inflate_obstacles(radius=cfg.inflate_radius_cells)

    logger.info(
        "Converted map to motion-planning grid: {w}x{h} cells ({m:.2f} m/cell)",
        w=width_cells,
        h=height_cells,
        m=cfg.meters_per_cell,
    )
    return grid


def count_obstacle_cells(grid: Grid) -> int:  # type: ignore[name-defined]
    """Count the number of obstacle cells in a grid.

    Args:
        grid: Grid to analyze.

    Returns:
        Number of cells marked as obstacles.

    Example:
        >>> obstacle_count = count_obstacle_cells(grid)
        >>> print(f"Grid has {obstacle_count} obstacle cells")
    """
    type_map_np = np.asarray(grid.type_map)
    return np.count_nonzero(type_map_np == TYPES.OBSTACLE)


def visualize_grid(
    grid: Grid,  # type: ignore[name-defined]
    output_path: Path | str | None = None,
    title: str = "Grid Map",
    equal_aspect: bool = True,
) -> None:
    """Visualize a grid map, optionally saving to file or showing interactively.

    Args:
        grid: Grid to visualize.
        output_path: Path where the visualization should be saved (e.g., "output/grid.png").
                    If None or empty string, shows the plot interactively instead.
        title: Title for the visualization window.
        equal_aspect: Whether to use equal aspect ratio for axes.

    Example:
        >>> from pathlib import Path
        >>> # Save to file
        >>> visualize_grid(grid, Path("output/plots/grid.png"), title="My Grid")
        >>> # Show interactively
        >>> visualize_grid(grid, None, title="My Grid")
    """
    meters_per_cell = getattr(grid, "meters_per_cell", None)
    cells_per_meter = getattr(grid, "cells_per_meter", None)
    vis = ClassicPlanVisualizer(
        title,
        meters_per_cell=meters_per_cell,
        cells_per_meter=cells_per_meter,
    )
    vis.plot_grid_map(
        grid,
        equal=equal_aspect,
        meters_per_cell=meters_per_cell,
    )

    if output_path and str(output_path).strip():
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis.fig.savefig(str(output_path))
        logger.success(f"Saved grid visualization to {output_path}")
    else:
        ensure_interactive_backend()
        vis.show()
        logger.info("Showing grid visualization interactively")

    vis.close()


def visualize_path(
    grid: Grid,  # type: ignore[name-defined]
    path: list[tuple[int, int]],
    output_path: Path | str | None = None,
    title: str = "Planned Path",
    equal_aspect: bool = True,
    path_style: str = "--",
    path_color: str = "C4",
    linewidth: float = 2.0,
    marker: str | None = None,
) -> None:
    """Visualize a planned path over a grid, optionally saving to disk.

    Args:
        grid: Grid to visualize.
        path: Path waypoints in grid/map coordinates.
        output_path: Optional path to save the figure; shows interactively when None.
        title: Title for the visualization window.
        equal_aspect: Whether to force equal aspect ratio.
        path_style: Matplotlib line style for the path.
        path_color: Matplotlib color for the path.
        linewidth: Path line width.
        marker: Optional marker for waypoints.
    """
    meters_per_cell = getattr(grid, "meters_per_cell", None)
    cells_per_meter = getattr(grid, "cells_per_meter", None)
    vis = ClassicPlanVisualizer(
        title,
        meters_per_cell=meters_per_cell,
        cells_per_meter=cells_per_meter,
    )
    vis.plot_grid_map(
        grid,
        equal=equal_aspect,
        meters_per_cell=meters_per_cell,
    )

    if path:
        vis.plot_path(
            path,
            style=path_style,
            color=path_color,
            linewidth=linewidth,
            marker=marker,
        )
    else:
        logger.warning("No path provided to visualize_path; rendering grid only.")

    if output_path and str(output_path).strip():
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis.fig.savefig(str(output_path))
        logger.success(f"Saved path visualization to {output_path}")
    else:
        ensure_interactive_backend()
        vis.show()
        logger.info("Showing path visualization interactively")

    vis.close()


def set_start_goal_on_grid(
    grid: Grid,  # type: ignore[name-defined]
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Grid:  # type: ignore[name-defined]
    """Set start and goal positions on the grid.

    Args:
        grid: Grid to modify.
        start: (x, y) tuple for start position in grid coordinates.
        goal: (x, y) tuple for goal position in grid coordinates.
    Returns:
        Grid with start and goal positions marked.
    """
    start_x, start_y = start
    goal_x, goal_y = goal
    grid.type_map[start_x][start_y] = TYPES.START
    grid.type_map[goal_x][goal_y] = TYPES.GOAL
    return grid


def get_obstacle_statistics(grid: Grid) -> dict[str, float]:  # type: ignore[name-defined]
    """Calculate statistics about obstacles in a grid.

    TODO: Percentage for obstacles is likely not calculated correctly.
    `INFO   | 29_osm_global_planner_test.py:55 | Planning grid: (513, 393), 0 obstacle cells (0.00%)`

    Args:
        grid: Grid to analyze.

    Returns:
        Dictionary containing:
            - obstacle_count: Number of obstacle cells
            - total_cells: Total number of cells in the grid
            - obstacle_percentage: Percentage of cells that are obstacles

    Example:
        >>> stats = get_obstacle_statistics(grid)
        >>> print(f"Grid is {stats['obstacle_percentage']:.1f}% occupied")
    """
    type_map_np = np.asarray(grid.type_map)
    obstacle_count = np.count_nonzero(type_map_np == TYPES.OBSTACLE)
    total_cells = type_map_np.size

    return {
        "obstacle_count": obstacle_count,
        "total_cells": total_cells,
        "obstacle_percentage": (obstacle_count / total_cells * 100) if total_cells > 0 else 0.0,
    }


__all__ = [
    "ClassicPlanVisualizer",
    "MotionPlanningGridConfig",
    "count_obstacle_cells",
    "get_obstacle_statistics",
    "map_definition_to_motion_planning_grid",
    "set_start_goal_on_grid",
    "visualize_grid",
    "visualize_path",
]

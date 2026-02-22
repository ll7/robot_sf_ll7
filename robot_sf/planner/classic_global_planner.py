"""Classic grid-based global planner using python_motion_planning algorithms.

This module provides a grid-based path planning approach using algorithms
from the python_motion_planning library (e.g., ThetaStar, A*). It operates
on rasterized grids converted from vector-based SVG maps.

Key Features:
    - Grid-based planning with configurable resolution
    - Support for multiple planning algorithms (ThetaStar, A*, etc.)
    - Integration with robot_sf MapDefinition format
    - Coordinate conversion between world space and grid indices

Example:
    >>> from robot_sf.nav.svg_map_parser import convert_map
    >>> from robot_sf.planner.classic_global_planner import ClassicGlobalPlanner
    >>> map_def = convert_map("maps/svg_maps/example.svg")
    >>> planner = ClassicGlobalPlanner(map_def, cells_per_meter=2.0)
    >>> path, info = planner.plan(start=(5.0, 5.0), goal=(45.0, 25.0))
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
from python_motion_planning.common import TYPES, Grid, Visualizer
from python_motion_planning.path_planner import AStar, ThetaStar
from shapely.affinity import scale
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity

from robot_sf.common import ensure_interactive_backend
from robot_sf.planner.theta_star_v2 import HighPerformanceThetaStar

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition

DEFAULT_VISUALIZATION_DPI = 300
MAX_VISUALIZATION_DPI = 3000


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
        """Initialize the classic planner visualizer with optional scaling metadata."""
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
    return math.floor(value * cells_per_meter)


def _mark_obstacle_cells(
    grid: Grid,
    polygon: Polygon,
    cells_per_meter: float,
) -> None:
    if polygon.is_empty:
        return

    type_map = grid.type_map
    width_cells = type_map.shape[0]
    height_cells = type_map.shape[1]
    scaled = scale(polygon, xfact=cells_per_meter, yfact=cells_per_meter, origin=(0, 0))
    minx, miny, maxx, maxy = scaled.bounds
    x_start = max(0, _world_to_grid(minx, 1.0))
    x_end = min(width_cells, math.ceil(maxx))
    y_start = max(0, _world_to_grid(miny, 1.0))
    y_end = min(height_cells, math.ceil(maxy))

    for y_world in range(y_start, y_end):
        for x_world in range(x_start, x_end):
            cell_poly = box(x_world, y_world, x_world + 1, y_world + 1)
            if scaled.intersects(cell_poly):
                grid_y = height_cells - 1 - y_world
                if 0 <= x_world < width_cells and 0 <= grid_y < height_cells:
                    type_map[x_world][grid_y] = TYPES.OBSTACLE


def map_definition_to_motion_planning_grid(
    map_def: MapDefinition, config: MotionPlanningGridConfig | None = None
) -> Grid:
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

    for idx, obstacle in enumerate(map_def.obstacles):
        poly = Polygon(obstacle.vertices)
        if not poly.is_valid:
            logger.warning(
                "Skipping invalid obstacle during grid rasterization "
                f"(index={idx}, valid={poly.is_valid}, reason={explain_validity(poly)})",
            )
            continue
        if poly.is_empty:
            logger.warning(
                "Skipping empty obstacle during grid rasterization (index={idx})", idx=idx
            )
            continue
        _mark_obstacle_cells(grid, poly, cfg.cells_per_meter)

    if cfg.inflate_radius_cells is not None:
        grid.inflate_obstacles(radius=cfg.inflate_radius_cells)

    logger.info(
        "Converted map to motion-planning grid: {w}x{h} cells ({m:.2f} m/cell)",
        w=width_cells,
        h=height_cells,
        m=cfg.meters_per_cell,
    )
    return grid


def count_obstacle_cells(grid: Grid) -> int:
    """Count obstacle cells in a grid.

    Args:
        grid: Grid to analyze.

    Returns:
        Number of cells marked as `TYPES.OBSTACLE`.

    Example:
        >>> obstacle_count = count_obstacle_cells(grid)
        >>> print(f"Grid has {obstacle_count} obstacle cells")
    """
    type_map_np = np.asarray(grid.type_map)
    return np.count_nonzero(type_map_np == TYPES.OBSTACLE)


def _sanitize_visualization_output_path(output_path: Path | str) -> Path:
    """Validate and normalize a visualization output path.

    This helper prevents parent-directory traversal segments in relative or absolute
    paths while preserving caller-controlled output locations.

    Returns:
        Sanitized `Path` object safe for visualization writes.
    """
    path_obj = Path(output_path)
    if any(part == ".." for part in path_obj.parts):
        raise ValueError(f"output_path contains unsupported parent traversal segments: {path_obj}")
    return path_obj


def _normalize_output_dpi(output_dpi: int | float) -> int:
    """Validate output DPI for saved visualizations.

    Args:
        output_dpi: Requested save DPI.

    Returns:
        Positive integer DPI value.
    """
    dpi_value = int(output_dpi)
    if dpi_value <= 0 or dpi_value > MAX_VISUALIZATION_DPI:
        raise ValueError(
            f"output_dpi must be in range (0, {MAX_VISUALIZATION_DPI}], got {output_dpi}"
        )
    return dpi_value


def visualize_grid(
    grid: Grid,
    output_path: Path | str | None = None,
    title: str = "Grid Map",
    equal_aspect: bool = True,
    output_dpi: int = DEFAULT_VISUALIZATION_DPI,
) -> None:
    """Visualize a grid map, optionally saving to file or showing interactively.

    Args:
        grid: Grid to visualize.
        output_path: Path where the visualization should be saved (e.g., "output/grid.png").
            If None or empty string, shows the plot interactively instead.
        title: Title for the visualization window.
        equal_aspect: Whether to use equal aspect ratio for axes.
        output_dpi: Export resolution for saved figures.

    Example:
        >>> from pathlib import Path
        >>> visualize_grid(grid, Path("output/plots/grid.png"), title="My Grid")
        >>> visualize_grid(grid, None, title="My Grid")
    """
    meters_per_cell = getattr(grid, "meters_per_cell", None)
    cells_per_meter = getattr(grid, "cells_per_meter", None)
    vis = ClassicPlanVisualizer(
        title,
        meters_per_cell=meters_per_cell,
        cells_per_meter=cells_per_meter,
    )
    vis.plot_grid_map(grid, equal=equal_aspect, meters_per_cell=meters_per_cell)

    if output_path and str(output_path).strip():
        resolved_output = _sanitize_visualization_output_path(output_path)
        dpi_value = _normalize_output_dpi(output_dpi)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        vis.fig.savefig(str(resolved_output), dpi=dpi_value)
        logger.success(f"Saved grid visualization to {resolved_output}")
    else:
        ensure_interactive_backend()
        vis.show()
        logger.info("Showing grid visualization interactively")
    vis.close()


def visualize_path(  # noqa: PLR0913
    grid: Grid,
    path: list[tuple[int, int]],
    output_path: Path | str | None = None,
    title: str = "Planned Path",
    equal_aspect: bool = True,
    path_style: str = "--",
    path_color: str = "C4",
    linewidth: float = 2.0,
    marker: str | None = None,
    output_dpi: int = DEFAULT_VISUALIZATION_DPI,
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
        output_dpi: Export resolution for saved figures.
    """
    meters_per_cell = getattr(grid, "meters_per_cell", None)
    cells_per_meter = getattr(grid, "cells_per_meter", None)
    vis = ClassicPlanVisualizer(
        title,
        meters_per_cell=meters_per_cell,
        cells_per_meter=cells_per_meter,
    )
    vis.plot_grid_map(grid, equal=equal_aspect, meters_per_cell=meters_per_cell)

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
        resolved_output = _sanitize_visualization_output_path(output_path)
        dpi_value = _normalize_output_dpi(output_dpi)
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        vis.fig.savefig(str(resolved_output), dpi=dpi_value)
        logger.success(f"Saved path visualization to {resolved_output}")
    else:
        ensure_interactive_backend()
        vis.show()
        logger.info("Showing path visualization interactively")
    vis.close()


def set_start_goal_on_grid(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Grid:
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


def get_obstacle_statistics(grid: Grid) -> dict[str, float]:
    """Calculate basic statistics about obstacle occupancy.

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


@dataclass(slots=True)
class ClassicPlannerConfig:
    """Configuration for the classic grid-based planner.

    Attributes:
        cells_per_meter: Grid resolution (cells per meter).
        inflate_radius_cells: Number of cells to inflate obstacles (for robot radius).
        add_boundary_obstacles: Whether to add obstacles at map boundaries.
        algorithm: Planning algorithm to use ('theta_star', 'a_star', etc.).
    """

    cells_per_meter: float = 2.0
    inflate_radius_cells: int | None = 0
    add_boundary_obstacles: bool = True
    algorithm: str = "theta_star_v2"

    def __post_init__(self) -> None:
        """Validate planner configuration.

        Raises:
            ValueError: If any configuration field is invalid.
        """
        if not isinstance(self.cells_per_meter, (int, float)) or not math.isfinite(
            self.cells_per_meter
        ):
            raise ValueError("cells_per_meter must be a finite number")
        if self.cells_per_meter <= 0:
            raise ValueError("cells_per_meter must be greater than zero")

        if self.inflate_radius_cells is not None:
            if (
                isinstance(self.inflate_radius_cells, float)
                and self.inflate_radius_cells.is_integer()
            ):
                self.inflate_radius_cells = int(self.inflate_radius_cells)
            if not isinstance(self.inflate_radius_cells, int) or self.inflate_radius_cells < 0:
                raise ValueError("inflate_radius_cells must be an int >= 0 or None")

        if not isinstance(self.add_boundary_obstacles, bool):
            raise ValueError("add_boundary_obstacles must be a bool")

        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise ValueError("algorithm must be a non-empty string")


class PlanningError(Exception):
    """Raised when path planning fails."""

    pass


class ClassicGlobalPlanner:
    """Classic grid-based global planner using python_motion_planning.

    This planner converts vector-based SVG maps to rasterized grids and
    uses algorithms from python_motion_planning for path planning. It supports
    multiple algorithms (Theta*, high-performance Theta*, A*) with per-call
    overrides, inflation fallbacks, cached grid reuse, and convenience
    visualization helpers.

    Attributes:
        map_def: The MapDefinition to plan in.
        config: Configuration for the planner.
        grid: The converted motion planning grid (created on first plan).
    """

    def __init__(
        self,
        map_def: MapDefinition,
        config: ClassicPlannerConfig | None = None,
    ) -> None:
        """Initialize the classic global planner.

        Args:
            map_def: MapDefinition containing obstacles and map bounds.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.map_def = map_def
        self.config = config or ClassicPlannerConfig()
        self._width_cells = math.ceil(self.map_def.width * self.config.cells_per_meter)
        self._height_cells = math.ceil(self.map_def.height * self.config.cells_per_meter)
        self._grid: Grid | None = None
        self._last_path_world: list[tuple[float, float]] | None = None
        self._last_path_grid: list[tuple[int, int]] | None = None
        self._last_path_info: dict | None = None

        logger.debug(
            "Initialized ClassicGlobalPlanner with {w}x{h} map, {cpm} cells/m",
            w=map_def.width,
            h=map_def.height,
            cpm=self.config.cells_per_meter,
        )

    @property
    def grid(self) -> Grid:
        """Get or create the motion planning grid.

        Returns:
            The motion planning Grid object.

        Raises:
            RuntimeError: If grid creation fails.
        """
        if self._grid is None:
            self._grid = self._build_grid(self.config.inflate_radius_cells)
        return self._grid

    def _build_grid(self, inflate_radius_cells: int | None) -> Grid:
        """Create a motion-planning grid from the map definition.

        Args:
            inflate_radius_cells: Number of cells to inflate obstacles with, or None.

        Returns:
            A populated motion-planning grid.
        """
        grid_config = MotionPlanningGridConfig(
            cells_per_meter=self.config.cells_per_meter,
            inflate_radius_cells=inflate_radius_cells,
            add_boundary_obstacles=self.config.add_boundary_obstacles,
        )
        grid = map_definition_to_motion_planning_grid(self.map_def, config=grid_config)
        logger.debug(
            "Created planning grid with shape {shape} and inflation {inflation}",
            shape=grid.type_map.shape,
            inflation=inflate_radius_cells,
        )
        return grid

    def _world_to_grid(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            world_x: X coordinate in world space (meters).
            world_y: Y coordinate in world space (meters).

        Returns:
            Tuple of (grid_x, grid_y) indices.
        """
        grid_x = math.floor(world_x * self.config.cells_per_meter)
        grid_y = self._height_cells - 1 - math.floor(world_y * self.config.cells_per_meter)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid indices to world coordinates.

        Args:
            grid_x: Grid X index.
            grid_y: Grid Y index.

        Returns:
            Tuple of (world_x, world_y) coordinates in meters.
        """
        world_x = (grid_x + 0.5) / self.config.cells_per_meter
        world_y = (self._height_cells - 1 - grid_y + 0.5) / self.config.cells_per_meter
        return world_x, world_y

    def _normalize_algorithm(self, override: str | None = None) -> str:
        """Normalize the algorithm name and validate support.

        Returns:
            Normalized algorithm identifier ('theta_star', 'theta_star_v2', or 'a_star').

        Args:
            override: Optional algorithm name overriding the config default.

        Raises:
            ValueError: If the requested algorithm is unsupported.
        """
        algo_raw = (override or self.config.algorithm).strip().lower()
        if algo_raw in {"theta_star", "thetastar", "theta"}:
            return "theta_star"
        if algo_raw in {"a_star", "astar", "a*", "a-star"}:
            return "a_star"
        if algo_raw in {"theta_star_v2", "theta_v2", "thetafast", "theta_star_fast", "theta2"}:
            return "theta_star_v2"
        msg = f"Unsupported algorithm: {override or self.config.algorithm}"
        raise ValueError(msg)

    def _validate_grid_point(self, name: str, gx: int, gy: int, grid: Grid) -> None:
        """Ensure a grid point is within bounds and not colliding.

        Args:
            name: Label for the point (e.g., "start" or "goal").
            gx: Grid X index.
            gy: Grid Y index.
            grid: Grid used for validation.

        Raises:
            PlanningError: When the point lies outside the grid bounds or inside an obstacle/inflation.
        """
        if not (0 <= gx < grid.type_map.shape[0]) or not (0 <= gy < grid.type_map.shape[1]):
            raise PlanningError(
                f"{name} out of grid bounds: grid=({gx}, {gy}), grid_shape={grid.type_map.shape}"
            )
        cell_value = grid.type_map[gx][gy]
        if cell_value in {TYPES.OBSTACLE, TYPES.INFLATION}:
            if cell_value == TYPES.OBSTACLE:
                cell_desc = "obstacle"
            elif cell_value == TYPES.INFLATION:
                cell_desc = "inflated area"
            else:
                cell_desc = "unknown"
            world_x, world_y = self._grid_to_world(gx, gy)
            raise PlanningError(
                f"{name} at ({world_x:.2f}, {world_y:.2f}) is in an invalid cell "
                f"({cell_desc}); choose a free/start/goal cell."
            )

    def validate_point(
        self, point_world: tuple[float, float], grid: Grid | None = None
    ) -> tuple[int, int]:
        """Validate a world-space point against the planning grid.

        Args:
            point_world: Coordinate (x, y) in world meters.
            grid: Optional grid to validate against; defaults to the planner's grid.

        Returns:
            tuple[int, int]: The validated grid indices corresponding to the world point.

        Raises:
            PlanningError: If the point is out of bounds or lies inside obstacles/inflation.
        """
        grid_obj = grid or self.grid
        gx, gy = self._world_to_grid(*point_world)
        self._validate_grid_point("point", gx, gy, grid_obj)
        return gx, gy

    def random_valid_point_on_grid(
        self,
        seed: int | None = None,
        grid: Grid | None = None,
        max_attempts: int = 1000,
        rng: random.Random | None = None,
    ) -> tuple[float, float]:
        """Sample a random valid world-space point on the planning grid.

        Args:
            seed: Optional random seed for reproducibility (ignored when rng is provided).
            grid: Optional grid to validate against; defaults to the planner's grid.
            max_attempts: Maximum number of sampling attempts before failing.
            rng: Optional pre-seeded random generator to reuse across calls.

        Returns:
            tuple[float, float]: World coordinates of a valid free cell.

        Raises:
            PlanningError: If no valid point is found within the allowed attempts.
        """
        grid_obj = grid or self.grid
        generator = rng if rng is not None else random.Random(seed)
        width, height = grid_obj.type_map.shape

        for _ in range(max_attempts):
            gx = generator.randrange(width)
            gy = generator.randrange(height)
            try:
                self._validate_grid_point("random point", gx, gy, grid_obj)
            except PlanningError:
                continue
            return self._grid_to_world(gx, gy)

        msg = f"Failed to sample a valid grid point after {max_attempts} attempts"
        raise PlanningError(msg)

    def _make_planner(self, algo: str, grid: Grid, start: tuple[int, int], goal: tuple[int, int]):
        """Instantiate the requested planner.

        Returns:
            Concrete planner instance for the requested algorithm.

        Args:
            algo: Normalized algorithm name.
            grid: Planning grid.
            start: Start cell (x, y).
            goal: Goal cell (x, y).

        Raises:
            ValueError: If the algorithm name is unsupported.
        """
        if algo == "theta_star":
            logger.warning("Theta_star can be roughly 20x slower than A_star on large grids.")
            return ThetaStar(map_=grid, start=start, goal=goal)
        if algo == "theta_star_v2":
            return HighPerformanceThetaStar(map_=grid, start=start, goal=goal)
        if algo == "a_star":
            return AStar(map_=grid, start=start, goal=goal)
        msg = f"Unsupported algorithm: {algo}"
        raise ValueError(msg)

    def _run_single_plan(
        self,
        start_grid: tuple[int, int],
        goal_grid: tuple[int, int],
        algo: str,
        inflation: int | None,
    ) -> tuple[Grid, list[tuple[int, int]], list[tuple[float, float]], dict | None]:
        """Execute one planning attempt for a given inflation radius.

        Returns:
            tuple[Grid, list[tuple[int, int]], list[tuple[float, float]], dict | None]:
                - grid: The constructed planning grid for this attempt.
                - path_grid: Waypoints in grid coordinates.
                - path_world: Waypoints in world coordinates.
                - path_info: Optional planner metadata scaled to meters.

        Args:
            start_grid: Start cell (x, y).
            goal_grid: Goal cell (x, y).
            algo: Normalized algorithm name.
            inflation: Inflation radius in cells for this attempt.
        """
        grid = self._build_grid(inflation)
        self._validate_grid_point("start", *start_grid, grid=grid)
        self._validate_grid_point("goal", *goal_grid, grid=grid)
        grid.type_map[start_grid[0]][start_grid[1]] = TYPES.START
        grid.type_map[goal_grid[0]][goal_grid[1]] = TYPES.GOAL

        planner = self._make_planner(algo, grid, start_grid, goal_grid)
        plan_result = planner.plan()
        path_grid, path_info = (
            plan_result if isinstance(plan_result, tuple) else (plan_result, None)
        )

        path_world = [self._grid_to_world(x, y) for x, y in path_grid] if path_grid else []
        scaled_info = self._scale_path_info(path_info, grid, inflation)
        return grid, path_grid, path_world, scaled_info

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        algorithm: str | None = None,
    ) -> tuple[list[tuple[float, float]], dict | None]:
        """Plan a path from start to goal with inflation fallback.

        Args:
            start: Start position (x, y) in world coordinates (meters).
            goal: Goal position (x, y) in world coordinates (meters).
            algorithm: Optional algorithm override ('theta_star', 'a_star'); defaults to config.

        Returns:
            tuple[list[tuple[float, float]], dict | None]:
                - path_world: Waypoints in world coordinates.
                - path_info: Optional planner metadata with length scaled to meters.

        Raises:
            PlanningError: If planning fails or coordinates are out of bounds.
        """
        start_grid = self._world_to_grid(*start)
        goal_grid = self._world_to_grid(*goal)
        algo = self._normalize_algorithm(algorithm)

        logger.debug(
            "Planning from {start} to {goal} (world coords: {start_w} → {goal_w})",
            start=start_grid,
            goal=goal_grid,
            start_w=start,
            goal_w=goal,
        )

        initial_inflation = self.config.inflate_radius_cells
        attempt_radii: list[int | None]
        if initial_inflation is None:
            attempt_radii = [None]
        else:
            start_radius = max(0, initial_inflation)
            attempt_radii = list(range(start_radius, -1, -1))

        last_error: Exception | None = None
        abort_due_to_invalid_cell = False

        for idx, inflation in enumerate(attempt_radii):
            try:
                grid, path_grid, path_world, scaled_info = self._run_single_plan(
                    start_grid, goal_grid, algo, inflation
                )
                if path_world:
                    self._grid = grid
                    self._last_path_grid = path_grid
                    self._last_path_world = path_world
                    self._last_path_info = scaled_info
                    logger.info(
                        "Found path with {n} waypoints from {start} to {goal} using {algo} and inflation {inflation}",
                        n=len(path_world),
                        start=start,
                        goal=goal,
                        algo=algo,
                        inflation=inflation,
                    )
                    return path_world, scaled_info

                logger.warning(
                    "No path found from {start} to {goal} with inflation {inflation}",
                    start=start,
                    goal=goal,
                    inflation=inflation,
                )
            except Exception as exc:  # noqa: BLE001 - broad catch to retry with smaller inflation
                last_error = exc
                logger.warning(
                    "Planning attempt failed with inflation {inflation}: {error}",
                    inflation=inflation,
                    error=exc,
                )
                if isinstance(exc, PlanningError) and "is in an invalid cell" in str(exc):
                    abort_due_to_invalid_cell = True
                    logger.warning(
                        "Start/goal lies in an invalid cell; aborting inflation fallback so caller can resample."
                    )
                    break

            is_last_attempt = idx == len(attempt_radii) - 1

            if not is_last_attempt:
                next_inflation = attempt_radii[idx + 1]
                logger.warning(
                    "Retrying planning with smaller inflation: {next_inflation}",
                    next_inflation=next_inflation,
                )

        if abort_due_to_invalid_cell:
            error_msg = "Planning aborted because start/goal is in an invalid cell; resample start/goal and retry."
        else:
            error_msg = (
                "Planning failed after trying inflation radii "
                f"{[r if r is not None else 'none' for r in attempt_radii]}"
            )
        if last_error is not None:
            error_msg = f"{error_msg}; last error: {last_error}"
        logger.error(error_msg)
        start_fmt = f"({start[0]:.2f}, {start[1]:.2f})"
        goal_fmt = f"({goal[0]:.2f}, {goal[1]:.2f})"
        logger.error(f"Planning from {start_fmt} to {goal_fmt} failed.")
        logger.error("Consider increasing the cells per meter value.")
        raise PlanningError(error_msg)

    def plan_random_path(
        self,
        algorithm: str | None = None,
        seed: int | None = None,
        max_attempts: int = 20,
    ) -> tuple[list[tuple[float, float]], dict | None, tuple[float, float], tuple[float, float]]:
        """Plan a path between two randomly sampled valid points.

        Args:
            algorithm: Optional algorithm override; defaults to the planner configuration.
            seed: Optional random seed for reproducibility.
            max_attempts: Maximum attempts to sample points and find a valid path.

        Returns:
            tuple[list[tuple[float, float]], dict | None, tuple[float, float], tuple[float, float]]:
                The planned path in world coordinates, path metadata, start, and goal.

        Raises:
            PlanningError: If no valid path can be found after the allowed attempts.
        """
        rng = random.Random(seed)
        last_error: PlanningError | None = None

        for attempt_idx in range(max_attempts):
            start = self.random_valid_point_on_grid(rng=rng)
            goal = self.random_valid_point_on_grid(rng=rng)
            if start == goal:
                continue

            try:
                path_world, path_info = self.plan(start=start, goal=goal, algorithm=algorithm)
                logger.info(
                    "Random path planned on attempt {attempt}: {start} -> {goal}",
                    attempt=attempt_idx + 1,
                    start=start,
                    goal=goal,
                )
                return path_world, path_info, start, goal
            except PlanningError as exc:
                last_error = exc
                logger.debug(
                    "Random planning attempt {attempt} failed: {error}",
                    attempt=attempt_idx + 1,
                    error=exc,
                )

        msg = f"Failed to plan random path after {max_attempts} attempts"
        if last_error is not None:
            msg = f"{msg}; last error: {last_error}"
        raise PlanningError(msg)

    def _scale_path_info(
        self,
        path_info: dict | None,
        grid: Grid,
        inflation: int | None,
    ) -> dict | None:
        """Convert planner path_info to world units and annotate metadata.

        Args:
            path_info: Raw path metadata from the planner.
            grid: Grid used for planning.
            inflation: Inflation radius (cells) applied during planning.

        Returns:
            Copy of path_info with length converted to meters and inflation annotated,
            or the original object when no conversion is possible.
        """
        if path_info is None:
            return None
        if not isinstance(path_info, dict):
            return path_info

        meters_per_cell = getattr(grid, "meters_per_cell", 1.0 / self.config.cells_per_meter)
        scaled_info = dict(path_info)
        length_cells = path_info.get("length")
        if isinstance(length_cells, (int, float)):
            scaled_info["length"] = float(length_cells) * meters_per_cell
        scaled_info["inflation_cells"] = inflation
        return scaled_info

    @staticmethod
    def _extract_expands(path_info: dict | None) -> dict | None:
        """Return the expand dictionary from path_info when available.

        Args:
            path_info: Planner metadata that may contain expand data.

        Returns:
            Dictionary of expanded nodes when present; otherwise None.
        """
        if not isinstance(path_info, dict):
            return None
        expands = path_info.get("expand")
        if isinstance(expands, dict):
            return expands
        return None

    def visualize_grid(
        self,
        output_path: Path | str | None = None,
        title: str = "Planning Grid",
        equal_aspect: bool = True,
        output_dpi: int = DEFAULT_VISUALIZATION_DPI,
    ) -> None:
        """Visualize the current planning grid.

        Args:
            output_path: Where to write the figure; shows interactively when None/empty.
            title: Title for the visualization window.
            equal_aspect: Whether to enforce equal aspect ratio.
            output_dpi: Export resolution for saved figures.
        """
        render_grid(
            self.grid,
            output_path=output_path,
            title=title,
            equal_aspect=equal_aspect,
            output_dpi=output_dpi,
        )

    def visualize_path(  # noqa: PLR0913
        self,
        path_world: list[tuple[float, float]] | None = None,
        output_path: Path | str | None = None,
        title: str = "Planned Path",
        equal_aspect: bool = True,
        path_style: str = "--",
        path_color: str = "C4",
        linewidth: float = 2.0,
        marker: str | None = "x",
        output_dpi: int = DEFAULT_VISUALIZATION_DPI,
        path_info: dict | None = None,
        show_expands: bool = True,
    ) -> None:
        """Visualize a planned path using the cached grid and path data.

        Args:
            path_world: Waypoints in world coordinates; defaults to last planned path.
            output_path: Where to write the figure; shows interactively when None/empty.
            title: Title for the visualization window.
            equal_aspect: Whether to enforce equal aspect ratio.
            path_style: Matplotlib linestyle for the path.
            path_color: Matplotlib color for the path.
            linewidth: Line width for the rendered path.
            marker: Optional marker for waypoints.
            output_dpi: Export resolution for saved figures.
            path_info: Optional planner metadata; defaults to the last plan result.
            show_expands: Whether to overlay expanded nodes when expand data is available.

        Raises:
            PlanningError: If no path is available to visualize.
        """
        if path_world is None:
            path_world = self._last_path_world
        if path_info is None:
            path_info = self._last_path_info
        if not path_world:
            msg = "No path available to visualize; run plan() or provide path_world."
            raise PlanningError(msg)

        grid = self.grid
        if show_expands:
            expands = self._extract_expands(path_info)
            if expands:
                # Create a shallow copy to avoid modifying the cached grid.
                grid = copy.copy(self.grid)
                grid.fill_expands(expands)
                logger.debug(
                    "Visualizing path with {n} expanded nodes overlayed",
                    n=len(expands),
                )
            else:
                logger.warning(
                    "Expanded area requested but path_info is missing expand data; skipping."
                )

        path_grid = [self._world_to_grid(x, y) for x, y in path_world]
        render_path(
            grid,
            path_grid,
            output_path=output_path,
            title=title,
            equal_aspect=equal_aspect,
            path_style=path_style,
            path_color=path_color,
            linewidth=linewidth,
            marker=marker,
            output_dpi=output_dpi,
        )


render_grid = visualize_grid
render_path = visualize_path


__all__ = [
    "ClassicGlobalPlanner",
    "ClassicPlanVisualizer",
    "ClassicPlannerConfig",
    "MotionPlanningGridConfig",
    "PlanningError",
    "count_obstacle_cells",
    "get_obstacle_statistics",
    "map_definition_to_motion_planning_grid",
    "set_start_goal_on_grid",
    "visualize_grid",
    "visualize_path",
]

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
    >>> planner = ClassicGlobalPlanner(map_def, cells_per_meter=1.0)
    >>> path, info = planner.plan(start=(5.0, 5.0), goal=(45.0, 25.0))
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from python_motion_planning.common import TYPES
from python_motion_planning.path_planner import AStar, ThetaStar

from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    map_definition_to_motion_planning_grid,
)
from robot_sf.nav.motion_planning_adapter import visualize_grid as render_grid
from robot_sf.nav.motion_planning_adapter import visualize_path as render_path

if TYPE_CHECKING:
    from pathlib import Path

    from python_motion_planning.common import Grid

    from robot_sf.nav.map_config import MapDefinition


@dataclass(slots=True)
class ClassicPlannerConfig:
    """Configuration for the classic grid-based planner.

    Attributes:
        cells_per_meter: Grid resolution (cells per meter).
        inflate_radius_cells: Number of cells to inflate obstacles (for robot radius).
        add_boundary_obstacles: Whether to add obstacles at map boundaries.
        algorithm: Planning algorithm to use ('theta_star', 'a_star', etc.).
    """

    cells_per_meter: float = 1.0
    inflate_radius_cells: int | None = 2
    add_boundary_obstacles: bool = True
    algorithm: str = "theta_star"

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
    uses algorithms from python_motion_planning for path planning.

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
        grid_y = math.floor(world_y * self.config.cells_per_meter)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> tuple[float, float]:
        """Convert grid indices to world coordinates.

        Args:
            grid_x: Grid X index.
            grid_y: Grid Y index.

        Returns:
            Tuple of (world_x, world_y) coordinates in meters.
        """
        world_x = grid_x / self.config.cells_per_meter
        world_y = grid_y / self.config.cells_per_meter
        return world_x, world_y

    def _normalize_algorithm(self, override: str | None = None) -> str:
        """Normalize the algorithm name and validate support."""
        algo_raw = (override or self.config.algorithm).strip().lower()
        if algo_raw in {"theta_star", "thetastar", "theta"}:
            return "theta_star"
        if algo_raw in {"a_star", "astar", "a*", "a-star"}:
            return "a_star"
        msg = f"Unsupported algorithm: {override or self.config.algorithm}"
        raise ValueError(msg)

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
            tuple[list[tuple[float, float]], dict | None]: Waypoints in world coordinates and
            optional path metadata with length scaled to meters.

        Raises:
            PlanningError: If planning fails or coordinates are out of bounds.
        """
        start_grid = self._world_to_grid(*start)
        goal_grid = self._world_to_grid(*goal)
        algo = self._normalize_algorithm(algorithm)

        def _validate_point(name: str, gx: int, gy: int) -> None:
            if not (0 <= gx < self.grid.type_map.shape[0]) or not (
                0 <= gy < self.grid.type_map.shape[1]
            ):
                raise PlanningError(
                    f"{name} out of grid bounds: world={start if name == 'start' else goal}, "
                    f"grid=({gx}, {gy}), grid_shape={self.grid.type_map.shape}"
                )

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

        for idx, inflation in enumerate(attempt_radii):
            grid = self._build_grid(inflation)

            # Mark start and goal in grid
            _validate_point("start", *start_grid)
            _validate_point("goal", *goal_grid)
            grid.type_map[start_grid[0]][start_grid[1]] = TYPES.START
            grid.type_map[goal_grid[0]][goal_grid[1]] = TYPES.GOAL

            if algo == "theta_star":
                planner = ThetaStar(map_=grid, start=start_grid, goal=goal_grid)
                logger.warning("Theta_star can be roughly 20x slower than A_star on large grids.")
            elif algo == "a_star":
                planner = AStar(map_=grid, start=start_grid, goal=goal_grid)
            else:
                msg = f"Unsupported algorithm: {algo}"
                raise ValueError(msg)

            try:
                plan_result = planner.plan()
                path_grid, path_info = (
                    plan_result if isinstance(plan_result, tuple) else (plan_result, None)
                )

                if path_grid:
                    path_world = [self._grid_to_world(x, y) for x, y in path_grid]
                    scaled_info = self._scale_path_info(path_info, grid, inflation)
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

            is_last_attempt = idx == len(attempt_radii) - 1
            if not is_last_attempt:
                next_inflation = attempt_radii[idx + 1]
                logger.warning(
                    "Retrying planning with smaller inflation: {next_inflation}",
                    next_inflation=next_inflation,
                )

        error_msg = (
            "Planning failed after trying inflation radii "
            f"{[r if r is not None else 'none' for r in attempt_radii]}"
        )
        if last_error is not None:
            error_msg = f"{error_msg}; last error: {last_error}"
        logger.error(error_msg)
        logger.error("Planning from {start} to {goal} failed.", start=start, goal=goal)
        logger.error("Consider increasing the cells per meter value.")
        raise PlanningError(error_msg)

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
    ) -> None:
        """Visualize the current planning grid.

        Args:
            output_path: Where to write the figure; shows interactively when None/empty.
            title: Title for the visualization window.
            equal_aspect: Whether to enforce equal aspect ratio.
        """
        render_grid(self.grid, output_path=output_path, title=title, equal_aspect=equal_aspect)

    def visualize_path(
        self,
        path_world: list[tuple[float, float]] | None = None,
        output_path: Path | str | None = None,
        title: str = "Planned Path",
        equal_aspect: bool = True,
        path_style: str = "--",
        path_color: str = "C4",
        linewidth: float = 2.0,
        marker: str | None = "x",
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
        )


__all__ = [
    "ClassicGlobalPlanner",
    "ClassicPlannerConfig",
    "PlanningError",
]

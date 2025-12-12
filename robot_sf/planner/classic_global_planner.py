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
    >>> path = planner.plan(start=(5.0, 5.0), goal=(45.0, 25.0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from python_motion_planning.common import TYPES
from python_motion_planning.path_planner import ThetaStar

from robot_sf.nav.motion_planning_adapter import (
    MotionPlanningGridConfig,
    map_definition_to_motion_planning_grid,
)

if TYPE_CHECKING:
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
            grid_config = MotionPlanningGridConfig(
                cells_per_meter=self.config.cells_per_meter,
                inflate_radius_cells=self.config.inflate_radius_cells,
                add_boundary_obstacles=self.config.add_boundary_obstacles,
            )
            self._grid = map_definition_to_motion_planning_grid(self.map_def, config=grid_config)
            logger.debug(
                "Created planning grid with shape {shape}",
                shape=self._grid.type_map.shape,
            )
        return self._grid

    def _world_to_grid(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert world coordinates to grid indices.

        Args:
            world_x: X coordinate in world space (meters).
            world_y: Y coordinate in world space (meters).

        Returns:
            Tuple of (grid_x, grid_y) indices.
        """
        grid_x = int(world_x * self.config.cells_per_meter)
        grid_y = int(world_y * self.config.cells_per_meter)
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

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Plan a path from start to goal.

        Args:
            start: Start position (x, y) in world coordinates (meters).
            goal: Goal position (x, y) in world coordinates (meters).

        Returns:
            List of waypoints [(x, y), ...] in world coordinates.
            Empty list if planning fails.

        Raises:
            PlanningError: If planning fails or coordinates are out of bounds.
        """
        # Convert to grid coordinates
        start_grid = self._world_to_grid(*start)
        goal_grid = self._world_to_grid(*goal)

        logger.debug(
            "Planning from {start} to {goal} (world coords: {start_w} → {goal_w})",
            start=start_grid,
            goal=goal_grid,
            start_w=start,
            goal_w=goal,
        )

        # Get or create grid
        grid = self.grid

        # Mark start and goal in grid
        grid.type_map[start_grid] = TYPES.START
        grid.type_map[goal_grid] = TYPES.GOAL

        # Select and run planner
        if self.config.algorithm == "theta_star":
            planner = ThetaStar(map_=grid, start=start_grid, goal=goal_grid)
        else:
            msg = f"Unsupported algorithm: {self.config.algorithm}"
            raise ValueError(msg)

        try:
            path_grid, _ = planner.plan()

            if not path_grid:
                logger.warning("No path found from {start} to {goal}", start=start, goal=goal)
                return []

            # Convert path back to world coordinates
            path_world = [self._grid_to_world(x, y) for x, y in path_grid]

            logger.info(
                "Found path with {n} waypoints from {start} to {goal}",
                n=len(path_world),
                start=start,
                goal=goal,
            )

            return path_world

        except Exception as e:
            msg = f"Planning failed: {e}"
            logger.error(msg)
            raise PlanningError(msg) from e


__all__ = [
    "ClassicGlobalPlanner",
    "ClassicPlannerConfig",
    "PlanningError",
]

"""Adapter to convert SVG map definitions into python_motion_planning Grid maps.

This helper discretizes ``MapDefinition`` obstacles into a grid representation that
python_motion_planning understands. It mirrors the occupancy grid discretization
strategy (meter-to-cell scaling, boundary padding) while remaining lightweight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger
from python_motion_planning.common import TYPES, Grid
from shapely.affinity import scale
from shapely.geometry import Polygon, box

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


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
    """Scale a world coordinate in meters to a grid cell index."""
    return int(math.floor(value * cells_per_meter))


def _mark_obstacle_cells(
    grid: Grid,  # type: ignore[name-defined]
    polygon: Polygon,
    cells_per_meter: float,
    types: object,
) -> None:
    """Rasterize a single obstacle polygon into the grid's type_map.

    Handles coordinate system conversion: world coords have Y-up, grid has Y-down.
    python_motion_planning Grid type_map is indexed [x][y], not [y][x].
    """
    if polygon.is_empty:
        return

    type_map = grid.type_map
    logger.debug(f"type_map type: {type(type_map)}, shape: {type_map.shape}, id: {id(type_map)}")
    # python_motion_planning GridTypeMap shape is (width, height) = (X, Y)
    # and it's indexed [y][x] in row-major order (standard numpy convention)
    width_cells = type_map.shape[0]  # X dimension (columns)
    height_cells = type_map.shape[1]  # Y dimension (rows)

    # Scale polygon coordinates to grid space
    scaled = scale(polygon, xfact=cells_per_meter, yfact=cells_per_meter, origin=(0, 0))
    minx, miny, maxx, maxy = scaled.bounds

    x_start = max(0, _world_to_grid(minx, 1.0))
    x_end = min(width_cells, int(math.ceil(maxx)))
    y_start = max(0, _world_to_grid(miny, 1.0))
    y_end = min(height_cells, int(math.ceil(maxy)))

    cells_marked = 0
    for y_world in range(y_start, y_end):
        for x_world in range(x_start, x_end):
            # Create cell polygon in world coordinates
            cell_poly = box(x_world, y_world, x_world + 1, y_world + 1)
            if scaled.intersects(cell_poly):
                # Convert world Y to grid: Y-axis is flipped
                # World: Y=0 at bottom, Y=height at top
                # Grid visualization: Y increases downward (row 0 at top)
                grid_y = height_cells - 1 - y_world
                # Ensure indices are within bounds
                if 0 <= x_world < width_cells and 0 <= grid_y < height_cells:
                    # python_motion_planning Grid: shape=(width, height), index as [x][y]
                    type_map[x_world][grid_y] = types.OBSTACLE
                    cells_marked += 1
                    # Debug: verify the value was set
                    if cells_marked == 1:
                        # Verify immediately after setting
                        actual_value = type_map[x_world][grid_y]
                        logger.debug(
                            "First cell marked at [{x}][{y}] = {val} (OBSTACLE={obs}, types.FREE={free})",
                            x=x_world,
                            y=grid_y,
                            val=actual_value,
                            obs=types.OBSTACLE,
                            free=types.FREE,
                        )

    if cells_marked > 0:
        logger.debug(
            "Marked {count} cells for obstacle",
            count=cells_marked,
        )
        # Check immediately after marking
        import numpy as np

        arr = np.asarray(type_map)
        unique, counts = np.unique(arr, return_counts=True)
        logger.debug(f"Unique values in type_map after marking: {dict(zip(unique, counts))}")


def map_definition_to_motion_planning_grid(
    map_def: MapDefinition, config: MotionPlanningGridConfig | None = None
) -> Grid:  # type: ignore[name-defined]
    """Convert a MapDefinition into a python_motion_planning Grid.

    Args:
        map_def: Parsed SVG map definition.
        config: Grid scaling and padding configuration.

    Returns:
        Configured Grid with obstacle cells marked.
    """
    cfg = config or MotionPlanningGridConfig()

    width_cells = math.ceil(map_def.width * cfg.cells_per_meter)
    height_cells = math.ceil(map_def.height * cfg.cells_per_meter)
    grid = Grid(bounds=[[0, width_cells], [0, height_cells]])

    if cfg.add_boundary_obstacles:
        grid.fill_boundary_with_obstacles()

    for obstacle in map_def.obstacles:
        poly = Polygon(obstacle.vertices)
        if not poly.is_valid or poly.is_empty:
            logger.warning("Skipping invalid obstacle during grid rasterization.")
            continue
        _mark_obstacle_cells(grid, poly, cfg.cells_per_meter, TYPES)

    # Debug: count obstacle cells before inflation
    import numpy as np

    type_map_np = np.asarray(grid.type_map)
    obstacle_count_before = np.count_nonzero(type_map_np == TYPES.OBSTACLE)
    logger.debug(
        "Grid has {obs} obstacle cells BEFORE inflation",
        obs=obstacle_count_before,
    )

    if cfg.inflate_radius_cells is not None:
        grid.inflate_obstacles(radius=cfg.inflate_radius_cells)

    # Debug: count obstacle cells after inflation
    type_map_np = np.asarray(grid.type_map)
    obstacle_count_after = np.count_nonzero(type_map_np == TYPES.OBSTACLE)
    logger.debug(
        "Grid has {obs} obstacle cells AFTER inflation ({pct:.2f}%)",
        obs=obstacle_count_after,
        pct=obstacle_count_after / type_map_np.size * 100 if type_map_np.size > 0 else 0,
    )

    logger.info(
        "Converted map to motion-planning grid: {w}x{h} cells ({m:.2f} m/cell)",
        w=width_cells,
        h=height_cells,
        m=cfg.meters_per_cell,
    )
    return grid


__all__ = ["MotionPlanningGridConfig", "map_definition_to_motion_planning_grid"]

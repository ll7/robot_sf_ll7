"""Rasterization functions for occupancy grid generation.

This module provides core rasterization algorithms to convert continuous
geometric primitives (line segments, circles) into discrete grid cells.

Functions:
- rasterize_line_segment: Convert line segment to occupied grid cells
- rasterize_circle: Convert circle to occupied grid cells
- rasterize_obstacles: Batch process obstacle line segments
- rasterize_pedestrians: Batch process pedestrian circles

Performance:
- Bresenham's algorithm for line rasterization: O(max(dx, dy))
- Discrete disk for circle rasterization: O(π * r²)
- Vectorized operations where possible via NumPy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from robot_sf.common.types import Circle2D, Line2D, RobotPose  # noqa: TC001
from robot_sf.nav.occupancy_grid_utils import get_affected_cells, world_to_grid_indices

if TYPE_CHECKING:
    from robot_sf.nav.occupancy_grid import GridConfig

try:  # Optional acceleration when shapely is available
    try:
        from shapely import contains_xy as _shp_contains_xy
    except ImportError:  # pragma: no cover - shapely <2.0 or variant
        from shapely.predicates import contains_xy as _shp_contains_xy  # type: ignore[assignment]
    from shapely import vectorized as _shp_vectorized
    from shapely.geometry import Point as _ShapelyPoint
    from shapely.geometry import Polygon as _ShapelyPolygon
    from shapely.prepared import prep as _shp_prep

    _SHAPELY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SHAPELY_AVAILABLE = False
    _ShapelyPolygon = None  # type: ignore[assignment]
    _ShapelyPoint = None  # type: ignore[assignment]
    _shp_vectorized = None  # type: ignore[assignment]
    _shp_prep = None  # type: ignore[assignment]
    _shp_contains_xy = None  # type: ignore[assignment]


def rasterize_line_segment(
    line: Line2D,
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> None:
    """Rasterize a line segment into the occupancy grid.

    Uses Bresenham's line algorithm for efficient discrete line drawing.

    Args:
        line: Line segment (start, end) coordinates
        grid_array: 2D grid array to modify [H, W]
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame
        grid_origin_y: Grid origin Y in world frame
        value: Occupancy value to set (default: 1.0)

    Modifies:
        grid_array: Sets cells along the line to `value`

    Performance:
        O(max(|dx|, |dy|)) where dx, dy are grid cell distances

    Example:
        >>> grid = np.zeros((100, 100))
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> line = Line2D((1.0, 1.0), (9.0, 9.0))
        >>> rasterize_line_segment(line, grid, config)
        >>> np.sum(grid > 0)  # Count occupied cells
        80
    """
    start, end = line

    # Clip the segment to the grid rectangle so that partially overlapping lines are rasterized
    min_x = grid_origin_x
    max_x = grid_origin_x + config.width
    min_y = grid_origin_y
    max_y = grid_origin_y + config.height
    clipped = _clip_line_to_rect(start, end, min_x, max_x, min_y, max_y)
    if clipped is None:
        logger.debug(f"Line segment {line} outside grid bounds, skipping")
        return
    start_clipped, end_clipped = clipped

    try:
        # Convert endpoints to grid indices
        row0, col0 = world_to_grid_indices(
            start_clipped[0], start_clipped[1], config, grid_origin_x, grid_origin_y
        )
        row1, col1 = world_to_grid_indices(
            end_clipped[0], end_clipped[1], config, grid_origin_x, grid_origin_y
        )
    except ValueError as e:
        # Endpoint outside grid bounds after clipping (should not happen but defend anyway)
        logger.debug(f"Line endpoint outside grid after clipping: {e}")
        return

    # Bresenham's line algorithm
    cells = _bresenham_line(row0, col0, row1, col1)

    # Set occupancy for all cells on the line
    for row, col in cells:
        if 0 <= row < config.grid_height and 0 <= col < config.grid_width:
            grid_array[row, col] = max(grid_array[row, col], value)


def _bresenham_line(row0: int, col0: int, row1: int, col1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm for discrete line rasterization.

    Args:
        row0: Starting row index.
        col0: Starting column index.
        row1: Ending row index.
        col1: Ending column index.

    Returns:
        List of (row, col) tuples along the line

    References:
        https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    cells = []

    dx = abs(col1 - col0)
    dy = abs(row1 - row0)
    sx = 1 if col0 < col1 else -1
    sy = 1 if row0 < row1 else -1
    err = dx - dy

    row, col = row0, col0

    while True:
        cells.append((row, col))

        if row == row1 and col == col1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            col += sx
        if e2 < dx:
            err += dx
            row += sy

    return cells


def _clip_line_to_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a line segment to an axis-aligned rectangle using Liang-Barsky.

    Returns the clipped segment (start, end) if it intersects the rectangle, otherwise None.

    Returns:
        tuple[tuple[float, float], tuple[float, float]] | None: Clipped line endpoints in world
        coordinates when the segment intersects the rectangle; ``None`` if fully outside.
    """
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0

    p = (-dx, dx, -dy, dy)
    q = (x0 - min_x, max_x - x0, y0 - min_y, max_y - y0)

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q, strict=False):
        if pi == 0:
            if qi < 0:
                return None  # Parallel and outside
            continue
        r = qi / pi
        if pi < 0:
            if r > u2:
                return None
            u1 = max(u1, r)
        else:
            if r < u1:
                return None
            u2 = min(u2, r)

    clipped_start = (x0 + u1 * dx, y0 + u1 * dy)
    clipped_end = (x0 + u2 * dx, y0 + u2 * dy)
    return clipped_start, clipped_end


def rasterize_circle(
    circle: Circle2D,
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> None:
    """Rasterize a circle into the occupancy grid.

    Uses discrete disk algorithm to fill all cells within the circle.
    Correctly handles circles whose centers are outside the grid bounds
    but which partially overlap the grid.

    Args:
        circle: Circle (center_x, center_y, radius)
        grid_array: 2D grid array to modify [H, W]
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame
        grid_origin_y: Grid origin Y in world frame
        value: Occupancy value to set (default: 1.0)

    Modifies:
        grid_array: Sets cells within circle to `value`

    Performance:
        O(π * (radius/resolution)²)

    Note:
        The function correctly rasterizes circles even when their centers
        are outside the grid, as long as they partially overlap the grid.
        This is handled by `get_affected_cells()` which performs proper
        bounding box intersection testing.

    Example:
        >>> grid = np.zeros((100, 100))
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> # Circle fully inside
        >>> circle = Circle2D((5.0, 5.0), 0.5)
        >>> rasterize_circle(circle, grid, config)
        >>> np.sum(grid > 0)  # Count occupied cells
        ~78  # Approximately π * 5² cells
        >>> # Circle center outside but overlapping
        >>> grid2 = np.zeros((100, 100))
        >>> circle2 = Circle2D((10.5, 5.0), 1.0)
        >>> rasterize_circle(circle2, grid2, config)
        >>> np.sum(grid2 > 0)  # Some cells from overlap
        15
    """
    center, radius = circle

    # Get all cells affected by the circle
    # The helper function handles bounds checking and clipping internally,
    # correctly detecting overlap even when circle center is outside grid
    try:
        affected_cells = get_affected_cells(
            center[0], center[1], radius, config, grid_origin_x, grid_origin_y
        )
    except ValueError as e:
        logger.debug(f"Circle {circle} does not intersect grid: {e}")
        return

    # Set occupancy for all affected cells
    for row, col in affected_cells:
        if 0 <= row < config.grid_height and 0 <= col < config.grid_width:
            grid_array[row, col] = max(grid_array[row, col], value)


def rasterize_obstacles(
    obstacles: list[Line2D],
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> int:
    """Rasterize multiple obstacle line segments into grid.

    Args:
        obstacles: List of line segment obstacles
        grid_array: 2D grid array to modify [H, W]
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame
        grid_origin_y: Grid origin Y in world frame
        value: Occupancy value to set (default: 1.0)

    Returns:
        Number of obstacles successfully rasterized

    Modifies:
        grid_array: Sets cells along obstacle lines to `value`

    Performance:
        O(N * max_line_length) where N = number of obstacles
    """
    count = 0
    for obstacle in obstacles:
        try:
            rasterize_line_segment(
                obstacle, grid_array, config, grid_origin_x, grid_origin_y, value
            )
            count += 1
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"Failed to rasterize obstacle {obstacle}: {e}")

    logger.debug(f"Rasterized {count}/{len(obstacles)} obstacles")
    return count


def rasterize_pedestrians(
    pedestrians: list[Circle2D],
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> int:
    """Rasterize multiple pedestrian circles into grid.

    Args:
        pedestrians: List of circular pedestrian objects
        grid_array: 2D grid array to modify [H, W]
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame
        grid_origin_y: Grid origin Y in world frame
        value: Occupancy value to set (default: 1.0)

    Returns:
        Number of pedestrians successfully rasterized

    Modifies:
        grid_array: Sets cells within pedestrian circles to `value`

    Performance:
        O(N * π * (max_radius/resolution)²) where N = number of pedestrians
    """
    count = 0
    for pedestrian in pedestrians:
        try:
            rasterize_circle(pedestrian, grid_array, config, grid_origin_x, grid_origin_y, value)
            count += 1
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"Failed to rasterize pedestrian {pedestrian}: {e}")

    logger.debug(f"Rasterized {count}/{len(pedestrians)} pedestrians")
    return count


def rasterize_robot(
    robot_pose: RobotPose,
    robot_radius: float,
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> bool:
    """Rasterize robot as a circle into grid.

    Args:
        robot_pose: Robot's pose (x, y, theta)
        robot_radius: Robot's radius (meters)
        grid_array: 2D grid array to modify [H, W]
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame
        grid_origin_y: Grid origin Y in world frame
        value: Occupancy value to set (default: 1.0)

    Returns:
        True if robot was successfully rasterized, False otherwise

    Modifies:
        grid_array: Sets cells within robot circle to `value`
    """
    # Extract position from RobotPose tuple ((x, y), theta)
    robot_position, _robot_orientation = robot_pose
    # Create Circle2D as tuple (center, radius)
    robot_circle: Circle2D = (robot_position, robot_radius)
    try:
        rasterize_circle(robot_circle, grid_array, config, grid_origin_x, grid_origin_y, value)
        return True
    except (ValueError, IndexError, TypeError) as e:
        logger.warning(f"Failed to rasterize robot at {robot_pose}: {e}")
        return False


def rasterize_polygon(
    polygon: list[tuple[float, float]],
    grid_array: np.ndarray,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
    value: float = 1.0,
) -> int:
    """Rasterize a filled polygon into the occupancy grid.

    Returns:
        int: Number of grid cells marked as occupied.
    """
    if len(polygon) < 3:
        return 0

    if polygon[0] != polygon[-1]:
        polygon = [*polygon, polygon[0]]

    xs, ys = zip(*polygon, strict=False)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    grid_min_x = grid_origin_x
    grid_max_x = grid_origin_x + config.width
    grid_min_y = grid_origin_y
    grid_max_y = grid_origin_y + config.height

    if max_x < grid_min_x or min_x > grid_max_x or max_y < grid_min_y or min_y > grid_max_y:
        return 0

    col_start = max(int((min_x - grid_origin_x) / config.resolution), 0)
    col_end = min(int((max_x - grid_origin_x) / config.resolution) + 1, config.grid_width)
    row_start = max(int((min_y - grid_origin_y) / config.resolution), 0)
    row_end = min(int((max_y - grid_origin_y) / config.resolution) + 1, config.grid_height)

    if col_start >= col_end or row_start >= row_end:
        return 0

    cols = np.arange(col_start, col_end)
    rows = np.arange(row_start, row_end)
    xv = grid_origin_x + (cols + 0.5) * config.resolution
    yv = grid_origin_y + (rows + 0.5) * config.resolution
    mesh_x, mesh_y = np.meshgrid(xv, yv)

    inside_mask = _points_in_polygon(mesh_x, mesh_y, polygon)
    if not inside_mask.any():
        return 0

    subgrid = grid_array[row_start:row_end, col_start:col_end]
    subgrid[inside_mask] = np.maximum(subgrid[inside_mask], value)
    return int(np.count_nonzero(inside_mask))


def _points_in_polygon(
    mesh_x: np.ndarray, mesh_y: np.ndarray, polygon: list[tuple[float, float]]
) -> np.ndarray:
    """Return a boolean mask of points inside a polygon."""
    if _SHAPELY_AVAILABLE:
        poly = _ShapelyPolygon(polygon)
        if not poly.is_valid:  # pragma: no cover - defensive
            poly = poly.buffer(0)
        if poly.is_empty:
            return np.zeros_like(mesh_x, dtype=bool)

        flat_x = mesh_x.ravel()
        flat_y = mesh_y.ravel()
        if _shp_contains_xy is not None:
            flat_mask = _shp_contains_xy(poly, flat_x, flat_y)
        elif _shp_vectorized is not None:
            flat_mask = _shp_vectorized.contains(poly, flat_x, flat_y)
        else:  # pragma: no cover - fallback for shapely without vectorized API
            prepared = _shp_prep(poly)
            flat_mask = np.fromiter(
                (
                    prepared.contains(_ShapelyPoint(x, y))
                    for x, y in zip(flat_x, flat_y, strict=False)
                ),
                dtype=bool,
                count=flat_x.size,
            )
        return flat_mask.reshape(mesh_x.shape)

    px = np.asarray([p[0] for p in polygon])
    py = np.asarray([p[1] for p in polygon])
    if px[0] == px[-1] and py[0] == py[-1]:
        px = px[:-1]
        py = py[:-1]

    x0 = px
    y0 = py
    x1 = np.roll(px, -1)
    y1 = np.roll(py, -1)

    flat_x = mesh_x.ravel()
    flat_y = mesh_y.ravel()

    cond1 = (y0 <= flat_y[:, None]) & (y1 > flat_y[:, None])
    cond2 = (y1 <= flat_y[:, None]) & (y0 > flat_y[:, None])
    intersects = cond1 | cond2
    x_intersect = x0 + (flat_y[:, None] - y0) * (x1 - x0) / (y1 - y0 + 1e-12)
    crossings = intersects & (flat_x[:, None] < x_intersect)
    inside = crossings.sum(axis=1) % 2 == 1
    return inside.reshape(mesh_x.shape)

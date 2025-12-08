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

import numpy as np  # noqa: TC002 - numpy is needed at runtime for array operations
from loguru import logger

from robot_sf.common.types import Circle2D, Line2D, RobotPose  # noqa: TC001
from robot_sf.nav.occupancy_grid_utils import (
    get_affected_cells,
    is_within_grid,
    world_to_grid_indices,
)

if TYPE_CHECKING:
    from robot_sf.nav.occupancy_grid import GridConfig


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

    # Check if line is within or intersects grid bounds
    if not (
        is_within_grid(start[0], start[1], config, grid_origin_x, grid_origin_y)
        or is_within_grid(end[0], end[1], config, grid_origin_x, grid_origin_y)
    ):
        # Line might still intersect grid even if endpoints are outside
        # For now, skip (could implement line clipping in future)
        logger.debug(f"Line segment {line} outside grid bounds, skipping")
        return

    try:
        # Convert endpoints to grid indices
        row0, col0 = world_to_grid_indices(start[0], start[1], config, grid_origin_x, grid_origin_y)
        row1, col1 = world_to_grid_indices(end[0], end[1], config, grid_origin_x, grid_origin_y)
    except ValueError as e:
        # Endpoint outside grid bounds
        logger.debug(f"Line endpoint outside grid: {e}")
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

"""Utility functions for occupancy grid operations.

This module provides foundational grid utilities:
1. Coordinate transformations (world ↔ grid, world ↔ ego)
2. Cell indexing and bounds checking
3. Distance computations for rasterization
4. Grid value interpolation and smoothing

These utilities support the core grid generation and query operations
while maintaining high performance (Numba JIT compilation where beneficial).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.common.types import RobotPose
    from robot_sf.nav.occupancy_grid import GridConfig


def world_to_grid_indices(
    world_x: float,
    world_y: float,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> tuple[int, int]:
    """Convert world coordinates to grid indices.

    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame (default: 0.0)
        grid_origin_y: Grid origin Y in world frame (default: 0.0)

    Returns:
        Tuple of (row, col) grid indices (0-based)

    Raises:
        ValueError: If coordinates are out of grid bounds

    Notes:
        - World frame origin is at (grid_origin_x, grid_origin_y)
        - Grid frame has origin at top-left (0, 0)
        - Row index increases downward (Y+ in world → row+ in grid)
        - Col index increases rightward (X+ in world → col+ in grid)

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> row, col = world_to_grid_indices(5.0, 5.0, config, 0.0, 0.0)
        >>> row, col
        (50, 50)
    """
    # Translate to grid origin
    local_x = world_x - grid_origin_x
    local_y = world_y - grid_origin_y

    # Check bounds
    if local_x < 0 or local_x > config.width:
        raise ValueError(
            f"X coordinate {world_x} out of bounds "
            f"[{grid_origin_x}, {grid_origin_x + config.width}]"
        )
    if local_y < 0 or local_y > config.height:
        raise ValueError(
            f"Y coordinate {world_y} out of bounds "
            f"[{grid_origin_y}, {grid_origin_y + config.height}]"
        )

    # Convert to grid indices
    col = int(local_x / config.resolution)
    row = int(local_y / config.resolution)

    # Clamp to valid range
    row = min(row, config.grid_height - 1)
    col = min(col, config.grid_width - 1)

    return row, col


def grid_indices_to_world(
    row: int,
    col: int,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> tuple[float, float]:
    """Convert grid indices to world coordinates.

    Args:
        row: Grid row index (0-based)
        col: Grid column index (0-based)
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame (default: 0.0)
        grid_origin_y: Grid origin Y in world frame (default: 0.0)

    Returns:
        Tuple of (world_x, world_y) coordinates

    Raises:
        ValueError: If indices are out of bounds

    Notes:
        - Converts grid cell center to world coordinates
        - Returns center of the cell at (row, col)

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> x, y = grid_indices_to_world(50, 50, config, 0.0, 0.0)
        >>> x, y
        (5.0, 5.0)
    """
    if not (0 <= row < config.grid_height):
        raise ValueError(f"Row index {row} out of bounds [0, {config.grid_height - 1}]")
    if not (0 <= col < config.grid_width):
        raise ValueError(f"Col index {col} out of bounds [0, {config.grid_width - 1}]")

    # Convert to world coordinates (cell center)
    local_x = (col + 0.5) * config.resolution
    local_y = (row + 0.5) * config.resolution

    world_x = local_x + grid_origin_x
    world_y = local_y + grid_origin_y

    return world_x, world_y


def is_within_grid(
    world_x: float,
    world_y: float,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> bool:
    """Check if world coordinates are within grid bounds.

    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame (default: 0.0)
        grid_origin_y: Grid origin Y in world frame (default: 0.0)

    Returns:
        True if coordinates are within grid bounds, False otherwise

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> is_within_grid(5.0, 5.0, config, 0.0, 0.0)
        True
        >>> is_within_grid(15.0, 5.0, config, 0.0, 0.0)
        False
    """
    local_x = world_x - grid_origin_x
    local_y = world_y - grid_origin_y

    return (0 <= local_x <= config.width) and (0 <= local_y <= config.height)


def world_to_ego(
    world_x: float,
    world_y: float,
    robot_pose: RobotPose,
) -> tuple[float, float]:
    """Transform world coordinates to robot's ego frame.

    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        robot_pose: Robot's pose (x, y, theta)

    Returns:
        Tuple of (ego_x, ego_y) in robot's local frame

    Notes:
        - Rotation: θ = -robot_pose.theta (inverse rotation)
        - Translation: Center at robot position
        - Ego frame: X+ forward, Y+ left, origin at robot center

    Example:
        >>> import math
        >>> pose = RobotPose(x=1.0, y=2.0, theta=0.0)
        >>> ego_x, ego_y = world_to_ego(2.0, 2.0, pose)
        >>> ego_x, ego_y
        (1.0, 0.0)
    """
    # Translate to robot position
    dx = world_x - robot_pose.x
    dy = world_y - robot_pose.y

    # Rotate by -theta
    cos_theta = np.cos(-robot_pose.theta)
    sin_theta = np.sin(-robot_pose.theta)

    ego_x = dx * cos_theta - dy * sin_theta
    ego_y = dx * sin_theta + dy * cos_theta

    return ego_x, ego_y


def ego_to_world(
    ego_x: float,
    ego_y: float,
    robot_pose: RobotPose,
) -> tuple[float, float]:
    """Transform ego frame coordinates to world frame.

    Args:
        ego_x: X coordinate in robot's ego frame
        ego_y: Y coordinate in robot's ego frame
        robot_pose: Robot's pose (x, y, theta)

    Returns:
        Tuple of (world_x, world_y) in world frame

    Notes:
        - Inverse of world_to_ego
        - Rotation: θ = robot_pose.theta (forward rotation)

    Example:
        >>> pose = RobotPose(x=1.0, y=2.0, theta=0.0)
        >>> world_x, world_y = ego_to_world(1.0, 0.0, pose)
        >>> world_x, world_y
        (2.0, 2.0)
    """
    # Rotate by theta
    cos_theta = np.cos(robot_pose.theta)
    sin_theta = np.sin(robot_pose.theta)

    dx = ego_x * cos_theta - ego_y * sin_theta
    dy = ego_x * sin_theta + ego_y * cos_theta

    world_x = dx + robot_pose.x
    world_y = dy + robot_pose.y

    return world_x, world_y


def get_grid_bounds(
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> tuple[float, float, float, float]:
    """Get the world-frame bounds of the grid.

    Args:
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame (default: 0.0)
        grid_origin_y: Grid origin Y in world frame (default: 0.0)

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) in world frame

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> bounds = get_grid_bounds(config, 0.0, 0.0)
        >>> bounds
        (0.0, 10.0, 0.0, 10.0)
    """
    min_x = grid_origin_x
    max_x = grid_origin_x + config.width
    min_y = grid_origin_y
    max_y = grid_origin_y + config.height

    return min_x, max_x, min_y, max_y


def clip_to_grid(
    world_x: float,
    world_y: float,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> tuple[float, float]:
    """Clip world coordinates to grid bounds.

    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        config: Grid configuration
        grid_origin_x: Grid origin X in world frame (default: 0.0)
        grid_origin_y: Grid origin Y in world frame (default: 0.0)

    Returns:
        Tuple of clipped (world_x, world_y)

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> x, y = clip_to_grid(15.0, 5.0, config, 0.0, 0.0)
        >>> x, y
        (10.0, 5.0)
    """
    min_x, max_x, min_y, max_y = get_grid_bounds(config, grid_origin_x, grid_origin_y)

    clipped_x = np.clip(world_x, min_x, max_x)
    clipped_y = np.clip(world_y, min_y, max_y)

    return clipped_x, clipped_y


def get_affected_cells(
    world_x: float,
    world_y: float,
    radius: float,
    config: GridConfig,
    grid_origin_x: float = 0.0,
    grid_origin_y: float = 0.0,
) -> list[tuple[int, int]]:
    """Get grid cells affected by a circular region.

    Args:
        world_x: Center X coordinate
        world_y: Center Y coordinate
        radius: Radius of the circular region
        config: Grid configuration
        grid_origin_x: Grid origin X (default: 0.0)
        grid_origin_y: Grid origin Y (default: 0.0)

    Returns:
        List of (row, col) tuples for affected cells

    Notes:
        - Uses "discrete disk" algorithm (all cells within euclidean distance)
        - Performance: O(π * (radius/resolution)²)

    Example:
        >>> config = GridConfig(resolution=0.1, width=10.0, height=10.0)
        >>> cells = get_affected_cells(5.0, 5.0, 0.3, config)
        >>> len(cells)  # Approx π * 3² = ~28 cells
        28
    """
    if radius <= 0:
        return []

    cell_radius = int(np.ceil(radius / config.resolution))
    center_row, center_col = world_to_grid_indices(
        world_x, world_y, config, grid_origin_x, grid_origin_y
    )

    affected = []
    for dr in range(-cell_radius, cell_radius + 1):
        for dc in range(-cell_radius, cell_radius + 1):
            row = center_row + dr
            col = center_col + dc

            # Check bounds
            if not (0 <= row < config.grid_height and 0 <= col < config.grid_width):
                continue

            # Check euclidean distance
            cell_world_x, cell_world_y = grid_indices_to_world(
                row, col, config, grid_origin_x, grid_origin_y
            )
            dist = np.sqrt((cell_world_x - world_x) ** 2 + (cell_world_y - world_y) ** 2)

            if dist <= radius:
                affected.append((row, col))

    return affected

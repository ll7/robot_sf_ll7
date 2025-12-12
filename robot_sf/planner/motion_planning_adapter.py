"""Adapter module to convert robot_sf MapDefinition to python-motion-planning map format.

This module provides conversion utilities to bridge between robot_sf's map representation
(MapDefinition with Obstacle objects) and python-motion-planning library's expected format.

The adapter handles:
- Converting obstacle vertices to the format expected by motion planning algorithms
- Building motion planning map objects with proper obstacle representation
- Maintaining coordinate system consistency

Typical usage:
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.planner.motion_planning_adapter import adapt_map_for_motion_planning

    map_def: MapDefinition = ...
    mp_map = adapt_map_for_motion_planning(map_def)
    planner = MotionPlanner(mp_map)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition


def adapt_map_for_motion_planning(map_def: MapDefinition) -> dict:
    """Convert a robot_sf MapDefinition to a python-motion-planning map format.

    Extracts obstacle geometry from the MapDefinition and converts it to the
    format expected by python-motion-planning library.

    Args:
        map_def: Robot SF map definition containing obstacles, dimensions, etc.

    Returns:
        Dictionary representing the map for motion planning with keys:
        - "width": Map width in meters
        - "height": Map height in meters
        - "obstacles": List of obstacle polygons as lists of vertices
        - "bounds": Map boundary as [x_min, y_min, x_max, y_max]

    Raises:
        ValueError: If map_def is invalid or contains no obstacles.
    """
    if not map_def:
        raise ValueError("map_def cannot be None")

    if map_def.width <= 0 or map_def.height <= 0:
        raise ValueError(f"Invalid map dimensions: width={map_def.width}, height={map_def.height}")

    # Convert obstacles to polygon format
    obstacles = []
    for obstacle in map_def.obstacles:
        # Each obstacle is a polygon represented by its vertices
        polygon = list(obstacle.vertices)
        if len(polygon) < 3:
            logger.warning(
                "Skipping degenerate obstacle with {count} vertices",
                count=len(polygon),
            )
            continue
        obstacles.append(polygon)

    logger.debug(
        "Converted map with dimensions {w}x{h} and {obs} obstacles",
        w=map_def.width,
        h=map_def.height,
        obs=len(obstacles),
    )

    return {
        "width": map_def.width,
        "height": map_def.height,
        "obstacles": obstacles,
        "bounds": [0, 0, map_def.width, map_def.height],
    }


def create_motion_planning_environment(map_def: MapDefinition) -> object:
    """Create a python-motion-planning environment from a robot_sf map definition.

    This is a placeholder for future integration. Currently returns the adapted map dict.

    Args:
        map_def: Robot SF map definition.

    Returns:
        Motion planning environment object (currently returns dict, will be upgraded).

    Raises:
        ImportError: If python-motion-planning is not installed.
    """
    try:
        import python_motion_planning  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "python-motion-planning not installed. Install with: uv add python-motion-planning"
        ) from e

    mp_map = adapt_map_for_motion_planning(map_def)
    logger.debug("Created motion planning environment from map definition")

    # TODO: Integrate with actual python-motion-planning environment creation
    # once the library API is explored
    return mp_map

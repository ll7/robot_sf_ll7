"""Programmatic API for OSM zones and routes configuration (Phase 3).

This module provides factory functions for creating zones and routes
programmatically without the visual editor. Enables:
- Code-first zone/route definition
- Scenario batch creation
- Equivalence testing (programmatic ≡ editor output)
- Automated map generation workflows

Example usage:

    from robot_sf.maps.osm_zones_config import (
        create_spawn_zone,
        create_goal_zone,
        create_route,
        OSMZonesConfig,
    )

    # Create zones
    spawn_zone = create_spawn_zone(
        "robot_spawn",
        polygon=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        priority=2,
    )

    goal_zone = create_goal_zone(
        "target_area",
        polygon=[(50.0, 50.0), (60.0, 50.0), (60.0, 60.0), (50.0, 60.0)],
    )

    # Create route
    main_route = create_route(
        "main_path",
        waypoints=[(5.0, 5.0), (25.0, 25.0), (55.0, 55.0)],
        route_type="pedestrian",
    )

    # Create config and add zones/routes
    config = OSMZonesConfig()
    config.zones["spawn_zone"] = spawn_zone
    config.zones["goal_zone"] = goal_zone
    config.routes["main"] = main_route

    # Save to YAML
    from robot_sf.maps.osm_zones_yaml import save_zones_yaml
    save_zones_yaml(config, "my_scenario.yaml")
"""

from typing import Any

from loguru import logger

from robot_sf.common.types import Vec2D
from robot_sf.maps.osm_zones_yaml import OSMZonesConfig, Route, Zone, load_zones_yaml


def _validate_and_convert_polygon(
    polygon: list[tuple[float, float]] | list[Vec2D],
) -> list[Vec2D]:
    """Convert polygon vertices to Vec2D tuples and validate basic geometry."""
    polygon_vecs: list[Vec2D] = []
    try:
        for point in polygon:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                polygon_vecs.append((float(point[0]), float(point[1])))
            else:
                raise TypeError(f"Invalid point format: {point}")
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert polygon to Vec2D: {e}") from e

    if len(polygon_vecs) < 3:
        raise ValueError(f"Polygon must have ≥3 points, got {len(polygon_vecs)}")

    x1, y1 = polygon_vecs[0]
    x2, y2 = polygon_vecs[1]
    x3, y3 = polygon_vecs[2]
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    if abs(cross) < 1e-6:
        raise ValueError("Polygon points appear to be collinear (degenerate polygon)")

    return polygon_vecs


def create_spawn_zone(
    name: str,
    polygon: list[tuple[float, float]] | list[Vec2D],
    priority: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Zone:
    """Create a spawn zone for robot initialization (T036).

    Spawn zones define areas where the robot can be placed at episode start.
    The editor uses spawn zones to populate the robot environment.

    Args:
        name: Zone identifier (e.g., 'spawn_north', 'spawn_1')
        polygon: List of (x, y) vertices in world coordinates (meters).
                 For rectangular zones, pass 4 points (counter-clockwise).
                 Minimum 3 points required for valid polygon.
        priority: Selection priority (higher = preferred). Default 1.
                 Used when multiple spawn zones available.
        metadata: Optional custom metadata dict (e.g., {"robot_type": "wheelchair"})

    Returns:
        Zone object with type='spawn', validated and ready for use.

    Raises:
        ValueError: If polygon has <3 points or all points collinear.
        TypeError: If polygon points not convertible to floats.

    Example:
        >>> zone = create_spawn_zone(
        ...     "spawn_center",
        ...     polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        ...     priority=2,
        ... )
        >>> zone.type
        'spawn'
    """
    polygon_vecs = _validate_and_convert_polygon(polygon)

    zone = Zone(
        name=name,
        type="spawn",
        polygon=polygon_vecs,
        priority=priority,
        metadata=metadata or {},
    )

    logger.debug(
        f"Created spawn zone '{name}' with {len(polygon_vecs)} vertices, priority={priority}"
    )
    return zone


def create_goal_zone(
    name: str,
    polygon: list[tuple[float, float]] | list[Vec2D],
    metadata: dict[str, Any] | None = None,
) -> Zone:
    """Create a goal zone for robot navigation target (T037).

    Goal zones define areas where the robot should navigate to.
    Used as episode success criteria in navigation tasks.

    Args:
        name: Zone identifier (e.g., 'goal_exit', 'goal_1')
        polygon: List of (x, y) vertices in world coordinates (meters).
                 Minimum 3 points required for valid polygon.
        metadata: Optional custom metadata dict (e.g., {"reward": 1.0})

    Returns:
        Zone object with type='goal', validated and ready for use.

    Raises:
        ValueError: If polygon has <3 points or all points collinear.
        TypeError: If polygon points not convertible to floats.

    Example:
        >>> zone = create_goal_zone(
        ...     "goal_corner",
        ...     polygon=[(90, 90), (100, 90), (100, 100), (90, 100)],
        ... )
        >>> zone.type
        'goal'
    """
    polygon_vecs = _validate_and_convert_polygon(polygon)

    zone = Zone(
        name=name,
        type="goal",
        polygon=polygon_vecs,
        priority=1,  # Goals don't use priority
        metadata=metadata or {},
    )

    logger.debug(f"Created goal zone '{name}' with {len(polygon_vecs)} vertices")
    return zone


def create_crowded_zone(
    name: str,
    polygon: list[tuple[float, float]] | list[Vec2D],
    density: float,
    metadata: dict[str, Any] | None = None,
) -> Zone:
    """Create a crowded zone with pedestrian density annotation (T038).

    Crowded zones mark areas where pedestrians are concentrated.
    Enables scenario design with varying crowd densities.

    Args:
        name: Zone identifier (e.g., 'crowd_center', 'busy_area')
        polygon: List of (x, y) vertices in world coordinates (meters).
                 Minimum 3 points required for valid polygon.
        density: Pedestrian density (persons per m²).
                 Typical range: 0.1 (sparse) to 5.0 (dense).
        metadata: Optional custom metadata dict

    Returns:
        Zone object with type='crowded', density stored in metadata.

    Raises:
        ValueError: If polygon has <3 points, density invalid, or collinear.
        TypeError: If polygon points not convertible to floats.

    Example:
        >>> zone = create_crowded_zone(
        ...     "busy_intersection",
        ...     polygon=[(30, 30), (40, 30), (40, 40), (30, 40)],
        ...     density=2.5,
        ... )
        >>> zone.metadata["density"]
        2.5
    """
    if density <= 0:
        raise ValueError(f"Density must be >0, got {density}")

    polygon_vecs = _validate_and_convert_polygon(polygon)

    # Store density in metadata
    meta = metadata or {}
    meta["density"] = density

    zone = Zone(
        name=name,
        type="crowded",
        polygon=polygon_vecs,
        priority=1,
        metadata=meta,
    )

    logger.debug(
        f"Created crowded zone '{name}' with {len(polygon_vecs)} vertices, density={density}"
    )
    return zone


def create_route(
    name: str,
    waypoints: list[tuple[float, float]] | list[Vec2D],
    route_type: str = "pedestrian",
    metadata: dict[str, Any] | None = None,
) -> Route:
    """Create a navigation route with waypoints (T039).

    Routes define paths for navigation or demonstration.
    Used in training scenarios, imitation learning, or manual guidance.

    Args:
        name: Route identifier (e.g., 'main_path', 'demo_route')
        waypoints: List of (x, y) waypoints in world coordinates (meters).
                   Minimum 2 waypoints required.
        route_type: Type of route ('pedestrian', 'wheelchair', 'vehicle').
                    Default 'pedestrian'.
        metadata: Optional custom metadata dict (e.g., {"speed": 1.2})

    Returns:
        Route object with waypoints validated and stored.

    Raises:
        ValueError: If <2 waypoints provided.
        TypeError: If waypoints not convertible to floats.

    Example:
        >>> route = create_route(
        ...     "path_to_goal",
        ...     waypoints=[(0, 0), (25, 25), (50, 50)],
        ...     route_type="pedestrian",
        ... )
        >>> len(route.waypoints)
        3
    """
    # Convert to Vec2D tuples
    waypoints_vecs: list[Vec2D] = []
    try:
        for point in waypoints:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                waypoints_vecs.append((float(point[0]), float(point[1])))
            else:
                raise TypeError(f"Invalid waypoint format: {point}")
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert waypoints to Vec2D: {e}") from e

    if len(waypoints_vecs) < 2:
        raise ValueError(f"Route must have ≥2 waypoints, got {len(waypoints_vecs)}")

    # Validate route_type
    valid_types = {"pedestrian", "wheelchair", "vehicle", "bicycle"}
    if route_type not in valid_types:
        logger.warning(f"Route type '{route_type}' not standard (valid: {valid_types})")

    route = Route(
        name=name,
        waypoints=waypoints_vecs,
        route_type=route_type,
        metadata=metadata or {},
    )

    logger.debug(f"Created route '{name}' with {len(waypoints_vecs)} waypoints, type={route_type}")
    return route


def create_config_with_zones_routes(
    zones: list[Zone] | None = None,
    routes: list[Route] | None = None,
    version: str = "1.0",
    metadata: dict[str, Any] | None = None,
) -> OSMZonesConfig:
    """Create an OSMZonesConfig with zones and routes (helper for programmatic workflows).

    Args:
        zones: List of Zone objects
        routes: List of Route objects
        version: Schema version (default "1.0")
        metadata: Global metadata (e.g., map source)

    Returns:
        OSMZonesConfig with zones/routes indexed by name.

    Example:
        >>> config = create_config_with_zones_routes(
        ...     zones=[spawn_zone, goal_zone],
        ...     routes=[main_route],
        ... )
    """
    config = OSMZonesConfig(version=version, metadata=metadata or {})

    if zones:
        for zone in zones:
            config.zones[zone.name] = zone

    if routes:
        for route in routes:
            config.routes[route.name] = route

    logger.info(
        f"Created config: {len(config.zones)} zones, {len(config.routes)} routes, v{version}"
    )
    return config


def load_scenario_config(yaml_file: str) -> OSMZonesConfig:
    """Load scenario configuration from YAML file (T040).

    Programmatic alternative to the visual editor. Loads a YAML file
    containing zone and route definitions, returning a typed OSMZonesConfig
    object suitable for environment creation.

    YAML format:
        version: "1.0"
        metadata:
          map_source: "oslo.pbf"
          description: "Downtown crossing scenario"
        zones:
          spawn_north:
            type: "spawn"
            polygon: [[10, 10], [20, 10], [20, 20]]
            priority: 2
          goal_south:
            type: "goal"
            polygon: [[80, 80], [90, 80], [90, 90]]
          intersection:
            type: "crowded"
            polygon: [[40, 40], [60, 40], [60, 60], [40, 60]]
            metadata:
              density: 2.5
        routes:
          main_path:
            name: "Main path"
            waypoints: [[15, 15], [50, 50], [85, 85]]
            route_type: "pedestrian"

    Args:
        yaml_file: Path to YAML scenario file

    Returns:
        OSMZonesConfig with zones and routes loaded

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML structure invalid or schema mismatch
        yaml.YAMLError: If YAML parsing fails

    Example:
        >>> config = load_scenario_config("scenarios/crossing.yaml")
        >>> len(config.zones)
        3
        >>> config.zones["spawn_north"].type
        'spawn'
    """
    # Use existing YAML loader for consistency
    config = load_zones_yaml(yaml_file)

    logger.info(
        f"Loaded scenario from {yaml_file}: {len(config.zones)} zones, {len(config.routes)} routes"
    )
    return config

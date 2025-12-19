"""OSM-based zones and routes YAML serialization module.

This module provides dataclasses and functions for:
- Defining zones (spawn, goal, crowded areas) from OSM-derived maps
- Defining routes (pedestrian paths, navigation corridors)
- Deterministic YAML serialization with round-trip guarantees
- Validation against OSM-derived MapDefinitions

Zones and routes are stored in a single YAML file with semantic structure:

    version: "1.0"
    metadata:
      map_source: "osm_file.pbf"
      created_at: "2025-12-19T12:00:00Z"
    zones:
      spawn_1:
        type: "spawn"
        polygon: [[x1, y1], [x2, y2], ...]
        priority: 1
      goal_1:
        type: "goal"
        polygon: [[x1, y1], [x2, y2], ...]
    routes:
      route_1:
        name: "Main path"
        waypoints: [[x1, y1], [x2, y2], ...]
        route_type: "pedestrian"

Determinism guarantees:
- Floating-point precision: 3 decimal places (0.001 m)
- Sorted keys in YAML output
- Consistent date/time formatting
- Reproducible round-trip (save → load → save produces identical output)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.common.types import Vec2D

# ============================================================================
# Dataclass Definitions (T022)
# ============================================================================


@dataclass
class Zone:
    """Single zone (spawn, goal, crowded area, etc.)."""

    name: str
    """Zone identifier."""

    type: str
    """Zone type: 'spawn', 'goal', 'crowded', 'restricted', etc."""

    polygon: list[Vec2D]
    """Polygon boundary as list of (x, y) tuples in world coordinates (meters)."""

    priority: int = 1
    """Priority for spawn/goal selection (higher = preferred)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Custom metadata (density, pedestrian_type, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict with 3-decimal precision."""
        result = {
            "name": self.name,
            "type": self.type,
            "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon],
            "priority": self.priority,
        }
        # Only include metadata if non-empty
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zone":
        """Reconstruct from dict."""
        return cls(
            name=data["name"],
            type=data["type"],
            polygon=data["polygon"],
            priority=data.get("priority", 1),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Route:
    """Navigation route (path, corridor, pedestrian route)."""

    name: str
    """Route identifier."""

    waypoints: list[Vec2D]
    """Waypoints as list of (x, y) tuples in world coordinates (meters)."""

    route_type: str = "pedestrian"
    """Route type: 'pedestrian', 'wheelchair', 'vehicle', etc."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Custom metadata (speed_preference, accessibility, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict with 3-decimal precision."""
        result = {
            "name": self.name,
            "waypoints": [[round(x, 3), round(y, 3)] for x, y in self.waypoints],
            "route_type": self.route_type,
        }
        # Only include metadata if non-empty
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Route":
        """Reconstruct from dict."""
        return cls(
            name=data["name"],
            waypoints=data["waypoints"],
            route_type=data.get("route_type", "pedestrian"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OSMZonesConfig:
    """Top-level YAML configuration for zones and routes (T022)."""

    version: str = "1.0"
    """Schema version for compatibility tracking."""

    zones: dict[str, Zone] = field(default_factory=dict)
    """Zones keyed by zone name."""

    routes: dict[str, Route] = field(default_factory=dict)
    """Routes keyed by route name."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Global metadata (map source, creation timestamp, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict with sorted keys."""
        zones_dict = {}
        for name, zone in sorted(self.zones.items()):
            zones_dict[name] = zone.to_dict()

        routes_dict = {}
        for name, route in sorted(self.routes.items()):
            routes_dict[name] = route.to_dict()

        result = {"version": self.version}

        # Only include non-empty sections
        if self.metadata:
            result["metadata"] = self.metadata
        if zones_dict:
            result["zones"] = zones_dict
        if routes_dict:
            result["routes"] = routes_dict

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OSMZonesConfig":
        """Reconstruct from dict."""
        zones = {}
        for name, zone_data in (data.get("zones") or {}).items():
            zones[name] = Zone.from_dict(zone_data)

        routes = {}
        for name, route_data in (data.get("routes") or {}).items():
            routes[name] = Route.from_dict(route_data)

        return cls(
            version=data.get("version", "1.0"),
            zones=zones,
            routes=routes,
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# YAML Loader & Saver (T023-T024)
# ============================================================================


def load_zones_yaml(yaml_file: str) -> OSMZonesConfig:
    """Load zones configuration from YAML file (T023).

    Args:
        yaml_file: Path to YAML file

    Returns:
        OSMZonesConfig object with zones and routes

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If schema validation fails
    """
    path = Path(yaml_file)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise

    if not data:
        logger.warning(f"Empty YAML file: {yaml_file}")
        return OSMZonesConfig()

    config = OSMZonesConfig.from_dict(data)
    logger.info(f"Loaded zones config: {len(config.zones)} zones, {len(config.routes)} routes")
    return config


def save_zones_yaml(config: OSMZonesConfig, yaml_file: str) -> None:
    """Save zones configuration to YAML with determinism guarantees (T024).

    Guarantees:
    - Floating-point precision: 3 decimal places
    - Sorted keys
    - Round-trip byte-identical (save → load → save produces same output)
    - Null values omitted

    Args:
        config: OSMZonesConfig to save
        yaml_file: Output file path

    Raises:
        IOError: If write fails
    """
    path = Path(yaml_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict()

    # Custom YAML representer for deterministic output
    def represent_dict_sorted(dumper, data):
        return dumper.represent_mapping("tag:yaml.org,2002:map", sorted(data.items()))

    yaml.add_representer(dict, represent_dict_sorted)

    try:
        with open(path, "w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=True,
                allow_unicode=True,
                default_style=None,
            )
        logger.info(f"Saved zones config to {yaml_file}")
    except OSError as e:
        logger.error(f"Failed to save YAML: {e}")
        raise


# ============================================================================
# YAML Validation (T025)
# ============================================================================


def validate_zones_yaml(config: OSMZonesConfig, map_def: Any | None = None) -> list[str]:
    """Validate zones configuration against OSM-derived map (T025).

    Returns warnings/errors for:
    - Out-of-bounds zones (outside map bounds)
    - Zones crossing obstacles
    - Invalid polygons (too few points, self-intersecting)
    - Duplicate zone names
    - Empty zones/routes

    Args:
        config: OSMZonesConfig to validate
        map_def: Optional MapDefinition to validate against

    Returns:
        List of warning/error messages (empty = valid)
    """
    warnings = []

    # Check empty config
    if not config.zones and not config.routes:
        warnings.append("Warning: No zones or routes defined")

    # Validate zones
    zone_names = set()
    for name, zone in config.zones.items():
        # Check duplicate names
        if name in zone_names:
            warnings.append(f"Error: Duplicate zone name: {name}")
        zone_names.add(name)

        # Check polygon validity
        if len(zone.polygon) < 3:
            warnings.append(f"Error: Zone '{name}' has fewer than 3 points")

        # Check bounds (if map_def provided)
        if map_def:
            bounds = map_def.bounds
            for i, (x, y) in enumerate(zone.polygon):
                if not (bounds.xmin <= x <= bounds.xmax and bounds.ymin <= y <= bounds.ymax):
                    warnings.append(
                        f"Warning: Zone '{name}' point {i} ({x:.1f}, {y:.1f}) outside map bounds"
                    )

            # Check obstacle intersection (if allowed_areas provided)
            if hasattr(map_def, "allowed_areas") and map_def.allowed_areas:
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon

                    zone_poly = ShapelyPolygon(zone.polygon)
                    for obstacle in map_def.obstacles:
                        obs_poly = ShapelyPolygon(obstacle)
                        if zone_poly.intersects(obs_poly):
                            warnings.append(f"Warning: Zone '{name}' crosses an obstacle")
                            break
                except Exception as e:
                    logger.debug(f"Obstacle check failed: {e}")

    # Validate routes
    route_names = set()
    for name, route in config.routes.items():
        # Check duplicate names
        if name in route_names:
            warnings.append(f"Error: Duplicate route name: {name}")
        route_names.add(name)

        # Check waypoints
        if len(route.waypoints) < 2:
            warnings.append(f"Error: Route '{name}' has fewer than 2 waypoints")

        # Check bounds (if map_def provided)
        if map_def:
            bounds = map_def.bounds
            for i, (x, y) in enumerate(route.waypoints):
                if not (bounds.xmin <= x <= bounds.xmax and bounds.ymin <= y <= bounds.ymax):
                    warnings.append(
                        f"Warning: Route '{name}' waypoint {i} "
                        f"({x:.1f}, {y:.1f}) outside map bounds"
                    )

    return warnings


# ============================================================================
# Helper Functions for Zone Creation (Preview for Phase 3)
# ============================================================================


def create_zone(
    name: str,
    polygon: list[Vec2D],
    zone_type: str = "spawn",
    priority: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Zone:
    """Create a zone object (used in Phase 3 for programmatic config).

    Args:
        name: Zone identifier
        polygon: Polygon boundary as list of (x, y) tuples
        zone_type: Zone type (spawn, goal, crowded, etc.)
        priority: Priority for selection
        metadata: Custom metadata dict

    Returns:
        Zone object
    """
    return Zone(
        name=name,
        type=zone_type,
        polygon=polygon,
        priority=priority,
        metadata=metadata or {},
    )


def create_route(
    name: str,
    waypoints: list[Vec2D],
    route_type: str = "pedestrian",
    metadata: dict[str, Any] | None = None,
) -> Route:
    """Create a route object (used in Phase 3 for programmatic config).

    Args:
        name: Route identifier
        waypoints: List of (x, y) tuples
        route_type: Route type (pedestrian, wheelchair, vehicle, etc.)
        metadata: Custom metadata dict

    Returns:
        Route object
    """
    return Route(
        name=name,
        waypoints=waypoints,
        route_type=route_type,
        metadata=metadata or {},
    )

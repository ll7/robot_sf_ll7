"""Type-aware reflections for post-loader release maps.

The release-scenario mirror diagnostic must transform a loaded
``MapDefinition`` after all loader overrides have been applied.  A raw YAML
reflection is unsafe because loading can add routes, normalize coordinates, and
attach scenario-owned metadata.  This module therefore reflects every
coordinate-bearing field of the runtime map model explicitly and constructs a
fresh map so the source map remains unchanged.

``axis="y"`` reflects across the horizontal centre line (``y -> height - y``),
matching the metamorphic ``mirror_y`` convention.  ``axis="x"`` reflects
across the vertical centre line (``x -> width - x``), matching
``mirror_x``.  Heading formulas follow the existing synthetic test helpers:
``-heading`` for the y reflection and ``pi - heading`` for the x reflection.
Angles are intentionally not wrapped; callers compare them modulo ``2*pi`` in
the same way as the existing trajectory tests.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite, pi
from typing import Literal

from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import (
    InfrastructureZone,
    MapDefinition,
    SinglePedestrianDefinition,
    SocialGroupDefinition,
)
from robot_sf.nav.nav_types import SemanticBoundary
from robot_sf.nav.obstacle import Obstacle

MirrorAxis = Literal["x", "y"]
_MAP_BOUNDS_TOLERANCE = 1e-8


def reflect_point(
    point: tuple[float, float] | list[float],
    *,
    width: float,
    height: float,
    axis: MirrorAxis,
) -> tuple[float, float]:
    """Reflect one finite point using the release-map mirror convention.

    ``axis="y"`` flips the y component and ``axis="x"`` flips the x
    component.  The dimensions are explicit so this helper cannot silently
    use a synthetic-map size when used by a release diagnostic.

    Returns:
        tuple[float, float]: The reflected finite point.
    """

    resolved_axis = _validate_axis(axis)
    resolved_width, resolved_height = _validate_dimensions(width, height)
    x, y = _point(point, "point")
    _validate_in_bounds((x, y), resolved_width, resolved_height, "point")
    if resolved_axis == "x":
        x = resolved_width - x
    else:
        y = resolved_height - y
    return x, y


def reflect_heading(heading: float, axis: MirrorAxis) -> float:
    """Reflect a finite heading according to the synthetic-map convention.

    Returns:
        float: The reflected heading, equivalent modulo ``2*pi``.
    """

    resolved_axis = _validate_axis(axis)
    value = _finite_number(heading, "heading")
    return pi - value if resolved_axis == "x" else -value


def reflect_vector(
    vector: tuple[float, float] | list[float], axis: MirrorAxis
) -> tuple[float, float]:
    """Reflect a world-frame vector and preserve its finite numeric payload.

    Returns:
        tuple[float, float]: The reflected vector.
    """

    resolved_axis = _validate_axis(axis)
    x, y = _point(vector, "vector")
    return (-x, y) if resolved_axis == "x" else (x, -y)


def reflect_map_definition(map_definition: MapDefinition, axis: MirrorAxis) -> MapDefinition:
    """Return a reflected copy of a post-loader :class:`MapDefinition`.

    All geometry is transformed explicitly: bounds, obstacles, zones, routes,
    POIs, allowed areas, semantic/infrastructure/social geometry, and
    single-pedestrian world coordinates.  Robot-relative role offsets preserve
    their forward component and flip their lateral component, which is the
    coordinate-frame equivalent of reflecting the resulting world target.

    Raises:
        TypeError: If ``map_definition`` is not a ``MapDefinition``.
        ValueError: If the axis, dimensions, known geometry, or coordinate
            container has an unsupported or non-finite shape.
    """

    if not isinstance(map_definition, MapDefinition):
        raise TypeError("map_definition must be a MapDefinition")
    resolved_axis = _validate_axis(axis)
    width, height = _validate_dimensions(map_definition.width, map_definition.height)
    _reject_unknown_public_fields(map_definition)

    bounds = _transform_bounds(map_definition.bounds, width, height, resolved_axis)
    obstacles = [
        _transform_obstacle(obstacle, width, height, resolved_axis, index=index)
        for index, obstacle in enumerate(_sequence(map_definition.obstacles, "obstacles"))
    ]

    robot_spawn_zones = _transform_rects(
        map_definition.robot_spawn_zones, width, height, resolved_axis, "robot_spawn_zones"
    )
    ped_spawn_zones = _transform_rects(
        map_definition.ped_spawn_zones, width, height, resolved_axis, "ped_spawn_zones"
    )
    robot_goal_zones = _transform_rects(
        map_definition.robot_goal_zones, width, height, resolved_axis, "robot_goal_zones"
    )
    ped_goal_zones = _transform_rects(
        map_definition.ped_goal_zones, width, height, resolved_axis, "ped_goal_zones"
    )
    ped_crowded_zones = _transform_rects(
        map_definition.ped_crowded_zones, width, height, resolved_axis, "ped_crowded_zones"
    )

    robot_routes = _transform_routes(
        map_definition.robot_routes, width, height, resolved_axis, "robot_routes"
    )
    ped_routes = _transform_routes(
        map_definition.ped_routes, width, height, resolved_axis, "ped_routes"
    )
    single_pedestrians = _transform_single_pedestrians(
        map_definition.single_pedestrians, width, height, resolved_axis
    )

    poi_positions = [
        reflect_point(point, width=width, height=height, axis=resolved_axis)
        for point in _sequence(map_definition.poi_positions, "poi_positions")
    ]
    poi_labels = _copy_mapping(map_definition.poi_labels, "poi_labels")
    allowed_areas = _transform_allowed_areas(
        map_definition.allowed_areas, width, height, resolved_axis
    )
    semantic_boundaries = _transform_semantic_boundaries(
        map_definition.semantic_boundaries, width, height, resolved_axis
    )
    infrastructure_zones = _transform_infrastructure_zones(
        map_definition.infrastructure_zones, width, height, resolved_axis
    )
    social_groups = _transform_social_groups(
        map_definition.social_groups, width, height, resolved_axis
    )

    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles,
        robot_spawn_zones=robot_spawn_zones,
        ped_spawn_zones=ped_spawn_zones,
        robot_goal_zones=robot_goal_zones,
        bounds=bounds,
        robot_routes=robot_routes,
        ped_goal_zones=ped_goal_zones,
        ped_crowded_zones=ped_crowded_zones,
        ped_routes=ped_routes,
        single_pedestrians=single_pedestrians,
        poi_positions=poi_positions,
        poi_labels=poi_labels,
        allowed_areas=allowed_areas,
        semantic_boundaries=semantic_boundaries,
        infrastructure_zones=infrastructure_zones,
        social_groups=social_groups,
        svg_geometry_contract=map_definition.svg_geometry_contract,
        goal_completion_policy=map_definition.goal_completion_policy,
    )


def mirror_map_definition(map_definition: MapDefinition, axis: MirrorAxis) -> MapDefinition:
    """Compatibility spelling for :func:`reflect_map_definition`.

    Returns:
        MapDefinition: A reflected copy of ``map_definition``.
    """

    return reflect_map_definition(map_definition, axis)


def _validate_axis(axis: MirrorAxis) -> MirrorAxis:
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    return axis


def _validate_dimensions(width: object, height: object) -> tuple[float, float]:
    resolved_width = _finite_number(width, "width")
    resolved_height = _finite_number(height, "height")
    if resolved_width <= 0.0 or resolved_height <= 0.0:
        raise ValueError(
            f"map width and height must be positive, got ({resolved_width}, {resolved_height})"
        )
    return resolved_width, resolved_height


def _reject_unknown_public_fields(map_definition: MapDefinition) -> None:
    """Reject public runtime extensions that this explicit transform cannot mirror."""

    known_fields = set(MapDefinition.__dataclass_fields__)
    unknown_fields = sorted(
        name
        for name in map_definition.__dict__
        if not name.startswith("_") and name not in known_fields
    )
    if unknown_fields:
        raise ValueError(
            "MapDefinition has unsupported public fields with unknown reflection semantics: "
            + ", ".join(unknown_fields)
        )


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{path} must be a finite numeric scalar, got {value!r}")
    try:
        resolved = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path} must be a finite numeric scalar, got {value!r}") from exc
    if not isfinite(resolved):
        raise ValueError(f"{path} must be finite, got {value!r}")
    return resolved


def _point(value: object, path: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{path} must be a 2-item coordinate, got {value!r}")
    return (
        _finite_number(value[0], f"{path}[0]"),
        _finite_number(value[1], f"{path}[1]"),
    )


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list or tuple, got {value!r}")
    return list(value)


def _copy_mapping(value: object, path: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping, got {value!r}")
    try:
        copied = deepcopy(value)
    except Exception as exc:  # pragma: no cover - defensive for arbitrary caller metadata
        raise ValueError(f"{path} could not be copied safely") from exc
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in copied.items()):
        raise ValueError(f"{path} must contain string keys and values")
    return copied


def _transform_rect(
    rect: object, width: float, height: float, axis: MirrorAxis, path: str
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = _sequence(rect, path)
    if len(points) != 3:
        raise ValueError(f"{path} must contain exactly three rectangle corners")
    return tuple(reflect_point(point, width=width, height=height, axis=axis) for point in points)  # type: ignore[return-value]


def _transform_rects(
    rects: object, width: float, height: float, axis: MirrorAxis, path: str
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    return [
        _transform_rect(rect, width, height, axis, f"{path}[{index}]")
        for index, rect in enumerate(_sequence(rects, path))
    ]


def _transform_bounds(
    bounds: object, width: float, height: float, axis: MirrorAxis
) -> list[tuple[float, float, float, float]]:
    transformed: list[tuple[float, float, float, float]] = []
    for index, bound in enumerate(_sequence(bounds, "bounds")):
        path = f"bounds[{index}]"
        if (
            isinstance(bound, (tuple, list))
            and len(bound) == 4
            and all(not isinstance(item, (tuple, list, dict)) for item in bound)
        ):
            x1, x2, y1, y2 = (
                _finite_number(bound[0], f"{path}[0]"),
                _finite_number(bound[1], f"{path}[1]"),
                _finite_number(bound[2], f"{path}[2]"),
                _finite_number(bound[3], f"{path}[3]"),
            )
            first = reflect_point((x1, y1), width=width, height=height, axis=axis)
            second = reflect_point((x2, y2), width=width, height=height, axis=axis)
            transformed.append((first[0], second[0], first[1], second[1]))
            continue
        if isinstance(bound, (tuple, list)) and len(bound) == 2:
            first = reflect_point(bound[0], width=width, height=height, axis=axis)
            second = reflect_point(bound[1], width=width, height=height, axis=axis)
            transformed.append((first[0], second[0], first[1], second[1]))
            continue
        raise ValueError(f"{path} has unsupported bound shape: {bound!r}")
    if len(transformed) != 4:
        raise ValueError(f"bounds must contain exactly four segments, got {len(transformed)}")
    return transformed


def _validate_geometry(
    geometry: object, path: str, width: float, height: float
) -> Polygon | MultiPolygon:
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"{path} must be a Polygon or MultiPolygon, got {type(geometry).__name__}")
    if geometry.is_empty:
        return geometry
    polygons = [geometry] if isinstance(geometry, Polygon) else list(geometry.geoms)
    for polygon_index, polygon in enumerate(polygons):
        rings = [polygon.exterior, *polygon.interiors]
        for ring_index, ring in enumerate(rings):
            for point_index, coordinate in enumerate(ring.coords):
                if len(coordinate) != 2:
                    raise ValueError(
                        f"{path}[{polygon_index}][{ring_index}][{point_index}] must be 2D"
                    )
                point = _point(
                    tuple(coordinate),
                    f"{path}[{polygon_index}][{ring_index}][{point_index}]",
                )
                _validate_in_bounds(
                    point,
                    width,
                    height,
                    f"{path}[{polygon_index}][{ring_index}][{point_index}]",
                )
    return geometry


def _transform_geometry(
    geometry: object, width: float, height: float, axis: MirrorAxis, path: str
) -> Polygon | MultiPolygon:
    validated = _validate_geometry(geometry, path, width, height)
    if validated.is_empty:
        return deepcopy(validated)
    origin = (width / 2.0, height / 2.0)
    x_factor = -1.0 if axis == "x" else 1.0
    y_factor = -1.0 if axis == "y" else 1.0
    transformed = scale(validated, xfact=x_factor, yfact=y_factor, origin=origin)
    return _validate_geometry(transformed, f"{path}.reflected", width, height)


def _validate_in_bounds(point: tuple[float, float], width: float, height: float, path: str) -> None:
    """Reject world points that do not fit the map's origin-based bounds."""

    x, y = point
    tolerance = _MAP_BOUNDS_TOLERANCE
    if x < -tolerance or x > width + tolerance or y < -tolerance or y > height + tolerance:
        raise ValueError(f"{path} lies outside map bounds [0, {width}] x [0, {height}]: {point!r}")


def _transform_obstacle(
    obstacle: object,
    width: float,
    height: float,
    axis: MirrorAxis,
    *,
    index: int,
) -> Obstacle:
    path = f"obstacles[{index}]"
    if not isinstance(obstacle, Obstacle):
        raise ValueError(f"{path} must be an Obstacle, got {type(obstacle).__name__}")
    vertices = [
        reflect_point(vertex, width=width, height=height, axis=axis)
        for vertex in _sequence(obstacle.vertices, f"{path}.vertices")
    ]
    geometry = obstacle.geometry
    if geometry is None:
        return Obstacle(vertices=vertices)
    reflected_geometry = _transform_geometry(geometry, width, height, axis, f"{path}.geometry")
    if reflected_geometry.is_empty:
        raise ValueError(f"{path}.geometry must not be empty")
    return Obstacle.from_geometry(reflected_geometry, representative_vertices=vertices)


def _transform_route(
    route: object, width: float, height: float, axis: MirrorAxis, path: str
) -> GlobalRoute:
    if not isinstance(route, GlobalRoute):
        raise ValueError(f"{path} must be a GlobalRoute, got {type(route).__name__}")
    waypoints = [
        reflect_point(point, width=width, height=height, axis=axis)
        for point in _sequence(route.waypoints, f"{path}.waypoints")
    ]
    return GlobalRoute(
        spawn_id=route.spawn_id,
        goal_id=route.goal_id,
        waypoints=waypoints,
        spawn_zone=_transform_rect(route.spawn_zone, width, height, axis, f"{path}.spawn_zone"),
        goal_zone=_transform_rect(route.goal_zone, width, height, axis, f"{path}.goal_zone"),
        source_path_id=route.source_path_id,
        source_label=route.source_label,
    )


def _transform_routes(
    routes: object, width: float, height: float, axis: MirrorAxis, path: str
) -> list[GlobalRoute]:
    return [
        _transform_route(route, width, height, axis, f"{path}[{index}]")
        for index, route in enumerate(_sequence(routes, path))
    ]


def _transform_role_offset(offset: object, axis: MirrorAxis, path: str) -> tuple[float, float]:
    forward, lateral = _point(offset, path)
    # The offset is (forward, lateral) in the target robot's local frame.  A
    # reflection changes the handedness of that frame, so only lateral flips.
    return forward, -lateral


def _transform_single_pedestrian(
    pedestrian: object, width: float, height: float, axis: MirrorAxis, path: str
) -> SinglePedestrianDefinition:
    if not isinstance(pedestrian, SinglePedestrianDefinition):
        raise ValueError(
            f"{path} must be a SinglePedestrianDefinition, got {type(pedestrian).__name__}"
        )
    trajectory = None
    if pedestrian.trajectory is not None:
        trajectory = [
            reflect_point(point, width=width, height=height, axis=axis)
            for point in _sequence(pedestrian.trajectory, f"{path}.trajectory")
        ]
    role_offset = (
        None
        if pedestrian.role_offset is None
        else _transform_role_offset(pedestrian.role_offset, axis, f"{path}.role_offset")
    )
    try:
        metadata = deepcopy(pedestrian.metadata)
        wait_at = deepcopy(pedestrian.wait_at)
    except Exception as exc:  # pragma: no cover - defensive for arbitrary caller metadata
        raise ValueError(f"{path} metadata/wait_at could not be copied safely") from exc
    return SinglePedestrianDefinition(
        id=pedestrian.id,
        start=reflect_point(pedestrian.start, width=width, height=height, axis=axis),
        goal=(
            None
            if pedestrian.goal is None
            else reflect_point(pedestrian.goal, width=width, height=height, axis=axis)
        ),
        trajectory=trajectory,
        speed_m_s=pedestrian.speed_m_s,
        wait_at=wait_at,
        start_delay_s=pedestrian.start_delay_s,
        note=pedestrian.note,
        role=pedestrian.role,
        role_target_id=pedestrian.role_target_id,
        role_offset=role_offset,
        hold_until_robot_within_m=pedestrian.hold_until_robot_within_m,
        hold_ref_point=(
            None
            if pedestrian.hold_ref_point is None
            else reflect_point(pedestrian.hold_ref_point, width=width, height=height, axis=axis)
        ),
        hold_timeout_s=pedestrian.hold_timeout_s,
        metadata=metadata,
    )


def _transform_single_pedestrians(
    pedestrians: object, width: float, height: float, axis: MirrorAxis
) -> list[SinglePedestrianDefinition]:
    return [
        _transform_single_pedestrian(
            pedestrian, width, height, axis, f"single_pedestrians[{index}]"
        )
        for index, pedestrian in enumerate(_sequence(pedestrians, "single_pedestrians"))
    ]


def _transform_allowed_areas(
    areas: object, width: float, height: float, axis: MirrorAxis
) -> list[Polygon | MultiPolygon] | None:
    if areas is None:
        return None
    return [
        _transform_geometry(area, width, height, axis, f"allowed_areas[{index}]")
        for index, area in enumerate(_sequence(areas, "allowed_areas"))
    ]


def _transform_semantic_boundaries(
    boundaries: object, width: float, height: float, axis: MirrorAxis
) -> list[SemanticBoundary]:
    transformed: list[SemanticBoundary] = []
    for index, boundary in enumerate(_sequence(boundaries, "semantic_boundaries")):
        path = f"semantic_boundaries[{index}]"
        if not isinstance(boundary, SemanticBoundary):
            raise ValueError(f"{path} must be a SemanticBoundary, got {type(boundary).__name__}")
        coordinates = tuple(
            reflect_point(point, width=width, height=height, axis=axis)
            for point in _sequence(boundary.coordinates, f"{path}.coordinates")
        )
        transformed.append(
            SemanticBoundary(
                coordinates=coordinates,
                label=boundary.label,
                id_=boundary.id_,
                vehicle_blocking=boundary.vehicle_blocking,
                pedestrian_passable=boundary.pedestrian_passable,
                occluding=boundary.occluding,
                spawn_edge=boundary.spawn_edge,
            )
        )
    return transformed


def _transform_infrastructure_zones(
    zones: object, width: float, height: float, axis: MirrorAxis
) -> list[InfrastructureZone]:
    transformed: list[InfrastructureZone] = []
    for index, zone in enumerate(_sequence(zones, "infrastructure_zones")):
        path = f"infrastructure_zones[{index}]"
        if not isinstance(zone, InfrastructureZone):
            raise ValueError(f"{path} must be an InfrastructureZone, got {type(zone).__name__}")
        vertices = [
            reflect_point(point, width=width, height=height, axis=axis)
            for point in _sequence(zone.vertices, f"{path}.vertices")
        ]
        transformed.append(
            InfrastructureZone(
                id=zone.id,
                zone_type=zone.zone_type,
                vertices=vertices,
                allowed_actor_types=tuple(zone.allowed_actor_types),
                note=zone.note,
            )
        )
    return transformed


def _transform_social_groups(
    groups: object, width: float, height: float, axis: MirrorAxis
) -> list[SocialGroupDefinition]:
    transformed: list[SocialGroupDefinition] = []
    for index, group in enumerate(_sequence(groups, "social_groups")):
        path = f"social_groups[{index}]"
        if not isinstance(group, SocialGroupDefinition):
            raise ValueError(f"{path} must be a SocialGroupDefinition, got {type(group).__name__}")
        polygon = None
        if group.o_space_polygon is not None:
            polygon = [
                reflect_point(point, width=width, height=height, axis=axis)
                for point in _sequence(group.o_space_polygon, f"{path}.o_space_polygon")
            ]
        try:
            metadata = deepcopy(group.metadata)
        except Exception as exc:  # pragma: no cover - defensive for arbitrary caller metadata
            raise ValueError(f"{path}.metadata could not be copied safely") from exc
        transformed.append(
            SocialGroupDefinition(
                group_id=group.group_id,
                type=group.type,
                members=tuple(group.members),
                formation=group.formation,
                centroid=reflect_point(group.centroid, width=width, height=height, axis=axis),
                radius=group.radius,
                o_space_polygon=polygon,
                metadata=metadata,
            )
        )
    return transformed


__all__ = [
    "MirrorAxis",
    "mirror_map_definition",
    "reflect_heading",
    "reflect_map_definition",
    "reflect_point",
    "reflect_vector",
]

"""Focused tests for post-loader release-map reflections."""

from __future__ import annotations

from copy import deepcopy
from math import pi

import pytest
from shapely.geometry import Point, Polygon, box

from robot_sf.benchmark.release_map_mirror import (
    mirror_map_definition,
    reflect_heading,
    reflect_map_definition,
    reflect_point,
    reflect_vector,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import (
    InfrastructureZone,
    MapDefinition,
    PedestrianWaitRule,
    SinglePedestrianDefinition,
    SocialGroupDefinition,
)
from robot_sf.nav.nav_types import SemanticBoundary
from robot_sf.nav.obstacle import Obstacle


def _base_map() -> MapDefinition:
    """Build a small map containing every post-loader geometry family."""

    spawn_zone = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0))
    goal_zone = ((8.0, 6.0), (9.0, 6.0), (9.0, 7.0))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (4.0, 3.0), (8.5, 6.5)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
        source_path_id="route-0",
        source_label="main",
    )
    obstacle_polygon = Polygon(
        [(3.0, 2.0), (5.0, 2.0), (5.0, 4.0), (3.0, 4.0)],
        holes=[[(3.5, 2.5), (4.5, 2.5), (4.5, 3.5), (3.5, 3.5)]],
    )
    pedestrian = SinglePedestrianDefinition(
        id="ped-0",
        start=(2.0, 6.0),
        trajectory=[(2.0, 5.0), (4.0, 4.0)],
        speed_m_s=1.2,
        wait_at=[PedestrianWaitRule(waypoint_index=1, wait_s=0.5, note="pause")],
        start_delay_s=0.25,
        note="test pedestrian",
        role="follow",
        role_target_id="robot:0",
        role_offset=(1.25, -0.5),
        hold_until_robot_within_m=1.0,
        hold_ref_point=(3.0, 4.0),
        hold_timeout_s=3.0,
        metadata={"source": "fixture", "non_geometric": {"value": 7}},
    )
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[
            Obstacle(
                vertices=[(3.0, 2.0), (5.0, 2.0), (5.0, 4.0), (3.0, 4.0)],
                geometry=obstacle_polygon,
            )
        ],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[((6.0, 1.0), (7.0, 1.0), (7.0, 2.0))],
        robot_goal_zones=[goal_zone],
        bounds=[
            (0.0, 10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0, 8.0),
            (10.0, 0.0, 8.0, 8.0),
            (0.0, 0.0, 8.0, 0.0),
        ],
        robot_routes=[route],
        ped_goal_zones=[((7.0, 6.0), (8.0, 6.0), (8.0, 7.0))],
        ped_crowded_zones=[((4.0, 5.0), (5.0, 5.0), (5.0, 6.0))],
        ped_routes=[],
        single_pedestrians=[pedestrian],
        poi_positions=[(2.5, 1.5)],
        poi_labels={"poi-0": "crossing"},
        allowed_areas=[box(0.5, 0.5, 9.5, 7.5)],
        semantic_boundaries=[
            SemanticBoundary(
                coordinates=((2.0, 2.0), (2.0, 6.0)),
                label="separator",
                id_="separator-0",
                vehicle_blocking=True,
                pedestrian_passable=True,
            )
        ],
        infrastructure_zones=[
            InfrastructureZone(
                id="ped-zone",
                zone_type="pedestrian_only",
                vertices=[(6.0, 2.0), (8.0, 2.0), (8.0, 4.0), (6.0, 4.0)],
                allowed_actor_types=("pedestrian",),
                note="metadata survives",
            )
        ],
        social_groups=[
            SocialGroupDefinition(
                group_id="group-0",
                type="conversation",
                members=("ped-0",),
                formation="circular_conversation",
                centroid=(6.0, 5.0),
                radius=0.75,
                o_space_polygon=[(5.5, 4.5), (6.5, 4.5), (6.5, 5.5), (5.5, 5.5)],
                metadata={"label": "social metadata"},
            )
        ],
        svg_geometry_contract="corrected",
    )


def test_reflect_point_heading_and_vector_follow_synthetic_conventions() -> None:
    """Both mirror axes use the same point, heading, and vector conventions as the harness."""

    assert reflect_point((2.0, 3.0), width=10.0, height=8.0, axis="y") == (2.0, 5.0)
    assert reflect_point((2.0, 3.0), width=10.0, height=8.0, axis="x") == (8.0, 3.0)
    assert reflect_heading(0.75, "y") == pytest.approx(-0.75)
    assert reflect_heading(0.75, "x") == pytest.approx(pi - 0.75)
    assert reflect_vector((1.5, -2.0), "y") == (1.5, 2.0)
    assert reflect_vector((1.5, -2.0), "x") == (-1.5, -2.0)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_reflection_transforms_all_coordinate_bearing_map_fields(axis: str) -> None:
    """The loaded map's geometry and local role vector are reflected together."""

    source = _base_map()
    reflected = reflect_map_definition(source, axis)  # type: ignore[arg-type]
    transform = lambda point: reflect_point(point, width=10.0, height=8.0, axis=axis)  # noqa: E731

    assert reflected.width == source.width
    assert reflected.height == source.height
    assert reflected.poi_positions == [transform(source.poi_positions[0])]
    assert reflected.robot_spawn_zones[0] == tuple(
        transform(point) for point in source.robot_spawn_zones[0]
    )
    assert reflected.ped_goal_zones[0] == tuple(
        transform(point) for point in source.ped_goal_zones[0]
    )
    assert reflected.bounds[0] == (
        transform((0.0, 0.0))[0],
        transform((10.0, 0.0))[0],
        transform((0.0, 0.0))[1],
        transform((10.0, 0.0))[1],
    )

    source_obstacle = source.obstacles[0]
    reflected_obstacle = reflected.obstacles[0]
    assert reflected_obstacle.vertices == [transform(point) for point in source_obstacle.vertices]
    assert reflected_obstacle.geometry.equals(
        Polygon([transform(point) for point in source_obstacle.vertices]).difference(
            Polygon(
                [transform(point) for point in source_obstacle.geometry.interiors[0].coords[:-1]]
            )
        )
    )

    source_route = source.robot_routes[0]
    reflected_route = reflected.robot_routes[0]
    assert reflected_route.waypoints == [transform(point) for point in source_route.waypoints]
    assert reflected_route.spawn_zone == tuple(
        transform(point) for point in source_route.spawn_zone
    )
    assert reflected_route.goal_zone == tuple(transform(point) for point in source_route.goal_zone)
    assert (reflected_route.source_path_id, reflected_route.source_label) == (
        source_route.source_path_id,
        source_route.source_label,
    )

    source_ped = source.single_pedestrians[0]
    reflected_ped = reflected.single_pedestrians[0]
    assert reflected_ped.start == transform(source_ped.start)
    assert reflected_ped.trajectory == [transform(point) for point in source_ped.trajectory]
    assert reflected_ped.hold_ref_point == transform(source_ped.hold_ref_point)
    assert reflected_ped.role_offset == (source_ped.role_offset[0], -source_ped.role_offset[1])

    assert reflected.allowed_areas[0].equals(
        box(
            transform((0.5, 0.5))[0],
            transform((0.5, 0.5))[1],
            transform((9.5, 7.5))[0],
            transform((9.5, 7.5))[1],
        )
    )
    assert reflected.semantic_boundaries[0].coordinates == tuple(
        transform(point) for point in source.semantic_boundaries[0].coordinates
    )
    assert reflected.infrastructure_zones[0].vertices == [
        transform(point) for point in source.infrastructure_zones[0].vertices
    ]
    assert reflected.social_groups[0].centroid == transform(source.social_groups[0].centroid)
    assert reflected.social_groups[0].o_space_polygon == [
        transform(point) for point in source.social_groups[0].o_space_polygon
    ]


@pytest.mark.parametrize("axis", ["x", "y"])
def test_double_reflection_is_identity_and_source_is_unchanged(axis: str) -> None:
    """Reflections are involutions and do not share mutable geometry with the source."""

    source = _base_map()
    source_snapshot = deepcopy(source)
    reflected = mirror_map_definition(source, axis)  # type: ignore[arg-type]
    restored = reflect_map_definition(reflected, axis)  # type: ignore[arg-type]

    assert restored.poi_positions == source_snapshot.poi_positions
    assert restored.robot_routes[0].waypoints == source_snapshot.robot_routes[0].waypoints
    assert restored.obstacles[0].geometry.equals(source_snapshot.obstacles[0].geometry)
    assert restored.single_pedestrians[0].metadata == source_snapshot.single_pedestrians[0].metadata
    assert (
        restored.social_groups[0].o_space_polygon
        == source_snapshot.social_groups[0].o_space_polygon
    )

    reflected.poi_positions[0] = (99.0, 99.0)
    reflected.single_pedestrians[0].metadata["source"] = "changed"
    assert source.poi_positions == source_snapshot.poi_positions
    assert source.single_pedestrians[0].metadata == source_snapshot.single_pedestrians[0].metadata


def test_reflection_fails_closed_for_nonfinite_bounds_and_unsupported_geometry() -> None:
    """Malformed coordinate-bearing fields cannot pass through unchanged."""

    nonfinite = _base_map()
    nonfinite.bounds[0] = (float("nan"), 10.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        reflect_map_definition(nonfinite, "y")

    unsupported = _base_map()
    unsupported.allowed_areas = [Point(1.0, 1.0)]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Polygon or MultiPolygon"):
        reflect_map_definition(unsupported, "x")

    unsupported_public_field = _base_map()
    unsupported_public_field.custom_coordinate_payload = {"point": (1.0, 2.0)}  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="unsupported public fields"):
        reflect_map_definition(unsupported_public_field, "y")


def test_reflection_fails_closed_for_out_of_bounds_points() -> None:
    """A geometrically complete reflection must also stay inside map bounds."""

    out_of_bounds = _base_map()
    out_of_bounds.poi_positions[0] = (10.1, 1.5)
    with pytest.raises(ValueError, match="outside map bounds"):
        reflect_map_definition(out_of_bounds, "x")

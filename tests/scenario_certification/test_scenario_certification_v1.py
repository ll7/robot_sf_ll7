"""Tests for scenario_cert.v1 certification contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.scenario_certification import certificate_to_dict, certify_map_definition
from robot_sf.scenario_certification.v1 import (
    DYNAMICALLY_OVERCONSTRAINED,
    GEOMETRICALLY_INFEASIBLE,
    HARD_BUT_SOLVABLE,
    INVALID,
    KINODYNAMICALLY_INFEASIBLE,
    KNIFE_EDGE,
    VALID,
    CertificationSettings,
)


def _zone(
    x: float, y: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return ((x - 0.2, y - 0.2), (x + 0.2, y - 0.2), (x + 0.2, y + 0.2))


def _bounds(width: float, height: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]


def _obstacle(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> Obstacle:
    return Obstacle([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


def _map(
    route_points: list[tuple[float, float]],
    *,
    width: float = 12.0,
    height: float = 8.0,
    obstacles: list[Obstacle] | None = None,
    pedestrians: list[SinglePedestrianDefinition] | None = None,
) -> MapDefinition:
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=route_points,
        spawn_zone=_zone(*route_points[0]),
        goal_zone=_zone(*route_points[-1]),
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles or [],
        robot_spawn_zones=[route.spawn_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[route.goal_zone],
        bounds=_bounds(width, height),
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=pedestrians or [],
    )


def _classify(
    map_def: MapDefinition,
    *,
    robot_config: object | None = None,
) -> str:
    certificate = certify_map_definition(
        map_def,
        robot_config=robot_config or DifferentialDriveSettings(radius=0.4),
        settings=CertificationSettings(planner_cells_per_meter=2.0),
    )
    return certificate.classification


def test_certificate_schema_validates_emitted_valid_certificate() -> None:
    """Verify emitted certificates satisfy the versioned JSON schema used by tools."""

    certificate = certify_map_definition(
        _map([(2.0, 2.0), (10.0, 2.0)]),
        robot_config=DifferentialDriveSettings(radius=0.4),
    )
    schema_path = Path("robot_sf/benchmark/schemas/scenario_cert.v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(certificate_to_dict(certificate), schema)
    assert certificate.classification == VALID


def test_invalid_route_fails_closed_when_start_is_outside_map() -> None:
    """Invalid start coordinates are excluded before planner evidence is considered."""

    assert _classify(_map([(-1.0, 2.0), (10.0, 2.0)])) == INVALID


def test_geometrically_infeasible_when_inflated_path_is_blocked() -> None:
    """A wall spanning the map separates start and goal with no inflated route."""

    blocked = _map(
        [(2.0, 4.0), (10.0, 4.0)],
        obstacles=[_obstacle(5.0, 0.0, 7.0, 8.0)],
    )
    assert _classify(blocked) == GEOMETRICALLY_INFEASIBLE


def test_kinodynamically_infeasible_for_tighter_turn_than_bicycle_limit() -> None:
    """Bicycle routes with turns tighter than the configured limit are excluded."""

    tight_turn = _map([(2.0, 2.0), (3.0, 2.0), (3.0, 3.0)])
    robot = BicycleDriveSettings(radius=0.4, wheelbase=4.0, max_steer=0.5)
    assert _classify(tight_turn, robot_config=robot) == KINODYNAMICALLY_INFEASIBLE


def test_static_pedestrian_on_route_is_dynamically_overconstrained() -> None:
    """Static pedestrians blocking the certified route are excluded as overconstrained."""

    blocked = _map(
        [(2.0, 2.0), (10.0, 2.0)],
        pedestrians=[SinglePedestrianDefinition(id="p0", start=(5.0, 2.0))],
    )
    assert _classify(blocked) == DYNAMICALLY_OVERCONSTRAINED


def test_low_clearance_solvable_route_is_knife_edge() -> None:
    """Solvable paths close to a static obstacle are marked stress-only."""

    narrow = _map(
        [(2.0, 2.0), (10.0, 2.0)],
        obstacles=[_obstacle(5.0, 2.55, 7.0, 4.0)],
    )
    assert _classify(narrow) == KNIFE_EDGE


def test_dynamic_crossing_is_hard_but_solvable() -> None:
    """Moving pedestrian interactions are represented without collapsing to pass/fail."""

    crossing = _map(
        [(2.0, 2.0), (10.0, 2.0)],
        pedestrians=[SinglePedestrianDefinition(id="p0", start=(5.0, 0.5), goal=(5.0, 4.0))],
    )
    assert _classify(crossing) == HARD_BUT_SOLVABLE


@pytest.mark.parametrize(
    ("classification", "eligibility"),
    [
        (VALID, "eligible"),
        (HARD_BUT_SOLVABLE, "eligible"),
        (KNIFE_EDGE, "stress_only"),
        (INVALID, "excluded"),
        (GEOMETRICALLY_INFEASIBLE, "excluded"),
        (KINODYNAMICALLY_INFEASIBLE, "excluded"),
        (DYNAMICALLY_OVERCONSTRAINED, "excluded"),
    ],
)
def test_schema_enum_documents_expected_statuses(classification: str, eligibility: str) -> None:
    """The public schema keeps the status taxonomy explicit and machine-readable."""

    schema = json.loads(Path("robot_sf/benchmark/schemas/scenario_cert.v1.json").read_text())
    assert classification in schema["$defs"]["classification"]["enum"]
    assert eligibility in schema["$defs"]["eligibility"]["enum"]

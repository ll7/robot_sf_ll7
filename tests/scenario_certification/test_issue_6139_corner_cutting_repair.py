"""Tests for issue #6139: reject A* certification paths whose swept envelope clips corners.

Issue #6139 repairs ``scenario_cert.v1`` so every accepted A* segment is validated
continuously against the same parsed obstacle geometry and robot envelope the
simulator uses. A grid-inflated A* path can still cut a diagonal corner that the
continuous robot disc cannot pass, so the certifier now:

* re-validate the planned path's full-polyline (swept-disc) clearance after A*,
* fail closed (``geometrically_infeasible`` / excluded) when the envelope clips a
  corner or the geometry cannot be verified,
* record the occupancy/A* verdict, continuous swept-envelope verdict, and
  executable runtime collision verdict together for the discriminating check.

These tests are certification-mechanism tests, not benchmark evidence. They assert
the corrected fail-closed contract on deterministic, programmatic map fixtures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import pytest
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.scenario_certification import certificate_to_dict, certify_map_definition
from robot_sf.scenario_certification.v1 import (
    GEOMETRICALLY_INFEASIBLE,
    VALID,
    CertificationSettings,
    _validate_planned_path_simulator_collision,
    _validate_planned_path_swept_envelope,
    measure_planned_path_clearance,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "robot_sf/benchmark/schemas/scenario_cert.v1.json"


def _zone(
    x: float, y: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return a small triangular spawn/goal zone centered near the given point."""
    return ((x - 0.2, y - 0.2), (x + 0.2, y - 0.2), (x + 0.2, y + 0.2))


def _bounds(width: float, height: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return rectangular map bounds for certification fixtures."""
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
    """Return an axis-aligned rectangular obstacle fixture."""
    return Obstacle([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])


def _map(
    route_points: list[tuple[float, float]],
    *,
    width: float = 14.0,
    height: float = 10.0,
    obstacles: list[Obstacle] | None = None,
) -> MapDefinition:
    """Build a minimal map definition around one robot route."""
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
        single_pedestrians=[],
        infrastructure_zones=[],
    )


def _route_certificate(
    map_def: MapDefinition,
    *,
    robot_radius: float = 1.0,
    cells_per_meter: float = 2.0,
):
    """Certify a fixture map and return its single route certificate."""
    certificate = certify_map_definition(
        map_def,
        robot_config=DifferentialDriveSettings(radius=robot_radius),
        settings=CertificationSettings(planner_cells_per_meter=cells_per_meter),
    )
    assert len(certificate.route_certificates) == 1
    return certificate, certificate.route_certificates[0]


def _load_schema() -> dict:
    """Load the versioned scenario certificate JSON schema."""
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Canonical corner-cutting measurement helper
# ---------------------------------------------------------------------------


def test_measure_planned_path_clearance_flags_negative_corner_clearance() -> None:
    """A planned vertex inside the robot-inflated obstacle reports negative clearance."""
    obstacle = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    clipped, min_vertex_clearance, min_path_clearance = measure_planned_path_clearance(
        [(1.0, 1.0), (10.0, 10.0)], obstacle, robot_radius=1.0
    )
    assert clipped == 1
    assert min_vertex_clearance is not None and min_vertex_clearance < 0.0
    assert min_path_clearance is not None and min_path_clearance < 0.0


def test_measure_planned_path_clearance_detects_segment_corner_cut() -> None:
    """A segment crossing an obstacle fails even when both vertices are clear."""
    obstacle = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
    clipped, min_vertex_clearance, min_path_clearance = measure_planned_path_clearance(
        [(-1.0, 1.0), (3.0, 1.0)], obstacle, robot_radius=0.25
    )
    assert clipped == 0
    assert min_vertex_clearance is not None and min_vertex_clearance > 0.0
    assert min_path_clearance is not None and min_path_clearance < 0.0


def test_measure_planned_path_clearance_returns_none_for_empty_obstacles() -> None:
    """An empty obstacle set yields no finite clearance for the caller to fail closed on."""
    from shapely.geometry import GeometryCollection

    clipped, min_vertex_clearance, min_path_clearance = measure_planned_path_clearance(
        [(0.0, 0.0), (1.0, 1.0)], GeometryCollection(), robot_radius=1.0
    )
    assert clipped == 0
    assert min_vertex_clearance is None
    assert min_path_clearance is None


# ---------------------------------------------------------------------------
# Swept-envelope verdict helper
# ---------------------------------------------------------------------------


def test_validate_swept_envelope_fails_closed_on_short_path() -> None:
    """A planned path with fewer than two waypoints cannot be validated."""
    obstacle = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    verdict = _validate_planned_path_swept_envelope(
        [(1.0, 1.0)], obstacle_union=obstacle, robot_radius=1.0
    )
    assert verdict["validated"] is False
    assert verdict["clips_obstacle"] is None
    assert "fewer_than_two" in verdict["blocker"]


def test_validate_swept_envelope_fails_closed_on_invalid_radius() -> None:
    """A non-finite or negative radius must fail closed before measuring clearance."""
    obstacle = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    verdict = _validate_planned_path_swept_envelope(
        [(0.0, 5.0), (5.0, 5.0)], obstacle_union=obstacle, robot_radius=float("nan")
    )
    assert verdict["validated"] is False
    assert "robot_radius" in verdict["blocker"]


def test_validate_swept_envelope_trivially_clear_without_obstacles() -> None:
    """With no static obstacles the swept envelope is trivially collision-free."""
    from shapely.geometry import GeometryCollection

    verdict = _validate_planned_path_swept_envelope(
        [(0.0, 0.0), (5.0, 5.0)], obstacle_union=GeometryCollection(), robot_radius=1.0
    )
    assert verdict["validated"] is True
    assert verdict["clips_obstacle"] is False
    assert verdict["clipped_vertex_count"] == 0


def test_runtime_collision_verdict_fails_closed_on_invalid_radius() -> None:
    """A non-finite runtime collision radius is unverifiable rather than collision-free."""
    map_def = _map([(1.0, 1.0), (10.0, 1.0)])

    verdict = _validate_planned_path_simulator_collision(
        [(1.0, 1.0), (10.0, 1.0)],
        map_def=map_def,
        goal=(10.0, 1.0),
        robot_radius=float("nan"),
    )

    assert verdict["validated"] is False
    assert verdict["collides_obstacle"] is None
    assert verdict["blocker"] == "robot_radius_must_be_finite_and_non_negative"


# ---------------------------------------------------------------------------
# Core #6139 acceptance criteria
# ---------------------------------------------------------------------------


def test_diagonal_corner_cutting_path_fails_closed() -> None:
    """A diagonally adjacent blocked-cell fixture cannot certify by corner cutting.

    A single central obstacle with a diagonal route around its corner produces an
    A* path (the occupancy grid has a free diagonal cell) whose continuous swept
    robot envelope clips the corner. The corrected certifier must reject it as
    ``geometrically_infeasible`` (excluded) rather than accepting it.
    """
    blocked = _map(
        [(2.0, 2.0), (12.0, 8.0)],
        obstacles=[_obstacle(5.0, 3.0, 9.0, 7.0)],
    )
    certificate, route_cert = _route_certificate(blocked, robot_radius=1.0)

    assert certificate.classification == GEOMETRICALLY_INFEASIBLE
    assert certificate.benchmark_eligibility == "excluded"
    # The occupancy/A* verdict must be corrected to False when the envelope clips.
    assert route_cert.checks["inflated_collision_free_path"] is False
    swept = route_cert.checks["swept_envelope"]
    assert swept["validated"] is True
    assert swept["clips_obstacle"] is True
    assert swept["clearance_m"] is not None and swept["clearance_m"] < 0.0
    assert swept["clipped_vertex_count"] >= 1
    assert any("swept_envelope_clips_obstacle" in reason for reason in route_cert.reasons)
    # The emitted certificate must still satisfy the versioned schema.
    jsonschema.validate(certificate_to_dict(certificate), _load_schema())


def test_discriminating_collision_verdicts_recorded_together() -> None:
    """The discriminating check records A*, swept-disc, and runtime verdicts together."""
    blocked = _map(
        [(2.0, 2.0), (12.0, 8.0)],
        obstacles=[_obstacle(5.0, 3.0, 9.0, 7.0)],
    )
    _certificate, route_cert = _route_certificate(blocked, robot_radius=1.0)

    checks = route_cert.checks
    # Occupancy-grid/A*, continuous swept-disc geometry, and the executable runtime
    # simulator collision verdict coexist on the same planned path.
    assert "inflated_collision_free_path" in checks
    assert "swept_envelope" in checks
    assert "simulator_obstacle_collision" in checks
    swept = checks["swept_envelope"]
    for key in ("validated", "clips_obstacle", "clearance_m", "clipped_vertex_count"):
        assert key in swept
    # The swept-disc clearance is finite (fail-closed on non-finite geometry).
    assert swept["validated"] is True
    assert isinstance(swept["clearance_m"], float)
    assert math.isfinite(swept["clearance_m"])
    simulator = checks["simulator_obstacle_collision"]
    assert simulator["validated"] is True
    assert simulator["collides_obstacle"] is True
    assert simulator["runtime_component"] == "ContinuousOccupancy.is_obstacle_collision"
    assert simulator["checked_sample_count"] > 0
    assert simulator["first_collision_sample_index"] is not None


def test_valid_path_has_finite_nonnegative_full_polyline_clearance() -> None:
    """Every accepted path keeps finite non-negative full-polyline clearance."""
    clear_map = _map([(2.0, 2.0), (12.0, 2.0)], obstacles=[_obstacle(5.0, 4.0, 7.0, 8.0)])
    certificate, route_cert = _route_certificate(clear_map, robot_radius=0.4)

    assert certificate.classification == VALID
    assert route_cert.checks["inflated_collision_free_path"] is True
    swept = route_cert.checks["swept_envelope"]
    assert swept["validated"] is True
    assert swept["clips_obstacle"] is False
    assert swept["clearance_m"] is not None
    assert math.isfinite(swept["clearance_m"])
    assert swept["clearance_m"] >= 0.0
    assert swept["clipped_vertex_count"] == 0
    simulator = route_cert.checks["simulator_obstacle_collision"]
    assert simulator["validated"] is True
    assert simulator["collides_obstacle"] is False


def test_clearance_is_computed_on_planned_path_not_authored_line() -> None:
    """The swept-envelope clearance is measured on the planned A* path, not the route line.

    The authored straight route line through the obstacle has negative authored-route
    clearance, but the planner routes around it. The swept-envelope verdict must
    follow the *planned* path so a corner-cutting planned path is still caught.
    """
    blocked = _map(
        [(2.0, 2.0), (12.0, 8.0)],
        obstacles=[_obstacle(5.0, 3.0, 9.0, 7.0)],
    )
    certificate, route_cert = _route_certificate(blocked, robot_radius=1.0)
    # The authored route line goes straight through the block: authored clearance < 0.
    assert route_cert.checks["minimum_static_clearance_m"] is not None
    assert route_cert.checks["minimum_static_clearance_m"] < 0.0
    # The planned path is rejected because it clips the corner.
    assert certificate.classification == GEOMETRICALLY_INFEASIBLE


# ---------------------------------------------------------------------------
# Frozen blind-corner regression (issue #6139 acceptance criterion #2)
# ---------------------------------------------------------------------------


def test_frozen_blind_corner_route_cannot_report_collision_free_when_clearance_negative() -> None:
    """The frozen blind-corner A* path clips the corner and must not certify collision-free.

    Loads the committed ``francis2023_blind_corner`` cell, re-plans the inflated A*
    path, and asserts the swept envelope clips the inner corner. Under the corrected
    certifier this route is ``geometrically_infeasible`` and never reports
    ``inflated_collision_free_path=True`` while the full-polyline clearance is negative.
    """
    pytest.importorskip("robot_sf.benchmark.map_runner")
    from robot_sf.scenario_certification.feasibility_oracle import (
        ISSUE_5596_BLIND_CORNER_SCENARIO_ID,
        _replan_astar_path,
        _resolve_scenario_map_and_route,
    )
    from robot_sf.scenario_certification.v1 import _obstacle_union
    from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

    manifest = _REPO_ROOT / "configs/scenarios/francis2023.yaml"
    if not manifest.exists():
        pytest.skip("committed francis2023.yaml manifest not available")
    scenarios = load_scenarios(manifest)
    cell = next(s for s in scenarios if s.get("name") == ISSUE_5596_BLIND_CORNER_SCENARIO_ID)
    config = build_robot_config_from_scenario(dict(cell), scenario_path=manifest)
    map_def, start, goal, blocker = _resolve_scenario_map_and_route(config, cell)
    assert blocker is None and start is not None and goal is not None
    robot_radius = float(getattr(config.robot_config, "radius", 1.0))
    obstacle_union = _obstacle_union(map_def)
    assert not obstacle_union.is_empty

    path = _replan_astar_path(map_def, start=start, goal=goal, robot_radius=robot_radius)
    assert path and len(path) >= 2
    # Independent continuous swept-disc check: buffer the planned polyline by the
    # robot radius and confirm it intersects the parsed obstacle geometry.
    swept = _validate_planned_path_swept_envelope(
        path, obstacle_union=obstacle_union, robot_radius=robot_radius
    )
    assert swept["validated"] is True
    assert swept["clips_obstacle"] is True
    assert swept["clearance_m"] is not None and swept["clearance_m"] < 0.0
    # Cross-check with an independent buffer-intersection measurement.
    envelope = LineString(path).buffer(robot_radius)
    assert not envelope.intersection(obstacle_union).is_empty

    # The corrected certifier must reject this route as geometrically infeasible.
    certificate = certify_map_definition(
        map_def,
        robot_config=DifferentialDriveSettings(radius=robot_radius),
        settings=CertificationSettings(planner_cells_per_meter=2.0),
        scenario={"name": ISSUE_5596_BLIND_CORNER_SCENARIO_ID},
    )
    assert certificate.classification == GEOMETRICALLY_INFEASIBLE
    assert certificate.benchmark_eligibility == "excluded"
    for route_cert in certificate.route_certificates:
        assert route_cert.checks["inflated_collision_free_path"] is False
        assert route_cert.checks["swept_envelope"]["clips_obstacle"] is True
        simulator = route_cert.checks["simulator_obstacle_collision"]
        assert simulator["validated"] is True
        assert simulator["collides_obstacle"] is True


# ---------------------------------------------------------------------------
# Independent geometry sanity checks
# ---------------------------------------------------------------------------


def test_swept_envelope_clearance_matches_independent_buffer_measurement() -> None:
    """The reported swept-disc clearance equals an independent line.distance() check."""
    obstacle = unary_union([Polygon([(5.0, 3.0), (9.0, 3.0), (9.0, 7.0), (5.0, 7.0)])])
    path = [(2.0, 2.0), (7.0, 5.0), (12.0, 8.0)]
    radius = 1.0
    _clipped, _vertex, reported = measure_planned_path_clearance(path, obstacle, radius)
    independent = float(LineString(path).distance(obstacle) - radius)
    assert reported is not None
    assert reported == pytest.approx(independent)

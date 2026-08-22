"""Focused contract tests for the archetype geometry checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from shapely.geometry import Point

from scripts.validation.check_scenario_archetype_geometry import (
    EndpointCheck,
    FragmentCheck,
    MapGeometryReport,
    _endpoint_checks,
    _fragment_checks,
    _missing_zone_kinds,
    _rect_centre,
    _rect_polygon,
    format_console_table,
    inspect_map_geometry,
    main,
)


@dataclass
class _FakeRoute:
    """Minimal duck-typed stand-in for GlobalRoute."""

    spawn_id: int
    goal_id: int
    waypoints: list[tuple[float, float]]
    spawn_zone: tuple | None
    goal_zone: tuple | None
    source_label: str = "robot_route_0_0"


@dataclass
class _FakeMapDef:
    """Minimal duck-typed stand-in for MapDefinition."""

    robot_spawn_zones: list = field(default_factory=list)
    ped_spawn_zones: list = field(default_factory=list)
    robot_goal_zones: list = field(default_factory=list)
    ped_goal_zones: list = field(default_factory=list)
    robot_routes: list = field(default_factory=list)
    ped_routes: list = field(default_factory=list)


# Three-corner rect spanning x in [0, 10], y in [0, 4]; centre (5, 2).
RECT = ((0.0, 4.0), (10.0, 4.0), (0.0, 0.0))


def test_rect_helpers_centre_and_polygon() -> None:
    assert _rect_centre(RECT) == (5.0, 2.0)
    poly = _rect_polygon(RECT)
    assert poly.contains(Point(5, 2))
    assert not poly.contains(Point(20, 20))


def test_endpoint_inside_zone_reports_ok() -> None:
    route = _FakeRoute(0, 0, [(5.0, 2.0), (6.0, 2.0)], RECT, RECT)
    checks = _endpoint_checks(_FakeMapDef(), "robot", [route], tolerance_m=0.5)
    assert [c.inside_zone for c in checks] == [True, True]
    assert checks[0].offset_to_centre_m == pytest.approx(0.0, abs=1e-6)


def test_endpoint_outside_beyond_tolerance_is_miss() -> None:
    route = _FakeRoute(0, 0, [(30.0, 30.0), (31.0, 30.0)], RECT, RECT)
    checks = _endpoint_checks(_FakeMapDef(), "robot", [route], tolerance_m=0.5)
    assert all(not c.inside_zone for c in checks)
    assert checks[0].offset_to_centre_m > 25.0


def test_disconnected_middle_fragment_detected() -> None:
    map_def = _FakeMapDef(
        robot_spawn_zones=[RECT],
        robot_goal_zones=[RECT],
    )
    # Segment 0 touches the spawn zone, segment 1 swings far outside every
    # zone (the cross-trap interior-fragment shape), segment 2 lands in the
    # goal zone.
    route = _FakeRoute(
        0,
        0,
        [(5.0, 2.0), (50.0, 50.0), (60.0, 60.0), (5.0, 2.5)],
        RECT,
        RECT,
    )
    checks = _fragment_checks(map_def, "robot", [route])
    assert len(checks) == 1
    assert checks[0].disconnected_fragment_count == 1
    assert checks[0].first_disconnected_segment == 1


def test_fully_connected_route_has_no_fragments() -> None:
    map_def = _FakeMapDef(robot_spawn_zones=[RECT], robot_goal_zones=[RECT])
    route = _FakeRoute(0, 0, [(5.0, 2.0), (6.0, 2.5), (7.0, 3.0)], RECT, RECT)
    checks = _fragment_checks(map_def, "robot", [route])
    assert checks[0].disconnected_fragment_count == 0


def test_missing_zone_kind_flagged() -> None:
    map_def = _FakeMapDef(ped_routes=[_FakeRoute(0, 0, [(0, 0), (1, 1)], None, None)])
    assert _missing_zone_kinds(map_def) == ["ped"]
    complete = _FakeMapDef(
        robot_spawn_zones=[RECT], robot_routes=[_FakeRoute(0, 0, [(5, 2)], RECT, RECT)]
    )
    assert _missing_zone_kinds(complete) == []


def test_report_violation_counting() -> None:
    report = MapGeometryReport(map_path="x.svg")
    report.endpoints.append(EndpointCheck("robot", "r", "start", "robot_spawn_zone", 0, False, 9.9))
    report.fragments.append(FragmentCheck("robot", "r", 2, 0))
    report.missing_zone_kinds.append("ped")
    assert report.violations == 3


def test_console_table_marks_findings() -> None:
    report = MapGeometryReport(map_path="m.svg")
    report.endpoints.append(
        EndpointCheck("robot", "robot_route_0_0", "start", "robot_spawn_zone", 0, False, 15.8)
    )
    text = format_console_table(report)
    assert "[MISS]" in text
    assert "15.800" in text


def test_integration_pinned_doorway_reproduces_audit(tmp_path: Path) -> None:
    """The doorway map's robot-route offsets match the dissertation audit."""

    repo_root = Path(__file__).resolve().parents[2]
    svg = repo_root / "maps/svg_maps/classic_doorway.svg"
    if not svg.exists():
        pytest.skip("pinned archetype maps not present")
    report = inspect_map_geometry(svg)
    robot = [e for e in report.endpoints if e.route_kind == "robot"]
    offsets = {e.end: e.offset_to_centre_m for e in robot}
    assert offsets["start"] == pytest.approx(2.398, abs=0.01)
    assert offsets["end"] == pytest.approx(2.490, abs=0.01)
    ped = [e for e in report.endpoints if e.route_kind == "ped"]
    assert all(e.inside_zone for e in ped)


def test_cli_default_exit_zero_and_fail_flag(tmp_path: Path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    svg = repo_root / "maps/svg_maps/classic_doorway.svg"
    if not svg.exists():
        pytest.skip("pinned archetype maps not present")
    assert main(["--map", str(svg)]) == 0
    assert main(["--map", str(svg), "--fail-on-violation"]) == 2
    out = capsys.readouterr().out
    assert "[MISS]" in out or "all checks informational-clean" in out

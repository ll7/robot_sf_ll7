"""Fail-closed contract for geometrically-impossible route clearance (issue #3628).

These tests pin the behaviour that the camera-ready benchmark must refuse to run a scenario whose
route centerline lies closer to a static obstacle than the robot's physical radius (a negative
clearance margin = guaranteed collision), while still allowing positive-but-narrow stress
geometry to pass with only an informational warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready_campaign import (
    RouteClearanceError,
    load_campaign_config,
    prepare_campaign_preflight,
)


def _write_campaign(
    tmp_path: Path,
    *,
    robot_radius: float,
    certification: bool = False,
) -> Path:
    """Write a minimal single-scenario campaign config and return its path.

    Returns:
        Path to the campaign config file.
    """
    map_path = (tmp_path / "scenario_map.svg").resolve()
    # Content is irrelevant: ``convert_map`` is monkeypatched per test to return fixed geometry.
    map_path.write_text("<svg></svg>\n", encoding="utf-8")

    scenario_lines = [
        "- name: clearance_case",
        f"  map_file: {map_path.as_posix()}",
        "  seeds: [111]",
        "  robot_config:",
        f"    radius: {robot_radius}",
    ]
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")

    config_lines = [
        "name: clearance_fail_closed",
        f"scenario_matrix: {scenario_path.as_posix()}",
        "planners:",
        "  - key: goal",
        "    algo: goal",
        "    planner_group: core",
    ]
    if certification:
        cert_lines = [
            "schema_version: route-clearance-certifications.v1",
            "certifications:",
            "  clearance_case:",
            "    status: certified_stress_geometry",
            "    claim_scope: stress geometry; planner-attribution claims require caveat",
            "    rationale: Narrow obstacle margin is intentional stress geometry.",
        ]
        cert_path = tmp_path / "certifications.yaml"
        cert_path.write_text("\n".join(cert_lines) + "\n", encoding="utf-8")
        config_lines.append(f"route_clearance_certifications: {cert_path.as_posix()}")

    config_path = tmp_path / "campaign.yaml"
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    return config_path


def _patch_geometry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    obstacle_x_min: float,
) -> None:
    """Patch ``convert_map`` so the centerline-to-obstacle distance equals ``obstacle_x_min``.

    The robot route is a segment ending at the origin (``(-1, 0) -> (0, 0)``). The obstacle is a
    box spanning the route's y-level (``y in [-1, 1]``) whose left edge is at ``x = obstacle_x_min``
    and whose x-range stays to the right of the segment, so the nearest approach is the horizontal
    gap from the origin to that left edge. Combined with the scenario robot radius this controls
    the sign of the clearance margin deterministically.
    """
    fake_map_def = SimpleNamespace(
        robot_routes=[SimpleNamespace(waypoints=[(-1.0, 0.0), (0.0, 0.0)])],
        obstacles=[
            SimpleNamespace(
                vertices=[
                    (obstacle_x_min, -1.0),
                    (obstacle_x_min + 1.0, -1.0),
                    (obstacle_x_min + 1.0, 1.0),
                    (obstacle_x_min, 1.0),
                ]
            )
        ],
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.convert_map",
        lambda _path: fake_map_def,
    )


def test_preflight_fails_closed_on_subradius_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A centerline closer than the robot radius must fail closed with RouteClearanceError."""
    # Obstacle edge at distance 0.5 from the centerline, robot radius 1.0 -> margin -0.5 (< 0).
    _patch_geometry(monkeypatch, obstacle_x_min=0.5)
    config_path = _write_campaign(tmp_path, robot_radius=1.0)
    cfg = load_campaign_config(config_path)

    with pytest.raises(RouteClearanceError) as excinfo:
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="fail")
    message = str(excinfo.value)
    assert "clearance_case" in message
    assert "geometrically impossible" in message


def test_preflight_fails_closed_even_with_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clearance certification must not excuse a negative (impossible) margin."""
    _patch_geometry(monkeypatch, obstacle_x_min=0.5)
    config_path = _write_campaign(tmp_path, robot_radius=1.0, certification=True)
    cfg = load_campaign_config(config_path)

    with pytest.raises(RouteClearanceError):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="fail-cert")


def test_preflight_passes_with_adequate_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route with margin above the warning threshold passes with no clearance warning."""
    # Obstacle edge at distance 2.0, robot radius 1.0 -> margin +1.0 (>= 0.5 warn threshold).
    _patch_geometry(monkeypatch, obstacle_x_min=2.0)
    config_path = _write_campaign(tmp_path, robot_radius=1.0)
    cfg = load_campaign_config(config_path)

    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="ok")
    payload = json.loads(Path(prepared["validate_config_path"]).read_text(encoding="utf-8"))
    assert payload["route_clearance_warning_count"] == 0
    assert payload["route_clearance_warnings"] == []


def test_preflight_warns_but_does_not_fail_on_narrow_positive_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A positive-but-narrow margin emits an informational warning without failing closed."""
    # Obstacle edge at distance 1.2, robot radius 1.0 -> margin +0.2 (>0 but < 0.5 warn threshold).
    _patch_geometry(monkeypatch, obstacle_x_min=1.2)
    config_path = _write_campaign(tmp_path, robot_radius=1.0)
    cfg = load_campaign_config(config_path)

    prepared = prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="warn")
    payload = json.loads(Path(prepared["validate_config_path"]).read_text(encoding="utf-8"))
    assert payload["route_clearance_warning_count"] == 1
    warning = payload["route_clearance_warnings"][0]
    assert warning["scenario"] == "clearance_case"
    assert warning["min_clearance_margin_m"] == pytest.approx(0.2)
    assert warning["min_clearance_margin_m"] > 0.0

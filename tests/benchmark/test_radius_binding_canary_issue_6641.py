"""Tests for the issue #6641 runtime radius-binding canary (benchmark 6600 Gate 1).

These tests cover the differential scan helper, each of the five binding-surface
probes (with controlled geometry so they are fast), the fail-closed negative
controls (a silently ignored radius must produce a no-go), the machine-readable
verdict schema, and one end-to-end run on the committed geometry-sensitive
``canary_corridor`` scenario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.radius_binding_canary import (
    CANARY_SURFACES,
    DEFAULT_SCENARIO_REL,
    RADIUS_BINDING_CANARY_SCHEMA,
    SURFACE_FEASIBILITY_ORACLE,
    SURFACE_METRIC_METADATA_ROWS,
    SURFACE_OBSTACLE_PEDESTRIAN_CONTACT,
    SURFACE_PLANNER_INPUTS,
    SURFACE_SIMULATOR_GEOMETRY,
    VERDICT_GO,
    VERDICT_NO_GO,
    CanaryGeometry,
    _scan_flip,
    canary_verdict_to_dict,
    load_canary_geometry,
    probe_feasibility_oracle,
    probe_metric_metadata_and_output_rows,
    probe_obstacle_pedestrian_contact,
    probe_planner_inputs,
    probe_simulator_collision_geometry,
    run_radius_binding_canary,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --- synthetic geometry fixture ---------------------------------------------


def _synthetic_geometry(wall_distance: float = 2.0) -> CanaryGeometry:
    """Build a small in-memory corridor geometry for fast unit tests.

    The robot anchors at (5, 5); a single wall segment at y = 5 + wall_distance
    spans the map so the nearest obstacle distance is exactly ``wall_distance``.
    """
    wall_y = 5.0 + wall_distance
    obstacle_lines = np.array([[0.0, wall_y, 10.0, wall_y]], dtype=float)
    return CanaryGeometry(
        scenario_id="synthetic-corridor",
        map_name="synthetic",
        route_point=(5.0, 5.0),
        goal_point=(9.0, 5.0),
        obstacle_lines_runtime=obstacle_lines,
        wall_distance_m=float(wall_distance),
        map_width=10.0,
        map_height=10.0,
        scenario={"name": "synthetic-corridor", "robot_config": {"radius": 0.3}},
        scenario_path=Path("synthetic"),
    )


# --- differential scan helper ------------------------------------------------


def test_scan_flip_finds_boundary_for_radius_responsive_predicate() -> None:
    """The scan locates the radius boundary of a radius-responsive predicate."""

    def predicate(radius: float) -> bool:
        return radius >= 1.7  # boundary at 1.7

    flip = _scan_flip(predicate, lo=0.0, hi=3.0, step=1e-3)
    assert flip is not None
    assert abs(flip - 1.7) <= 2e-3


def test_scan_flip_returns_none_when_predicate_never_flips() -> None:
    """A predicate that stays False yields no boundary."""

    def predicate(_radius: float) -> bool:
        return False

    assert _scan_flip(predicate, lo=0.0, hi=2.0, step=1e-3) is None


def test_scan_flip_returns_none_when_initially_true() -> None:
    """A predicate already True at the low end has no detectable boundary."""

    def predicate(_radius: float) -> bool:
        return True

    assert _scan_flip(predicate, lo=0.0, hi=2.0, step=1e-3) is None


# --- simulator collision geometry -------------------------------------------


def test_simulator_collision_geometry_passes_on_synthetic_corridor() -> None:
    """The runtime obstacle-collision flip tracks the wall distance."""
    geometry = _synthetic_geometry(wall_distance=2.0)
    verdict = probe_simulator_collision_geometry(geometry, scan_step_m=1e-3)
    assert verdict.surface == SURFACE_SIMULATOR_GEOMETRY
    assert verdict.status == "pass"
    assert abs(verdict.evidence["collision_flip_radius_m"] - 2.0) <= 5e-3


def test_simulator_collision_geometry_fails_when_radius_ignored() -> None:
    """A collision geometry that ignores the radius is flagged as no-go.

    This is the fail-closed negative control: replace the runtime component's
    obstacle lines with an empty set so no radius ever collides, then confirm the
    probe records a fail rather than a silent pass.
    """
    geometry = CanaryGeometry(
        scenario_id="empty",
        map_name="empty",
        route_point=(5.0, 5.0),
        goal_point=(9.0, 5.0),
        obstacle_lines_runtime=np.empty((0, 4), dtype=float),
        wall_distance_m=2.0,
        map_width=10.0,
        map_height=10.0,
        scenario={"name": "empty"},
        scenario_path=Path("empty"),
    )
    verdict = probe_simulator_collision_geometry(geometry, scan_step_m=1e-3)
    assert verdict.status == "fail"


# --- obstacle and pedestrian contact ----------------------------------------


def test_obstacle_pedestrian_contact_passes_for_responsive_radii() -> None:
    """Pedestrian contact flips at robot_radius + ped_radius for both pairs."""
    geometry = _synthetic_geometry(wall_distance=5.0)
    verdict = probe_obstacle_pedestrian_contact(
        geometry,
        selected_robot_radius_m=0.3,
        selected_ped_radius_m=0.4,
        scan_step_m=1e-3,
    )
    assert verdict.surface == SURFACE_OBSTACLE_PEDESTRIAN_CONTACT
    assert verdict.status == "pass"
    for pair in verdict.evidence["pairs"]:
        assert pair["pass"] is True
        assert abs(pair["contact_flip_distance_m"] - pair["expected_contact_distance_m"]) <= 5e-3


# --- feasibility oracle ------------------------------------------------------


def test_feasibility_oracle_probe_requires_real_scenario(scenario_geometry) -> None:
    """The oracle probe tracks the envelope radius on the real scenario geometry."""
    geometry: CanaryGeometry = scenario_geometry
    wall = geometry.wall_distance_m
    radius_a = 0.25 * wall
    radius_b = 0.6 * wall
    verdict = probe_feasibility_oracle(
        geometry,
        radius_a_m=radius_a,
        radius_b_m=radius_b,
    )
    assert verdict.surface == SURFACE_FEASIBILITY_ORACLE
    assert verdict.status == "pass"
    expected_delta = abs(radius_b - radius_a)
    observed_delta = abs(verdict.evidence["clearance_delta_m"])
    assert abs(observed_delta - expected_delta) <= max(5e-3, expected_delta * 1e-6)


def test_feasibility_oracle_probe_rejects_equal_radii() -> None:
    """Equal envelope radii cannot probe a delta and must fail closed."""
    geometry = _synthetic_geometry()
    verdict = probe_feasibility_oracle(geometry, radius_a_m=0.5, radius_b_m=0.5)
    assert verdict.status == "fail"


# --- metric metadata and output rows ----------------------------------------


def test_metric_metadata_probe_passes_for_responsive_metric() -> None:
    """human_collisions responds to the recorded radii and the resolver reads them."""
    verdict = probe_metric_metadata_and_output_rows(
        selected_robot_radius_m=0.3,
        selected_ped_radius_m=0.4,
    )
    assert verdict.surface == SURFACE_METRIC_METADATA_ROWS
    assert verdict.status == "pass"
    assert verdict.evidence["collisions_responsive_to_radius"] is True
    assert verdict.evidence["resolver_responsive"] is True


# --- planner inputs ----------------------------------------------------------


def test_planner_inputs_probe_passes_for_responsive_observation() -> None:
    """The runner observation builder carries the configured radii."""
    verdict = probe_planner_inputs(
        selected_robot_radius_m=0.3,
        selected_ped_radius_m=0.4,
    )
    assert verdict.surface == SURFACE_PLANNER_INPUTS
    assert verdict.status == "pass"
    assert verdict.evidence["robot_payload_carries_radius"] is True
    assert verdict.evidence["agent_payloads_carry_radius"] is True


# --- orchestrator and schema -------------------------------------------------


def test_canary_surfaces_are_the_five_required_binding_surfaces() -> None:
    """The canary covers exactly the five surfaces named in the issue contract."""
    assert set(CANARY_SURFACES) == {
        SURFACE_SIMULATOR_GEOMETRY,
        SURFACE_OBSTACLE_PEDESTRIAN_CONTACT,
        SURFACE_FEASIBILITY_ORACLE,
        SURFACE_METRIC_METADATA_ROWS,
        SURFACE_PLANNER_INPUTS,
    }


def test_verdict_dict_is_machine_readable_and_schema_tagged(scenario_geometry) -> None:
    """The serialized verdict is JSON-safe and carries the canary schema tag."""
    verdict = run_radius_binding_canary(geometry=scenario_geometry)
    payload = canary_verdict_to_dict(verdict)
    assert payload["schema_version"] == RADIUS_BINDING_CANARY_SCHEMA
    assert payload["verdict"] in {VERDICT_GO, VERDICT_NO_GO}
    assert payload["evidence_status"] == "diagnostic-only"
    assert [s["surface"] for s in payload["surfaces"]] == list(CANARY_SURFACES)
    # Must be JSON-serializable.
    import json

    json.dumps(payload)


@pytest.mark.timeout(120, method="thread")
def test_run_radius_binding_canary_go_on_committed_corridor_scenario() -> None:
    """End-to-end: the committed geometry-sensitive scenario yields a go verdict.

    This is the executable proof for the issue contract: the selected radius
    propagates to all five binding surfaces on a real geometry-sensitive scenario.
    """
    scenario_path = _REPO_ROOT / DEFAULT_SCENARIO_REL
    assert scenario_path.exists(), f"missing canary scenario {scenario_path}"
    verdict = run_radius_binding_canary(scenario_path=scenario_path)
    assert verdict.verdict == VERDICT_GO, (
        f"expected go verdict, got {verdict.verdict}; surfaces="
        f"{[(s.surface, s.status, s.observed) for s in verdict.surfaces]}"
    )
    assert len(verdict.surfaces) == 5
    assert all(s.status == "pass" for s in verdict.surfaces)
    # The geometry-sensitive scenario must have a real, finite wall distance.
    assert verdict.scenario["wall_distance_m"] > 0.0
    assert verdict.scenario["obstacle_segment_count"] > 0


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(scope="module")
def scenario_geometry() -> CanaryGeometry:
    """Load the committed canary corridor geometry once for the module."""
    return load_canary_geometry(_REPO_ROOT / DEFAULT_SCENARIO_REL)

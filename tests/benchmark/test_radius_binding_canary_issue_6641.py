"""Tests for the #6641 Gate 1 runtime radius-binding canary.

The canary proves that a declared robot collision-envelope radius propagates consistently
to the five binding surfaces (simulator collision geometry, obstacle/pedestrian contact
logic, feasibility/oracle, metric metadata and output rows, and planner inputs) on a
geometry-sensitive scenario, and that it fails closed when any binding diverges or cannot
be observed.

These tests run the real canary on the geometry-sensitive ``francis2023_narrow_doorway``
scenario (the narrow-doorway family referenced by #6600) and add negative controls that
simulate a silently divergent binding and an unobservable surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.radius_binding_canary import (
    BINDING_SURFACES,
    CAMPAIGN_ENVELOPE_RADII_M,
    CANARY_SCHEMA,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    probe_metric_metadata,
    probe_sim_collision_geometry,
    run_radius_binding_canary,
)
from robot_sf.scenario_certification.feasibility_oracle import make_envelope_scenario
from robot_sf.training.scenario_loader import load_scenarios
from scripts.benchmark.run_radius_binding_canary_issue_6641 import build_report

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml"


@pytest.fixture(scope="module")
def narrow_doorway_scenario() -> dict:
    """Load the geometry-sensitive narrow-doorway scenario used by the canary."""
    return dict(load_scenarios(_SCENARIO_PATH)[0])


def test_campaign_radii_match_issue_6600_treatment() -> None:
    """The canary's default radii match the #6600 fixed treatment (0.5, 0.8, 1.0 m)."""
    assert CAMPAIGN_ENVELOPE_RADII_M == (0.5, 0.8, 1.0)


def test_binding_surfaces_are_the_five_gate1_surfaces() -> None:
    """The canary covers exactly the five Gate 1 binding surfaces."""
    assert BINDING_SURFACES == (
        "simulator_collision_geometry",
        "obstacle_pedestrian_contact_logic",
        "feasibility_oracle",
        "metric_metadata_and_output_rows",
        "planner_inputs",
    )


@pytest.mark.parametrize("radius", CAMPAIGN_ENVELOPE_RADII_M)
def test_canary_go_on_narrow_doorway(narrow_doorway_scenario: dict, radius: float) -> None:
    """Every binding surface binds the declared radius on the geometry-sensitive scenario."""
    verdict = run_radius_binding_canary(
        narrow_doorway_scenario, radius, scenario_path=_SCENARIO_PATH
    )

    assert verdict.go is True
    assert verdict.schema == CANARY_SCHEMA
    assert verdict.target_radius_m == radius
    assert verdict.claim_boundary == DIAGNOSTIC_CLAIM_BOUNDARY
    assert [surface.surface for surface in verdict.surfaces] == list(BINDING_SURFACES)
    for surface in verdict.surfaces:
        assert surface.bound is True, surface.note
        assert surface.observed_radius_m == pytest.approx(radius)
        assert surface.note == ""

    by_surface = {surface.surface: surface for surface in verdict.surfaces}
    assert by_surface["simulator_collision_geometry"].evidence["runtime_component"] == (
        "Simulator.robots[0].config.radius"
    )
    assert by_surface["obstacle_pedestrian_contact_logic"].evidence["runtime_component"] == (
        "ContinuousOccupancy.agent_radius/ped_radius + "
        "is_obstacle_collision/is_pedestrian_collision"
    )
    contact_evidence = by_surface["obstacle_pedestrian_contact_logic"].evidence
    assert contact_evidence["runtime_obstacle_boundary_inside"] is True
    assert contact_evidence["runtime_obstacle_boundary_outside"] is False
    assert contact_evidence["runtime_pedestrian_boundary_inside"] is True
    assert contact_evidence["runtime_pedestrian_boundary_outside"] is False
    assert by_surface["planner_inputs"].evidence["runtime_component"] == (
        "Simulator.pysf_sim.forces[].config.robot_radius"
    )


def test_canary_verdict_serializes_to_json(narrow_doorway_scenario: dict) -> None:
    """The verdict is machine-readable (JSON-serializable) per the stop condition."""
    from robot_sf.benchmark.radius_binding_canary import canary_verdict_to_dict

    verdict = run_radius_binding_canary(narrow_doorway_scenario, 0.5, scenario_path=_SCENARIO_PATH)
    payload = canary_verdict_to_dict(verdict)

    rendered = json.dumps(payload, sort_keys=True)
    roundtrip = json.loads(rendered)
    assert roundtrip["schema"] == CANARY_SCHEMA
    assert roundtrip["go"] is True
    assert {surface["surface"] for surface in roundtrip["surfaces"]} == set(BINDING_SURFACES)
    assert all(surface["bound"] for surface in roundtrip["surfaces"])


def test_canary_oracle_surface_records_geometric_envelope(narrow_doorway_scenario: dict) -> None:
    """The oracle surface evidence carries the envelope radius and diameter binding."""
    verdict = run_radius_binding_canary(narrow_doorway_scenario, 0.5, scenario_path=_SCENARIO_PATH)
    oracle = next(
        surface for surface in verdict.surfaces if surface.surface == "feasibility_oracle"
    )

    assert oracle.evidence["geometric_envelope_radius_m"] == pytest.approx(0.5)
    assert oracle.evidence["geometric_envelope_diameter_m"] == pytest.approx(1.0)
    assert oracle.evidence["injected_robot_config_radius_m"] == pytest.approx(0.5)


def test_probe_fail_closed_when_target_diverges_from_declared(
    narrow_doorway_scenario: dict,
) -> None:
    """A probe reports no-go when the claimed target diverges from the bound radius."""
    declared = make_envelope_scenario(narrow_doorway_scenario, envelope_radius_m=0.5)

    verdict = probe_sim_collision_geometry(
        declared,
        1.0,
        scenario_path=_SCENARIO_PATH,  # target differs from declared 0.5
    )

    assert verdict.bound is False
    assert verdict.observed_radius_m == pytest.approx(0.5)
    assert verdict.note


def test_canary_fail_closed_on_silently_divergent_metric_surface(
    narrow_doorway_scenario: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently divergent metric-metadata binding stops the canary with a no-go.

    Simulates the failure mode the canary exists to catch: the runner's output-row radius
    extraction returns a stale default instead of the declared ``robot_config.radius``.
    """
    from robot_sf.benchmark import runner

    monkeypatch.setattr(runner, "_scenario_robot_radius_m", lambda _scenario: 0.3)

    verdict = run_radius_binding_canary(narrow_doorway_scenario, 0.5, scenario_path=_SCENARIO_PATH)

    assert verdict.go is False
    metric = next(
        surface
        for surface in verdict.surfaces
        if surface.surface == "metric_metadata_and_output_rows"
    )
    assert metric.bound is False
    assert metric.observed_radius_m == pytest.approx(0.3)
    # The other four surfaces still bind; only the divergent surface is no-go.
    assert sum(1 for surface in verdict.surfaces if not surface.bound) == 1


def test_probe_fail_closed_when_surface_unobservable() -> None:
    """A probe fails closed (no-go) when the robot config cannot be built."""
    bogus_scenario = {"name": "bogus", "robot_config": {"type": "not_a_real_robot"}}
    declared = make_envelope_scenario(bogus_scenario, envelope_radius_m=0.5)

    verdict = probe_sim_collision_geometry(declared, 0.5, scenario_path=_SCENARIO_PATH)

    assert verdict.bound is False
    assert verdict.observed_radius_m is None
    assert "error" in verdict.evidence
    assert verdict.note


def test_canary_fail_closed_when_shared_config_is_unobservable() -> None:
    """A shared setup failure emits five no-go surfaces instead of a usage exception."""
    bogus_scenario = {"name": "bogus", "robot_config": {"type": "not_a_real_robot"}}

    verdict = run_radius_binding_canary(bogus_scenario, 0.5, scenario_path=_SCENARIO_PATH)

    assert verdict.go is False
    assert len(verdict.surfaces) == len(BINDING_SURFACES)
    assert all(surface.bound is False for surface in verdict.surfaces)
    assert all("error" in surface.evidence for surface in verdict.surfaces)


def test_metric_probe_binds_runner_and_orchestrator_paths(narrow_doorway_scenario: dict) -> None:
    """The metric surface verifies both the runner row and orchestrator metric bindings."""
    declared = make_envelope_scenario(narrow_doorway_scenario, envelope_radius_m=0.8)

    verdict = probe_metric_metadata(declared, 0.8, scenario_path=_SCENARIO_PATH)

    assert verdict.bound is True
    assert verdict.evidence["runner_row_robot_radius_m"] == pytest.approx(0.8)
    assert verdict.evidence["episode_data_robot_radius_m"] == pytest.approx(0.8)
    assert verdict.evidence["simulation_config_robot_radius_m"] == pytest.approx(0.8)


def test_canary_rejects_non_positive_target(narrow_doorway_scenario: dict) -> None:
    """A non-positive target radius is rejected before any probe runs."""
    with pytest.raises(ValueError, match="target_radius_m"):
        run_radius_binding_canary(narrow_doorway_scenario, 0.0, scenario_path=_SCENARIO_PATH)


def test_report_rejects_empty_radius_probe(narrow_doorway_scenario: dict) -> None:
    """An empty radius list cannot produce a fail-open go verdict."""
    with pytest.raises(ValueError, match="at least one target radius"):
        build_report(
            narrow_doorway_scenario,
            scenario_path=_SCENARIO_PATH,
            radii=[],
            tolerance=1e-9,
        )

"""Tests for the feasibility-oracle surface of the #6641 radius-binding canary.

The Gate 1 canary (``robot_sf/benchmark/radius_binding_canary.py``) proves the declared
collision-envelope radius propagates to the feasibility/oracle surface: the oracle's
scenario injection (``make_envelope_scenario`` -> ``robot_config.radius``) and the
planner-free geometric inflation (``envelope_radius_m`` / ``envelope_diameter_m``).

These tests exercise that surface with a deterministic stub certifier for unit coverage
and one real end-to-end run on the geometry-sensitive ``francis2023_narrow_doorway``
scenario for integration coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.radius_binding_canary import (
    probe_feasibility_oracle,
    run_radius_binding_canary,
)
from robot_sf.scenario_certification.feasibility_oracle import make_envelope_scenario
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    VALID,
    RouteCertificate,
    ScenarioCertificate,
)
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml"


def _stub_certificate(*, minimum_static_clearance_m: float = 0.5) -> ScenarioCertificate:
    """Build a scenario certificate with a controlled route clearance."""
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id="francis2023_narrow_doorway",
        source="fixture",
        classification=VALID,
        benchmark_eligibility="eligible",
        reasons=[],
        checks={"route_count": 1},
        route_certificates=[
            RouteCertificate(
                route_id="route_0",
                spawn_id=0,
                goal_id=0,
                classification=VALID,
                benchmark_eligibility="eligible",
                reasons=[],
                checks={
                    "robot_radius_m": 1.0,
                    "minimum_static_clearance_m": minimum_static_clearance_m,
                    "shortest_path_length_m": 20.0,
                    "inflated_collision_free_path": True,
                },
            )
        ],
    )


def _stub_certifier(certificate: ScenarioCertificate):
    """Return a certifier callable that ignores inputs and returns the stub certificate."""

    def _certify(_scenario: Any, _scenario_path: Path) -> ScenarioCertificate:
        return certificate

    return _certify


@pytest.fixture(scope="module")
def narrow_doorway_scenario() -> dict:
    """Load the geometry-sensitive narrow-doorway scenario."""
    return dict(load_scenarios(_SCENARIO_PATH)[0])


@pytest.mark.parametrize("radius", [0.5, 0.8, 1.0])
def test_oracle_probe_binds_envelope_radius_with_stub_certifier(
    narrow_doorway_scenario: dict, radius: float
) -> None:
    """The oracle surface binds the declared envelope radius and inflates the diameter."""
    declared = make_envelope_scenario(narrow_doorway_scenario, envelope_radius_m=radius)

    verdict = probe_feasibility_oracle(
        declared,
        radius,
        scenario_path=_SCENARIO_PATH,
        certifier=_stub_certifier(_stub_certificate()),
    )

    assert verdict.bound is True, verdict.note
    assert verdict.observed_radius_m == pytest.approx(radius)
    assert verdict.evidence["geometric_envelope_diameter_m"] == pytest.approx(2.0 * radius)
    assert verdict.evidence["injected_robot_config_radius_m"] == pytest.approx(radius)
    assert verdict.evidence["verdict_envelope_radius_m"] == pytest.approx(radius)


def test_oracle_probe_fail_closed_on_divergent_envelope(narrow_doorway_scenario: dict) -> None:
    """The oracle surface reports no-go when the target diverges from the injected radius."""
    declared = make_envelope_scenario(narrow_doorway_scenario, envelope_radius_m=0.5)

    verdict = probe_feasibility_oracle(
        declared,
        1.0,  # target differs from the injected 0.5 m envelope
        scenario_path=_SCENARIO_PATH,
        certifier=_stub_certifier(_stub_certificate()),
    )

    assert verdict.bound is False
    assert verdict.observed_radius_m == pytest.approx(0.5)
    assert verdict.note


def test_oracle_probe_binds_on_real_narrow_doorway_map(narrow_doorway_scenario: dict) -> None:
    """Integration: the real certifier binds the envelope radius on the narrow-doorway map."""
    declared = make_envelope_scenario(narrow_doorway_scenario, envelope_radius_m=0.5)

    # certifier=None -> canonical certifier runs the real planner-free geometric margin.
    verdict = probe_feasibility_oracle(declared, 0.5, scenario_path=_SCENARIO_PATH)

    assert verdict.bound is True, verdict.note
    assert verdict.observed_radius_m == pytest.approx(0.5)
    assert verdict.evidence["geometric_envelope_diameter_m"] == pytest.approx(1.0)
    # The narrow-doorway geometry is radius-sensitive: the corridor margin is reported.
    assert "min_corridor_width_m" in verdict.evidence


def test_canary_go_includes_oracle_surface_on_real_map(narrow_doorway_scenario: dict) -> None:
    """The whole canary is go on the narrow-doorway scenario including the oracle surface."""
    verdict = run_radius_binding_canary(narrow_doorway_scenario, 0.5, scenario_path=_SCENARIO_PATH)

    assert verdict.go is True
    oracle = next(
        surface for surface in verdict.surfaces if surface.surface == "feasibility_oracle"
    )
    assert oracle.bound is True

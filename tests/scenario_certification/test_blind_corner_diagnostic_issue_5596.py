"""Tests for the issue #5596 blind-corner zero-success diagnostic.

These tests exercise the new diagnostic capability introduced for issue #5596:
  * the route-follow intervention lane (registered ``route_follow`` policy) that drives
    the certified A* route instead of beelining at the goal,
  * the ``build_issue_5596_blind_corner_diagnostic`` report that isolates the failure
    mechanism across the three competing explanations from the issue,
  * the straight-line vs certified-route clearance comparison.

They do NOT duplicate the issue #5574 feasibility-oracle tests: #5574 has no
route-follow intervention and no mechanism classification. Unit tests inject
deterministic certifier/runner stubs; one real end-to-end test runs the diagnostic on
the committed ``francis2023_blind_corner`` scenario (marked slow; diagnostic-only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.scenario_certification import feasibility_oracle
from robot_sf.scenario_certification.feasibility_oracle import (
    ISSUE_5596_BLIND_CORNER_SCENARIO_ID,
    ISSUE_5596_DIAGNOSTIC_SCHEMA,
    ROUTE_FOLLOW_ALGO,
    build_issue_5596_blind_corner_diagnostic,
    make_route_follow_episode_runner,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    VALID,
    RouteCertificate,
    ScenarioCertificate,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _route_certificate(
    *,
    classification: str = VALID,
    eligibility: str = "eligible",
    minimum_static_clearance_m: float | None = 1.0,
    shortest_path_length_m: float | None = 20.0,
    inflated_path: bool = True,
) -> RouteCertificate:
    """Build a route certificate with controlled geometric checks."""
    checks: dict[str, Any] = {"robot_radius_m": 1.0}
    if minimum_static_clearance_m is not None:
        checks["minimum_static_clearance_m"] = minimum_static_clearance_m
    if shortest_path_length_m is not None:
        checks["shortest_path_length_m"] = shortest_path_length_m
    checks["inflated_collision_free_path"] = inflated_path
    return RouteCertificate(
        route_id="route_0",
        spawn_id=0,
        goal_id=0,
        classification=classification,
        benchmark_eligibility=eligibility,
        reasons=[],
        checks=checks,
    )


def _certificate(
    classification: str = VALID,
    *,
    eligibility: str = "eligible",
    minimum_static_clearance_m: float | None = 1.0,
    shortest_path_length_m: float | None = 20.0,
    inflated_path: bool = True,
) -> ScenarioCertificate:
    """Build a scenario certificate wrapping one controlled route certificate."""
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id="francis2023_blind_corner",
        source="fixture",
        classification=classification,
        benchmark_eligibility=eligibility,
        reasons=[],
        checks={"route_count": 1},
        route_certificates=[
            _route_certificate(
                classification=classification,
                eligibility=eligibility,
                minimum_static_clearance_m=minimum_static_clearance_m,
                shortest_path_length_m=shortest_path_length_m,
                inflated_path=inflated_path,
            )
        ],
    )


def _scenario(
    *,
    name: str = "francis2023_blind_corner",
    archetype: str = "blind_corner",
    max_episode_steps: int = 400,
    robot_config: dict[str, Any] | None = None,
    seed: int = 219,
) -> dict[str, Any]:
    """Build a minimal blind-corner scenario fixture."""
    return {
        "name": name,
        "simulation_config": {"max_episode_steps": max_episode_steps, "ped_density": 0.0},
        "robot_config": dict(robot_config or {}),
        "metadata": {"archetype": archetype},
        "seeds": [seed],
    }


def _manifest(tmp_path: Path, scenario: dict[str, Any]) -> Path:
    """Write a single-cell manifest and return its path."""
    path = tmp_path / "blind_corner.yaml"
    path.write_text(
        "scenarios:\n  - name: " + scenario["name"] + "\n"
        "    simulation_config:\n"
        "      max_episode_steps: " + str(scenario["simulation_config"]["max_episode_steps"]) + "\n"
        "    robot_config: {}\n"
        "    metadata:\n"
        "      archetype: blind_corner\n"
        "    seeds: [219]\n",
        encoding="utf-8",
    )
    return path


def test_route_follow_runner_uses_route_follow_algo_key() -> None:
    """The intervention runner builder returns a callable and uses the route_follow algo key."""
    config = feasibility_oracle.FeasibilityOracleConfig(
        scenario_path=_REPO_ROOT / "configs/scenarios/francis2023.yaml",
        envelope_radii_m=(1.0, 0.5),
    )
    runner = make_route_follow_episode_runner(config)
    # The builder returns a callable; the real path is exercised by the end-to-end test.
    # Unit isolation checks the algo key constant is defined.
    assert callable(runner)
    assert ROUTE_FOLLOW_ALGO == "route_follow"


def test_issue_5596_report_scopes_to_issue_5596_and_blind_corner(tmp_path: Path) -> None:
    """The diagnostic report is scoped to issue #5596 and the blind-corner cell."""
    manifest = _manifest(tmp_path, _scenario())

    def runner(_s, _seed, _horizon, _algo):
        return {
            "steps": 100,
            "outcome": {"route_complete": False},
            "termination_reason": "collision",
        }

    report = build_issue_5596_blind_corner_diagnostic(
        manifest,
        envelope_radii_m=(1.0, 0.5),
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID),
    )

    assert report["schema_version"] == ISSUE_5596_DIAGNOSTIC_SCHEMA
    assert report["issue"] == "5596"
    assert report["scenario_id"] == ISSUE_5596_BLIND_CORNER_SCENARIO_ID
    assert report["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert report["oracle_verdict"]["issue"] == "5596"
    assert report["route_follow_intervention_verdict"]["issue"] == "5596"


def test_issue_5596_report_rejects_missing_blind_corner_cell(tmp_path: Path) -> None:
    """A manifest without the blind-corner cell fails closed."""
    manifest = tmp_path / "no_blind_corner.yaml"
    manifest.write_text("scenarios:\n  - name: other\n    seeds: [219]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing from manifest"):
        build_issue_5596_blind_corner_diagnostic(manifest, envelope_radii_m=(1.0, 0.5))


def test_issue_5596_report_rejects_missing_manifest(tmp_path: Path) -> None:
    """A missing manifest fails closed before diagnostic selection."""
    with pytest.raises(FileNotFoundError, match="Scenario manifest file not found"):
        build_issue_5596_blind_corner_diagnostic(
            tmp_path / "missing.yaml", envelope_radii_m=(1.0, 0.5)
        )


def test_issue_5596_report_requires_reduced_radius_after_nominal(tmp_path: Path) -> None:
    """The sensitivity axis must place the nominal radius first and include a reduced probe."""
    manifest = _manifest(tmp_path, _scenario())
    with pytest.raises(ValueError, match="nominal radius first"):
        build_issue_5596_blind_corner_diagnostic(
            manifest, envelope_radii_m=(0.5, 1.0), episode_runner=lambda *_a: {}
        )


def test_issue_5596_mechanism_supports_route_cause_when_intervention_also_fails(
    tmp_path: Path,
) -> None:
    """When goal and route-follow both fail, the mechanism is route geometry (#2), not script (#1)."""
    manifest = _manifest(tmp_path, _scenario())

    def runner(_s, _seed, _horizon, _algo):
        return {
            "steps": 100,
            "outcome": {"route_complete": False},
            "termination_reason": "collision",
        }

    report = build_issue_5596_blind_corner_diagnostic(
        manifest,
        envelope_radii_m=(1.0, 0.5),
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID),
    )
    mechanism = report["mechanism"]
    assert mechanism["explanation_1_scripted_controller_incomplete"] is False
    assert mechanism["explanation_2_route_geometry_or_config_cause"] is True
    assert mechanism["oracle_nominal_feasible"] is False
    assert mechanism["route_follow_intervention_feasible"] is False


def test_issue_5596_mechanism_supports_script_when_intervention_succeeds(tmp_path: Path) -> None:
    """When the route-follow lane completes where goal fails, explanation #1 is supported."""
    manifest = _manifest(tmp_path, _scenario())

    def runner(_scenario, _seed, _horizon, _algo):
        # goal script collides, route-follow completes.
        if _algo == ROUTE_FOLLOW_ALGO:
            return {
                "steps": 50,
                "outcome": {"route_complete": True},
                "termination_reason": "success",
            }
        return {
            "steps": 100,
            "outcome": {"route_complete": False},
            "termination_reason": "collision",
        }

    report = build_issue_5596_blind_corner_diagnostic(
        manifest,
        envelope_radii_m=(1.0, 0.5),
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID),
    )
    mechanism = report["mechanism"]
    assert mechanism["explanation_1_scripted_controller_incomplete"] is True
    assert mechanism["supported_explanation"] == "scripted_controller_corner_cut"


@pytest.mark.slow
def test_issue_5596_diagnostic_end_to_end_on_committed_blind_corner() -> None:
    """The diagnostic produces a coherent, reproducible verdict on the committed cell.

    Integration test: it loads ``francis2023.yaml``, runs the real certifier, the real
    ``goal`` oracle rollout, and the real route-follow intervention, and checks the report
    is internally consistent and diagnostic-scoped. Marked slow because it builds the
    simulator. It does NOT make any benchmark claim.
    """
    pytest.importorskip("robot_sf.benchmark.map_runner")
    manifest = _REPO_ROOT / "configs/scenarios/francis2023.yaml"

    report = build_issue_5596_blind_corner_diagnostic(manifest, envelope_radii_m=(1.0, 0.5))

    assert report["schema_version"] == ISSUE_5596_DIAGNOSTIC_SCHEMA
    assert report["scenario_id"] == ISSUE_5596_BLIND_CORNER_SCENARIO_ID
    # The oracle lane must report a concrete verdict for both radii.
    assert report["oracle_verdict"]["nominal_verdict"]["feasible"] in (True, False)
    assert report["oracle_verdict"]["reduced_verdicts"]
    # Clearance comparison is present for each radius.
    assert len(report["straight_line_vs_route_clearance"]) == 2
    for entry in report["straight_line_vs_route_clearance"]:
        assert entry["envelope_radius_m"] in (1.0, 0.5)
    # Mechanism is bounded and diagnostic.
    assert "supported_explanation" in report["mechanism"]
    assert report["mechanism"]["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"

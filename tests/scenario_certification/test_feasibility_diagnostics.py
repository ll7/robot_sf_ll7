"""Tests for issue #3484 executable feasibility diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.scenario_certification.failure_cause import (
    DYNAMIC_BLOCKING_OR_DEADLOCK,
    PLANNER_LIMITED,
)
from robot_sf.scenario_certification.feasibility_diagnostics import (
    DIAGNOSTIC_CLAIM_BOUNDARY,
    FEASIBILITY_DIAGNOSTICS_SCHEMA,
    FeasibilityDiagnosticConfig,
    make_actor_free_scenario,
    make_extended_time_scenario,
    route_clearance_lane,
    run_feasibility_diagnostics,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    GEOMETRICALLY_INFEASIBLE,
    VALID,
    RouteCertificate,
    ScenarioCertificate,
)


def _certificate(classification: str, eligibility: str = "eligible") -> ScenarioCertificate:
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id="classic_bottleneck_low",
        source="fixture",
        classification=classification,
        benchmark_eligibility=eligibility,
        reasons=[],
        checks={},
        route_certificates=[
            RouteCertificate(
                route_id="route_0",
                spawn_id=0,
                goal_id=0,
                classification=classification,
                benchmark_eligibility=eligibility,
                reasons=[],
                checks={},
            )
        ],
    )


def test_route_clearance_maps_geometric_infeasibility_to_false() -> None:
    """No inflated collision-free path must fail the route-clearance lane."""

    lane = route_clearance_lane(
        {"name": "classic_bottleneck_low", "metadata": {"archetype": "bottleneck"}},
        scenario_path=Path("configs/scenarios/classic_interactions.yaml"),
        certifier=lambda _scenario, _path: _certificate(GEOMETRICALLY_INFEASIBLE, "excluded"),
    )

    assert lane.passed is False
    assert lane.status == "failed"
    assert lane.evidence["classification"] == GEOMETRICALLY_INFEASIBLE


def test_route_clearance_maps_valid_certificate_to_true() -> None:
    """Valid route certificates become positive route-feasibility evidence."""

    lane = route_clearance_lane(
        {"name": "classic_bottleneck_low", "metadata": {"archetype": "bottleneck"}},
        scenario_path=Path("configs/scenarios/classic_interactions.yaml"),
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    assert lane.passed is True
    assert lane.status == "passed"


def test_actor_free_scenario_mutation_removes_pedestrians_preserves_route_fields() -> None:
    """Actor-free variants remove dynamic pedestrians without editing route identity."""

    scenario = {
        "name": "classic_cross_trap_high",
        "map_id": "classic_cross_trap",
        "simulation_config": {
            "max_episode_steps": 600,
            "ped_density": 0.08,
            "single_pedestrians": [{"spawn": [0, 0], "goal": [1, 1]}],
            "pedestrian_flows": [{"spawn_zone": 0}],
        },
        "metadata": {"archetype": "cross_trap"},
        "seeds": [101, 102],
    }

    mutated = make_actor_free_scenario(scenario)

    assert mutated["name"] == "classic_cross_trap_high"
    assert mutated["map_id"] == "classic_cross_trap"
    assert mutated["simulation_config"]["ped_density"] == 0.0
    assert "single_pedestrians" not in mutated["simulation_config"]
    assert "pedestrian_flows" not in mutated["simulation_config"]
    assert mutated["single_pedestrians"] == []
    assert mutated["social_groups"] == []
    assert mutated["metadata"]["diagnostic_variant"] == "actor_free"
    assert mutated["metadata"]["diagnostic_claim_boundary"] == DIAGNOSTIC_CLAIM_BOUNDARY


def test_extended_time_mutation_changes_only_horizon_and_metadata() -> None:
    """Extended-time variants multiply horizon in memory and record provenance."""

    scenario = {
        "name": "classic_head_on_corridor_low",
        "simulation_config": {"max_episode_steps": 500, "ped_density": 0.02},
        "metadata": {"archetype": "head_on_corridor"},
    }

    mutated, horizon = make_extended_time_scenario(scenario, multiplier=2.0)

    assert horizon == 1000
    assert mutated["simulation_config"]["max_episode_steps"] == 1000
    assert mutated["simulation_config"]["ped_density"] == 0.02
    assert mutated["metadata"]["diagnostic_variant"] == "extended_time"
    assert scenario["simulation_config"]["max_episode_steps"] == 500


def test_report_aggregates_three_diagnostic_lanes_and_keeps_verdict_fail_closed(
    monkeypatch,
) -> None:
    """Three-lane report can classify planner limitation when scripted rollout succeeds."""

    scenarios = [
        {
            "name": "classic_bottleneck_low",
            "simulation_config": {"max_episode_steps": 500, "ped_density": 0.02},
            "metadata": {"archetype": "bottleneck"},
            "seeds": [131],
        }
    ]
    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: scenarios,
    )

    def runner(
        _scenario: dict[str, Any], seed: int, horizon: int | None, algo: str
    ) -> dict[str, Any]:
        return {"success": True, "seed": seed, "termination_reason": "success", "algo": algo}

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(scenario_path=Path("fixture.yaml"), families=("bottleneck",)),
        episode_runner=runner,
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    assert report["schema_version"] == FEASIBILITY_DIAGNOSTICS_SCHEMA
    assert report["claim_boundary"] == DIAGNOSTIC_CLAIM_BOUNDARY
    assert report["scenario_rows"][0]["route_feasible"]["passed"] is True
    assert report["scenario_rows"][0]["actor_free_solved"]["passed"] is True
    assert report["scenario_rows"][0]["oracle_solved"]["passed"] is True
    assert report["scenario_rows"][0]["extended_time_solved"]["passed"] is None
    assert report["family_verdicts"][0]["failure_cause_verdict"]["cause"] == PLANNER_LIMITED


def test_report_can_include_extended_time_for_time_limited_verdict(
    monkeypatch,
) -> None:
    """Optional extended-time success maps to the established time-limited verdict."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_bottleneck_low",
                "simulation_config": {"max_episode_steps": 500, "ped_density": 0.02},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [131],
            }
        ],
    )

    def runner(
        _scenario: dict[str, Any], _seed: int, _horizon: int | None, _algo: str
    ) -> dict[str, Any]:
        return {"success": True, "termination_reason": "success"}

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("bottleneck",),
            run_extended_time=True,
        ),
        episode_runner=runner,
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    assert report["family_verdicts"][0]["failure_cause_verdict"]["cause"] == "time_limited"
    assert report["family_verdicts"][0]["failure_cause_verdict"]["comparable_for_ranking"] is False


def test_extended_time_blocks_when_seed_missing(monkeypatch) -> None:
    """Requested extended-time lane reports missing seed instead of not-run."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_bottleneck_low",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
            }
        ],
    )

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("bottleneck",),
            run_extended_time=True,
        ),
        episode_runner=lambda *_args, **_kwargs: {"success": True},
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    extended = report["scenario_rows"][0]["extended_time_solved"]
    assert extended["status"] == "blocked"
    assert extended["blocker"] == "scenario_has_no_seed"


def test_multi_seed_request_fails_closed(monkeypatch) -> None:
    """This diagnostic slice rejects unsupported multi-seed configuration."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_bottleneck_low",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [1, 2],
            }
        ],
    )

    with pytest.raises(ValueError, match="seeds_per_scenario"):
        run_feasibility_diagnostics(
            FeasibilityDiagnosticConfig(
                scenario_path=Path("fixture.yaml"),
                families=("bottleneck",),
                seeds_per_scenario=2,
            ),
            episode_runner=lambda *_args, **_kwargs: {"success": True},
            certifier=lambda _scenario, _path: _certificate(VALID),
        )


def test_metrics_false_classifies_lane_failure(monkeypatch) -> None:
    """Explicit false metric fields count as observed rollout failure."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_bottleneck_low",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [1],
            }
        ],
    )

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("bottleneck",),
            run_actor_free=False,
            run_oracle=True,
        ),
        episode_runner=lambda *_args, **_kwargs: {"metrics": {"success": False}},
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    assert report["scenario_rows"][0]["oracle_solved"]["passed"] is False
    assert report["scenario_rows"][0]["oracle_solved"]["evidence"]["success_reason"] == (
        "metrics_success_false"
    )


def test_family_verdict_false_if_any_scenario_lane_fails(monkeypatch) -> None:
    """Family aggregation is conservative when one scenario lane fails."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_cross_trap_medium",
                "simulation_config": {"max_episode_steps": 600},
                "metadata": {"archetype": "cross_trap"},
                "seeds": [101],
            }
        ],
    )

    def runner(
        scenario: dict[str, Any], _seed: int, _horizon: int | None, _algo: str
    ) -> dict[str, Any]:
        if scenario.get("metadata", {}).get("diagnostic_variant") == "actor_free":
            return {"success": True, "termination_reason": "success"}
        return {"success": False, "termination_reason": "timeout"}

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("cross_trap",),
            run_extended_time=True,
        ),
        episode_runner=runner,
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    verdict = report["family_verdicts"][0]["failure_cause_verdict"]
    assert verdict["cause"] == DYNAMIC_BLOCKING_OR_DEADLOCK
    assert verdict["comparable_for_ranking"] is False


def test_report_summarizes_difficulty_ramp_first_failure(monkeypatch) -> None:
    """Difficulty ramp reports ordered observed scenario variants only."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_bottleneck_high",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [1],
            },
            {
                "name": "classic_bottleneck_low",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [1],
            },
            {
                "name": "classic_bottleneck_medium",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "bottleneck"},
                "seeds": [1],
            },
        ],
    )

    def runner(
        scenario: dict[str, Any], _seed: int, _horizon: int | None, _algo: str
    ) -> dict[str, Any]:
        if scenario["name"].endswith("_low"):
            return {"success": True, "termination_reason": "success"}
        return {"success": False, "termination_reason": "timeout"}

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("bottleneck",),
            run_extended_time=True,
        ),
        episode_runner=runner,
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    ramp = report["difficulty_ramp"][0]
    assert ramp["family_id"] == "bottleneck"
    assert ramp["claim_boundary"] == DIAGNOSTIC_CLAIM_BOUNDARY
    assert [level["difficulty_level"] for level in ramp["levels"]] == [
        "low",
        "medium",
        "high",
    ]
    assert ramp["first_actor_free_failure_level"] == "medium"
    assert ramp["first_oracle_failure_level"] == "medium"


def test_report_uses_metadata_difficulty_before_name_suffix(monkeypatch) -> None:
    """Difficulty labels prefer explicit scenario metadata when present."""

    monkeypatch.setattr(
        "robot_sf.scenario_certification.feasibility_diagnostics.load_scenarios",
        lambda _path: [
            {
                "name": "classic_head_on_corridor_low",
                "simulation_config": {"max_episode_steps": 500},
                "metadata": {"archetype": "head_on_corridor", "difficulty": "custom"},
                "seeds": [1],
            }
        ],
    )

    report = run_feasibility_diagnostics(
        FeasibilityDiagnosticConfig(
            scenario_path=Path("fixture.yaml"),
            families=("head_on_corridor",),
        ),
        episode_runner=lambda *_args, **_kwargs: {"success": True},
        certifier=lambda _scenario, _path: _certificate(VALID),
    )

    assert report["scenario_rows"][0]["difficulty_level"] == "custom"
    assert report["difficulty_ramp"][0]["levels"][0]["difficulty_level"] == "custom"

"""Tests for adversarial route generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.nav.adversarial_route_generation import (
    AdversarialRouteGenerationConfig,
    CandidateRouteSet,
    OptimizationResult,
    evaluate_route_set,
    generate_candidate_route_set,
    optimize_route_set,
    write_route_override_artifact,
)
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig, PlanningError

if TYPE_CHECKING:
    from pathlib import Path


def test_generate_candidate_routes_stay_in_map_bounds(
    test_map: MapDefinition,
    test_planner: ClassicGlobalPlanner,
) -> None:
    """Generated route waypoints should remain inside map bounds."""
    config = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        trial_count=2,
        robot_route_count=1,
        ped_route_count=1,
        seed=17,
    )
    candidate, reason = generate_candidate_route_set(test_map, test_planner, config)
    assert reason is None
    assert candidate is not None
    for route in [*candidate.robot_routes, *candidate.ped_routes]:
        for x, y in route.waypoints:
            assert 0.0 <= x <= test_map.width
            assert 0.0 <= y <= test_map.height


def test_generate_candidate_rejects_invalid_points_with_feasibility_filter(
    test_map: MapDefinition,
    test_planner: ClassicGlobalPlanner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feasibility filter should reject candidates with invalid start/goal points."""
    config = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        trial_count=2,
        robot_route_count=1,
        ped_route_count=0,
    )

    def _raise_invalid(_point, grid=None):  # type: ignore[no-untyped-def]
        raise PlanningError("point invalid")

    monkeypatch.setattr(test_planner, "validate_point", _raise_invalid)
    candidate, reason = generate_candidate_route_set(test_map, test_planner, config)
    assert candidate is None
    assert reason == "invalid_start_or_goal"


def test_generate_candidate_raises_invalid_points_when_feasibility_filter_disabled(
    test_map: MapDefinition,
    test_planner: ClassicGlobalPlanner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation errors should propagate when feasibility filtering is disabled."""
    config = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        trial_count=2,
        robot_route_count=1,
        ped_route_count=0,
        feasibility_filter=False,
    )

    def _raise_invalid(_point, grid=None):  # type: ignore[no-untyped-def]
        raise PlanningError("point invalid")

    monkeypatch.setattr(test_planner, "validate_point", _raise_invalid)
    with pytest.raises(PlanningError, match="point invalid"):
        generate_candidate_route_set(test_map, test_planner, config)


def test_evaluate_route_set_modes_are_deterministic(test_map: MapDefinition) -> None:
    """Objective mode evaluation should be deterministic for fixed input routes."""
    candidate = CandidateRouteSet(
        robot_routes=test_map.robot_routes, ped_routes=test_map.ped_routes
    )

    composite_cfg = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        objective_mode="composite",
        ped_route_count=1,
    )
    failure_cfg = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        objective_mode="failure_only",
        ped_route_count=1,
    )
    near_cfg = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        objective_mode="near_miss_only",
        ped_route_count=1,
    )
    first = evaluate_route_set(candidate, test_map, composite_cfg)
    second = evaluate_route_set(candidate, test_map, composite_cfg)
    assert first.score == pytest.approx(second.score)
    assert evaluate_route_set(candidate, test_map, failure_cfg).score == pytest.approx(
        first.failure_proxy
    )
    assert evaluate_route_set(candidate, test_map, near_cfg).score == pytest.approx(
        first.near_miss_stress
    )


def test_write_route_override_artifact_roundtrip(tmp_path: Path, test_map: MapDefinition) -> None:
    """Artifact writer should emit the expected YAML schema keys."""
    candidate = CandidateRouteSet(
        robot_routes=test_map.robot_routes, ped_routes=test_map.ped_routes
    )
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="demo", map_file="maps/demo.svg", ped_route_count=1
    )
    evaluation = evaluate_route_set(candidate, test_map, cfg)
    result = OptimizationResult(
        map_def=test_map,
        config=cfg,
        best_candidate=candidate,
        best_evaluation=evaluation,
        failed_trials=0,
        feasibility_rejection_counts={},
        top_k_scores=[evaluation.score],
        valid_trial_count=1,
        valid_candidates_by_trial={0: candidate},
    )
    artifacts = write_route_override_artifact(result, output_root=tmp_path)
    assert artifacts["artifact_path"].exists()
    assert artifacts["overlay_plot_path"].exists()
    assert artifacts["overlay_plot_path"].stat().st_size > 0
    data = yaml.safe_load(artifacts["artifact_path"].read_text(encoding="utf-8"))
    assert data["scenario_id"] == "demo"
    assert data["optimizer"] == "optuna_tpe"
    assert "route_payload" in data
    assert "robot_routes" in data["route_payload"]
    assert "ped_routes" in data["route_payload"]
    report = artifacts["report_path"].read_text(encoding="utf-8")
    assert "trajectories_overlay.png" in report


def test_optimize_route_set_is_deterministic_for_fixed_seed(test_map: MapDefinition) -> None:
    """Optimization should produce deterministic best payload for fixed seed."""
    planner_config = ClassicPlannerConfig(
        cells_per_meter=1.0,
        inflate_radius_meters=0.0,
        add_boundary_obstacles=False,
        algorithm="a_star",
    )
    planner_a = ClassicGlobalPlanner(test_map, planner_config)
    planner_b = ClassicGlobalPlanner(test_map, planner_config)
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="deterministic",
        map_file="maps/demo.svg",
        trial_count=6,
        seed=11,
        robot_route_count=1,
        ped_route_count=1,
    )

    res_a = optimize_route_set(test_map, planner_a, cfg)
    res_b = optimize_route_set(test_map, planner_b, cfg)
    payload_a = {
        "robot_routes": [
            [tuple(point) for point in route.waypoints]
            for route in res_a.best_candidate.robot_routes
        ],
        "ped_routes": [
            [tuple(point) for point in route.waypoints] for route in res_a.best_candidate.ped_routes
        ],
    }
    payload_b = {
        "robot_routes": [
            [tuple(point) for point in route.waypoints]
            for route in res_b.best_candidate.robot_routes
        ],
        "ped_routes": [
            [tuple(point) for point in route.waypoints] for route in res_b.best_candidate.ped_routes
        ],
    }
    assert payload_a == payload_b
    assert res_a.best_evaluation.score == pytest.approx(res_b.best_evaluation.score)


def test_config_rejects_invalid_scenario_id_fields() -> None:
    """Config should reject unsafe scenario/map identity fields."""
    with pytest.raises(ValueError, match="scenario_id"):
        AdversarialRouteGenerationConfig(
            scenario_id="../unsafe",
            map_file="maps/demo.svg",
        )
    with pytest.raises(ValueError, match="scenario_id"):
        AdversarialRouteGenerationConfig(
            scenario_id="a/b",
            map_file="maps/demo.svg",
        )
    with pytest.raises(ValueError, match="scenario_id"):
        AdversarialRouteGenerationConfig(
            scenario_id="a\\b",
            map_file="maps/demo.svg",
        )


def test_config_rejects_invalid_map_file_field() -> None:
    """Config should reject unsafe map path fields."""
    with pytest.raises(ValueError, match="map_file"):
        AdversarialRouteGenerationConfig(
            scenario_id="safe_id",
            map_file="../maps/demo.svg",
        )


def test_evaluate_route_set_skips_short_routes_for_inefficiency(test_map: MapDefinition) -> None:
    """Inefficiency scoring should skip routes with fewer than two waypoints."""
    short_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(2.0, 2.0)],
        spawn_zone=test_map.robot_spawn_zones[0],
        goal_zone=test_map.robot_goal_zones[0],
    )
    candidate = CandidateRouteSet(
        robot_routes=[short_route],
        ped_routes=[],
    )
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="short_route_test",
        map_file="maps/demo.svg",
        objective_mode="composite",
        ped_route_count=0,
    )

    evaluation = evaluate_route_set(candidate, test_map, cfg)
    assert evaluation.path_inefficiency == 0.0


def test_optimize_route_set_preserves_zero_trial_scores(
    test_map: MapDefinition,
    test_planner: ClassicGlobalPlanner,
) -> None:
    """Optimizer top-k extraction should keep legitimate 0.0 trial scores."""
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="zero_score_test",
        map_file="maps/demo.svg",
        objective_mode="failure_only",
        trial_count=6,
        seed=19,
        robot_route_count=1,
        ped_route_count=0,
        top_k=3,
    )

    result = optimize_route_set(test_map, test_planner, cfg)
    assert result.top_k_scores
    assert all(score == pytest.approx(0.0) for score in result.top_k_scores)


def test_config_rejects_weights_not_summing_to_one() -> None:
    """Config should enforce normalized objective weights."""
    with pytest.raises(ValueError, match="sum to 1.0"):
        AdversarialRouteGenerationConfig(
            scenario_id="weights_invalid",
            map_file="maps/demo.svg",
            failure_weight=0.5,
            delay_weight=0.3,
            inefficiency_weight=0.2,
            near_miss_weight=0.2,
        )


def test_write_route_override_artifact_writes_overlay_plot(
    tmp_path: Path,
    test_map: MapDefinition,
) -> None:
    """Overlay plot should be created from feasible trials plus best candidate."""
    base_candidate = CandidateRouteSet(
        robot_routes=test_map.robot_routes,
        ped_routes=test_map.ped_routes,
    )
    alt_robot_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (7.0, 7.0), (8.2, 8.5)],
        spawn_zone=test_map.robot_spawn_zones[0],
        goal_zone=test_map.robot_goal_zones[0],
    )
    alt_candidate = CandidateRouteSet(
        robot_routes=[alt_robot_route],
        ped_routes=test_map.ped_routes,
    )
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="overlay_demo",
        map_file="maps/demo.svg",
        ped_route_count=1,
    )
    evaluation = evaluate_route_set(base_candidate, test_map, cfg)
    result = OptimizationResult(
        map_def=test_map,
        config=cfg,
        best_candidate=base_candidate,
        best_evaluation=evaluation,
        failed_trials=0,
        feasibility_rejection_counts={},
        top_k_scores=[evaluation.score],
        valid_trial_count=2,
        valid_candidates_by_trial={0: alt_candidate, 1: base_candidate},
    )

    artifacts = write_route_override_artifact(result, output_root=tmp_path)
    assert artifacts["overlay_plot_path"].exists()
    assert artifacts["overlay_plot_path"].stat().st_size > 0


def test_overlay_plot_ignores_short_routes(
    tmp_path: Path,
    test_map: MapDefinition,
) -> None:
    """Overlay plotting should ignore short routes and still render."""
    short_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(3.0, 3.0)],
        spawn_zone=test_map.robot_spawn_zones[0],
        goal_zone=test_map.robot_goal_zones[0],
    )
    candidate = CandidateRouteSet(
        robot_routes=[short_route],
        ped_routes=test_map.ped_routes,
    )
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="overlay_short_routes",
        map_file="maps/demo.svg",
        ped_route_count=1,
    )
    evaluation = evaluate_route_set(candidate, test_map, cfg)
    result = OptimizationResult(
        map_def=test_map,
        config=cfg,
        best_candidate=candidate,
        best_evaluation=evaluation,
        failed_trials=0,
        feasibility_rejection_counts={},
        top_k_scores=[evaluation.score],
        valid_trial_count=1,
        valid_candidates_by_trial={0: candidate},
    )

    artifacts = write_route_override_artifact(result, output_root=tmp_path)
    assert artifacts["overlay_plot_path"].exists()
    assert artifacts["overlay_plot_path"].stat().st_size > 0

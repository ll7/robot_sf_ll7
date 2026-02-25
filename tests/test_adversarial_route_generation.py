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


def _build_test_map() -> MapDefinition:
    """Create a simple map definition with both robot and pedestrian routes."""
    robot_spawn_zone = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    robot_goal_zone = ((8.0, 8.0), (9.0, 8.0), (8.0, 9.0))
    ped_spawn_zone = ((1.0, 8.0), (2.0, 8.0), (1.0, 9.0))
    ped_goal_zone = ((8.0, 1.0), (9.0, 1.0), (8.0, 2.0))
    bounds = [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 10.0, 0.0, 10.0),
    ]
    robot_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (8.5, 8.5)],
        spawn_zone=robot_spawn_zone,
        goal_zone=robot_goal_zone,
    )
    ped_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 8.5), (8.5, 1.5)],
        spawn_zone=ped_spawn_zone,
        goal_zone=ped_goal_zone,
    )
    return MapDefinition(
        width=10.0,
        height=10.0,
        obstacles=[],
        robot_spawn_zones=[robot_spawn_zone],
        ped_spawn_zones=[ped_spawn_zone],
        robot_goal_zones=[robot_goal_zone],
        bounds=bounds,
        robot_routes=[robot_route],
        ped_goal_zones=[ped_goal_zone],
        ped_crowded_zones=[],
        ped_routes=[ped_route],
        single_pedestrians=[],
    )


def _build_planner(map_def: MapDefinition) -> ClassicGlobalPlanner:
    """Create a deterministic planner for tests."""
    return ClassicGlobalPlanner(
        map_def,
        ClassicPlannerConfig(
            cells_per_meter=1.0,
            inflate_radius_meters=0.0,
            add_boundary_obstacles=False,
            algorithm="a_star",
        ),
    )


def test_generate_candidate_routes_stay_in_map_bounds() -> None:
    """Generated route waypoints should remain inside map bounds."""
    map_def = _build_test_map()
    planner = _build_planner(map_def)
    config = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        trial_count=2,
        robot_route_count=1,
        ped_route_count=1,
        seed=17,
    )
    candidate, reason = generate_candidate_route_set(map_def, planner, config)
    assert reason is None
    assert candidate is not None
    for route in [*candidate.robot_routes, *candidate.ped_routes]:
        for x, y in route.waypoints:
            assert 0.0 <= x <= map_def.width
            assert 0.0 <= y <= map_def.height


def test_generate_candidate_rejects_invalid_points_with_feasibility_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feasibility filter should reject candidates with invalid start/goal points."""
    map_def = _build_test_map()
    planner = _build_planner(map_def)
    config = AdversarialRouteGenerationConfig(
        scenario_id="test",
        map_file="test.svg",
        trial_count=2,
        robot_route_count=1,
        ped_route_count=0,
    )

    def _raise_invalid(_point, grid=None):  # type: ignore[no-untyped-def]
        raise PlanningError("point invalid")

    monkeypatch.setattr(planner, "validate_point", _raise_invalid)
    candidate, reason = generate_candidate_route_set(map_def, planner, config)
    assert candidate is None
    assert reason == "invalid_start_or_goal"


def test_generate_candidate_raises_invalid_points_when_feasibility_filter_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation errors should propagate when feasibility filtering is disabled."""
    map_def = _build_test_map()
    planner = _build_planner(map_def)
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

    monkeypatch.setattr(planner, "validate_point", _raise_invalid)
    with pytest.raises(PlanningError, match="point invalid"):
        generate_candidate_route_set(map_def, planner, config)


def test_evaluate_route_set_modes_are_deterministic() -> None:
    """Objective mode evaluation should be deterministic for fixed input routes."""
    map_def = _build_test_map()
    candidate = CandidateRouteSet(robot_routes=map_def.robot_routes, ped_routes=map_def.ped_routes)

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
    first = evaluate_route_set(candidate, map_def, composite_cfg)
    second = evaluate_route_set(candidate, map_def, composite_cfg)
    assert first.score == pytest.approx(second.score)
    assert evaluate_route_set(candidate, map_def, failure_cfg).score == pytest.approx(
        first.failure_proxy
    )
    assert evaluate_route_set(candidate, map_def, near_cfg).score == pytest.approx(
        first.near_miss_stress
    )


def test_write_route_override_artifact_roundtrip(tmp_path: Path) -> None:
    """Artifact writer should emit the expected YAML schema keys."""
    map_def = _build_test_map()
    candidate = CandidateRouteSet(robot_routes=map_def.robot_routes, ped_routes=map_def.ped_routes)
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="demo", map_file="maps/demo.svg", ped_route_count=1
    )
    evaluation = evaluate_route_set(candidate, map_def, cfg)
    result = OptimizationResult(
        config=cfg,
        best_candidate=candidate,
        best_evaluation=evaluation,
        failed_trials=0,
        feasibility_rejection_counts={},
        top_k_scores=[evaluation.score],
        valid_trial_count=1,
    )
    artifacts = write_route_override_artifact(result, output_root=tmp_path)
    assert artifacts["artifact_path"].exists()
    data = yaml.safe_load(artifacts["artifact_path"].read_text(encoding="utf-8"))
    assert data["scenario_id"] == "demo"
    assert data["optimizer"] == "optuna_tpe"
    assert "route_payload" in data
    assert "robot_routes" in data["route_payload"]
    assert "ped_routes" in data["route_payload"]


def test_optimize_route_set_is_deterministic_for_fixed_seed() -> None:
    """Optimization should produce deterministic best payload for fixed seed."""
    map_def = _build_test_map()
    planner_a = _build_planner(map_def)
    planner_b = _build_planner(map_def)
    cfg = AdversarialRouteGenerationConfig(
        scenario_id="deterministic",
        map_file="maps/demo.svg",
        trial_count=6,
        seed=11,
        robot_route_count=1,
        ped_route_count=1,
    )

    res_a = optimize_route_set(map_def, planner_a, cfg)
    res_b = optimize_route_set(map_def, planner_b, cfg)
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


def test_config_rejects_invalid_identity_fields() -> None:
    """Config should reject unsafe scenario/map identity fields."""
    with pytest.raises(ValueError, match="scenario_id"):
        AdversarialRouteGenerationConfig(
            scenario_id="../unsafe",
            map_file="maps/demo.svg",
        )
    with pytest.raises(ValueError, match="map_file"):
        AdversarialRouteGenerationConfig(
            scenario_id="safe_id",
            map_file="../maps/demo.svg",
        )

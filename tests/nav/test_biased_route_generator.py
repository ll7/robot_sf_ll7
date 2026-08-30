"""Unit and contract tests for deterministic biased route generator (Issue #8033)."""

from __future__ import annotations

import itertools
import time

import pytest

from robot_sf.benchmark.route_choice_observability import (
    DIAGNOSTIC_SCHEMA_VERSION,
)
from robot_sf.nav.biased_route_generator import (
    BiasedRouteConfig,
    build_corridor_fixture,
    build_crossing_fixture,
    build_doorway_fixture,
    evaluate_route_observability_sequence,
    generate_biased_route,
    generate_corridor_homotopy_routes,
    generate_doorway_homotopy_routes,
    rasterize_route_to_grid,
)


def test_config_validation_rejects_invalid_values() -> None:
    """Test fail-closed parameter validation on BiasedRouteConfig."""
    cfg = BiasedRouteConfig()
    assert cfg.bias_mode == "neutral"
    assert cfg.lateral_bias_m == 1.0

    with pytest.raises(ValueError, match="Invalid bias_mode"):
        BiasedRouteConfig(bias_mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lateral_bias_m"):
        BiasedRouteConfig(lateral_bias_m=-0.5)

    with pytest.raises(ValueError, match="lateral_bias_m"):
        BiasedRouteConfig(lateral_bias_m=float("nan"))

    with pytest.raises(ValueError, match="num_points"):
        BiasedRouteConfig(num_points=1)

    with pytest.raises(ValueError, match="Invalid profile"):
        BiasedRouteConfig(profile="unknown")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tolerance_m"):
        BiasedRouteConfig(tolerance_m=-0.1)

    with pytest.raises(ValueError, match="neutral_band_m"):
        BiasedRouteConfig(neutral_band_m=-0.2)

    with pytest.raises(ValueError, match="progress_interval"):
        BiasedRouteConfig(progress_interval=(0.8, 0.2))

    with pytest.raises(ValueError, match="coordinate_frame"):
        BiasedRouteConfig(coordinate_frame="")
    with pytest.raises(ValueError, match="units"):
        BiasedRouteConfig(units="   ")


def test_generate_biased_route_rejects_degenerate_inputs() -> None:
    """Test degenerate or non-finite inputs fail closed."""
    with pytest.raises(ValueError, match="start"):
        generate_biased_route((float("nan"), 0.0), (10.0, 0.0))

    with pytest.raises(ValueError, match="goal"):
        generate_biased_route((0.0, 0.0), (10.0, float("inf")))

    with pytest.raises(ValueError, match="start"):
        generate_biased_route((0.0, 0.0, 0.0), (10.0, 0.0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="degenerate"):
        generate_biased_route((5.0, 5.0), (5.0, 5.0))


def test_deterministic_replay_invariance() -> None:
    """Re-generating routes with same inputs produces bit-identical path coordinates."""
    start = (2.5, -1.0)
    goal = (18.0, 7.5)
    cfg = BiasedRouteConfig(bias_mode="left", lateral_bias_m=1.5, num_points=40)

    res1 = generate_biased_route(start, goal, cfg)
    res2 = generate_biased_route(start, goal, cfg)

    assert res1.path == res2.path
    assert res1.length_m == res2.length_m
    assert res1.max_lateral_offset_m == res2.max_lateral_offset_m
    assert res1.side_report.side == "left"
    assert res1.side_report.side == res2.side_report.side


@pytest.mark.parametrize(
    "bias_mode,expected_side",
    [
        ("neutral", "neutral"),
        ("left", "left"),
        ("right", "right"),
    ],
)
def test_route_side_classification_matches_bias_mode(
    bias_mode: str,
    expected_side: str,
) -> None:
    """Biased routes along canonical horizontal axis classify into expected side categories."""
    start = (0.0, 0.0)
    goal = (20.0, 0.0)
    cfg = BiasedRouteConfig(
        bias_mode=bias_mode,  # type: ignore[arg-type]
        lateral_bias_m=1.2,
        num_points=50,
        neutral_band_m=0.2,
        tolerance_m=0.05,
    )
    result = generate_biased_route(start, goal, cfg)

    assert result.bias_mode == bias_mode
    assert result.side_report.side == expected_side
    assert result.side_report.reason is None
    assert result.start == start
    assert result.goal == goal
    assert result.path[0] == pytest.approx(start)
    assert result.path[-1] == pytest.approx(goal)

    if bias_mode == "neutral":
        assert result.max_lateral_offset_m == pytest.approx(0.0)
    else:
        assert result.max_lateral_offset_m > 0.5


@pytest.mark.parametrize("profile", ["smooth_sine", "hann", "cubic", "trapezoid"])
def test_profile_shapes_produce_continuous_bounded_paths(profile: str) -> None:
    """All profile shapes produce smooth paths satisfying boundary and clearance bounds."""
    start = (0.0, 0.0)
    goal = (10.0, 0.0)
    cfg = BiasedRouteConfig(
        bias_mode="left",
        lateral_bias_m=2.0,
        num_points=60,
        profile=profile,  # type: ignore[arg-type]
    )
    result = generate_biased_route(start, goal, cfg)

    assert len(result.path) == 60
    assert result.path[0] == pytest.approx(start)
    assert result.path[-1] == pytest.approx(goal)
    assert result.max_lateral_offset_m <= 2.0 + 1e-6
    assert result.side_report.side == "left"


def test_rotation_and_translation_invariance() -> None:
    """Left and right biases rotate properly under arbitrary directed start-to-goal axes."""
    start = (10.0, 10.0)
    goal = (20.0, 20.0)

    cfg_left = BiasedRouteConfig(bias_mode="left", lateral_bias_m=1.5, num_points=30)
    res_left = generate_biased_route(start, goal, cfg_left)
    assert res_left.side_report.side == "left"

    cfg_right = BiasedRouteConfig(bias_mode="right", lateral_bias_m=1.5, num_points=30)
    res_right = generate_biased_route(start, goal, cfg_right)
    assert res_right.side_report.side == "right"

    start_rev = (20.0, 5.0)
    goal_rev = (0.0, 5.0)
    res_rev_left = generate_biased_route(start_rev, goal_rev, cfg_left)
    assert res_rev_left.side_report.side == "left"


def test_canonical_fixtures_and_homotopy_routes() -> None:
    """Test canonical corridor, doorway, and crossing fixture route generators."""
    corridor_fix = build_corridor_fixture()
    assert corridor_fix.environment_type == "corridor"
    assert len(corridor_fix.obstacles) >= 3

    corridor_routes = generate_corridor_homotopy_routes(corridor_fix)
    assert set(corridor_routes.keys()) == {"neutral", "left", "right"}
    assert corridor_routes["neutral"].side_report.side == "neutral"
    assert corridor_routes["left"].side_report.side == "left"
    assert corridor_routes["right"].side_report.side == "right"

    door_fix = build_doorway_fixture()
    assert door_fix.environment_type == "doorway"
    door_routes = generate_doorway_homotopy_routes(door_fix)
    assert door_routes["left"].side_report.side == "left"
    assert door_routes["right"].side_report.side == "right"

    cross_fix = build_crossing_fixture()
    assert cross_fix.environment_type == "crossing"
    assert len(cross_fix.obstacles) == 1


def test_rasterize_route_to_grid_step_validity() -> None:
    """Rasterizing continuous path to grid produces strictly 8-connected valid cell sequence."""
    start = (0.0, 0.0)
    goal = (10.0, 10.0)
    cfg = BiasedRouteConfig(bias_mode="left", lateral_bias_m=2.0, num_points=50)
    result = generate_biased_route(start, goal, cfg)

    grid_path = rasterize_route_to_grid(
        result.path,
        grid_origin=(-2.0, -2.0),
        grid_resolution=0.2,
        grid_shape=(100, 100),
    )

    assert len(grid_path) >= 50
    for (r1, c1), (r2, c2) in itertools.pairwise(grid_path):
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        assert max(dr, dc) <= 1, f"Non-adjacent step jump from ({r1}, {c1}) to ({r2}, {c2})"

    assert rasterize_route_to_grid([], (0.0, 0.0), 0.2, (50, 50)) == []

    with pytest.raises(ValueError, match="Invalid grid_resolution"):
        rasterize_route_to_grid(result.path, (0.0, 0.0), 0.0, (50, 50))


def test_evaluate_route_observability_sequence() -> None:
    """Evaluating a temporal route sequence generates consistent diagnostic reports."""
    start = (0.0, 0.0)
    goal = (20.0, 0.0)

    left_routes = [
        generate_biased_route(start, goal, BiasedRouteConfig(bias_mode="left", lateral_bias_m=1.0))
        for _ in range(5)
    ]
    eval_result = evaluate_route_observability_sequence(left_routes)

    assert eval_result["schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert eval_result["status"] == "available"
    assert len(eval_result["route_side_observations"]) == 5
    assert eval_result["route_side_observations"][0]["side"] == "left"

    consistency = eval_result["temporal_consistency"]
    assert consistency["consistency_fraction"] == 1.0
    assert consistency["side_transition_count"] == 0
    assert consistency["dominant_side"] == "left"
    assert consistency["valid_count"] == 5

    empty_eval = evaluate_route_observability_sequence([])
    assert empty_eval["status"] == "not_available"
    assert empty_eval["temporal_consistency"]["consistency_fraction"] == 0.0


def test_generation_runtime_performance() -> None:
    """Route generation executes in under 5ms per route (< 50ms requirement)."""
    start = (0.0, 0.0)
    goal = (25.0, 5.0)
    cfg = BiasedRouteConfig(bias_mode="right", lateral_bias_m=1.5, num_points=100)

    t0 = time.perf_counter()
    for _ in range(20):
        generate_biased_route(start, goal, cfg)
    elapsed = time.perf_counter() - t0
    per_route_ms = (elapsed / 20.0) * 1000.0

    assert per_route_ms < 5.0, f"Route generation too slow: {per_route_ms:.2f}ms per route"


def test_biased_route_result_as_dict() -> None:
    """BiasedRouteResult and CanonicalFixtureTopology serialize to clean dictionaries."""
    start = (0.0, 0.0)
    goal = (10.0, 0.0)
    res = generate_biased_route(start, goal, BiasedRouteConfig(bias_mode="neutral"))
    d = res.as_dict()

    assert d["bias_mode"] == "neutral"
    assert d["start"] == [0.0, 0.0]
    assert d["goal"] == [10.0, 0.0]
    assert "side_report" in d
    assert d["side_report"]["side"] == "neutral"

    fix = build_corridor_fixture()
    fix_dict = fix.as_dict()
    assert fix_dict["environment_type"] == "corridor"
    assert "obstacles" in fix_dict

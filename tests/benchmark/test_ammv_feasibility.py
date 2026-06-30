"""Tests for the AMMV command-feasibility / tip-over evaluator (issue #3466)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.ammv_feasibility import (
    AMMV_FEASIBILITY_SCHEMA,
    AmmvFeasibilityParams,
    evaluate_artifact_command_feasibility,
    evaluate_command_feasibility,
)
from robot_sf.benchmark.map_runner_batch_runner import _initial_feasibility_totals
from robot_sf.benchmark.map_runner_batch_summary import accumulate_batch_metadata


def test_feasible_command_sequence_is_classified_feasible() -> None:
    """Gentle commands within stability and curvature limits must be feasible."""
    velocities = np.array([0.5, 0.6, 0.5])
    turn_rates = np.array([0.2, 0.1, 0.2])  # |omega| <= max_curvature * v; low lateral accel

    report = evaluate_command_feasibility(velocities, turn_rates)

    assert report["schema_version"] == AMMV_FEASIBILITY_SCHEMA
    assert report["feasible"] is True
    assert report["tip_over_violation"] is False
    assert report["n_curvature_violations"] == 0
    assert report["min_stability_margin"] > 0.0


def test_over_yaw_command_trips_tip_over() -> None:
    """A high speed*yaw command exceeding the critical lateral accel must flag tip-over."""
    velocities = np.array([0.5, 2.0, 0.5])
    turn_rates = np.array([0.2, 2.0, 0.2])  # step 1: a_y = 4.0 > a_y,crit ~ 2.73

    report = evaluate_command_feasibility(velocities, turn_rates)

    assert report["tip_over_violation"] is True
    assert report["n_tip_over_steps"] == 1
    assert report["rollover_event"] == "ROLLOVER_CRITICAL"
    assert report["min_stability_margin"] == 0.0
    assert report["feasible"] is False


def test_excess_curvature_is_flagged_infeasible() -> None:
    """A tight turn beyond the proxy non-holonomic curvature limit must be flagged."""
    params = AmmvFeasibilityParams(max_curvature_per_m=1.0)
    velocities = np.array([0.5])
    turn_rates = np.array([0.9])  # |omega|/v = 1.8 > 1.0; but a_y=0.45 small (no tip-over)

    report = evaluate_command_feasibility(velocities, turn_rates, params)

    assert report["n_curvature_violations"] == 1
    assert report["tip_over_violation"] is False
    assert report["feasible"] is False


def test_in_place_rotation_uses_in_place_yaw_limit() -> None:
    """At near-zero speed the in-place yaw limit applies instead of curvature."""
    params = AmmvFeasibilityParams(in_place_yaw_rate_max=0.5)
    ok = evaluate_command_feasibility(np.array([0.0]), np.array([0.4]), params)
    too_fast = evaluate_command_feasibility(np.array([0.0]), np.array([0.9]), params)

    assert ok["feasible"] is True
    assert too_fast["n_curvature_violations"] == 1


def test_matches_benchmark_surface_stability_margin() -> None:
    """The min stability margin must match metrics.evaluate_stability_margin for the same params."""
    from robot_sf.benchmark.metrics import evaluate_stability_margin

    params = AmmvFeasibilityParams()
    report = evaluate_command_feasibility(np.array([1.0]), np.array([1.5]), params)
    expected = evaluate_stability_margin(
        1.0,
        1.5,
        t_w=params.track_width_m,
        L=params.wheelbase_m,
        h_c=params.cog_height_m,
        a=params.front_axle_to_cog_m,
    )

    assert report["min_stability_margin"] == pytest.approx(expected)


def test_length_mismatch_and_empty_are_rejected() -> None:
    """Mismatched lengths and empty sequences must fail closed."""
    with pytest.raises(ValueError):
        evaluate_command_feasibility(np.array([1.0, 2.0]), np.array([0.1]))
    with pytest.raises(ValueError):
        evaluate_command_feasibility(np.array([]), np.array([]))


def test_non_finite_commands_fail_closed() -> None:
    """NaN/Inf commands must raise rather than be silently reported feasible."""
    with pytest.raises(ValueError):
        evaluate_command_feasibility(np.array([1.0, np.nan]), np.array([0.1, 0.1]))
    with pytest.raises(ValueError):
        evaluate_command_feasibility(np.array([1.0, np.inf]), np.array([0.1, 0.1]))
    with pytest.raises(ValueError):
        evaluate_command_feasibility(np.array([1.0, 1.0]), np.array([0.1, np.nan]))


def test_artifact_command_feasibility_extracts_versioned_fields() -> None:
    """Per-step artifact selected actions expose AMMV feasibility telemetry."""
    report = evaluate_artifact_command_feasibility(
        [
            {"linear_velocity": 0.5, "angular_velocity": 0.2},
            {"linear_velocity": 0.6, "angular_velocity": 0.1},
        ]
    )

    assert report["schema_version"] == AMMV_FEASIBILITY_SCHEMA
    assert report["proxy_kind"] == "internal_non_hardware"
    assert report["status"] == "available"
    assert report["n_commands"] == 2
    assert report["tip_over_violation"] is False
    assert report["n_curvature_violations"] == 0
    assert report["feasible"] is True


def test_artifact_command_feasibility_fails_closed_for_missing_yaw_rate() -> None:
    """Holonomic artifact actions cannot be promoted as AMMV-feasible command evidence."""
    report = evaluate_artifact_command_feasibility([{"vx": 0.5, "vy": 0.0}])

    assert report["schema_version"] == AMMV_FEASIBILITY_SCHEMA
    assert report["proxy_kind"] == "internal_non_hardware"
    assert report["status"] == "missing_inputs"
    assert report["tip_over_violation"] is True
    assert report["n_curvature_violations"] is None
    assert report["feasible"] is False


def test_artifact_command_feasibility_fails_closed_for_invalid_inputs() -> None:
    """Malformed artifact command payloads stay diagnostic and infeasible."""
    reports = [
        evaluate_artifact_command_feasibility("not-a-command-sequence"),
        evaluate_artifact_command_feasibility([]),
        evaluate_artifact_command_feasibility(
            [{"linear_velocity": 0.1, "angular_velocity": 0.0}, 1]
        ),
        evaluate_artifact_command_feasibility(
            [{"linear_velocity": float("nan"), "angular_velocity": 0.0}]
        ),
    ]

    assert all(report["schema_version"] == AMMV_FEASIBILITY_SCHEMA for report in reports)
    assert all(report["status"] == "missing_inputs" for report in reports)
    assert all(report["tip_over_violation"] is True for report in reports)
    assert all(report["feasible"] is False for report in reports)


def test_batch_metadata_folds_ammv_feasibility_fields() -> None:
    """Batch summaries expose AMMV artifact telemetry without changing existing rows."""
    totals = _initial_feasibility_totals()
    requested_seen, native_steps, adapted_steps = accumulate_batch_metadata(
        {
            "algorithm_metadata": {
                "ammv_feasibility": {
                    "schema_version": AMMV_FEASIBILITY_SCHEMA,
                    "proxy_kind": "internal_non_hardware",
                    "n_commands": 3,
                    "min_stability_margin": 0.25,
                    "tip_over_violation": False,
                    "n_curvature_violations": 1,
                    "feasible": False,
                }
            }
        },
        feasibility_totals=totals,
    )

    assert (requested_seen, native_steps, adapted_steps) == (False, 0, 0)
    assert totals["ammv_episode_count"] == 1
    assert totals["ammv_commands_evaluated"] == 3
    assert totals["ammv_min_stability_margin"] == pytest.approx(0.25)
    assert totals["ammv_tip_over_episode_count"] == 0
    assert totals["ammv_curvature_violation_count"] == 1
    assert totals["ammv_feasible_episode_count"] == 0


def test_invalid_params_rejected() -> None:
    """Non-physical proxy params must fail closed."""
    with pytest.raises(ValueError):
        AmmvFeasibilityParams(max_curvature_per_m=0.0)

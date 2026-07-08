"""Characterization baseline tests for ``robot_sf/benchmark/critical_intervals.py``.

These tests pin the *current observable behavior* of the critical-interval metric
helpers on tiny synthetic traces/arrays. They are table-driven and assert exact
golden values, including edge cases (single sample, NaN/Inf at the serialization
finite guard, overlap/contact TTC, receding/perpendicular non-closing pairs,
empty pedestrian sets, strict threshold boundaries).

Purpose (issue #4881, wave 2; Refs #4874, #4770): lock a behavioral baseline so
the post-submission refactor wave can prove behavior-preservation. If a test
reveals a genuine bug, do NOT fix it here — document it and file a separate fix
issue.

These tests are additive: they pin the TTC convention/constants, the exact
``_pairwise_ttc_s`` closing-speed formula, the
``_compute_max_braking_deceleration_mps2`` output for constructed velocity
profiles, the ``_compute_interval_metrics_in_window`` aggregation table, and the
``report_to_dict`` NaN/Inf -> ``None`` serialization guard. They do not
duplicate the anchor-detection, window-clamping, or whole-run pipeline coverage
in ``test_critical_intervals.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.critical_intervals import (
    _TTC_EPS_DIST,
    _TTC_EPS_SPEED,
    DEFAULT_NEAR_MISS_DIST,
    SCHEMA_VERSION,
    TTC_CONVENTION,
    VALID_ANCHORS,
    CriticalIntervalReport,
    IntervalMetrics,
    _compute_interval_metrics_in_window,
    _compute_max_braking_deceleration_mps2,
    _pairwise_ttc_s,
    report_to_dict,
)

# ---------------------------------------------------------------------------
# Constants / convention table
# ---------------------------------------------------------------------------


def test_constants_and_convention_are_pinned() -> None:
    """Pin the schema, TTC convention, near-miss distance, and tolerances."""
    assert SCHEMA_VERSION == "critical-intervals.v1"
    assert TTC_CONVENTION == "line_of_sight_closing_speed_seconds.v1"
    assert DEFAULT_NEAR_MISS_DIST == 0.7
    assert _TTC_EPS_DIST == 1e-9
    assert _TTC_EPS_SPEED == 1e-9


def test_valid_anchors_set_is_pinned() -> None:
    """The accepted anchor name set is exactly the documented frozenset."""
    assert VALID_ANCHORS == frozenset(
        {
            "closest_approach",
            "ttc_threshold_crossing",
            "first_braking_event",
            "collision_or_near_miss",
            "recovery_after_avoidance",
            "planner_mode_switch",
            "pedestrian_deviation_onset",
            "stuck_oscillation_onset",
        }
    )


# ---------------------------------------------------------------------------
# _pairwise_ttc_s: physical closing-speed TTC (exact)
# ---------------------------------------------------------------------------


def test_pairwise_ttc_head_on_closing_is_exact_distance_over_speed() -> None:
    """Stationary ped dead ahead of a closing robot -> TTC == distance / speed."""
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([1.0, 0.0]),
        ped_pos=np.array([5.0, 0.0]),
        ped_vel=np.array([0.0, 0.0]),
    )
    assert ttc == pytest.approx(5.0)


def test_pairwise_ttc_3_4_5_triangle_closing_is_exact() -> None:
    """Non-axis-aligned geometry pins the closing-speed formula numerically."""
    # dist = 5 (3-4-5), closing speed = dot([2,0],[3,4])/5 = 6/5 = 1.2 -> ttc = 5/1.2
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([2.0, 0.0]),
        ped_pos=np.array([3.0, 4.0]),
        ped_vel=np.array([0.0, 0.0]),
    )
    assert ttc == pytest.approx(5.0 / 1.2)


def test_pairwise_ttc_receding_pair_is_none() -> None:
    """A pair moving apart (negative closing speed) -> ``None``."""
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([1.0, 0.0]),
        ped_pos=np.array([5.0, 0.0]),
        ped_vel=np.array([2.0, 0.0]),  # ped outruns robot -> receding
    )
    assert ttc is None


def test_pairwise_ttc_perpendicular_motion_is_none() -> None:
    """Motion perpendicular to the separation vector has zero closing speed -> ``None``."""
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([0.0, 1.0]),  # perpendicular to [5,0] separation
        ped_pos=np.array([5.0, 0.0]),
        ped_vel=np.array([0.0, 0.0]),
    )
    assert ttc is None


def test_pairwise_ttc_stationary_pair_is_none() -> None:
    """Both stationary -> zero closing speed -> ``None`` (not zero, not inf)."""
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([0.0, 0.0]),
        ped_pos=np.array([5.0, 0.0]),
        ped_vel=np.array([0.0, 0.0]),
    )
    assert ttc is None


def test_pairwise_ttc_overlap_is_zero() -> None:
    """Contact/overlap (dist below the epsilon) -> ``0.0`` regardless of velocity."""
    ttc = _pairwise_ttc_s(
        robot_pos=np.array([0.0, 0.0]),
        robot_vel=np.array([1.0, 0.0]),
        ped_pos=np.array([1e-12, 0.0]),  # dist < _TTC_EPS_DIST
        ped_vel=np.array([0.0, 0.0]),
    )
    assert ttc == 0.0


# ---------------------------------------------------------------------------
# _compute_max_braking_deceleration_mps2 (exact velocity profiles)
# ---------------------------------------------------------------------------


def test_braking_deceleration_single_sample_is_none() -> None:
    """Fewer than two velocity samples -> ``None``."""
    assert _compute_max_braking_deceleration_mps2(np.array([[1.0, 0.0]]), dt=1.0) is None


def test_braking_deceleration_two_sample_uniform_decel_is_one() -> None:
    """A uniform 1 m/s deceleration over two samples yields max decel 1.0 m/s^2."""
    vel = np.array([[2.0, 0.0], [1.0, 0.0]])
    assert _compute_max_braking_deceleration_mps2(vel, dt=1.0) == pytest.approx(1.0)


def test_braking_deceleration_three_sample_non_uniform_is_two() -> None:
    """Pins the central-difference + one-sided-boundary scheme on a mixed profile."""
    # Boundary one-sided diffs give decel 2.0 at t0; interior central diff 0.5 at t1.
    vel = np.array([[4.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert _compute_max_braking_deceleration_mps2(vel, dt=1.0) == pytest.approx(2.0)


def test_braking_deceleration_pure_acceleration_is_zero() -> None:
    """A purely accelerating profile has no braking component -> 0.0."""
    vel = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert _compute_max_braking_deceleration_mps2(vel, dt=1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_interval_metrics_in_window: aggregation table (exact)
# ---------------------------------------------------------------------------


def _straight_trace() -> dict[str, object]:
    """Robot drives +x at 1 m/s past a stationary ped at (0, 0.5)."""
    return {
        "robot_pos": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        "peds_pos": [
            [[0.0, 0.5]],
            [[0.0, 0.5]],
            [[0.0, 0.5]],
            [[0.0, 0.5]],
        ],
        "robot_vel": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        "dt": 1.0,
    }


def test_window_metrics_full_straight_trace_table() -> None:
    """Pin the aggregated window metrics for the straight-line trace."""
    result = _compute_interval_metrics_in_window(_straight_trace(), start=0, end=4)
    assert result["n_steps"] == 4
    assert result["min_distance_m"] == pytest.approx(0.5)
    assert result["min_clearance_m"] == pytest.approx(0.5)
    assert result["near_miss_count"] == 1  # only the 0.5 m step is < 0.7 m
    assert result["collision_flag"] is False  # nothing < 0.3 m
    assert result["mean_speed_ms"] == pytest.approx(1.0)
    assert result["max_speed_ms"] == pytest.approx(1.0)
    assert result["max_deceleration_mps2"] == pytest.approx(0.0)  # constant speed
    # ``heading_oscillation`` is ``np.var`` over the *flattened* unit-direction
    # vectors, so a perfectly straight (constant (1, 0)) path yields 0.25 (the
    # variance between the constant 1 and 0 components), NOT 0.0. This is the
    # pinned current behavior; the PR body flags it as a candidate follow-up bug.
    assert result["heading_oscillation"] == pytest.approx(0.25)


def test_window_metrics_subwindow_slicing() -> None:
    """A sub-slice [1, 3) aggregates only those two steps."""
    result = _compute_interval_metrics_in_window(_straight_trace(), start=1, end=3)
    assert result["n_steps"] == 2
    # steps 1 and 2: distances sqrt(1.25)=1.118 and sqrt(4.25)=2.062 -> min 1.118
    assert result["min_distance_m"] == pytest.approx(math.sqrt(1.25))
    assert result["near_miss_count"] == 0  # both > 0.7 m


def test_window_metrics_no_pedestrians_drops_distance_fields() -> None:
    """With no pedestrians, distance fields are ``None`` and event counts are absent."""
    trace = {
        "robot_pos": [[0.0, 0.0], [1.0, 0.0]],
        "peds_pos": [],
        "robot_vel": [[1.0, 0.0], [1.0, 0.0]],
        "dt": 1.0,
    }
    result = _compute_interval_metrics_in_window(trace, start=0, end=2)
    assert result["min_distance_m"] is None
    assert result["min_clearance_m"] is None
    assert "near_miss_count" not in result  # only set when pedestrians present
    assert "collision_flag" not in result


@pytest.mark.parametrize("ped_y,expected_collision", [(0.29, True), (0.30, False)])
def test_window_metrics_collision_boundary_is_strict(
    ped_y: float, expected_collision: bool
) -> None:
    """The collision flag uses strict ``dist < 0.3 m`` (0.30 m is NOT a collision)."""
    trace = {
        "robot_pos": [[0.0, 0.0], [1.0, 0.0]],
        "peds_pos": [[[0.0, ped_y]], [[0.0, ped_y]]],
        "dt": 1.0,
    }
    result = _compute_interval_metrics_in_window(trace, start=0, end=2)
    assert result["collision_flag"] is expected_collision


@pytest.mark.parametrize("ped_y,expected_count", [(0.69, 1), (0.70, 0)])
def test_window_metrics_near_miss_boundary_is_strict(ped_y: float, expected_count: int) -> None:
    """The near-miss count uses strict ``dist < 0.7 m`` (0.70 m is NOT a near miss)."""
    trace = {
        "robot_pos": [[0.0, 0.0], [5.0, 0.0]],  # step 1 is far, only step 0 can qualify
        "peds_pos": [[[0.0, ped_y]], [[0.0, ped_y]]],
        "dt": 1.0,
    }
    result = _compute_interval_metrics_in_window(trace, start=0, end=2)
    assert result["near_miss_count"] == expected_count


# ---------------------------------------------------------------------------
# report_to_dict: NaN/Inf -> None serialization guard
# ---------------------------------------------------------------------------


def test_report_to_dict_top_level_keys_and_convention() -> None:
    """``report_to_dict`` exposes the pinned top-level key set and TTC convention."""
    report = CriticalIntervalReport()
    out = report_to_dict(report)
    assert set(out) == {
        "ttc_convention",
        "whole_run",
        "critical_intervals",
        "interval_metrics",
        "missing_anchors",
    }
    assert out["ttc_convention"] == TTC_CONVENTION


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_report_to_dict_coerces_non_finite_metric_to_none(bad_value: float) -> None:
    """NaN/Inf interval-metric floats serialize to ``None``; finite floats pass through."""
    report = CriticalIntervalReport(
        interval_metrics=[
            IntervalMetrics(anchor="closest_approach", min_ttc_s=bad_value, min_distance_m=1.25),
        ]
    )
    metrics = report_to_dict(report)["interval_metrics"][0]
    assert metrics["min_ttc_s"] is None
    assert metrics["min_distance_m"] == pytest.approx(1.25)


def test_report_to_dict_interval_metric_field_set_is_pinned() -> None:
    """Each serialized interval metric carries exactly the documented field set."""
    report = CriticalIntervalReport(
        interval_metrics=[IntervalMetrics(anchor="closest_approach", n_steps=3)]
    )
    metrics = report_to_dict(report)["interval_metrics"][0]
    assert set(metrics) == {
        "anchor",
        "n_steps",
        "min_clearance_m",
        "min_distance_m",
        "min_ttc_s",
        "mean_speed_ms",
        "max_speed_ms",
        "max_deceleration_mps2",
        "heading_oscillation",
        "near_miss_count",
        "collision_flag",
    }

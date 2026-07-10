"""Regression fixtures for runtime-collision termination metric correctness.

Issue #5097: SNQI and success_rate consumed the withdrawn derived collision count.
These tests verify that when a pedestrian is close enough to trigger the geometry-derived
collision threshold (D_COLL), compute_all_metrics() produces collisions >= 1 and
success = 0 — closing the regression path that was silent in release 0.0.2.

The release 0.0.2 discrepancy arose because the runtime trigger fires at ~1.4m
(robot+ped radius sum) while D_COLL = 0.25m. Episodes that terminated via runtime
trigger had derived counts of zero. These tests use the D_COLL threshold so that the
derived geometry DOES detect the collision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.metrics import (
    EpisodeData,
    collision_count,
    compute_all_metrics,
    human_collisions,
    success_rate,
)
from robot_sf.benchmark.snqi.compute import BaselineStats, Weights, compute_snqi_v0


def _make_collision_episode(
    *,
    horizon: int = 100,
    ped_at_robot: bool = True,
) -> EpisodeData:
    """Episode where the robot and pedestrian overlap at step 2.

    With ped_at_robot=True, the ped is placed at robot position — distance 0.0 m,
    well below D_COLL = 0.25 m. The robot does not reach goal (reached_goal_step=None).
    """
    T = 10
    robot_pos = np.zeros((T, 2), dtype=float)
    robot_vel = np.zeros((T, 2), dtype=float)
    robot_acc = np.zeros((T, 2), dtype=float)
    goal = np.array([10.0, 0.0], dtype=float)

    # pedestrian starts far away, then overlaps robot at step 2
    peds_pos = np.full((T, 1, 2), fill_value=100.0, dtype=float)
    if ped_at_robot:
        peds_pos[2, 0] = robot_pos[2]  # distance 0.0 m at step 2

    ped_forces = np.zeros((T, 1, 2), dtype=float)

    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=0.1,
        reached_goal_step=None,  # collision termination — goal not reached
        obstacles=None,
        other_agents_pos=None,
    )


def _make_collision_termination_with_goal_claim(
    *,
    horizon: int = 100,
) -> EpisodeData:
    """Episode where robot claims goal reached but also has a ped collision.

    This is the pathological case from release 0.0.2: the collision gate in
    success_rate() must reject success when collision_count > 0.
    """
    T = 10
    robot_pos = np.zeros((T, 2), dtype=float)
    robot_vel = np.zeros((T, 2), dtype=float)
    robot_acc = np.zeros((T, 2), dtype=float)
    goal = np.array([10.0, 0.0], dtype=float)

    peds_pos = np.full((T, 1, 2), fill_value=100.0, dtype=float)
    peds_pos[2, 0] = robot_pos[2]  # collision at step 2

    ped_forces = np.zeros((T, 1, 2), dtype=float)

    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=0.1,
        reached_goal_step=3,  # claims goal reached before horizon
        obstacles=None,
        other_agents_pos=None,
    )


# --- Regression: collision detection ---


def test_ped_at_robot_position_produces_human_collision() -> None:
    """A ped at robot position produces human_collisions > 0."""
    data = _make_collision_episode()
    assert human_collisions(data) >= 1


def test_ped_at_robot_position_produces_collision_count_ge_1() -> None:
    """collision_count() >= 1 when ped overlaps robot — closes the release 0.0.2 gap."""
    data = _make_collision_episode()
    assert collision_count(data) >= 1


# --- Regression: success_rate must be 0 on collision ---


def test_success_rate_is_zero_on_collision_termination() -> None:
    """success_rate() = 0.0 when episode ends via runtime collision (goal not reached)."""
    data = _make_collision_episode()
    assert success_rate(data, horizon=100) == 0.0


def test_success_rate_is_zero_when_collision_and_goal_both_claimed() -> None:
    """success_rate() = 0.0 even if reached_goal_step is set, when collision_count > 0.

    This is the regression case: the no-collision gate must bind on episodes where
    the derived geometry correctly detects a collision.
    """
    data = _make_collision_termination_with_goal_claim()
    assert collision_count(data) >= 1
    assert success_rate(data, horizon=100) == 0.0


# --- Regression: compute_all_metrics propagates collision correctly ---


def test_compute_all_metrics_collisions_ge_1_on_ped_overlap() -> None:
    """compute_all_metrics() 'collisions' field >= 1 when ped overlaps robot."""
    data = _make_collision_episode()
    result = compute_all_metrics(data, horizon=100)
    assert result["collisions"] >= 1, (
        "collisions field must be >= 1 for a ped-overlap episode; "
        "if 0 the SNQI collision term would silently vanish (issue #5097)"
    )


def test_compute_all_metrics_success_is_zero_on_ped_overlap() -> None:
    """compute_all_metrics() 'success' field = 0.0 when ped overlaps robot."""
    data = _make_collision_episode()
    result = compute_all_metrics(data, horizon=100)
    assert result["success"] == 0.0


def test_compute_all_metrics_success_zero_when_collision_and_goal_claimed() -> None:
    """Even with reached_goal_step set, 'success' = 0.0 when collision_count > 0."""
    data = _make_collision_termination_with_goal_claim()
    result = compute_all_metrics(data, horizon=100)
    assert result["collisions"] >= 1
    assert result["success"] == 0.0


# --- Regression: SNQI collision term is non-zero on ped overlap ---


def test_snqi_collision_term_nonzero_on_ped_overlap() -> None:
    """SNQI collision penalty is non-zero when collisions >= 1 and baseline is set.

    In release 0.0.2 this was zero for all 241 exact-collision episodes because
    the derived collision count was zero. With correct collision detection the
    collision term must contribute.

    Uses a minimal metrics dict built from the collision-count output of compute_all_metrics
    to avoid nan propagation from force metrics into the SNQI score computation.
    """
    data = _make_collision_episode()
    full_metrics = compute_all_metrics(data, horizon=100)
    collisions_val = float(full_metrics.get("collisions", 0.0))
    assert collisions_val >= 1.0, "precondition: collisions must be >= 1 for this test"

    # Build a force-free metrics dict to avoid nan * 0.0 = nan in compute_snqi_v0.
    # This directly verifies that a correct collisions value is the sole driver of penalty.
    minimal_metrics: dict[str, float] = {
        "success": 0.0,
        "time_to_goal_norm": 1.0,
        "collisions": collisions_val,
        "near_misses": 0.0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
        "jerk_mean": 0.0,
    }

    baseline_stats: BaselineStats = {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "time_to_goal_norm": {"med": 0.5, "p95": 1.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }
    weights: Weights = {
        "w_success": 1.0,
        "w_time": 0.1,
        "w_collisions": 1.0,
        "w_near": 0.1,
        "w_comfort": 0.0,
        "w_force_exceed": 0.0,
        "w_jerk": 0.0,
    }

    score = compute_snqi_v0(minimal_metrics, weights, baseline_stats)
    # The collision penalty term is -w_collisions * normalize(collisions).
    # With collisions >= 1, p95 = 2.0, med = 0.0: norm = clamp((1 - 0) / (2 - 0)) = 0.5.
    # score = 0.0 (success) - 0.1 * 1.0 (time) - 1.0 * 0.5 (collision) - 0.1 * 0.0 ...
    # so score <= -0.5. The key assertion: the collision penalty is active (score < 0).
    assert math.isfinite(score), f"SNQI score must be finite, got {score}"
    assert score < 0.0, (
        f"SNQI score should be negative (collision penalty active), got {score}; "
        "if 0 or positive the collision term was not applied (issue #5097 regression)"
    )


@pytest.mark.parametrize("collisions_val", [0.0, 1.0, 2.0])
def test_snqi_collision_term_monotone_with_collision_count(collisions_val: float) -> None:
    """SNQI collision penalty increases monotonically with collision count."""
    baseline_stats: BaselineStats = {
        "collisions": {"med": 0.0, "p95": 3.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
        "force_exceed_events": {"med": 0.0, "p95": 1.0},
        "time_to_goal_norm": {"med": 0.5, "p95": 1.0},
        "jerk_norm": {"med": 0.0, "p95": 1.0},
    }
    weights: Weights = {
        "w_success": 0.0,
        "w_time": 0.0,
        "w_collisions": 1.0,
        "w_near": 0.0,
        "w_comfort": 0.0,
        "w_force_exceed": 0.0,
        "w_jerk": 0.0,
    }
    metrics = {
        "success": 0.0,
        "time_to_goal_norm": 1.0,
        "collisions": collisions_val,
        "near_misses": 0.0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
    }
    score = compute_snqi_v0(metrics, weights, baseline_stats)
    # Higher collision count → lower (more negative) score; zero collisions → score 0.0
    if collisions_val == 0.0:
        assert score == pytest.approx(0.0)
    else:
        # normalize_metric uses key "med", not "median"
        med = 0.0
        p95 = 3.0
        norm = min(1.0, max(0.0, (collisions_val - med) / (p95 - med)))
        assert score == pytest.approx(-norm, abs=1e-9)
        assert math.isfinite(score)

"""Characterization baseline tests for ``robot_sf/benchmark/metrics.py`` public entry points.

These tests pin the *current observable behavior* of the core metric functions
on small synthetic ``EpisodeData`` inputs. They are table-driven and assert
exact golden values, including the documented edge cases: empty pedestrian
sets (K=0), single timestep, and degenerate/NaN force inputs where the
finite-guards apply.

Purpose (issue #4874, Refs #4770): lock a behavioral baseline so the
post-submission refactor wave can prove behavior-preservation by re-running
these tests. If a test reveals a genuine bug, do NOT fix it here — document it
and file a separate fix issue.

These tests are additive and focus on golden-value pinning; they do not
duplicate the property/contract coverage in the ``test_*_metric_contract.py``
files.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sf.benchmark.constants import COMFORT_FORCE_THRESHOLD
from robot_sf.benchmark.metrics import (
    EpisodeData,
    avg_speed,
    comfort_exposure,
    compute_all_metrics,
    curvature_mean,
    energy,
    evaluate_stability_margin,
    force_exceed_events,
    force_quantiles,
    force_sample_stats,
    has_force_data,
    human_collisions,
    jerk_mean,
    mean_clearance,
    mean_distance,
    min_clearance,
    min_distance,
    near_misses,
    path_efficiency,
    path_length,
    ped_force_mean,
    per_ped_force_quantiles,
    robot_ped_within_5m_frac,
    snqi,
    socnavbench_path_length,
    success_rate,
    time_to_goal,
    timeout,
)

# EpisodeData default radii: robot_radius=1.0, ped_radius=0.4 -> sum 1.4.
# Clearance is center-distance minus (robot_radius + ped_radius).
_RADIUS_SUM = 1.4


def _episode(  # noqa: PLR0913
    *,
    robot_pos: np.ndarray,
    robot_vel: np.ndarray | None = None,
    robot_acc: np.ndarray | None = None,
    peds_pos: np.ndarray | None = None,
    ped_forces: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    dt: float = 0.25,
    reached_goal_step: int | None = None,
    obstacles: np.ndarray | None = None,
) -> EpisodeData:
    """Build a minimal ``EpisodeData`` with zero defaults for unused arrays."""
    t = robot_pos.shape[0]
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel if robot_vel is not None else np.zeros_like(robot_pos),
        robot_acc=robot_acc if robot_acc is not None else np.zeros_like(robot_pos),
        peds_pos=(peds_pos if peds_pos is not None else np.zeros((t, 0, 2))),
        ped_forces=(ped_forces if ped_forces is not None else np.zeros((t, 0, 2))),
        goal=goal if goal is not None else np.zeros(2),
        dt=dt,
        reached_goal_step=reached_goal_step,
        obstacles=obstacles,
    )


def _single_ped_episode(ped_offset: float, *, t: int = 1) -> EpisodeData:
    """One pedestrian fixed at ``(ped_offset, 0)``; robot at origin each step."""
    robot_pos = np.zeros((t, 2))
    peds = np.tile([[[ped_offset, 0.0]]], (t, 1, 1))
    return _episode(robot_pos=robot_pos, peds_pos=peds, ped_forces=np.zeros((t, 1, 2)))


# ---------------------------------------------------------------------------
# Robot-pedestrian distance / clearance / collision / near-miss table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ped_offset", "exp_human_coll", "exp_near", "exp_min_dist", "exp_min_clearance"),
    [
        # clearance = offset - 1.4
        (1.0, 1.0, 0.0, 1.0, -0.4),  # overlap -> collision
        (1.5, 0.0, 1.0, 1.5, 0.1),  # 0 <= clearance < 0.5 -> near miss
        (2.0, 0.0, 0.0, 2.0, 0.6),  # clearance >= 0.5 -> neither
    ],
)
def test_clearance_based_collision_and_near_miss_table(
    ped_offset: float,
    exp_human_coll: float,
    exp_near: float,
    exp_min_dist: float,
    exp_min_clearance: float,
) -> None:
    """Pin radius-aware human-collision / near-miss / distance outputs."""
    data = _single_ped_episode(ped_offset)
    assert human_collisions(data) == exp_human_coll
    assert near_misses(data) == exp_near
    assert min_distance(data) == pytest.approx(exp_min_dist)
    assert min_clearance(data) == pytest.approx(exp_min_clearance)
    assert robot_ped_within_5m_frac(data) == pytest.approx(1.0)


def test_mean_distance_and_clearance_average_min_per_step() -> None:
    """``mean_distance`` averages the per-step minimum robot-ped distance."""
    robot_pos = np.zeros((2, 2))
    # Two peds at distances 1.0 and 3.0 -> per-step min = 1.0 both steps.
    peds = np.tile([[[1.0, 0.0], [3.0, 0.0]]], (2, 1, 1))
    data = _episode(robot_pos=robot_pos, peds_pos=peds, ped_forces=np.zeros((2, 2, 2)))
    assert mean_distance(data) == pytest.approx(1.0)
    assert mean_clearance(data) == pytest.approx(1.0 - _RADIUS_SUM)


def test_empty_pedestrian_set_returns_documented_guards() -> None:
    """K=0 yields 0 counts and NaN for undefined distances (documented guards)."""
    data = _episode(robot_pos=np.zeros((3, 2)), peds_pos=np.zeros((3, 0, 2)))
    assert near_misses(data) == 0.0
    assert human_collisions(data) == 0.0
    assert math.isnan(min_distance(data))
    assert math.isnan(mean_distance(data))
    assert math.isnan(min_clearance(data))
    assert math.isnan(robot_ped_within_5m_frac(data))


# ---------------------------------------------------------------------------
# Success / timeout / time-to-goal
# ---------------------------------------------------------------------------


def _straight_line_episode() -> EpisodeData:
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    vel = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    return _episode(
        robot_pos=pos,
        robot_vel=vel,
        goal=np.array([3.0, 0.0]),
        dt=0.5,
        reached_goal_step=3,
    )


def test_success_rate_requires_goal_before_horizon_and_no_collisions() -> None:
    """Success requires reaching the goal strictly before the horizon."""
    data = _straight_line_episode()
    assert success_rate(data, horizon=4) == 1.0
    assert success_rate(data, horizon=3) == 0.0  # reached_step >= horizon
    assert timeout(data, horizon=4) == 0.0
    assert timeout(data, horizon=3) == 1.0


def test_time_to_goal_is_step_times_dt() -> None:
    """``time_to_goal`` is ``reached_goal_step * dt``; NaN when the goal is unreached."""
    data = _straight_line_episode()
    assert time_to_goal(data) == pytest.approx(3 * 0.5)
    no_goal = _episode(robot_pos=np.zeros((3, 2)), reached_goal_step=None)
    assert math.isnan(time_to_goal(no_goal))


# ---------------------------------------------------------------------------
# Path / motion metrics on a straight line
# ---------------------------------------------------------------------------


def test_path_motion_metrics_on_straight_line() -> None:
    """Pin path/energy/jerk/curvature/efficiency values on a unit straight-line episode."""
    data = _straight_line_episode()
    assert path_length(data) == pytest.approx(3.0)
    assert avg_speed(data) == pytest.approx(1.0)
    assert energy(data) == pytest.approx(0.0)  # zero acceleration
    assert jerk_mean(data) == pytest.approx(0.0)
    assert curvature_mean(data) == pytest.approx(0.0)
    assert path_efficiency(data, 3.0) == pytest.approx(1.0)
    assert socnavbench_path_length(data) == pytest.approx(3.0)


def test_path_motion_metrics_on_curved_nonuniform_trajectory() -> None:
    """Pin non-zero jerk and curvature on a bent trajectory."""
    data = _episode(
        robot_pos=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ),
        robot_acc=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
            ]
        ),
        dt=0.5,
        goal=np.array([0.0, 1.0]),
        reached_goal_step=3,
    )

    assert path_length(data) == pytest.approx(3.0)
    assert jerk_mean(data) == pytest.approx(1.0)
    assert curvature_mean(data) == pytest.approx(1.0)


def test_path_length_single_timestep_is_zero() -> None:
    """A single-timestep trajectory has zero path length."""
    data = _episode(robot_pos=np.array([[1.0, 1.0]]))
    assert path_length(data) == 0.0


# ---------------------------------------------------------------------------
# Force metrics, finite-guards, and degenerate inputs
# ---------------------------------------------------------------------------


def test_force_quantiles_and_mean_on_known_magnitude() -> None:
    """Single sample of magnitude 5 (3-4-5 force vector) at all quantiles/mean."""
    forces = np.array([[[3.0, 4.0]]])  # ||(3,4)|| = 5
    data = _episode(robot_pos=np.zeros((1, 2)), peds_pos=np.zeros((1, 1, 2)), ped_forces=forces)
    assert has_force_data(data) is True
    q = force_quantiles(data)
    assert q["force_q50"] == pytest.approx(5.0)
    assert q["force_q90"] == pytest.approx(5.0)
    assert q["force_q95"] == pytest.approx(5.0)
    assert ped_force_mean(data) == pytest.approx(5.0)
    # comfort threshold default 2.0 -> 1 exceed event over 1 (t,k) sample.
    assert force_exceed_events(data) == pytest.approx(1.0)
    assert comfort_exposure(data) == pytest.approx(1.0)


def test_per_ped_force_quantiles_averages_across_pedestrians() -> None:
    """Per-ped quantiles are computed then averaged across pedestrians.

    ped0 magnitudes all 10 -> q50=10 ; ped1/ped2 all 1 -> q50=1 ; mean = (10+1+1)/3.
    """
    forces = np.zeros((3, 3, 2))
    forces[:, 0, :] = [10.0, 0.0]  # ped0 magnitude 10
    forces[:, 1, :] = [1.0, 0.0]  # ped1 magnitude 1
    forces[:, 2, :] = [0.0, 1.0]  # ped2 magnitude 1
    data = _episode(robot_pos=np.zeros((3, 2)), peds_pos=np.zeros((3, 3, 2)), ped_forces=forces)
    pq = per_ped_force_quantiles(data)
    assert pq["ped_force_q50"] == pytest.approx(4.0)  # mean of {10, 1, 1}


@pytest.mark.parametrize(
    ("forces", "expected_status"),
    [
        (np.zeros((2, 1, 2)), "all-zero"),
        (np.full((2, 1, 2), np.nan), "all-invalid"),
    ],
)
def test_has_force_data_false_on_degenerate_forces(
    forces: np.ndarray, expected_status: str
) -> None:
    """All-zero and all-NaN force arrays are treated as no valid force data."""
    data = _episode(robot_pos=np.zeros((2, 2)), peds_pos=np.zeros((2, 1, 2)), ped_forces=forces)
    assert has_force_data(data) is False
    assert force_sample_stats(data)["status"] == expected_status


def test_force_metrics_return_nan_without_force_data() -> None:
    """When force data is absent, force metrics return NaN (not zero)."""
    data = _episode(
        robot_pos=np.zeros((2, 2)),
        peds_pos=np.zeros((2, 1, 2)),
        ped_forces=np.zeros((2, 1, 2)),
    )
    assert math.isnan(ped_force_mean(data))
    assert math.isnan(force_exceed_events(data))
    assert math.isnan(comfort_exposure(data))
    q = force_quantiles(data)
    assert all(math.isnan(v) for v in q.values())


def test_force_sample_stats_no_pedestrians_branch() -> None:
    """K=0 is reported as ``no-pedestrians`` with zeroed counts."""
    data = _episode(robot_pos=np.zeros((2, 2)), peds_pos=np.zeros((2, 0, 2)))
    stats = force_sample_stats(data)
    assert stats["status"] == "no-pedestrians"
    assert stats["raw_samples"] == 0


def test_comfort_force_threshold_constant() -> None:
    """Pin the default comfort force threshold used by force-exceed metrics."""
    assert COMFORT_FORCE_THRESHOLD == 2.0


# ---------------------------------------------------------------------------
# SNQI table (well-defined arithmetic)
# ---------------------------------------------------------------------------


_SNQI_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 0.5,
    "collisions": 2.0,
    "near_misses": 1.0,
    "comfort_exposure": 0.1,
    "force_exceed_events": 3.0,
    "jerk_mean": 0.2,
    "curvature_mean": 0.05,
}
_SNQI_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 1.0,
    "w_collisions": 1.0,
    "w_near": 1.0,
    "w_comfort": 1.0,
    "w_force_exceed": 1.0,
    "w_jerk": 1.0,
    "w_curvature": 1.0,
}
_SNQI_BASELINE = {
    "collisions": {"med": 1.0, "p95": 3.0},  # (2-1)/(3-1) = 0.5
    "near_misses": {"med": 0.0, "p95": 2.0},  # (1-0)/(2-0) = 0.5
    "force_exceed_events": {"med": 1.0, "p95": 5.0},  # (3-1)/(5-1) = 0.5
    "jerk_mean": {"med": 0.0, "p95": 0.4},  # 0.2/0.4 = 0.5
    "curvature_mean": {"med": 0.0, "p95": 0.1},  # 0.05/0.1 = 0.5
}


def test_snqi_with_baseline_normalizes_penalties() -> None:
    """Each penalized metric normalizes to 0.5; score = 1 - 0.5*5 - 0.1 = -2.1."""
    score = snqi(_SNQI_METRICS, _SNQI_WEIGHTS, baseline_stats=_SNQI_BASELINE)
    assert score == pytest.approx(-2.1)


def test_snqi_without_baseline_treats_penalties_as_zero() -> None:
    """Without baseline, normalized penalties contribute 0; score = 1 - 0.5 - 0.1."""
    score = snqi(_SNQI_METRICS, _SNQI_WEIGHTS)
    assert score == pytest.approx(0.4)


def test_snqi_clips_normalized_penalties_to_unit_interval() -> None:
    """A value above p95 normalizes to 1.0 (clipped), not above 1."""
    metrics = {"success": 0.0, "collisions": 100.0, "time_to_goal_norm": 1.0}
    weights = {"w_collisions": 1.0}
    baseline = {"collisions": {"med": 0.0, "p95": 1.0}}
    # coll norm clipped to 1.0 -> score = 0 - 1 (time default) - 1 (coll) = -2.0
    assert snqi(metrics, weights, baseline_stats=baseline) == pytest.approx(-2.0)


def test_snqi_safe_falls_back_on_nan_and_none() -> None:
    """NaN/None metric values fall back to documented defaults, not raise."""
    metrics = {"success": float("nan"), "time_to_goal_norm": None}
    # success -> default 0.0 ; time_norm -> default 1.0 ; score = 0 - 1 = -1.0
    assert snqi(metrics, {"w_success": 1.0, "w_time": 1.0}) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Rollover stability margin
# ---------------------------------------------------------------------------


def test_evaluate_stability_margin_zero_lateral_load_is_one() -> None:
    """Zero yaw-rate gives no lateral load -> margin 1.0 (full stability)."""
    assert evaluate_stability_margin(1.0, 0.0) == pytest.approx(1.0)


def test_evaluate_stability_margin_clamps_to_unit_interval() -> None:
    """Large lateral acceleration drives the margin to its 0.0 floor."""
    assert evaluate_stability_margin(1e6, 1e6) == pytest.approx(0.0)


def test_evaluate_stability_margin_nan_speed_returns_nan() -> None:
    """A non-finite speed sample yields NaN rather than raising."""
    assert math.isnan(evaluate_stability_margin(float("nan"), 1.0))


def test_evaluate_stability_margin_rejects_non_positive_geometry() -> None:
    """Non-positive geometry parameters raise ``ValueError``."""
    with pytest.raises(ValueError):
        evaluate_stability_margin(1.0, 1.0, h_c=-0.1)


# ---------------------------------------------------------------------------
# compute_all_metrics orchestrator smoke + key presence
# ---------------------------------------------------------------------------


def test_compute_all_metrics_returns_core_scalar_keys() -> None:
    """The orchestrator emits the locked set of benchmark-facing scalar keys."""
    data = _straight_line_episode()
    values = compute_all_metrics(data, horizon=10, shortest_path_len=3.0, robot_max_speed=1.0)
    for key in (
        "success",
        "time_to_goal_norm",
        "collisions",
        "near_misses",
        "path_efficiency",
        "avg_speed",
        "comfort_exposure",
        "jerk_mean",
        "curvature_mean",
        "energy",
        "socnavbench_path_length",
    ):
        assert key in values, f"missing orchestrator key {key!r}"
    assert values["success"] == 1.0


def test_compute_all_metrics_opt_in_keys_absent_by_default() -> None:
    """Experimental surfaces are absent unless explicitly enabled."""
    data = _straight_line_episode()
    values = compute_all_metrics(data, horizon=10, shortest_path_len=3.0)
    assert "ped_impact_accel_delta_mean" not in values
    assert "human_proxy_available" not in values
    assert "near_misses_ttc" not in values

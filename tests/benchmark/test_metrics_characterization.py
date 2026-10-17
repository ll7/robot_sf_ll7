"""Core metric behavior on small synthetic episodes.

Continuous values use controlled differences and geometric invariants (#10089).
Boundary counts, missing-data guards and independently calculated SNQI arithmetic
remain explicit contract oracles. A small release-pinned motion sentinel lives in
``test_release_metric_oracles.py``; it must not be rebaselined for moving main.
"""

from __future__ import annotations

import math
from dataclasses import replace

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


@pytest.mark.parametrize("metric", [min_distance, mean_distance, min_clearance, mean_clearance])
def test_distance_metrics_respond_only_to_nearest_pedestrian(metric) -> None:
    """Moving a far pedestrian has no effect; moving the closest changes one sample."""
    data = _episode(
        robot_pos=np.zeros((2, 2)),
        peds_pos=np.array([[[2.0, 0.0], [8.0, 0.0]], [[4.0, 0.0], [9.0, 0.0]]]),
        ped_forces=np.zeros((2, 2, 2)),
    )
    farther = replace(data, peds_pos=data.peds_pos.copy())
    farther.peds_pos[:, 1, 0] += 3.0
    assert metric(farther) == pytest.approx(metric(data))
    closer = replace(data, peds_pos=data.peds_pos.copy())
    closer.peds_pos[0, 0, 0] -= 0.75
    expected_delta = 0.75 if metric in (min_distance, min_clearance) else 0.75 / 2
    assert metric(data) - metric(closer) == pytest.approx(expected_delta)


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


def test_time_to_goal_is_completed_steps_times_dt() -> None:
    """``time_to_goal`` is ``(reached_goal_step + 1) * dt``; NaN when the goal is unreached."""
    data = _straight_line_episode()
    assert time_to_goal(data) == pytest.approx(4 * 0.5)
    no_goal = _episode(robot_pos=np.zeros((3, 2)), reached_goal_step=None)
    assert math.isnan(time_to_goal(no_goal))


# ---------------------------------------------------------------------------
# Path / motion metrics on a straight line
# ---------------------------------------------------------------------------


def _motion_episode() -> EpisodeData:
    """Bent path with nonuniform speeds and accelerations; every metric is nonzero."""
    return _episode(
        robot_pos=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
        robot_vel=np.array([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-0.5, 0.0]]),
        robot_acc=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 2.0]]),
        peds_pos=np.array([[[5.0, 1.0]], [[6.0, 2.0]], [[7.0, 3.0]], [[8.0, 4.0]]]),
        ped_forces=np.zeros((4, 1, 2)),
        goal=np.array([1.0, 1.0]),
        dt=0.5,
        reached_goal_step=3,
    )


_MOTION_METRICS = [
    path_length,
    socnavbench_path_length,
    avg_speed,
    energy,
    jerk_mean,
    curvature_mean,
]


@pytest.mark.parametrize("metric", _MOTION_METRICS)
def test_motion_metrics_follow_length_unit_scaling(metric) -> None:
    """Changing length units scales vectors; curvature has inverse-length units."""
    data = _motion_episode()
    scale = 3.0
    scaled = replace(
        data,
        robot_pos=data.robot_pos * scale,
        robot_vel=data.robot_vel * scale,
        robot_acc=data.robot_acc * scale,
        peds_pos=data.peds_pos * scale,
        goal=data.goal * scale,
        robot_radius=data.robot_radius * scale,
        ped_radius=data.ped_radius * scale,
    )
    before = metric(data)
    assert before > 0
    multiplier = 1 / scale if metric is curvature_mean else scale
    assert metric(scaled) == pytest.approx(before * multiplier)


@pytest.mark.parametrize("metric", [min_distance, mean_distance, min_clearance, mean_clearance])
def test_distance_metrics_follow_length_unit_scaling(metric) -> None:
    """Scale center separation and both radii together, retaining surface clearance units."""
    data = _single_ped_episode(2.0, t=2)
    scale = 3.0
    scaled = replace(
        data,
        robot_pos=data.robot_pos * scale,
        peds_pos=data.peds_pos * scale,
        robot_radius=data.robot_radius * scale,
        ped_radius=data.ped_radius * scale,
    )
    before = metric(data)
    assert before > 0
    assert metric(scaled) == pytest.approx(before * scale)


@pytest.mark.parametrize("transform", ["translation", "rotation"])
@pytest.mark.parametrize(
    "metric", [*_MOTION_METRICS, min_distance, mean_distance, min_clearance, mean_clearance]
)
def test_metric_invariance_under_rigid_transform(metric, transform) -> None:
    """World origin and orientation cannot change scalar motion or separation metrics."""
    data = _motion_episode()
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([7.25, -3.5])
    if transform == "translation":
        transformed = replace(
            data,
            robot_pos=data.robot_pos + offset,
            peds_pos=data.peds_pos + offset,
            goal=data.goal + offset,
        )
    else:
        transformed = replace(
            data,
            robot_pos=data.robot_pos @ rotation.T,
            peds_pos=data.peds_pos @ rotation.T,
            goal=data.goal @ rotation.T,
            robot_vel=data.robot_vel @ rotation.T,
            robot_acc=data.robot_acc @ rotation.T,
        )
    assert metric(transformed) == pytest.approx(metric(data))


def test_detour_increases_length_and_reduces_efficiency() -> None:
    """Move one intermediate point while retaining endpoints and reference distance."""
    direct = _straight_line_episode()
    detour = replace(direct, robot_pos=direct.robot_pos.copy())
    detour.robot_pos[1, 1] = 2.0
    assert path_length(detour) > path_length(direct)
    assert path_efficiency(detour, 3.0) < path_efficiency(direct, 3.0)


def test_path_length_single_timestep_is_zero() -> None:
    """A single-timestep trajectory has zero path length."""
    data = _episode(robot_pos=np.array([[1.0, 1.0]]))
    assert path_length(data) == 0.0


# ---------------------------------------------------------------------------
# Force metrics, finite-guards, and degenerate inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["force_q50", "force_q90", "force_q95", "mean"])
def test_force_metrics_scale_with_force_magnitude(key) -> None:
    """Multiply only force vectors: each quantile and the mean scale by the same factor."""
    forces = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[2.0, 1.0]]])
    data = _episode(robot_pos=np.zeros((3, 2)), peds_pos=np.zeros((3, 1, 2)), ped_forces=forces)
    scaled = replace(data, ped_forces=forces * 2.5)
    if key == "mean":
        before, after = ped_force_mean(data), ped_force_mean(scaled)
    else:
        before, after = force_quantiles(data)[key], force_quantiles(scaled)[key]
    assert before > 0
    assert after == pytest.approx(before * 2.5)


@pytest.mark.parametrize("key", ["force_q50", "force_q90", "force_q95", "mean"])
def test_force_metrics_are_invariant_under_rotation(key) -> None:
    """Changing the orientation of force vectors preserves magnitude summaries."""
    forces = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[2.0, 1.0]]])
    data = _episode(robot_pos=np.zeros((3, 2)), peds_pos=np.zeros((3, 1, 2)), ped_forces=forces)
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = replace(data, ped_forces=forces @ rotation.T)
    if key == "mean":
        assert ped_force_mean(rotated) == pytest.approx(ped_force_mean(data))
    else:
        assert force_quantiles(rotated)[key] == pytest.approx(force_quantiles(data)[key])


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

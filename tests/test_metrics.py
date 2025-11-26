"""Tests for metrics stubs ensuring interface stability and basic behaviors.

We synthesize tiny episodes for three edge cases:
1. Empty crowd (K=0) -> collisions, near_misses should stay NaN (stub) but keys exist.
2. All collisions scenario (robot overlapping pedestrians every step) -> still returns NaNs (stub) but keys exist.
3. Partial success (goal not reached) -> success key present.

Once metrics are implemented these tests can be adapted to assert numeric values;
for now they guard against accidental signature/key regressions.
"""

from __future__ import annotations

import math

import numpy as np

from robot_sf.benchmark.metrics import (
    METRIC_NAMES,
    EpisodeData,
    compute_all_metrics,
    snqi,
)


def _make_episode(T: int, K: int) -> EpisodeData:
    robot_pos = np.zeros((T, 2))
    robot_vel = np.zeros((T, 2))
    robot_acc = np.zeros((T, 2))
    peds_pos = np.zeros((T, K, 2)) if K > 0 else np.zeros((T, 0, 2))
    ped_forces = np.zeros_like(peds_pos)
    goal = np.array([5.0, 0.0])
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=goal,
        dt=0.1,
        reached_goal_step=None,
    )


def test_metrics_keys_empty_crowd():
    ep = _make_episode(T=5, K=0)
    values = compute_all_metrics(ep, horizon=10)
    for name in METRIC_NAMES:
        assert name in values, f"Missing metric key {name}"


def test_metrics_keys_all_collisions():
    ep = _make_episode(T=5, K=3)
    # Overwrite positions to simulate overlap (robot at origin, peds too)
    ep.peds_pos[:] = 0.0
    values = compute_all_metrics(ep, horizon=10)
    for name in METRIC_NAMES:
        assert name in values
    # Collisions every timestep -> collisions == T
    assert values["collisions"] == 5
    # Near misses zero because all are strict collisions
    assert values["near_misses"] == 0
    # Min distance exactly zero
    assert values["min_distance"] == 0.0
    # Mean distance also zero in this constructed case
    assert values["mean_distance"] == 0.0


def test_metrics_partial_success_flag_present():
    ep = _make_episode(T=5, K=2)
    vals = compute_all_metrics(ep, horizon=10)
    assert "success" in vals


def test_near_miss_region_only():
    # Craft positions so robot at origin; pedestrians at 0.3m (>0.25 collision) and 0.45m
    T, K = 4, 2
    ep = _make_episode(T=T, K=K)
    # Robot stays at origin
    # Place ped 0 at (0.3,0), ped 1 at (0.45,0) constant over time
    ep.peds_pos[:, 0, 0] = 0.3
    ep.peds_pos[:, 1, 0] = 0.45
    values = compute_all_metrics(ep, horizon=10)
    # No collisions
    assert values["collisions"] == 0
    # Near miss each timestep because min dist 0.3 inside [0.25,0.5)
    assert values["near_misses"] == T
    # Min distance 0.3
    assert np.isclose(values["min_distance"], 0.3)
    # Mean distance also 0.3 (constant over time)
    assert np.isclose(values["mean_distance"], 0.3)


def test_mixed_collision_and_near_miss():
    # First two timesteps collision (<0.25), next two near-miss (0.3)
    T = 4
    ep = _make_episode(T=T, K=1)
    dists = [0.1, 0.2, 0.3, 0.3]
    for t, d in enumerate(dists):
        ep.peds_pos[t, 0, 0] = d
    values = compute_all_metrics(ep, horizon=10)
    assert values["collisions"] == 2
    assert values["near_misses"] == 2
    assert np.isclose(values["min_distance"], 0.1)
    # Mean of per-timestep minimum distances: (0.1 + 0.2 + 0.3 + 0.3)/4 = 0.225
    assert np.isclose(values["mean_distance"], 0.225)


def test_success_and_time_to_goal_norm_success_case():
    # Robot moves linearly to goal without collisions
    T = 6
    ep = _make_episode(T=T, K=0)
    # Create linear motion towards goal x=5.0 reached at step 5 (< horizon 10)
    xs = np.linspace(0, 5.0, T)
    ep.robot_pos[:, 0] = xs
    ep.reached_goal_step = 5
    vals = compute_all_metrics(ep, horizon=10)
    assert vals["success"] == 1.0
    assert np.isclose(vals["time_to_goal_norm"], 5 / 10)
    # path_efficiency should be 1 for straight line
    assert np.isclose(vals["path_efficiency"], 1.0)


def test_success_failure_due_to_collision():
    T = 5
    ep = _make_episode(T=T, K=1)
    # Robot moves, but pedestrian collides at step 1
    ep.robot_pos[:, 0] = np.linspace(0, 1.0, T)
    ep.peds_pos[:, 0, 0] = 0.0  # always at origin -> collision early
    ep.reached_goal_step = 4
    vals = compute_all_metrics(ep, horizon=10)
    assert vals["collisions"] > 0
    assert vals["success"] == 0.0
    assert vals["time_to_goal_norm"] == 1.0  # failure case


def test_path_efficiency_curved_path_less_than_one():
    # Robot zig-zags to goal increasing actual length
    T = 6
    ep = _make_episode(T=T, K=0)
    # Start (0,0) to goal (5,0); zig zag in y
    xs = np.linspace(0, 5.0, T)
    ys = np.array([0, 0.5, -0.5, 0.5, -0.5, 0])
    ep.robot_pos[:, 0] = xs
    ep.robot_pos[:, 1] = ys
    ep.reached_goal_step = 5
    vals = compute_all_metrics(ep, horizon=10)
    straight = 5.0
    # Recompute actual path length for assertion reference
    diffs = ep.robot_pos[1:] - ep.robot_pos[:-1]
    actual = np.linalg.norm(diffs, axis=1).sum()
    expected_eff = min(1.0, straight / actual)
    assert np.isclose(vals["path_efficiency"], expected_eff)


def test_force_metrics_basic():
    T, K = 5, 3
    ep = _make_episode(T=T, K=K)
    # Populate forces with increasing pattern
    for t in range(T):
        for k in range(K):
            ep.ped_forces[t, k] = np.array([t + 1, k + 1])  # magnitude grows
    vals = compute_all_metrics(ep, horizon=10)
    # Quantiles should be finite
    assert (
        np.isfinite(vals["force_q50"])
        and np.isfinite(vals["force_q90"])
        and np.isfinite(vals["force_q95"])
    )
    # Exceed events with high placeholder threshold 5.0 should count some
    assert vals["force_exceed_events"] > 0
    # Comfort exposure normalized in [0,1]
    assert 0 <= vals["comfort_exposure"] <= 1


def test_force_metrics_no_peds():
    ep = _make_episode(T=4, K=0)
    vals = compute_all_metrics(ep, horizon=10)
    assert np.isnan(vals["force_q50"])  # no pedestrians
    assert vals["force_exceed_events"] == 0
    assert vals["comfort_exposure"] == 0


def test_per_ped_force_quantiles_no_peds():
    """T001: Verify K=0 returns NaN for all per-ped quantile keys."""
    ep = _make_episode(T=4, K=0)
    vals = compute_all_metrics(ep, horizon=10)
    assert np.isnan(vals["ped_force_q50"]), "Expected NaN for ped_force_q50 with no peds"
    assert np.isnan(vals["ped_force_q90"]), "Expected NaN for ped_force_q90 with no peds"
    assert np.isnan(vals["ped_force_q95"]), "Expected NaN for ped_force_q95 with no peds"


def test_per_ped_force_quantiles_single_ped():
    """T002: Verify single ped quantiles equal that pedestrian's individual quantiles."""
    T, K = 5, 1
    ep = _make_episode(T=T, K=K)
    # Single ped with varying forces: magnitudes will be [1, 5, 10, 5, 1]
    forces = [1.0, 5.0, 10.0, 5.0, 1.0]
    for t, mag in enumerate(forces):
        ep.ped_forces[t, 0] = np.array([mag, 0.0])  # Force in x direction only

    vals = compute_all_metrics(ep, horizon=10)

    # For single ped, per-ped quantiles should equal individual quantiles
    # Expected quantiles of [1, 5, 10, 5, 1]: q50=5, q90=9.5, q95=9.75
    expected_q50 = np.quantile(forces, 0.5)
    expected_q90 = np.quantile(forces, 0.9)
    expected_q95 = np.quantile(forces, 0.95)

    assert np.isclose(vals["ped_force_q50"], expected_q50), (
        f"Expected ped_force_q50={expected_q50}, got {vals['ped_force_q50']}"
    )
    assert np.isclose(vals["ped_force_q90"], expected_q90), (
        f"Expected ped_force_q90={expected_q90}, got {vals['ped_force_q90']}"
    )
    assert np.isclose(vals["ped_force_q95"], expected_q95), (
        f"Expected ped_force_q95={expected_q95}, got {vals['ped_force_q95']}"
    )


def test_per_ped_force_quantiles_multi_ped_varying():
    """T003: Verify multi-ped with varying forces shows per-ped mean differs from aggregated."""
    T, K = 3, 3
    ep = _make_episode(T=T, K=K)

    # Ped 0: consistently high forces [10, 10, 10]
    # Ped 1, 2: consistently low forces [1, 1, 1]
    for t in range(T):
        ep.ped_forces[t, 0] = np.array([10.0, 0.0])
        ep.ped_forces[t, 1] = np.array([1.0, 0.0])
        ep.ped_forces[t, 2] = np.array([1.0, 0.0])

    vals = compute_all_metrics(ep, horizon=10)

    # Per-ped medians: ped0=10, ped1=1, ped2=1 → mean = (10+1+1)/3 = 4
    expected_per_ped_median = (10.0 + 1.0 + 1.0) / 3.0
    assert np.isclose(vals["ped_force_q50"], expected_per_ped_median), (
        f"Expected per-ped median ~{expected_per_ped_median}, got {vals['ped_force_q50']}"
    )

    # Aggregated median (existing force_q50) should be 1.0 (6 samples of 1, 3 samples of 10)
    # This demonstrates the difference between per-ped and aggregated approaches
    assert vals["force_q50"] == 1.0, "Aggregated median should be 1.0"
    assert vals["ped_force_q50"] > vals["force_q50"], (
        "Per-ped median should be higher than aggregated median in this case"
    )


def test_per_ped_force_quantiles_all_identical():
    """T004: Verify all identical forces yield identical quantiles."""
    T, K = 4, 2
    ep = _make_episode(T=T, K=K)

    # All forces = 5.0
    for t in range(T):
        for k in range(K):
            ep.ped_forces[t, k] = np.array([5.0, 0.0])

    vals = compute_all_metrics(ep, horizon=10)

    # All quantiles should equal 5.0
    assert np.isclose(vals["ped_force_q50"], 5.0), f"Expected 5.0, got {vals['ped_force_q50']}"
    assert np.isclose(vals["ped_force_q90"], 5.0), f"Expected 5.0, got {vals['ped_force_q90']}"
    assert np.isclose(vals["ped_force_q95"], 5.0), f"Expected 5.0, got {vals['ped_force_q95']}"


def test_per_ped_force_quantiles_in_compute_all():
    """T005: Verify keys present in compute_all_metrics output."""
    T, K = 3, 2
    ep = _make_episode(T=T, K=K)

    # Populate forces to ensure metrics treat data as present
    ep.ped_forces[..., 0] = 1.0

    vals = compute_all_metrics(ep, horizon=10)

    # Verify all three keys are present
    assert "ped_force_q50" in vals, "ped_force_q50 key missing from metrics"
    assert "ped_force_q90" in vals, "ped_force_q90 key missing from metrics"
    assert "ped_force_q95" in vals, "ped_force_q95 key missing from metrics"

    # Verify values are finite (not NaN) since we have pedestrians
    assert np.isfinite(vals["ped_force_q50"]), "ped_force_q50 should be finite with K>0"
    assert np.isfinite(vals["ped_force_q90"]), "ped_force_q90 should be finite with K>0"
    assert np.isfinite(vals["ped_force_q95"]), "ped_force_q95 should be finite with K>0"


def test_force_metrics_missing_force_data_flagged():
    """Force metrics return NaN when pedestrian force data is absent."""

    ep = _make_episode(T=4, K=2)  # ped_forces default to zeros (treated as missing)
    vals = compute_all_metrics(ep, horizon=10)

    assert math.isnan(vals["force_q50"])
    assert math.isnan(vals["ped_force_q50"])
    assert math.isnan(vals["force_exceed_events"])
    assert math.isnan(vals["comfort_exposure"])


def test_energy_and_jerk_mean():
    T = 6
    ep = _make_episode(T=T, K=0)
    # Construct acceleration as linearly increasing in x: a_t = t
    for t in range(T):
        ep.robot_acc[t, 0] = t
    vals = compute_all_metrics(ep, horizon=10)
    # Energy = sum |a_t| = sum_{t=0}^{5} t = 15
    assert np.isclose(vals["energy"], 15.0)
    # Jerk differences: a_{t+1} - a_t = 1 for t=0..4 -> vector [1,0]; norms=1 (5 values)
    # Using first T-2=4 jerk vectors => average = 4 / 4 =1
    assert np.isclose(vals["jerk_mean"], 1.0)


def test_curvature_mean():
    T = 6
    ep = _make_episode(T=T, K=0)

    # Create a simple circular arc path for known curvature
    # For a circle of radius R, curvature κ = 1/R
    R = 2.0  # radius

    # Generate positions for a quarter circle
    angles = np.linspace(0, np.pi / 2, T)
    for t in range(T):
        ep.robot_pos[t, 0] = R * np.cos(angles[t])
        ep.robot_pos[t, 1] = R * np.sin(angles[t])

    vals = compute_all_metrics(ep, horizon=10)

    # For a circle of radius 2, expected curvature should be 1/2 = 0.5
    # Due to discrete approximation, we allow some tolerance
    expected_curvature = 1.0 / R
    assert vals["curvature_mean"] > 0.0, "Curvature should be positive for curved path"
    assert abs(vals["curvature_mean"] - expected_curvature) < 0.5, (
        f"Expected curvature ~{expected_curvature}, got {vals['curvature_mean']}"
    )


def test_curvature_mean_straight_line():
    T = 6
    ep = _make_episode(T=T, K=0)

    # Create a straight line path (zero curvature)
    for t in range(T):
        ep.robot_pos[t, 0] = t * 1.0  # moving in x direction
        ep.robot_pos[t, 1] = 0.0  # constant y

    vals = compute_all_metrics(ep, horizon=10)

    # Straight line should have zero curvature
    assert np.isclose(vals["curvature_mean"], 0.0, atol=1e-6), (
        f"Expected zero curvature for straight line, got {vals['curvature_mean']}"
    )


def test_curvature_mean_insufficient_points():
    # Test with fewer than 4 points (should return 0.0)
    T = 3
    ep = _make_episode(T=T, K=0)

    vals = compute_all_metrics(ep, horizon=10)

    # Should return 0.0 for insufficient points
    assert vals["curvature_mean"] == 0.0, (
        f"Expected 0.0 for insufficient points, got {vals['curvature_mean']}"
    )


def test_curvature_mean_invalid_dt_zero():
    # dt == 0 should safely return 0.0
    T = 6
    ep = _make_episode(T=T, K=0)
    ep.dt = 0.0
    # simple motion
    ep.robot_pos[:, 0] = np.linspace(0, 1.0, T)
    vals = compute_all_metrics(ep, horizon=10)
    assert vals["curvature_mean"] == 0.0


def test_curvature_mean_invalid_dt_nan():
    # dt NaN should safely return 0.0
    T = 6
    ep = _make_episode(T=T, K=0)
    ep.dt = float("nan")
    ep.robot_pos[:, 0] = np.linspace(0, 1.0, T)
    vals = compute_all_metrics(ep, horizon=10)
    assert vals["curvature_mean"] == 0.0


def test_force_gradient_norm_mean():
    # Create a simple linear force field Fx = x, Fy = y so |F| = sqrt(x^2+y^2).
    # Gradient norm of |F| is 1 everywhere except at origin where it's undefined (we exclude by path).
    nx, ny = 6, 4
    xs = np.linspace(0, 5, nx)
    ys = np.linspace(0, 3, ny)
    X, Y = np.meshgrid(xs, ys)
    Fx = X.copy()
    Fy = Y.copy()
    T = 5
    ep = _make_episode(T=T, K=0)
    # path along diagonal away from origin to avoid singularity at (0,0)
    ep.robot_pos[:, 0] = np.linspace(0.5, 4.5, T)
    ep.robot_pos[:, 1] = np.linspace(0.5, 2.5, T)
    ep.force_field_grid = {"X": X, "Y": Y, "Fx": Fx, "Fy": Fy}
    vals = compute_all_metrics(ep, horizon=10)
    g = vals["force_gradient_norm_mean"]
    # Expect close to 1
    assert np.isfinite(g)
    assert 0.9 <= g <= 1.1


def test_snqi_scoring():
    # Construct two metric dicts: one ideal, one poor
    good = {
        "success": 1.0,
        "time_to_goal_norm": 0.2,
        "collisions": 0.0,
        "near_misses": 1.0,
        "comfort_exposure": 0.05,
        "force_exceed_events": 2.0,
        "jerk_mean": 0.5,
        "curvature_mean": 0.1,
    }
    bad = {
        "success": 0.0,
        "time_to_goal_norm": 1.0,
        "collisions": 8.0,
        "near_misses": 12.0,
        "comfort_exposure": 0.4,
        "force_exceed_events": 20.0,
        "jerk_mean": 3.0,
        "curvature_mean": 2.5,
    }
    baseline = {
        "collisions": {"med": 1.0, "p95": 6.0},
        "near_misses": {"med": 2.0, "p95": 15.0},
        "force_exceed_events": {"med": 3.0, "p95": 25.0},
        "jerk_mean": {"med": 0.3, "p95": 2.5},
        "curvature_mean": {"med": 0.2, "p95": 2.0},
    }
    weights = {
        "w_success": 1.0,
        "w_time": 0.5,
        "w_collisions": 0.8,
        "w_near": 0.3,
        "w_comfort": 0.6,
        "w_force_exceed": 0.4,
        "w_jerk": 0.2,
        "w_curvature": 0.3,
    }
    s_good = snqi(good, weights, baseline_stats=baseline)
    s_bad = snqi(bad, weights, baseline_stats=baseline)
    assert s_good > s_bad
    # Ensure scores are finite
    assert np.isfinite(s_good) and np.isfinite(s_bad)


# =============================================================================
# Paper Metrics Tests (2306.16740v4)
# =============================================================================


def test_success_rate():
    """Test success_rate metric - binary success indicator."""
    from robot_sf.benchmark.metrics import success_rate

    # Success case: reached goal before horizon with no collisions
    ep = _make_episode(T=5, K=0)
    ep.reached_goal_step = 3
    assert success_rate(ep, horizon=10) == 1.0

    # Failure: timeout (no goal reached)
    ep.reached_goal_step = None
    assert success_rate(ep, horizon=10) == 0.0

    # Failure: reached but after horizon
    ep.reached_goal_step = 11
    assert success_rate(ep, horizon=10) == 0.0


def test_collision_count():
    """Test collision_count metric - sum of all collision types."""
    from robot_sf.benchmark.metrics import collision_count

    ep = _make_episode(T=5, K=2)
    # No obstacles or other agents -> only human collisions
    # Make pedestrians very close
    ep.peds_pos[:, 0, :] = ep.robot_pos + 0.01
    result = collision_count(ep)
    assert result >= 0.0
    assert np.isfinite(result)


def test_wall_collisions():
    """Test wall_collisions metric."""
    from robot_sf.benchmark.metrics import wall_collisions

    ep = _make_episode(T=5, K=0)

    # No obstacles -> 0.0
    ep.obstacles = None
    assert wall_collisions(ep) == 0.0

    # Add obstacles far away -> 0.0
    ep.obstacles = np.array([[10.0, 10.0], [15.0, 15.0]])
    assert wall_collisions(ep) == 0.0

    # Add obstacle close to robot -> should detect collision
    ep.obstacles = np.array([[0.01, 0.01]])  # Very close to origin
    result = wall_collisions(ep)
    assert result >= 0.0


def test_agent_collisions():
    """Test agent_collisions metric."""
    from robot_sf.benchmark.metrics import agent_collisions

    ep = _make_episode(T=5, K=0)

    # No other agents -> 0.0
    ep.other_agents_pos = None
    assert agent_collisions(ep) == 0.0

    # Add agents far away -> 0.0
    ep.other_agents_pos = np.ones((5, 2, 2)) * 10.0
    assert agent_collisions(ep) == 0.0

    # Add agent close to robot
    ep.other_agents_pos = np.zeros((5, 1, 2)) + 0.01
    result = agent_collisions(ep)
    assert result >= 0.0


def test_human_collisions():
    """Test human_collisions metric."""
    from robot_sf.benchmark.metrics import human_collisions

    ep = _make_episode(T=5, K=0)
    # No pedestrians -> 0.0
    assert human_collisions(ep) == 0.0

    # Add pedestrians far away
    ep = _make_episode(T=5, K=2)
    ep.peds_pos[:, :, :] = 10.0
    assert human_collisions(ep) == 0.0


def test_timeout():
    """Test timeout metric - binary timeout indicator."""
    from robot_sf.benchmark.metrics import timeout

    ep = _make_episode(T=5, K=0)

    # Goal reached -> no timeout
    ep.reached_goal_step = 3
    assert timeout(ep, horizon=10) == 0.0

    # Goal not reached -> timeout
    ep.reached_goal_step = None
    assert timeout(ep, horizon=10) == 1.0

    # Reached after horizon -> timeout
    ep.reached_goal_step = 11
    assert timeout(ep, horizon=10) == 1.0


def test_failure_to_progress():
    """Test failure_to_progress metric."""
    from robot_sf.benchmark.metrics import failure_to_progress

    # Robot moving toward goal -> low failures
    T = 50
    ep = _make_episode(T=T, K=0)
    ep.robot_pos[:, 0] = np.linspace(0, 4.0, T)
    ep.goal = np.array([5.0, 0.0])
    result = failure_to_progress(ep, distance_threshold=0.1, time_threshold=1.0)
    assert result >= 0.0
    assert np.isfinite(result)

    # Too short trajectory -> 0.0
    short_ep = _make_episode(T=2, K=0)
    assert failure_to_progress(short_ep, time_threshold=5.0) == 0.0


def test_stalled_time():
    """Test stalled_time metric."""
    from robot_sf.benchmark.metrics import stalled_time

    ep = _make_episode(T=10, K=0)

    # No velocity -> all time stalled
    ep.robot_vel[:, :] = 0.0
    result = stalled_time(ep, velocity_threshold=0.05)
    assert result == 10 * 0.1  # T * dt

    # High velocity -> no stalling
    ep.robot_vel[:, :] = 1.0
    result = stalled_time(ep, velocity_threshold=0.05)
    assert result == 0.0


def test_time_to_goal():
    """Test time_to_goal metric."""
    from robot_sf.benchmark.metrics import time_to_goal

    ep = _make_episode(T=10, K=0)

    # Goal reached at step 5
    ep.reached_goal_step = 5
    assert time_to_goal(ep) == 5 * 0.1

    # Goal not reached -> NaN
    ep.reached_goal_step = None
    assert np.isnan(time_to_goal(ep))


def test_path_length():
    """Test path_length metric."""
    from robot_sf.benchmark.metrics import path_length

    # Straight line path
    ep = _make_episode(T=11, K=0)
    ep.robot_pos[:, 0] = np.linspace(0, 10, 11)
    result = path_length(ep)
    assert np.isclose(result, 10.0, atol=0.01)

    # Single timestep -> 0.0
    ep_short = _make_episode(T=1, K=0)
    assert path_length(ep_short) == 0.0


def test_success_path_length():
    """Test success_path_length (SPL) metric."""
    from robot_sf.benchmark.metrics import success_path_length

    # Success with optimal path
    ep = _make_episode(T=11, K=0)
    ep.robot_pos[:, 0] = np.linspace(0, 5, 11)
    ep.reached_goal_step = 10
    ep.goal = np.array([5.0, 0.0])
    result = success_path_length(ep, horizon=20, optimal_length=5.0)
    assert result == 1.0  # Perfect efficiency

    # Failure -> 0.0
    ep.reached_goal_step = None
    result = success_path_length(ep, horizon=20, optimal_length=5.0)
    assert result == 0.0


def test_velocity_statistics():
    """Test velocity_min, velocity_avg, velocity_max metrics."""
    from robot_sf.benchmark.metrics import velocity_avg, velocity_max, velocity_min

    ep = _make_episode(T=10, K=0)
    # Varying velocity
    ep.robot_vel[:5, 0] = 1.0  # speed = 1.0
    ep.robot_vel[5:, 0] = 2.0  # speed = 2.0

    v_min = velocity_min(ep)
    v_avg = velocity_avg(ep)
    v_max = velocity_max(ep)

    assert v_min == 1.0
    assert v_max == 2.0
    assert 1.0 <= v_avg <= 2.0

    # Empty trajectory -> NaN
    ep_empty = _make_episode(T=0, K=0)
    ep_empty.robot_vel = np.empty((0, 2))
    ep_empty.robot_pos = np.empty((0, 2))
    ep_empty.robot_acc = np.empty((0, 2))
    ep_empty.peds_pos = np.empty((0, 0, 2))
    ep_empty.ped_forces = np.empty((0, 0, 2))
    assert np.isnan(velocity_min(ep_empty))


def test_acceleration_statistics():
    """Test acceleration_min, acceleration_avg, acceleration_max metrics."""
    from robot_sf.benchmark.metrics import acceleration_avg, acceleration_max, acceleration_min

    ep = _make_episode(T=10, K=0)
    ep.robot_acc[:5, 0] = 0.5
    ep.robot_acc[5:, 0] = 1.5

    a_min = acceleration_min(ep)
    a_avg = acceleration_avg(ep)
    a_max = acceleration_max(ep)

    assert a_min == 0.5
    assert a_max == 1.5
    assert 0.5 <= a_avg <= 1.5


def test_jerk_statistics():
    """Test jerk_min, jerk_avg, jerk_max metrics."""
    from robot_sf.benchmark.metrics import jerk_avg, jerk_max, jerk_min

    ep = _make_episode(T=10, K=0)
    # Linear acceleration change -> constant jerk
    ep.robot_acc[:, 0] = np.linspace(0, 1, 10)

    j_min = jerk_min(ep)
    j_avg = jerk_avg(ep)
    j_max = jerk_max(ep)

    assert np.isfinite(j_min)
    assert np.isfinite(j_avg)
    assert np.isfinite(j_max)
    assert j_min <= j_avg <= j_max

    # Too few timesteps -> NaN
    ep_short = _make_episode(T=1, K=0)
    assert np.isnan(jerk_min(ep_short))


def test_clearing_distance_statistics():
    """Test clearing_distance_min and clearing_distance_avg metrics."""
    from robot_sf.benchmark.metrics import clearing_distance_avg, clearing_distance_min

    ep = _make_episode(T=5, K=0)

    # No obstacles -> NaN
    ep.obstacles = None
    assert np.isnan(clearing_distance_min(ep))
    assert np.isnan(clearing_distance_avg(ep))

    # Add obstacles at varying distances
    ep.obstacles = np.array([[2.0, 0.0], [5.0, 0.0]])
    cd_min = clearing_distance_min(ep)
    cd_avg = clearing_distance_avg(ep)

    assert np.isfinite(cd_min)
    assert np.isfinite(cd_avg)
    assert cd_min <= cd_avg


def test_space_compliance():
    """Test space_compliance metric."""
    from robot_sf.benchmark.metrics import space_compliance

    # No pedestrians -> NaN
    ep = _make_episode(T=10, K=0)
    assert np.isnan(space_compliance(ep))

    # Pedestrians far away -> low compliance (0.0)
    ep = _make_episode(T=10, K=2)
    ep.peds_pos[:, :, :] = 10.0
    result = space_compliance(ep, threshold=0.5)
    assert result == 0.0

    # Pedestrians close -> high compliance (> 0)
    ep.peds_pos[:, 0, :] = 0.3  # Within 0.5m threshold
    result = space_compliance(ep, threshold=0.5)
    assert result > 0.0


def test_distance_to_human_min():
    """Test distance_to_human_min metric."""
    from robot_sf.benchmark.metrics import distance_to_human_min

    # No pedestrians -> NaN
    ep = _make_episode(T=5, K=0)
    assert np.isnan(distance_to_human_min(ep))

    # Pedestrians at known distances
    ep = _make_episode(T=5, K=2)
    ep.peds_pos[:, 0, :] = np.array([1.0, 0.0])
    ep.peds_pos[:, 1, :] = np.array([2.0, 0.0])
    result = distance_to_human_min(ep)
    assert np.isclose(result, 1.0, atol=0.01)


def test_time_to_collision_min():
    """Test time_to_collision_min metric."""
    from robot_sf.benchmark.metrics import time_to_collision_min

    # No pedestrians -> NaN
    ep = _make_episode(T=5, K=0)
    assert np.isnan(time_to_collision_min(ep))

    # Pedestrians with no approaching motion -> NaN
    ep = _make_episode(T=10, K=1)
    ep.robot_vel[:, :] = 0.0
    ep.peds_pos[:, 0, :] = np.array([5.0, 0.0])
    result = time_to_collision_min(ep)
    assert np.isnan(result)  # No relative motion or not approaching


def test_aggregated_time():
    """Test aggregated_time metric."""
    from robot_sf.benchmark.metrics import aggregated_time

    ep = _make_episode(T=10, K=0)
    ep.reached_goal_step = 8

    # Should return time_to_goal for single robot
    result = aggregated_time(ep)
    assert result == 8 * 0.1

    # No goal reached -> NaN
    ep.reached_goal_step = None
    result = aggregated_time(ep)
    assert np.isnan(result)


def test_all_paper_metrics_smoke():
    """Smoke test: all 22 paper metrics are callable and return float/NaN."""
    from robot_sf.benchmark.metrics import (
        acceleration_avg,
        acceleration_max,
        acceleration_min,
        agent_collisions,
        aggregated_time,
        clearing_distance_avg,
        clearing_distance_min,
        collision_count,
        distance_to_human_min,
        failure_to_progress,
        human_collisions,
        jerk_avg,
        jerk_max,
        jerk_min,
        path_length,
        space_compliance,
        stalled_time,
        success_path_length,
        success_rate,
        time_to_collision_min,
        time_to_goal,
        timeout,
        velocity_avg,
        velocity_max,
        velocity_min,
        wall_collisions,
    )

    ep = _make_episode(T=10, K=2)
    ep.reached_goal_step = 5
    ep.obstacles = np.array([[5.0, 5.0]])
    ep.other_agents_pos = np.ones((10, 1, 2)) * 8.0

    metrics = [
        (success_rate, {"horizon": 20}),
        (collision_count, {}),
        (wall_collisions, {}),
        (agent_collisions, {}),
        (human_collisions, {}),
        (timeout, {"horizon": 20}),
        (failure_to_progress, {}),
        (stalled_time, {}),
        (time_to_goal, {}),
        (path_length, {}),
        (success_path_length, {"horizon": 20, "optimal_length": 5.0}),
        (velocity_min, {}),
        (velocity_avg, {}),
        (velocity_max, {}),
        (acceleration_min, {}),
        (acceleration_avg, {}),
        (acceleration_max, {}),
        (jerk_min, {}),
        (jerk_avg, {}),
        (jerk_max, {}),
        (clearing_distance_min, {}),
        (clearing_distance_avg, {}),
        (space_compliance, {}),
        (distance_to_human_min, {}),
        (time_to_collision_min, {}),
        (aggregated_time, {}),
    ]

    for func, kwargs in metrics:
        result = func(ep, **kwargs)
        # Should return float or NaN (both are float type)
        assert isinstance(result, float), f"{func.__name__} did not return float"

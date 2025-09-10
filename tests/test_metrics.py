"""Tests for metrics stubs ensuring interface stability and basic behaviors.

We synthesize tiny episodes for three edge cases:
1. Empty crowd (K=0) -> collisions, near_misses should stay NaN (stub) but keys exist.
2. All collisions scenario (robot overlapping pedestrians every step) -> still returns NaNs (stub) but keys exist.
3. Partial success (goal not reached) -> success key present.

Once metrics are implemented these tests can be adapted to assert numeric values;
for now they guard against accidental signature/key regressions.
"""

from __future__ import annotations

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
    dt = ep.dt
    
    # Generate positions for a quarter circle
    angles = np.linspace(0, np.pi/2, T)
    for t in range(T):
        ep.robot_pos[t, 0] = R * np.cos(angles[t])
        ep.robot_pos[t, 1] = R * np.sin(angles[t])
    
    vals = compute_all_metrics(ep, horizon=10)
    
    # For a circle of radius 2, expected curvature should be 1/2 = 0.5
    # Due to discrete approximation, we allow some tolerance
    expected_curvature = 1.0 / R
    assert vals["curvature_mean"] > 0.0, "Curvature should be positive for curved path"
    assert abs(vals["curvature_mean"] - expected_curvature) < 0.5, f"Expected curvature ~{expected_curvature}, got {vals['curvature_mean']}"


def test_curvature_mean_straight_line():
    T = 6
    ep = _make_episode(T=T, K=0)
    
    # Create a straight line path (zero curvature)
    for t in range(T):
        ep.robot_pos[t, 0] = t * 1.0  # moving in x direction
        ep.robot_pos[t, 1] = 0.0      # constant y
    
    vals = compute_all_metrics(ep, horizon=10)
    
    # Straight line should have zero curvature
    assert np.isclose(vals["curvature_mean"], 0.0, atol=1e-6), f"Expected zero curvature for straight line, got {vals['curvature_mean']}"


def test_curvature_mean_insufficient_points():
    # Test with fewer than 4 points (should return 0.0)
    T = 3
    ep = _make_episode(T=T, K=0)
    
    vals = compute_all_metrics(ep, horizon=10)
    
    # Should return 0.0 for insufficient points
    assert vals["curvature_mean"] == 0.0, f"Expected 0.0 for insufficient points, got {vals['curvature_mean']}"


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

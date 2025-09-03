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

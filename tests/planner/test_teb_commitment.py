"""Tests for the testing-only TEB-inspired commitment planner."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.teb_commitment import (
    TEBCommitmentConfig,
    TEBCommitmentPlannerAdapter,
    build_teb_commitment_config,
)


def _obs(*, robot=(0.0, 0.0), heading=0.0, speed=0.0, goal=(2.0, 0.0), obstacle_cells=None):
    obstacle_cells = obstacle_cells or []
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    for row, col in obstacle_cells:
        grid[0, row, col] = 1.0
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=float),
            "velocities": np.zeros((0, 2), dtype=float),
            "count": np.asarray([0.0], dtype=float),
            "radius": 0.25,
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-2.0, -2.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([4.0, 4.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }


def test_build_teb_commitment_config_overrides_fields() -> None:
    """Config builder should thread explicit planner settings through."""
    cfg = build_teb_commitment_config(
        {"commit_gain": 0.9, "probe_distance": 1.4, "symmetry_bias": -0.25}
    )
    assert cfg.commit_gain == 0.9
    assert cfg.probe_distance == 1.4
    assert cfg.symmetry_bias == -0.25


def test_teb_commitment_planner_moves_toward_goal_in_open_space() -> None:
    """Open-space behavior should remain goal directed and bounded."""
    planner = TEBCommitmentPlannerAdapter(TEBCommitmentConfig())
    v, w = planner.plan(_obs(goal=(3.0, 0.0)))
    assert v > 0.0
    assert abs(w) <= planner.config.max_angular_speed


def test_teb_commitment_planner_commits_side_when_blocked() -> None:
    """Blocked forward corridor should trigger a non-zero steering command."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(commit_gain=0.8, commit_persistence_steps=5)
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(2, 3), (2, 2)])
    _v, w = planner.plan(obs)
    assert abs(w) > 1e-3
    assert planner._commit_ttl > 0


def test_teb_goal_change_resets_commit_state() -> None:
    """Switching to a new waypoint should drop stale side commitment state."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(commit_gain=0.8, commit_persistence_steps=5)
    )
    blocked = _obs(goal=(3.0, 0.0), obstacle_cells=[(2, 3), (2, 2)])
    planner.plan(blocked)
    assert planner._commit_ttl > 0

    planner.plan(_obs(goal=(0.0, 3.0)))

    assert planner._commit_ttl == 0
    assert planner._commit_side == 0


def test_teb_blocked_commit_ttl_ages_each_blocked_step() -> None:
    """Persistent blockage should eventually force a fresh side evaluation."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(commit_gain=0.8, commit_persistence_steps=3, symmetry_bias=0.2)
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(2, 3), (2, 2)])

    planner.plan(obs)
    assert planner._commit_ttl == 2
    planner.plan(obs)
    assert planner._commit_ttl == 1
    planner.plan(obs)
    assert planner._commit_ttl == 2


def test_teb_symmetry_bias_drives_deterministic_side_choice() -> None:
    """Positive symmetry_bias should commit left (side=1) in a symmetric, obstacle-free stall."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            symmetry_bias=0.3,
            commit_persistence_steps=2,
            low_speed_threshold=0.15,
            progress_epsilon=0.05,
        )
    )
    # Obstacle-free scene: symmetry_bias is the only side-score contributor.
    obs = _obs(goal=(3.0, 0.0), speed=0.05)
    planner.plan(obs)  # prime _last_goal_distance
    planner.plan(obs)  # stall detected → _choose_side called with score = symmetry_bias

    # Positive bias with no grid asymmetry → left (side=1) must be committed.
    assert planner._commit_side == 1


def test_teb_negative_symmetry_bias_commits_right() -> None:
    """Negative symmetry_bias should commit right (side=-1) in a symmetric, obstacle-free stall."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            symmetry_bias=-0.3,
            commit_persistence_steps=2,
            low_speed_threshold=0.15,
            progress_epsilon=0.05,
        )
    )
    obs = _obs(goal=(3.0, 0.0), speed=0.05)
    planner.plan(obs)
    planner.plan(obs)

    assert planner._commit_side == -1


def test_teb_stall_triggers_progress_recovery_commitment() -> None:
    """A low-speed, no-progress stall should activate side commitment even without obstacles."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            symmetry_bias=0.1,
            commit_persistence_steps=4,
            low_speed_threshold=0.15,
            progress_epsilon=0.05,
        )
    )
    # First step primes _last_goal_distance; robot moves slowly.
    obs = _obs(goal=(3.0, 0.0), speed=0.05)
    planner.plan(obs)
    assert planner._commit_ttl == 0  # no stall yet on first step

    # Second step: same distance, same low speed → stalled → should commit.
    planner.plan(obs)

    assert planner._commit_ttl > 0
    assert planner._commit_side != 0


def test_teb_reset_clears_stall_and_commitment_state() -> None:
    """reset() must wipe all per-episode state so a fresh episode starts clean."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(commit_persistence_steps=5, symmetry_bias=0.1)
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(2, 3)])
    planner.plan(obs)
    planner.plan(obs)  # build up state

    planner.reset()

    assert planner._commit_ttl == 0
    assert planner._commit_side == 0
    assert planner._last_goal_distance is None

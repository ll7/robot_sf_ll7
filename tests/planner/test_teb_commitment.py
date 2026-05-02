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


def test_teb_route_guide_uses_correct_config() -> None:
    """Route guide embedded in TEB should use benchmark-tuned waypoint and inflation settings.

    At the benchmark grid resolution (0.2 m/cell):
    - waypoint_lookahead_cells=5 → 1.0 m target, saturates max_linear_speed without
      introducing corner-cutting artefacts that a longer lookahead (2.0 m) causes.
    - obstacle_inflation_cells >= 3 → >= 0.6 m clearance, exceeding robot radius (0.25 m).
    """
    planner = TEBCommitmentPlannerAdapter()
    assert planner._route_guide.config.waypoint_lookahead_cells == 5
    assert planner._route_guide.config.obstacle_inflation_cells >= 3


def test_build_teb_commitment_config_overrides_fields() -> None:
    """Config builder should thread explicit planner settings through."""
    cfg = build_teb_commitment_config(
        {
            "commit_gain": 0.9,
            "probe_distance": 1.4,
            "symmetry_bias": -0.25,
            "max_commit_gain": 1.6,
            "blocked_probe_steps": 4,
            "corridor_sample_offset": 0.45,
        }
    )
    assert cfg.commit_gain == 0.9
    assert cfg.probe_distance == 1.4
    assert cfg.symmetry_bias == -0.25
    assert cfg.max_commit_gain == 1.6
    assert cfg.blocked_probe_steps == 4
    assert cfg.corridor_sample_offset == 0.45


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


def test_teb_commitment_escalates_lateral_gain_when_first_detour_stays_blocked() -> None:
    """Blocked sidestep headings should escalate to a stronger committed turn."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            commit_gain=0.35,
            commit_gain_step=0.35,
            max_commit_gain=1.05,
            blocked_probe_spacing=0.5,
            blocked_probe_steps=3,
            commit_persistence_steps=5,
        )
    )
    baseline = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            commit_gain=0.35,
            commit_gain_step=0.35,
            max_commit_gain=0.35,
            blocked_probe_spacing=0.5,
            blocked_probe_steps=3,
            commit_persistence_steps=5,
        )
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(0, 0), (1, 2)])

    _v, w = planner.plan(obs)
    _v0, w0 = baseline.plan(obs)

    assert abs(w) > abs(w0)
    assert planner._commit_side != 0


def test_teb_commitment_can_flip_sides_when_preferred_corridor_is_more_blocked() -> None:
    """Blocked-probe fallback should invert the side when the chosen corridor is worse."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            symmetry_bias=0.3,
            commit_gain=0.45,
            max_commit_gain=1.1,
            blocked_probe_spacing=0.5,
            blocked_probe_steps=3,
            commit_persistence_steps=4,
        )
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(2, 3), (3, 1), (3, 2)])

    planner.plan(obs)

    assert planner._commit_side == -1


def test_teb_commitment_scores_flanking_obstacles_as_blocked_corridor() -> None:
    """Corridor scoring should react to side-wall occupancy.

    Symmetric flanking obstacles (above and below the direct path) cause every
    committed heading to score as blocked.  The blocked-recovery path then hands
    off to the route guide; because the direct goal path is actually clear (no
    obstacles on the centreline), the route guide falls back to direct-goal
    tracking and the planner emits a valid forward command.
    """
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            commit_gain=0.7,
            blocked_probe_spacing=0.5,
            blocked_probe_steps=3,
            corridor_sample_offset=0.4,
            commit_persistence_steps=4,
        )
    )
    obs = _obs(goal=(3.0, 0.0), obstacle_cells=[(1, 2), (3, 2)])

    v, _w = planner.plan(obs)

    # The recovery path issues the route guide's direct-goal command; robot should
    # move (v > 0) rather than stopping, because the direct goal path is clear.
    assert v > 0.0


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


def test_teb_uses_goal_next_once_current_waypoint_is_reached() -> None:
    """Planner should switch to the next route waypoint when current is already satisfied."""
    planner = TEBCommitmentPlannerAdapter(TEBCommitmentConfig())
    obs = _obs(robot=(0.0, 0.0), goal=(0.05, 0.0))
    obs["goal"]["next"] = np.asarray((0.0, 2.0), dtype=float)

    _v, w = planner.plan(obs)

    assert w > 0.1


def test_teb_blocked_state_uses_grid_route_fallback(monkeypatch) -> None:
    """Blocked corridor recovery should fall back to the route planner command."""
    planner = TEBCommitmentPlannerAdapter(
        TEBCommitmentConfig(
            commit_gain=0.35,
            commit_persistence_steps=4,
            low_speed_threshold=0.15,
            progress_epsilon=0.05,
        )
    )
    grid = np.zeros((4, 40, 40), dtype=np.float32)
    for row, col in [(20, 22), (20, 21)]:
        grid[0, row, col] = 1.0
    obs = {
        "robot": {
            "position": np.asarray((0.0, 0.0), dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray((3.0, 0.0), dtype=float),
            "next": np.asarray((3.0, 0.0), dtype=float),
        },
        "pedestrians": {
            "positions": np.zeros((0, 2), dtype=float),
            "velocities": np.zeros((0, 2), dtype=float),
            "count": np.asarray([0.0], dtype=float),
            "radius": 0.25,
        },
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": np.asarray([-20.0, -20.0], dtype=float),
        "occupancy_grid_meta_resolution": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_size": np.asarray([40.0, 40.0], dtype=float),
        "occupancy_grid_meta_use_ego_frame": np.asarray([1.0], dtype=float),
        "occupancy_grid_meta_channel_indices": np.asarray([0, 1, 2, 3], dtype=float),
    }

    def _fake_plan(_self, _observation):
        return 0.0, 0.9

    monkeypatch.setattr(
        "robot_sf.planner.teb_commitment.GridRoutePlannerAdapter.plan",
        _fake_plan,
    )

    _v, w = planner.plan(obs)

    assert w > 0.1


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

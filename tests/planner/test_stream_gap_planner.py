"""Tests for the stream-gap planner."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.stream_gap import StreamGapPlannerAdapter, StreamGapPlannerConfig


def _obs(
    *, robot=(0.0, 0.0), heading=0.0, goal=(2.0, 0.0), ped_positions=None, ped_velocities=None
):
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
        },
    }


def test_stream_gap_commits_in_open_space() -> None:
    """Planner should commit when no pedestrian blocks the corridor."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(commit_speed=0.9))
    v, w = planner.plan(_obs(goal=(4.0, 0.0)))
    assert v >= 0.89
    assert abs(w) <= planner.config.max_angular_speed


def test_stream_gap_waits_for_blocking_crossing_pedestrian() -> None:
    """Planner should wait when a pedestrian currently blocks the goal corridor."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig())
    v, _w = planner.plan(
        _obs(
            goal=(4.0, 0.0),
            ped_positions=[(0.8, 0.0)],
            ped_velocities=[(0.0, 0.0)],
        )
    )
    assert v == 0.0


def test_stream_gap_approaches_when_gap_is_soon() -> None:
    """Planner should creep/approach when a free window starts shortly ahead."""
    planner = StreamGapPlannerAdapter(
        StreamGapPlannerConfig(
            safe_gap_time=0.8,
            approach_gap_time=0.8,
            corridor_half_width=0.5,
            sample_horizon=2.0,
            sample_dt=0.2,
            approach_speed=0.33,
        )
    )
    v, _w = planner.plan(
        _obs(
            goal=(4.0, 0.0),
            ped_positions=[(0.7, 0.45)],
            ped_velocities=[(0.0, 1.0)],
        )
    )
    assert 0.3 <= v <= 0.34


def test_stream_gap_commit_hold_persists_after_gap_opening() -> None:
    """Commit mode should persist briefly once the planner decides to go."""
    planner = StreamGapPlannerAdapter(StreamGapPlannerConfig(commit_hold_steps=3, commit_speed=0.8))
    open_obs = _obs(goal=(4.0, 0.0))
    v1, _ = planner.plan(open_obs)
    v2, _ = planner.plan(open_obs)
    assert v1 >= 0.79
    assert v2 >= 0.79

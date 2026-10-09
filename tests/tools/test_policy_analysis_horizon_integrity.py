"""Terminal event precedence must survive a collision at the horizon."""

import numpy as np
import pytest

from scripts.tools import policy_analysis_run as analysis


@pytest.mark.parametrize("planner", ["ppo", "goal", "orca"])
@pytest.mark.parametrize("terminal", ["collision", "success", "timeout"])
def test_horizon_preserves_terminal_event(monkeypatch, planner, terminal):
    """The real record assembler emits exactly one outcome at the last step."""
    collision = terminal == "collision"
    success = terminal == "success"
    monkeypatch.setattr(
        analysis,
        "compute_all_metrics",
        lambda *a, **kw: {"success": float(success), "collisions": float(collision)},
    )
    monkeypatch.setattr(analysis, "post_process_metrics", lambda metrics, **kw: metrics)
    monkeypatch.setattr(analysis, "sample_obstacle_points", lambda *a: None)
    monkeypatch.setattr(analysis, "compute_shortest_path_length", lambda *a: 1.0)
    trajectory = analysis.EpisodeTrajectory(
        robot_positions=[np.array([0.0, 0.0]), np.array([1.0, 0.0])],
        ped_positions=[],
        ped_forces=[],
    )
    map_def = type("Map", (), {"obstacles": [], "bounds": ((0.0, 0.0), (2.0, 2.0))})()
    record = analysis._build_episode_record(
        {"id": "horizon"},
        seed=1001,
        policy_name=planner,
        map_def=map_def,
        goal_vec=np.array([1.0, 0.0]),
        trajectory=trajectory,
        reached_goal_step=1 if success else None,
        wall_time=1.0,
        max_steps=1,
        dt=0.1,
        robot_max_speed=1.0,
        robot_radius=0.3,
        ped_radius=0.3,
        ts_start="2026-10-09T00:00:00+00:00",
        video_path=None,
        terminated=terminal != "timeout",
        truncated=terminal == "timeout",
        reached_max_steps=True,
        last_info={
            "meta": {
                "is_obstacle_collision": collision,
                "is_route_complete": success,
                "is_timesteps_exceeded": True,
            }
        },
    )
    assert record["termination_reason"] == terminal
    assert record["outcome"]["timeout_event"] is (terminal == "timeout")
    assert record["outcome"]["collision_event"] is collision
    assert record["outcome"]["route_complete"] is success

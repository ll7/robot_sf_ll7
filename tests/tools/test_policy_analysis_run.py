"""Tests for policy analysis CLI helper behavior."""

from __future__ import annotations

import argparse
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from loguru import logger

from scripts.tools import policy_analysis_run

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_policies_rejects_invalid_names():
    """Reject invalid policy names in sweeps to prevent silent typos."""
    args = argparse.Namespace(
        policy="ppo",
        policy_sweep=True,
        policies="fast_pysf_planner,typo",
    )
    with pytest.raises(ValueError, match="Invalid policies"):
        policy_analysis_run._resolve_policies(args)


def test_run_frame_extraction_logs_timeout(monkeypatch, tmp_path: Path):
    """Log and return cleanly when frame extraction hits a timeout."""
    report_json = tmp_path / "report.json"
    report_json.write_text("{}", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        exc = subprocess.TimeoutExpired(
            cmd="extract_failure_frames.py",
            timeout=60,
            output="stdout message",
            stderr="stderr message",
        )
        if not hasattr(exc, "stdout"):
            try:
                exc.stdout = exc.output
            except AttributeError:
                pass
        raise exc

    monkeypatch.setattr(policy_analysis_run.subprocess, "run", _boom)
    captured: list[str] = []
    handle = logger.add(lambda message: captured.append(str(message)), level="WARNING")
    try:
        policy_analysis_run._run_frame_extraction(report_json, output_root=tmp_path)
    finally:
        logger.remove(handle)

    joined = "\n".join(captured)
    assert "timed out" in joined.lower()
    assert str(report_json) in joined


def test_resolve_termination_reason_filters_rejects_overlap() -> None:
    """Overlapping include/exclude reasons should raise a ValueError."""
    args = argparse.Namespace(
        termination_reason=["collision", "success"],
        exclude_termination_reason=["collision"],
    )
    with pytest.raises(ValueError, match="both include and exclude"):
        policy_analysis_run._resolve_termination_reason_filters(args)


def test_record_matches_termination_reason_filter() -> None:
    """Termination-reason filtering should enforce include/exclude constraints."""
    record = {"termination_reason": "collision"}
    assert policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include=set(),
        exclude=set(),
    )
    assert policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include={"collision"},
        exclude=set(),
    )
    assert not policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include={"success"},
        exclude=set(),
    )
    assert not policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include=set(),
        exclude={"collision"},
    )


def test_summarize_records_includes_reason_counts() -> None:
    """Summary payload should include per-reason count and rate fields."""
    records = [
        {
            "status": "success",
            "termination_reason": "success",
            "metrics": {"success": 1, "collisions": 0},
        },
        {
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {"success": 0, "collisions": 1},
        },
        {
            "status": "failure",
            "termination_reason": "max_steps",
            "metrics": {"success": 0, "collisions": 0},
        },
    ]
    summary = policy_analysis_run._summarize_records(records)
    reason_counts = summary["termination_reason_counts"]
    reason_rates = summary["termination_reason_rates"]

    assert reason_counts["success"] == 1
    assert reason_counts["collision"] == 1
    assert reason_counts["max_steps"] == 1
    assert reason_rates["success"] == pytest.approx(1 / 3)
    assert reason_rates["collision"] == pytest.approx(1 / 3)


def test_summarize_records_prefers_termination_reason_for_rates() -> None:
    """Success/collision rates should follow termination_reason, not raw metrics."""
    records = [
        {
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {"success": 1, "collisions": 0},
        }
    ]
    summary = policy_analysis_run._summarize_records(records)

    assert summary["success_rate"] == pytest.approx(0.0)
    assert summary["collision_rate"] == pytest.approx(1.0)
    assert summary["termination_reason_counts"]["collision"] == 1


def test_summarize_records_does_not_count_waypoint_only_success() -> None:
    """Waypoint-level success must not be treated as route completion success."""
    records = [
        {
            "status": "failure",
            "termination_reason": "max_steps",
            "metrics": {"success": 0},
        }
    ]
    summary = policy_analysis_run._summarize_records(records)
    assert summary["success_rate"] == pytest.approx(0.0)


def test_summarize_records_counts_all_non_success_non_collision_as_failures() -> None:
    """Failure count should include all non-success/collision termination outcomes."""
    records = [
        {"status": "success", "termination_reason": "success", "metrics": {"success": 1}},
        {"status": "collision", "termination_reason": "collision", "metrics": {"success": 0}},
        {"status": "failure", "termination_reason": "max_steps", "metrics": {"success": 0}},
        {"status": "failure", "termination_reason": "terminated", "metrics": {"success": 0}},
        {"status": "failure", "termination_reason": "error", "metrics": {"success": 0}},
    ]
    summary = policy_analysis_run._summarize_records(records)
    assert summary["failures"] == 3


def test_apply_video_termination_suffix_renames_file_and_updates_metadata(tmp_path: Path) -> None:
    """Episode videos should be renamed with a termination suffix for fast visual triage."""
    src = tmp_path / "scenario_seed123_ppo.mp4"
    src.write_bytes(b"video")
    record = {"video": {"path": str(src)}}

    policy_analysis_run._apply_video_termination_suffix(
        video_path=src,
        termination_reason="collision",
        record=record,
    )

    target = tmp_path / "scenario_seed123_ppo_collision.mp4"
    assert target.exists()
    assert not src.exists()
    assert record["video"]["path"] == str(target)


def test_apply_video_termination_suffix_noop_when_video_missing(tmp_path: Path) -> None:
    """Missing videos should be ignored without mutating record metadata."""
    src = tmp_path / "missing.mp4"
    record = {"video": {"path": str(src)}}

    policy_analysis_run._apply_video_termination_suffix(
        video_path=src,
        termination_reason="success",
        record=record,
    )

    assert record["video"]["path"] == str(src)


def test_build_error_episode_record_marks_prediction_planner_as_adapter() -> None:
    """Error records should keep prediction_planner execution mode as adapter."""
    record = policy_analysis_run._build_error_episode_record(
        {"id": "s1"},
        seed=42,
        policy_name="prediction_planner",
        max_steps=100,
        ts_start="2026-03-02T00:00:00+00:00",
        error=RuntimeError("boom"),
    )
    planner_meta = record.get("algorithm_metadata", {}).get("planner_kinematics", {})
    assert planner_meta.get("execution_mode") == "adapter"


def test_build_error_episode_record_marks_fast_pysf_planner_as_adapter() -> None:
    """Error records should keep fast_pysf planners execution mode as adapter."""
    record = policy_analysis_run._build_error_episode_record(
        {"id": "s1"},
        seed=43,
        policy_name="fast_pysf_planner",
        max_steps=100,
        ts_start="2026-03-02T00:00:00+00:00",
        error=RuntimeError("boom"),
    )
    planner_meta = record.get("algorithm_metadata", {}).get("planner_kinematics", {})
    assert planner_meta.get("execution_mode") == "adapter"


def test_build_episode_record_collision_overrides_route_complete_flag(monkeypatch) -> None:
    """Collision on a route-complete step should not produce contradictory outcome payload."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 1.0, "time_to_goal_norm": 1.0},
    )
    monkeypatch.setattr(
        policy_analysis_run,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )
    monkeypatch.setattr(policy_analysis_run, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(policy_analysis_run, "compute_shortest_path_length", lambda *args: 1.0)

    class _Map:
        obstacles = []
        bounds = ((0.0, 0.0), (1.0, 1.0))

    traj = policy_analysis_run.EpisodeTrajectory(
        robot_positions=[np.array([0.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)],
        ped_positions=[],
        ped_forces=[],
    )
    record = policy_analysis_run._build_episode_record(
        {"id": "s1"},
        seed=7,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=1,
        wall_time=1.0,
        max_steps=10,
        dt=0.1,
        robot_max_speed=1.0,
        robot_radius=1.0,
        ped_radius=0.4,
        ts_start="2026-03-02T00:00:00+00:00",
        video_path=None,
        terminated=True,
        truncated=False,
        last_info={"meta": {"is_route_complete": True, "is_obstacle_collision": True}},
        reached_max_steps=False,
    )
    assert record["termination_reason"] == "collision"
    assert record["outcome"]["collision_event"] is True
    assert record["outcome"]["route_complete"] is False


def test_build_episode_record_route_complete_success_without_terminal_flags(monkeypatch) -> None:
    """Route completion in info metadata should resolve to success when no collision is present."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 1.0, "collisions": 0.0, "time_to_goal_norm": 0.2},
    )
    monkeypatch.setattr(
        policy_analysis_run,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )
    monkeypatch.setattr(policy_analysis_run, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(policy_analysis_run, "compute_shortest_path_length", lambda *args: 1.0)

    class _Map:
        obstacles = []
        bounds = ((0.0, 0.0), (1.0, 1.0))

    traj = policy_analysis_run.EpisodeTrajectory(
        robot_positions=[np.array([0.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)],
        ped_positions=[],
        ped_forces=[],
    )
    record = policy_analysis_run._build_episode_record(
        {"id": "s1"},
        seed=11,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=1,
        wall_time=1.0,
        max_steps=10,
        dt=0.1,
        robot_max_speed=1.0,
        robot_radius=1.0,
        ped_radius=0.4,
        ts_start="2026-03-02T00:00:00+00:00",
        video_path=None,
        terminated=False,
        truncated=False,
        last_info={"meta": {"is_route_complete": True, "is_obstacle_collision": False}},
        reached_max_steps=False,
    )
    assert record["termination_reason"] == "success"
    assert record["outcome"]["route_complete"] is True
    assert record["outcome"]["collision_event"] is False


def test_build_episode_record_metric_collision_overrides_route_complete(monkeypatch) -> None:
    """Metric collision evidence must override route-complete metadata."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {"success": 0.0, "collisions": 2.0, "time_to_goal_norm": 1.0},
    )
    monkeypatch.setattr(
        policy_analysis_run,
        "post_process_metrics",
        lambda metrics, **kwargs: metrics,
    )
    monkeypatch.setattr(policy_analysis_run, "sample_obstacle_points", lambda *args: None)
    monkeypatch.setattr(policy_analysis_run, "compute_shortest_path_length", lambda *args: 1.0)

    class _Map:
        obstacles = []
        bounds = ((0.0, 0.0), (1.0, 1.0))

    traj = policy_analysis_run.EpisodeTrajectory(
        robot_positions=[np.array([0.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)],
        ped_positions=[],
        ped_forces=[],
    )
    record = policy_analysis_run._build_episode_record(
        {"id": "s1"},
        seed=12,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=1,
        wall_time=1.0,
        max_steps=10,
        dt=0.1,
        robot_max_speed=1.0,
        robot_radius=1.0,
        ped_radius=0.4,
        ts_start="2026-03-02T00:00:00+00:00",
        video_path=None,
        terminated=True,
        truncated=False,
        last_info={"meta": {"is_route_complete": True, "is_obstacle_collision": False}},
        reached_max_steps=False,
    )
    assert record["termination_reason"] == "collision"
    assert record["outcome"]["route_complete"] is False
    assert record["outcome"]["collision_event"] is True


def test_collect_episode_trajectories_snapshots_mutable_simulator_buffers() -> None:
    """Trajectory collection must copy mutable simulator arrays per timestep."""

    class _Sim:
        def __init__(self) -> None:
            self._robot = np.array([0.0, 0.0], dtype=float)
            self._peds = np.array([[0.0, 0.0]], dtype=float)
            self._forces = np.array([[0.0, 0.0]], dtype=float)

        @property
        def robot_pos(self):
            return [self._robot]

        @property
        def ped_pos(self):
            return self._peds

        @property
        def last_ped_forces(self):
            return self._forces

    class _Env:
        def __init__(self) -> None:
            self.simulator = _Sim()
            self._step = 0

        def step(self, _action):
            self._step += 1
            # Mutate shared arrays in-place (aliasing hazard).
            self.simulator._robot[:] = [float(self._step), 0.0]
            self.simulator._peds[:] = [[float(self._step), 1.0]]
            self.simulator._forces[:] = [[0.0, float(self._step)]]
            terminated = self._step >= 3
            info = {"meta": {"is_route_complete": False, "is_obstacle_collision": False}}
            return {}, 0.0, terminated, False, info

        def render(self):
            return None

    class _Adapter:
        def action(self, _obs, _env, *, robot_speed: float):
            _ = robot_speed
            return np.zeros(2, dtype=float)

    outcome = policy_analysis_run._collect_episode_trajectories(
        _Env(),
        {},
        policy_adapter=_Adapter(),
        policy_model=None,
        policy_obs_adapter=None,
        robot_speed=1.0,
        record_forces=True,
        max_steps=5,
        videos=False,
    )

    robot_x = [float(step[0]) for step in outcome.trajectory.robot_positions]
    ped_x = [float(step[0, 0]) for step in outcome.trajectory.ped_positions]
    force_y = [float(step[0, 1]) for step in outcome.trajectory.ped_forces]

    assert robot_x == [1.0, 2.0, 3.0]
    assert ped_x == [1.0, 2.0, 3.0]
    assert force_y == [1.0, 2.0, 3.0]

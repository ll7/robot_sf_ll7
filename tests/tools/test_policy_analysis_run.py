"""Tests for policy analysis CLI helper behavior."""

from __future__ import annotations

import argparse
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from gymnasium import spaces
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


def test_summarize_records_backfills_collision_means_from_termination_reason() -> None:
    """Collision summaries should not stay zero when the episode terminated as collision."""
    records = [
        {
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {
                "success": 0.0,
                "collisions": 0.0,
                "ped_collision_count": 0.0,
                "obstacle_collision_count": 0.0,
                "agent_collision_count": 0.0,
            },
        }
    ]
    summary = policy_analysis_run._summarize_records(records)

    assert summary["collision_rate"] == pytest.approx(1.0)
    assert summary["total_collision_rate"] == pytest.approx(1.0)
    assert summary["unclassified_collision_rate"] == pytest.approx(1.0)
    assert summary["warnings"]


def test_summarize_records_zero_fills_missing_collision_split_metrics() -> None:
    """Collision split means should use all episodes, not only records with explicit counters."""
    records = [
        {
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {
                "success": 0.0,
                "collisions": 1.0,
                "ped_collision_count": 1.0,
                "obstacle_collision_count": 0.0,
                "agent_collision_count": 0.0,
            },
        },
        {
            "status": "error",
            "termination_reason": "error",
            "metrics": {
                "success": 0.0,
                "collisions": 0.0,
            },
        },
    ]
    summary = policy_analysis_run._summarize_records(records)

    assert summary["total_collision_rate"] == pytest.approx(0.5)
    assert summary["ped_collision_rate"] == pytest.approx(0.5)
    assert summary["obstacle_collision_rate"] == pytest.approx(0.0)
    assert summary["agent_collision_rate"] == pytest.approx(0.0)


def test_resolved_total_collision_value_prefers_split_counts_over_termination_fallback() -> None:
    """Split collision counters should be used before forcing a 1.0 termination fallback."""
    total, used_fallback = policy_analysis_run._resolved_total_collision_value(
        {
            "termination_reason": "collision",
            "metrics": {
                "collisions": 0.0,
                "ped_collision_count": 0.0,
                "obstacle_collision_count": 1.0,
                "agent_collision_count": 0.0,
            },
        }
    )

    assert total == pytest.approx(1.0)
    assert used_fallback is False


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


def test_policy_parser_supports_new_orca_variant_choices() -> None:
    """The CLI parser accepts the newly added ORCA variant policy names."""
    parser = policy_analysis_run._build_parser()
    for policy_name in [
        "socnav_orca_nonholonomic",
        "socnav_orca_dd",
        "socnav_orca_relaxed",
        "socnav_hrvo",
    ]:
        namespace = parser.parse_args(["--policy", policy_name])
        assert namespace.policy == policy_name


def test_policy_parser_preserves_existing_socnav_sacadrl_choice() -> None:
    """Adding ORCA variants must not remove existing CLI policy choices."""
    parser = policy_analysis_run._build_parser()

    namespace = parser.parse_args(["--policy", "socnav_sacadrl"])

    assert namespace.policy == "socnav_sacadrl"


def test_resolve_policies_accepts_sacadrl_and_orca_variant_in_sweep() -> None:
    """Policy sweep parsing should accept old and new SocNav policy names together."""
    args = argparse.Namespace(
        policy="goal",
        policy_sweep=True,
        policies="socnav_sacadrl,socnav_orca_dd",
    )

    policies = policy_analysis_run._resolve_policies(args)

    assert policies == ["socnav_sacadrl", "socnav_orca_dd"]


def test_build_socnav_policy_returns_valid_orca_variant_policies() -> None:
    """Variant policy names should produce a SocNav policy with the correct adapter type."""
    variant_checks = {
        "socnav_orca_nonholonomic": {
            "config_attrs": {
                "orca_heading_slowdown": 0.8,
                "orca_commit_distance": 1.8,
                "orca_commit_lateral_gain": 0.6,
            },
        },
        "socnav_orca_dd": {
            "config_attrs": {
                "orca_time_horizon": 3.0,
                "orca_neighbor_dist": 8.0,
                "orca_max_neighbors": 6,
                "orca_stall_speed_threshold": 0.1,
            },
        },
        "socnav_orca_relaxed": {
            "config_attrs": {
                "orca_time_horizon": 8.0,
                "orca_obstacle_range": 8.0,
                "orca_obstacle_threshold": 0.6,
                "orca_head_on_bias": 0.4,
                "orca_symmetry_bias": 0.15,
            },
        },
    }

    for policy_name, check in variant_checks.items():
        policy = policy_analysis_run._build_socnav_policy(
            policy_name,
            socnav_root=None,
            orca_time_horizon=None,
            orca_neighbor_dist=None,
            socnav_allow_fallback=False,
        )
        assert policy is not None
        assert hasattr(policy, "adapter")
        for attr, expected in check["config_attrs"].items():
            actual = getattr(policy.adapter.config, attr)
            assert actual == expected


def test_build_socnav_policy_returns_hrvo_policy() -> None:
    """The HRVO policy name should return a valid HRVO policy instance."""
    policy = policy_analysis_run._build_socnav_policy(
        "socnav_hrvo",
        socnav_root=None,
        orca_time_horizon=None,
        orca_neighbor_dist=None,
        socnav_allow_fallback=False,
    )
    assert policy is not None
    assert hasattr(policy, "adapter")
    assert policy.adapter.__class__.__name__ == "HRVOPlannerAdapter"


def test_build_socnav_policy_preserves_sacadrl_policy() -> None:
    """Issue 768 additions must not break the existing SA-CADRL builder path."""
    policy = policy_analysis_run._build_socnav_policy(
        "socnav_sacadrl",
        socnav_root=None,
        orca_time_horizon=None,
        orca_neighbor_dist=None,
        socnav_allow_fallback=False,
    )

    assert policy is not None
    assert hasattr(policy, "adapter")


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


def test_build_episode_record_backfills_collision_split_metrics_from_meta(monkeypatch) -> None:
    """Collision split counters should be inferred from terminal meta flags when metrics miss them."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {
            "success": 0.0,
            "collisions": 0.0,
            "ped_collision_count": 0.0,
            "obstacle_collision_count": 0.0,
            "agent_collision_count": 0.0,
            "time_to_goal_norm": 1.0,
        },
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
        seed=13,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=None,
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
        last_info={"meta": {"is_route_complete": False, "is_obstacle_collision": True}},
        reached_max_steps=False,
    )

    assert record["termination_reason"] == "collision"
    assert record["metrics"]["collisions"] == pytest.approx(1.0)
    assert record["metrics"]["obstacle_collision_count"] == pytest.approx(1.0)
    assert record["metrics"]["ped_collision_count"] == pytest.approx(0.0)
    assert record["metrics"]["agent_collision_count"] == pytest.approx(0.0)
    assert record["metrics"]["wall_collisions"] == pytest.approx(1.0)
    assert record["metrics"]["success"] is False


def test_build_episode_record_backfills_split_metrics_when_total_collision_is_present(
    monkeypatch,
) -> None:
    """Split collision counters should backfill from meta even when aggregate collisions are present."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {
            "success": 0.0,
            "collisions": 1.0,
            "ped_collision_count": 0.0,
            "obstacle_collision_count": 0.0,
            "agent_collision_count": 0.0,
            "time_to_goal_norm": 1.0,
        },
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
        seed=14,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=None,
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
        last_info={"meta": {"is_route_complete": False, "is_obstacle_collision": True}},
        reached_max_steps=False,
    )

    assert record["termination_reason"] == "collision"
    assert record["metrics"]["collisions"] == pytest.approx(1.0)
    assert record["metrics"]["obstacle_collision_count"] == pytest.approx(1.0)
    assert record["metrics"]["ped_collision_count"] == pytest.approx(0.0)
    assert record["metrics"]["agent_collision_count"] == pytest.approx(0.0)
    assert record["metrics"]["wall_collisions"] == pytest.approx(1.0)
    assert record["metrics"]["success"] is False


def test_build_episode_record_backfills_missing_terminal_split_when_other_splits_exist(
    monkeypatch,
) -> None:
    """Terminal collision subtype should backfill even if another split counter is already present."""

    monkeypatch.setattr(
        policy_analysis_run,
        "compute_all_metrics",
        lambda *args, **kwargs: {
            "success": 0.0,
            "collisions": 2.0,
            "ped_collision_count": 1.0,
            "obstacle_collision_count": 0.0,
            "agent_collision_count": 0.0,
            "wall_collisions": 0.0,
            "time_to_goal_norm": 1.0,
        },
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
        seed=15,
        policy_name="goal",
        map_def=_Map(),
        goal_vec=np.array([1.0, 0.0], dtype=float),
        trajectory=traj,
        reached_goal_step=None,
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
        last_info={"meta": {"is_route_complete": False, "is_obstacle_collision": True}},
        reached_max_steps=False,
    )

    assert record["termination_reason"] == "collision"
    assert record["metrics"]["collisions"] == pytest.approx(2.0)
    assert record["metrics"]["ped_collision_count"] == pytest.approx(1.0)
    assert record["metrics"]["obstacle_collision_count"] == pytest.approx(1.0)
    assert record["metrics"]["wall_collisions"] == pytest.approx(1.0)
    assert record["metrics"]["success"] is False


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


def test_reset_env_aligns_dict_observation_to_loaded_policy_space() -> None:
    """Policy-analysis reset path should trim extra dict keys for MultiInput PPO."""

    class _Env:
        def reset(self, *, seed: int):
            assert seed == 123
            return (
                {
                    "robot_speed": [0.0, 0.0],
                    "goal_current": [1.0, 0.0],
                    "robot_velocity_xy": [0.0, 0.0],
                },
                {},
            )

    policy_model = SimpleNamespace(
        observation_space=spaces.Dict(
            {
                "robot_speed": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "goal_current": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )
    )
    adapter = policy_analysis_run.resolve_policy_obs_adapter(policy_model)

    obs = policy_analysis_run._reset_env(
        _Env(),
        seed=123,
        policy_model=policy_model,
        policy_obs_adapter=adapter,
    )

    assert set(obs) == {"robot_speed", "goal_current"}

"""Tests for force-sample diagnostics in benchmark metric outputs."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData, compute_all_metrics, post_process_metrics


def _episode(*, ped_count: int) -> EpisodeData:
    """Build a tiny EpisodeData fixture with configurable pedestrian count."""
    steps = 3
    robot_pos = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], dtype=float)
    robot_vel = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
    robot_acc = np.zeros_like(robot_vel)
    peds_pos = np.zeros((steps, ped_count, 2), dtype=float)
    ped_forces = np.zeros_like(peds_pos)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        obstacles=np.zeros((0, 2), dtype=float),
        goal=np.array([1.0, 0.0], dtype=float),
        dt=0.1,
        reached_goal_step=None,
    )


def test_force_sample_stats_reports_no_pedestrian_case() -> None:
    """No-ped episodes should expose explicit zero-count force diagnostics."""
    ep = _episode(ped_count=0)
    metrics_raw = compute_all_metrics(ep, horizon=5)
    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)

    stats = metrics["force_sample_stats"]
    assert stats["status"] == "no-pedestrians"
    assert stats["raw_samples"] == 0
    assert stats["invalid_samples"] == 0


def test_force_sample_stats_reports_all_invalid_samples() -> None:
    """All-NaN force data should surface as all-invalid with explicit counts."""
    ep = _episode(ped_count=2)
    ep.ped_forces[:] = np.nan

    metrics_raw = compute_all_metrics(ep, horizon=5)
    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)

    stats = metrics["force_sample_stats"]
    assert stats["status"] == "all-invalid"
    assert stats["raw_samples"] == 6
    assert stats["finite_samples"] == 0
    assert stats["invalid_samples"] == 6


def test_force_sample_stats_reports_mixed_validity() -> None:
    """Mixed finite/invalid force samples should report filtered invalid contribution."""
    ep = _episode(ped_count=1)
    ep.ped_forces[:] = np.array([[[0.3, 0.0]], [[np.nan, np.nan]], [[0.0, 0.0]]], dtype=float)

    metrics_raw = compute_all_metrics(ep, horizon=5)
    metrics = post_process_metrics(metrics_raw, snqi_weights=None, snqi_baseline=None)

    stats = metrics["force_sample_stats"]
    assert stats["status"] == "ok"
    assert stats["raw_samples"] == 3
    assert stats["finite_samples"] == 2
    assert stats["invalid_samples"] == 1
    assert stats["nonzero_force_samples"] == 1

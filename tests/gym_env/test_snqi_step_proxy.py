"""Tests for RobotEnv's extracted step-level SNQI proxy collaborator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_sf.gym_env.snqi_proxy import StepSNQIProxy, extract_robot_xy


def _simulator(robot_xy: tuple[float, float], *, ped_pos=None, ped_forces=None) -> SimpleNamespace:
    """Build a compact simulator stub exposing only SNQI proxy inputs."""
    return SimpleNamespace(
        robot_poses=[(robot_xy, 0.0)],
        ped_pos=np.asarray(ped_pos if ped_pos is not None else [], dtype=float).reshape(-1, 2),
        last_ped_forces=np.asarray(
            ped_forces if ped_forces is not None else [],
            dtype=float,
        ).reshape(-1, 2),
    )


def test_snqi_proxy_reports_near_miss_and_force_exposure() -> None:
    """Step proxy should expose SNQI-aligned scalar metadata from simulator state."""
    proxy = StepSNQIProxy()
    proxy.prime(_simulator((0.0, 0.0)))

    metrics = proxy.compute_step_metrics(
        _simulator(
            (0.0, 0.0),
            ped_pos=[(0.30, 0.0), (2.0, 0.0)],
            ped_forces=[(3.0, 0.0), (0.1, 0.0)],
        ),
        dt=0.1,
    )

    assert metrics["near_misses"] == 1.0
    assert metrics["force_exceed_events"] == 1.0
    assert metrics["comfort_exposure"] == 0.5
    assert metrics["jerk_mean"] == 0.0


def test_snqi_proxy_accumulates_running_jerk_mean() -> None:
    """Step proxy should keep finite-difference jerk state across calls."""
    proxy = StepSNQIProxy()
    proxy.prime(_simulator((0.0, 0.0)))

    proxy.compute_step_metrics(_simulator((1.0, 0.0)), dt=1.0)
    proxy.compute_step_metrics(_simulator((3.0, 0.0)), dt=1.0)
    metrics = proxy.compute_step_metrics(_simulator((6.5, 0.0)), dt=1.0)

    assert metrics["jerk_mean"] == 0.5


def test_extract_robot_xy_accepts_backend_pose_shapes() -> None:
    """Robot pose extraction should stay tolerant of simulator backend payloads."""
    assert np.allclose(extract_robot_xy(((1.0, 2.0), 0.25)), np.array([1.0, 2.0]))
    assert np.allclose(extract_robot_xy(np.array([3.0, 4.0, 1.57])), np.array([3.0, 4.0]))
    assert np.allclose(extract_robot_xy(np.array([5.0])), np.zeros(2))

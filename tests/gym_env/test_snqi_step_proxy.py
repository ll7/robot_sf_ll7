"""Tests for RobotEnv's extracted step-level SNQI proxy collaborator."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_sf.gym_env.snqi_proxy import StepSNQIProxy, coerce_xy_rows, extract_robot_xy


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
            ped_pos=[(1.50, 0.0), (2.0, 0.0)],
            ped_forces=[(3.0, 0.0), (0.1, 0.0)],
        ),
        dt=0.1,
    )

    assert metrics["near_misses"] == 1.0
    assert metrics["min_distance"] == 1.5
    assert metrics["min_clearance"] == 0.10000000000000009
    assert metrics["center_distance_near_miss_diagnostic"] == 0.0
    assert metrics["force_exceed_events"] == 1.0
    assert metrics["comfort_exposure"] == 0.5
    assert metrics["jerk_mean"] == 0.0


def test_snqi_proxy_uses_ped_position_override_without_reading_simulator_ped_pos() -> None:
    """A step-local pedestrian snapshot should avoid a second simulator ped_pos read."""

    class SimulatorWithGuardedPedPos:
        """Simulator stub whose pedestrian positions must come from the override."""

        robot_poses = [((0.0, 0.0), 0.0)]
        last_ped_forces = np.array([[3.0, 0.0], [0.1, 0.0]], dtype=float)

        @property
        def ped_pos(self):
            raise AssertionError("ped_pos should not be read when an override is provided")

    proxy = StepSNQIProxy()

    metrics = proxy.compute_step_metrics(
        SimulatorWithGuardedPedPos(),
        dt=0.1,
        ped_positions_override=np.array([[1.50, 0.0], [2.0, 0.0]], dtype=float),
    )

    assert metrics["near_misses"] == 1.0
    assert metrics["center_distance_near_miss_diagnostic"] == 0.0
    assert metrics["force_exceed_events"] == 1.0
    assert metrics["comfort_exposure"] == 0.5


def test_snqi_proxy_names_center_distance_band_as_diagnostic_only() -> None:
    """Raw center-distance bands are no longer safety-facing near-miss metrics."""
    proxy = StepSNQIProxy()
    proxy.prime(_simulator((0.0, 0.0)))

    metrics = proxy.compute_step_metrics(
        _simulator((0.0, 0.0), ped_pos=[(0.30, 0.0)]),
        dt=0.1,
    )

    assert metrics["near_misses"] == 0.0
    assert metrics["min_clearance"] < 0.0
    assert metrics["center_distance_near_miss_diagnostic"] == 1.0


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
    assert np.allclose(extract_robot_xy(None), np.zeros(2))
    assert np.allclose(extract_robot_xy("invalid"), np.zeros(2))


def test_snqi_proxy_accepts_tuple_and_array_robot_pose_sequences() -> None:
    """Robot pose payloads may be lists, tuples, or arrays depending on backend."""
    proxy = StepSNQIProxy()

    proxy.prime(SimpleNamespace(robot_poses=(((1.0, 2.0), 0.0),)))
    assert np.allclose(proxy.state.prev_robot_pos, np.array([1.0, 2.0]))

    metrics = proxy.compute_step_metrics(
        SimpleNamespace(
            robot_poses=np.array([[1.5, 2.0, 0.0]]),
            ped_pos=None,
            last_ped_forces=None,
        ),
        dt=0.1,
    )

    assert metrics["near_misses"] == 0.0
    assert metrics["force_exceed_events"] == 0.0


def test_coerce_xy_rows_handles_malformed_payloads() -> None:
    """Malformed simulator payloads should coerce to empty rows."""
    assert coerce_xy_rows(None).shape == (0, 2)
    assert coerce_xy_rows("invalid").shape == (0, 2)
    assert coerce_xy_rows({"x": 1.0}).shape == (0, 2)

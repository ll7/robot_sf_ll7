"""D-055 path geometry regressions; synthetic arrays never step a simulator."""

import math

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData, curvature_mean


def _episode(positions: list[list[float]] | np.ndarray, dt: float = 0.1) -> EpisodeData:
    """Build aligned metric inputs without an environment or random seed."""
    pos = np.asarray(positions, dtype=float)
    return EpisodeData(
        robot_pos=pos,
        robot_vel=np.zeros_like(pos),
        robot_acc=np.zeros_like(pos),
        peds_pos=np.empty((len(pos), 0, 2)),
        ped_forces=np.empty((len(pos), 0, 2)),
        goal=np.zeros(2),
        dt=dt,
    )


def test_creep_segment_is_bounded() -> None:
    """A perpendicular 1e-6 m/s creep must not dominate two straight path legs."""
    # At dt=.1, the middle displacement is 1e-7 m. Old cross/speed^3
    # contributes ~1e13 to the time mean; the two counted legs point east.
    data = _episode([[0, 0], [1, 0], [1, 1e-7], [2, 1e-7]])
    result = curvature_mean(data)
    assert math.isfinite(result)
    assert 0.0 <= result <= 1.0
    assert result == 0.0


def test_straight_line() -> None:
    """Collinear forward steps have no turning."""
    assert curvature_mean(_episode([[0, 0], [1, 0], [2, 0], [3, 0]])) == 0.0


@pytest.mark.parametrize("radius", [0.5, 2.0, 10.0])
def test_circle(radius: float) -> None:
    """A finely sampled circle tends to reciprocal radius."""
    angle = np.linspace(0, 2 * np.pi, 4001)
    pos = radius * np.column_stack((np.cos(angle), np.sin(angle)))
    # For radius=.5, use coarser sampling so every step exceeds 1 mm.
    if radius == 0.5:
        pos = pos[::2]
    assert curvature_mean(_episode(pos)) == pytest.approx(1 / radius, rel=1e-3)


def test_stop_and_rotate_then_leave_counts_one_turn() -> None:
    """An arbitrary in-place dwell separates two 1 m legs meeting at 90 degrees."""
    data = _episode([[0, 0], [1, 0], [1, 0], [1, 0], [1, 1]])
    assert curvature_mean(data) == pytest.approx(np.pi / 4)


def test_reversal_counts_pi() -> None:
    """Reversing after a stop contributes pi radians over two metres."""
    data = _episode([[0, 0], [1, 0], [1, 0], [0, 0]])
    assert curvature_mean(data) == pytest.approx(np.pi / 2)


def test_short_path_length_floor() -> None:
    """A 0.4 m path turns pi/2 and divides by 1 m, not by 0.4 m."""
    data = _episode([[0, 0], [0.1, 0], [0.2, 0], [0.2, 0.2]])
    assert curvature_mean(data) == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("second_step", [0.000999, 0.001])
def test_displacement_threshold_is_inclusive(second_step: float) -> None:
    """Only a second step of at least 1 mm can contribute a turn."""
    data = _episode([[0, 0], [0.001, 0], [0.001, second_step]])
    expected = np.pi / 2 if second_step == 0.001 else 0.0
    assert curvature_mean(data) == pytest.approx(expected)


@pytest.mark.parametrize("positions", [[], [[0, 0]], [[0, 0], [1, 0]], [[0, 0]] * 4])
def test_fewer_than_two_counted_steps(positions: list[list[float]]) -> None:
    """Empty, stationary and single-step paths have no defined turning."""
    assert curvature_mean(_episode(np.asarray(positions).reshape(-1, 2))) == 0.0


@pytest.mark.parametrize("dt", [0.01, 1.0, 0.0, float("nan")])
def test_timestep_does_not_change_path_geometry(dt: float) -> None:
    """Sampling time must not rescale the same recorded geometric path."""
    assert curvature_mean(_episode([[0, 0], [1, 0], [1, 1]], dt=dt)) == pytest.approx(np.pi / 4)


def test_angle_wrap_across_pi_boundary() -> None:
    """Directions +135 and -135 degrees meet at 90, rather than 270, degrees."""
    data = _episode([[0, 0], [-1, 1], [-2, 0]])
    assert curvature_mean(data) == pytest.approx(np.pi / (4 * np.sqrt(2)))


def test_reset_to_first_step_is_part_of_path() -> None:
    """Post-step recording and reset-inclusive recording describe the same path."""
    post = _episode([[1, 0], [1, 1]])
    post.initial_robot_pos = np.array([0.0, 0.0])
    inclusive = _episode([[0, 0], [1, 0], [1, 1]])
    inclusive.initial_robot_pos = np.array([0.0, 0.0])
    inclusive.robot_pos_includes_reset = True
    assert curvature_mean(post) == pytest.approx(np.pi / 4)
    assert curvature_mean(inclusive) == curvature_mean(post)


def test_nonfinite_positions_remain_finite() -> None:
    """Invalid displacement samples cannot produce NaN curvature."""
    data = _episode([[0, 0], [1, 0], [float("nan"), 0], [1, 0], [1, 1]])
    assert curvature_mean(data) == pytest.approx(np.pi / 4)


@pytest.mark.parametrize(
    ("positions", "dt", "expected"),
    [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], 0.5, 1.0),
        ([[0, 0], [1, 0], [1, 1e-7], [2, 1e-7]], 0.1, 5e13),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], 0.0, 0.0),
        ([[0, 0], [1, 0], [1, 1]], 0.1, 0.0),
    ],
)
def test_historical_v1_rows_keep_exact_values(
    positions: list[list[float]], dt: float, expected: float
) -> None:
    """Explicit and unmarked historical rows dispatch to unchanged v1 geometry."""
    from robot_sf.benchmark.metric_definitions import metric_schema_version

    data = _episode(positions, dt)
    # Values captured from the starting head, not derived by the new implementation.
    for row in ({}, {"metric_schema_version": "robot-sf-metrics.v1"}):
        assert curvature_mean(data, metric_schema_version=metric_schema_version(row)) == expected


def test_curvature_is_declared_as_changed_metric() -> None:
    """Definition comparisons must suppress cross-version curvature effects."""
    from robot_sf.benchmark.metric_definitions import changed_metric_field

    assert changed_metric_field("metrics.curvature_mean")

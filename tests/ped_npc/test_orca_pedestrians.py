"""Native ORCA contracts with analytic free motion and geometric clearance oracles."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.ped_npc import orca_pedestrians


def _step(state, speeds=None, **overrides):
    """Use fixed small-fixture settings without any RNG or SFM providers."""
    settings = {
        "dt": 0.1,
        "radius": 0.2,
        "obstacle_segments": np.empty((0, 4)),
        "robot_discs": np.empty((0, 5)),
        **overrides,
    }
    if speeds is None:
        speeds = np.ones(len(state))
    return orca_pedestrians.step_orca_pedestrians(state, speeds, **settings)


def test_native_free_motion_caps_speed_and_does_not_overshoot_goal() -> None:
    """Independent actors move by v*dt, stop at a near goal, and preserve metadata."""
    state = np.array(
        [
            [0, 0, 0, 0, 10, 0, 0.5],
            [0, 20, 0, 0, 0.02, 20, 0.5],
            [0, 40, 0, 0, 10, 40, 0.5],
            [0, 60, 0, 0, 0, 60, 0.5],
        ],
        dtype=float,
    )
    result = _step(state, np.array([0.5, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        result[:, :4],
        [
            [0.05, 0, 0.5, 0],
            [0.02, 20, 0.2, 0],
            [0, 40, 0, 0],
            [0, 60, 0, 0],
        ],
        atol=1e-6,
    )
    np.testing.assert_array_equal(result[:, 4:], state[:, 4:])
    assert state[0, 0] == 0.0


def test_native_head_on_crowd_and_wall_keep_disc_clearance() -> None:
    """Two head-on discs and a goal behind a wall do not penetrate their geometry."""
    state = np.array([[-1, 0, 1, 0, 10, 0], [1, 0, -1, 0, -10, 0]], dtype=float)
    for _ in range(30):
        state = _step(state)
        assert np.linalg.norm(state[0, :2] - state[1, :2]) >= 0.4 - 1e-6
        assert np.max(np.linalg.norm(state[:, 2:4], axis=1)) <= 1.0 + 1e-6
    wall_actor = np.array([[0, 0, 1, 0, 3, 0]], dtype=float)
    for _ in range(30):
        wall_actor = _step(wall_actor, obstacle_segments=np.array([[1, -2, 1, 2]]))
        assert wall_actor[0, 0] <= 0.8 + 1e-6
    assert wall_actor[0, 0] > 0.0


def test_native_robot_disc_changes_pedestrian_motion() -> None:
    """The current robot disc constrains motion without changing robot dynamics."""
    state = np.array([[0, 0, 1, 0, 10, 0]], dtype=float)
    robots = np.array([[1, 0, 0, 0, 0.3]])
    result = _step(state, robot_discs=robots)
    assert result[0, 0] < 0.1 - 1e-6
    np.testing.assert_array_equal(robots, [[1, 0, 0, 0, 0.3]])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"state": np.zeros((1, 5))}, "state"),
        ({"speeds": np.array([-1.0])}, "max_speeds"),
        ({"speeds": np.array([1.0, 1.0])}, "max_speeds"),
        ({"obstacle_segments": np.zeros((1, 3))}, "obstacle_segments"),
        ({"robot_discs": np.zeros((1, 5))}, "robot_discs"),
        ({"robot_discs": np.zeros((1, 4))}, "robot_discs"),
        ({"speeds": np.array([np.nan])}, "finite"),
        ({"dt": 0.0}, "dt and radius"),
        ({"radius": -1.0}, "dt and radius"),
    ],
)
def test_invalid_inputs_fail_before_native_solver(overrides, message) -> None:
    """Malformed native inputs are rejected before C++ can consume them."""
    values = {"state": np.zeros((1, 6)), **overrides}
    with pytest.raises(ValueError, match=message):
        _step(**values)


def test_empty_crowd_and_degenerate_wall_are_supported() -> None:
    """Empty populations retain shape; zero-length wall markers never enter RVO2."""
    assert _step(np.empty((0, 7))).shape == (0, 7)
    result = _step(np.array([[0, 0, 0, 0, 10, 0]]), obstacle_segments=np.zeros((1, 4)))
    assert result[0, 0] == pytest.approx(0.1, abs=1e-6)


def test_missing_native_backend_fails_without_sfm_fallback(monkeypatch) -> None:
    """A selected independent model cannot silently fall back to Social Force."""

    def unavailable(_name):
        raise ImportError("not installed")

    monkeypatch.setattr(orca_pedestrians, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="requires the native rvo2 backend"):
        _step(np.zeros((1, 6)))

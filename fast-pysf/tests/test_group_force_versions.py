"""Behaviour regressions for opt-in Moussaid group interactions."""

import numpy as np
import pytest
from pysocialforce.config import GroupGazeForceConfig, GroupReplusiveForceConfig, SceneConfig
from pysocialforce.forces import GroupGazeForceAlt, GroupRepulsiveForce
from pysocialforce.scene import PedState


def _peds(companion, *, velocity=(1.0, 0.0), goal=(10.0, 0.0)):
    return PedState(
        np.array([[0.0, 0.0, *velocity, *goal, 0.5], [*companion, 1.0, 0.0, 10.0, 0.0, 0.5]]),
        [[0, 1]],
        SceneConfig(),
    )


def _gaze(peds, phi=90.0):
    config = GroupGazeForceConfig(factor=4.0, fov_phi=phi)
    # Existing configs accept attributes: on base this selector is silently ignored,
    # so red proof fails on the physical result rather than a missing new API.
    config.law_version = "moussaid_2010_v2"
    return GroupGazeForceAlt(config, peds)()


def test_gaze_companions_ahead_do_not_propel_and_behind_brake():
    """Gaze companions ahead do not propel and behind brake."""
    assert _gaze(_peds((1.0, 0.0)))[0] == pytest.approx([0.0, 0.0])
    assert _gaze(_peds((-1.0, 0.0)))[0] == pytest.approx([-2.0 * np.pi, 0.0])


def test_gaze_respects_fov_and_brakes_actual_velocity_without_goal_singularity():
    """Gaze respects fov and brakes actual velocity without goal singularity."""
    # A companion at 90 degrees is visible at phi=90, needs pi/4 turn at phi=45.
    assert _gaze(_peds((0.0, 1.0)), 90.0)[0] == pytest.approx([0.0, 0.0])
    assert _gaze(_peds((0.0, 1.0)), 45.0)[0] == pytest.approx([-np.pi, 0.0])
    # Velocity, not the perpendicular waypoint direction, defines gaze and braking.
    for goal in [(0.0, 10.0), (0.0, 1e-8), (0.0, 0.0)]:
        peds = _peds((0.0, -1.0), velocity=(0.0, 2.0), goal=goal)
        assert _gaze(peds)[0] == pytest.approx([0.0, -4.0 * np.pi])
    assert _gaze(_peds((-1.0, 0.0), velocity=(0.0, 0.0)))[0] == pytest.approx([0.0, 0.0])


def test_repulsion_keeps_unit_magnitude_as_separation_shrinks():
    """Repulsion keeps unit magnitude as separation shrinks."""
    config = GroupReplusiveForceConfig(factor=1.0, threshold=0.55)
    config.law_version = "unit_vectors_v2"
    for separation in [0.5, 0.1, 1e-9]:
        assert GroupRepulsiveForce(config, _peds((separation, 0.0)))() == pytest.approx(
            np.array([[-1.0, 0.0], [1.0, 0.0]])
        )
    assert GroupRepulsiveForce(config, _peds((0.56, 0.0)))() == pytest.approx(np.zeros((2, 2)))
    assert np.isfinite(GroupRepulsiveForce(config, _peds((0.0, 0.0)))()).all()

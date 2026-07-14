"""Tests for the test_forces module."""

import sys
from collections.abc import Callable
from threading import Event, Thread

import numpy as np
import pytest
from pysocialforce import Simulator_v2 as Simulator
from pysocialforce import forces
from pysocialforce.config import SimulatorConfig
from pysocialforce.map_config import MapDefinition
from pysocialforce.ped_grouping import PedestrianGroupings, PedestrianStates


def _assert_releases_gil(call: Callable[[], None]) -> None:
    """Assert that ``call`` lets a ready sibling Python thread run."""
    ready = Event()
    start = Event()
    progressed = Event()
    stop = Event()

    def _sibling_worker() -> None:
        ready.set()
        start.wait()
        progressed.set()
        stop.wait()

    worker = Thread(target=_sibling_worker, daemon=True)
    worker.start()
    assert ready.wait(timeout=1.0)

    previous_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1.0)
    try:
        start.set()
        call()
        assert progressed.is_set(), "compiled force call held the GIL for its complete runtime"
    finally:
        sys.setswitchinterval(previous_switch_interval)
        stop.set()
        worker.join(timeout=1.0)


@pytest.fixture()
def generate_scene():
    """Test case for the force calculation functionality."""
    raw_states = np.zeros((5, 7))
    raw_states[:, :4] = np.array(
        [[1, 1, 1, 0], [1, 1.1, 0, 1], [3, 3, 1, 1], [3, 3.01, 1, 2], [3, 4, 3, 1]]
    )

    def populate(sim_config: SimulatorConfig, map_def: MapDefinition):
        """Create pedestrian states and groupings for testing.

        Args:
            sim_config: Simulator configuration parameters.
            map_def: Map definition with obstacles and routes.
        """
        states = PedestrianStates(raw_states)
        groupings = PedestrianGroupings(states, {})
        return states, groupings, []

    scene = Simulator(populate=populate)
    return scene


@pytest.fixture()
def generate_scene_with_groups():
    """Test case for the force calculation functionality."""
    groups = {0: {1, 0}, 1: {3, 2}}
    raw_states = np.zeros((5, 7))
    raw_states[:, :4] = np.array(
        [[1, 1, 1, 0], [1, 1.1, 0, 1], [3, 3, 1, 1], [3, 3.01, 1, 2], [3, 4, 3, 1]]
    )

    def populate(sim_config: SimulatorConfig, map_def: MapDefinition):
        """Create pedestrian states and groupings for testing.

        Args:
            sim_config: Simulator configuration parameters.
            map_def: Map definition with obstacles and routes.
        """
        states = PedestrianStates(raw_states)
        groupings = PedestrianGroupings(states, groups)
        return states, groupings, []

    scene = Simulator(populate=populate)
    return scene


def test_desired_force(generate_scene: Simulator):
    """Test desired force computation.

    Verifies force calculations against known expected values.

    Args:
        generate_scene: Fixture that creates a test scene.
    """
    scene = generate_scene
    config = scene.config
    f = forces.DebuggableForce(forces.DesiredForce(config.desired_force_config, scene.peds))
    config.desired_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array(
            [
                [-3.83847763, -1.83847763],
                [-1.74894926, -3.92384419],
                [-4.6, -4.6],
                [-6.10411508, -8.11779546],
                [-10.93315315, -8.57753753],
            ]
        )
    )


def test_social_force(generate_scene: Simulator):
    """Test social force computation.

    Verifies force calculations against known expected values.

    Args:
        generate_scene: Fixture that creates a test scene.
    """
    scene = generate_scene
    config = scene.config
    f = forces.DebuggableForce(forces.SocialForce(config.social_force_config, scene.peds))
    config.social_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array(
            [
                [3.18320152e-12, 1.74095049e-12],
                [-3.64726290e-05, -6.76204532e-05],
                [7.86014187e-03, 1.66840389e-04],
                [7.70524167e-03, -3.03788477e-05],
                [9.12767677e-06, 1.23117582e-05],
            ]
        )
    )


def test_gil_releasing_social_force_preserves_output_and_allows_parallel_steps():
    """The opt-in kernel must be exact and let a sibling Python worker progress."""
    num_pedestrians = 2_000
    positions = np.column_stack(
        (
            np.arange(num_pedestrians, dtype=np.float64) * 0.01,
            np.zeros(num_pedestrians, dtype=np.float64),
        )
    )
    velocities = np.zeros_like(positions)
    args = (positions, velocities, 3.0, 2, 3, 2.0, 0.35)
    expected = forces.social_force(*args)
    actual = forces._social_force_gil_releasing(*args)

    np.testing.assert_array_equal(actual, expected)
    _assert_releases_gil(lambda: forces._social_force_gil_releasing(*args))


def test_social_force_context_selects_gil_releasing_kernel(generate_scene, monkeypatch):
    """The alternate dispatcher must remain opt-in and scoped to its context."""
    scene = generate_scene
    calculator = forces.SocialForce(scene.config.social_force_config, scene.peds)
    calls = {"default": 0, "gil_releasing": 0}

    def _record_default(*args):
        calls["default"] += 1
        return np.zeros_like(args[0])

    def _record_gil_releasing(*args):
        calls["gil_releasing"] += 1
        return np.zeros_like(args[0])

    monkeypatch.setattr(forces, "social_force", _record_default)
    monkeypatch.setattr(forces, "_social_force_gil_releasing", _record_gil_releasing)

    calculator()
    with forces.social_force_gil_releasing_context():
        calculator()
    calculator()

    assert calls == {"default": 2, "gil_releasing": 1}


def test_group_rep_force(generate_scene_with_groups: Simulator):
    """Test group repulsive force computation.

    Verifies force calculations against known expected values.

    Args:
        generate_scene_with_groups: Fixture that creates a test scene with groups.
    """
    scene = generate_scene_with_groups
    print(scene)
    config = scene.config
    scene.peds.groups = [[1, 0], [3, 2]]
    f = forces.DebuggableForce(
        forces.GroupRepulsiveForce(config.group_repulsive_force_config, scene.peds)
    )
    config.group_repulsive_force_config.factor = 1.0
    assert f(debug=True) == pytest.approx(
        np.array([[0.0, -0.1], [0.0, 0.1], [0.0, -0.01], [0.0, 0.01], [0.0, 0.0]])
    )


def test_group_gaze_force_handles_zero_distance():
    """Group gaze force should not divide by zero when ped distance is zero."""
    member_pos = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    member_directions = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    member_dist = np.array([0.0, 0.0], dtype=np.float32)
    forces_out = forces.group_gaze_force(member_pos, member_directions, member_dist)
    assert np.all(np.isfinite(forces_out))
    assert np.allclose(forces_out, 0.0)

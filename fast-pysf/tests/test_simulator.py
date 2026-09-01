"""Integration-style simulator smoke tests for the fast-pysf package."""

import warnings

import numpy as np
import pysocialforce as pysf
import pytest
from pysocialforce.config import SceneConfig
from pysocialforce.force_trace import annotate_force_component
from pysocialforce.ped_grouping import PedestrianGroupings, PedestrianStates
from pysocialforce.scene import PedState


class _CountingForce:
    """Callable force fixture whose evaluations are externally observable."""

    def __init__(self, values: np.ndarray, calls: list[str], name: str) -> None:
        self.values = values
        self.calls = calls
        self.name = name

    def __call__(self) -> np.ndarray:
        """Return the configured force and record exactly one evaluation."""
        self.calls.append(self.name)
        return self.values.copy()


def test_can_simulate_with_empty_map_no_peds():
    """Simulator should step without errors on an empty default map."""
    simulator = pysf.Simulator_v2()
    result = simulator.compute_force_components()
    assert result.base_total.shape == (0, 2)
    for _ in range(10):
        simulator.step()
        print(simulator)


def test_can_simulate_with_populated_map():
    """Simulator should step without errors when obstacles/routes/zones are present."""
    obstacle01 = pysf.map_config.Obstacle([(10, 10), (15, 10), (15, 15), (10, 15)])
    obstacle02 = pysf.map_config.Obstacle([(20, 10), (25, 10), (25, 15), (20, 15)])

    route01 = pysf.map_config.GlobalRoute([(0, 0), (10, 10), (20, 10), (30, 0)])
    crowded_zone01 = ((10, 10), (20, 10), (20, 20))

    map_def = pysf.map_config.MapDefinition(
        obstacles=[obstacle01, obstacle02], routes=[route01], crowded_zones=[crowded_zone01]
    )

    simulator = pysf.Simulator_v2(map_def)

    for _ in range(10):
        simulator.step()
        print(simulator.states.ped_positions)


def test_compute_forces_accumulates_multiple_force_components():
    """compute_forces should sum each force component explicitly into an array."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    def make_forces(_, __):
        return [
            lambda: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            lambda: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        ]

    simulator = pysf.Simulator(state=state, make_forces=make_forces)
    forces = simulator.compute_forces()

    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 2)
    assert np.array_equal(forces, np.array([[6.0, 8.0], [10.0, 12.0]], dtype=float))


def test_compute_force_components_evaluates_each_force_once_and_preserves_default_sum():
    """The diagnostic roster must be exact without changing aggregate force semantics."""
    state = np.array(
        [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]],
        dtype=float,
    )
    calls: list[str] = []
    first = annotate_force_component(
        _CountingForce(np.array([[1.0, 2.0]]), calls, "first"),
        component_id="goal",
        component_type="desired",
    )
    second = annotate_force_component(
        _CountingForce(np.array([[3.0, 4.0]]), calls, "second"),
        component_id="social",
        component_type="social",
    )

    simulator = pysf.Simulator(state=state, make_forces=lambda _, __: [first, second])
    result = simulator.compute_force_components()

    assert calls == ["first", "second"]
    assert [component.component_id for component in result.components] == ["goal", "social"]
    np.testing.assert_array_equal(result.base_total, np.array([[4.0, 6.0]]))
    np.testing.assert_array_equal(result.component_sum, result.base_total)
    assert result.base_total.flags.writeable is False

    # The historical aggregate path remains available and returns the same force.
    np.testing.assert_array_equal(simulator.compute_forces(), result.base_total)
    assert calls == ["first", "second", "first", "second"]


def test_compute_force_components_rejects_nonfinite_force_output():
    """Unavailable numerical output must fail closed before it can become a label."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]], dtype=float)
    simulator = pysf.Simulator(
        state=state,
        make_forces=lambda _, __: [lambda: np.array([[np.nan, 0.0]])],
    )

    with pytest.raises(ValueError, match="force component values must be finite"):
        simulator.compute_force_components()


def test_simulator_v2_step_accumulates_multiple_force_components():
    """Simulator_v2 stepping should feed the combined force array to pedestrian state."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    captured_forces = []

    def populate(_, __):
        states = PedestrianStates(state)
        groupings = PedestrianGroupings(states, {})
        return states, groupings, []

    def make_forces(_, __):
        return [
            lambda: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            lambda: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        ]

    simulator = pysf.Simulator_v2(make_forces=make_forces, populate=populate)
    result = simulator.compute_force_components()

    np.testing.assert_array_equal(result.base_total, np.array([[6.0, 8.0], [10.0, 12.0]]))
    np.testing.assert_array_equal(result.component_sum, result.base_total)

    simulator.peds.step = lambda force: captured_forces.append(force.copy())

    simulator.step()

    assert len(captured_forces) == 1
    assert np.array_equal(captured_forces[0], np.array([[6.0, 8.0], [10.0, 12.0]], dtype=float))


def test_capped_velocity_handles_zero_desired_speed_without_runtime_warning():
    """Velocity capping should remain finite when desired speed is zero."""
    desired_velocity = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    max_velocity = np.array([0.0, 0.5], dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        capped = PedState.capped_velocity(desired_velocity, max_velocity)

    assert np.all(np.isfinite(capped))
    assert np.allclose(capped[0], np.array([0.0, 0.0], dtype=float))
    assert np.allclose(capped[1], np.array([0.5, 0.0], dtype=float))


@pytest.mark.parametrize(
    ("integration_scheme", "expected_position"),
    [("explicit_euler", 0.1), ("semi_implicit_euler", 0.05)],
)
def test_step_diagnostics_capture_uncapped_cap_and_position_integration(
    integration_scheme: str,
    expected_position: float,
):
    """Diagnostics expose the exact values used by both supported integrators."""
    state = np.array([[0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0]], dtype=float)
    peds = PedState(
        state,
        [],
        SceneConfig(dt_secs=0.1, integration_scheme=integration_scheme),
    )
    peds.assign_desired_speeds(np.array([0.5]))
    before = peds.state.copy()

    diagnostics = peds.compute_step_diagnostics(np.array([[10.0, 0.0]]))

    np.testing.assert_array_equal(peds.state, before)
    np.testing.assert_allclose(diagnostics.uncapped_velocity, [[2.0, 0.0]])
    np.testing.assert_allclose(diagnostics.uncapped_speed_mps, [2.0])
    np.testing.assert_allclose(diagnostics.applied_velocity, [[0.5, 0.0]])
    np.testing.assert_array_equal(diagnostics.cap_active, [True])
    np.testing.assert_allclose(diagnostics.position_velocity, [[expected_position / 0.1, 0.0]])

    peds.step(np.array([[10.0, 0.0]]), capture_diagnostics=True)
    assert peds.last_step_diagnostics is not None
    np.testing.assert_array_equal(
        peds.last_step_diagnostics.applied_velocity,
        diagnostics.applied_velocity,
    )
    np.testing.assert_array_equal(
        peds.last_step_diagnostics.position_velocity,
        diagnostics.position_velocity,
    )
    np.testing.assert_allclose(peds.pos(), [[expected_position, 0.0]])
    np.testing.assert_allclose(peds.vel(), [[0.5, 0.0]])

    # A normal legacy step must not leave a stale privileged snapshot behind.
    peds.step(np.zeros((1, 2), dtype=float))
    assert peds.last_step_diagnostics is None

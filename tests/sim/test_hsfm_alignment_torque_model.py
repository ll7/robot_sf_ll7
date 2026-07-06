"""Tests for the opt-in HSFM body-orientation alignment-torque pedestrian model.

The alignment torque (issue #3481) decouples pedestrian body orientation ``phi`` from the
instantaneous total-force / velocity direction: instead of snapping to the desired direction
each step, ``phi`` relaxes toward it via a damped second-order torque with a bounded turn rate.
These tests pin the pure-math contract, the config validation surface, and the simulator seam.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ALIGNMENT_TORQUE_V1,
    HSFM_TOTAL_FORCE_V1,
    step_alignment_torque_heading,
    wrap_to_pi,
)
from robot_sf.sim.sim_config import AlignmentTorqueConfig, SimulationSettings
from robot_sf.sim.simulator import Simulator
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


class _FakePedState:
    def __init__(self, state: np.ndarray) -> None:
        self.state = state
        self.max_speeds = np.full(state.shape[0], 10.0, dtype=float)
        self.updated_state: np.ndarray | None = None
        self.updated_groups: list[list[int]] | None = None

    def update(self, state: np.ndarray, groups: list[list[int]]) -> None:
        self.updated_state = state
        self.updated_groups = groups


def _alignment_oracle(
    headings: np.ndarray,
    angular_velocities: np.ndarray,
    target_headings: np.ndarray,
    *,
    dt: float,
    k_theta: float,
    k_omega: float,
    max_angular_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent per-element reference for the semi-implicit torque integrator."""
    next_heading = np.empty_like(headings, dtype=float)
    next_omega = np.empty_like(angular_velocities, dtype=float)
    for i in range(headings.shape[0]):
        error = np.mod(target_headings[i] - headings[i] + np.pi, 2.0 * np.pi) - np.pi
        omega = angular_velocities[i] + dt * (k_theta * error - k_omega * angular_velocities[i])
        omega = float(np.clip(omega, -max_angular_speed, max_angular_speed))
        phi = np.mod(headings[i] + dt * omega + np.pi, 2.0 * np.pi) - np.pi
        next_heading[i] = phi
        next_omega[i] = omega
    return next_heading, next_omega


def test_wrap_to_pi_folds_to_half_open_interval() -> None:
    """Angles wrap into ``(-pi, pi]`` with the -pi boundary folded to +pi."""
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(-np.pi) == pytest.approx(np.pi)
    assert wrap_to_pi(3.0 * np.pi) == pytest.approx(np.pi)
    assert float(wrap_to_pi(1.5 * np.pi)) == pytest.approx(-0.5 * np.pi)
    wrapped = wrap_to_pi(np.array([2.0 * np.pi, -2.5 * np.pi]))
    assert wrapped == pytest.approx([0.0, -0.5 * np.pi])


def test_alignment_torque_holds_when_already_aligned() -> None:
    """A pedestrian already facing the target neither turns nor accumulates angular speed."""
    headings = np.array([0.3, -1.2], dtype=float)
    next_heading, next_omega = step_alignment_torque_heading(
        headings,
        np.zeros_like(headings),
        headings.copy(),
        dt=0.1,
        k_theta=4.0,
        k_omega=4.0,
        max_angular_speed=np.pi,
    )
    assert next_heading == pytest.approx(headings)
    assert next_omega == pytest.approx([0.0, 0.0])


def test_alignment_torque_does_not_snap_to_target_in_one_step() -> None:
    """Unlike the instant-snap HSFM heading, a large error only partially turns per step.

    This is the decoupling contract: the body orientation lags the desired direction rather
    than jumping to it, so a planner cannot exploit an instantaneous heading flip.
    """
    heading = np.array([0.0], dtype=float)
    target = np.array([np.pi / 2], dtype=float)
    next_heading, next_omega = step_alignment_torque_heading(
        heading,
        np.zeros_like(heading),
        target,
        dt=0.1,
        k_theta=4.0,
        k_omega=4.0,
        max_angular_speed=np.pi,
    )
    # Turned toward the target but well short of it, and with a positive angular velocity.
    assert 0.0 < float(next_heading[0]) < float(target[0])
    assert float(next_omega[0]) > 0.0


def test_alignment_torque_matches_semi_implicit_oracle() -> None:
    """The vectorized integrator matches an independent per-element reference."""
    headings = np.array([0.0, 1.0, -2.5, 3.0], dtype=float)
    angular_velocities = np.array([0.0, -0.5, 0.2, 1.0], dtype=float)
    targets = np.array([np.pi / 2, -1.0, 2.5, -3.0], dtype=float)
    kwargs = {"dt": 0.1, "k_theta": 6.0, "k_omega": 4.9, "max_angular_speed": 2.0}

    actual_heading, actual_omega = step_alignment_torque_heading(
        headings, angular_velocities, targets, **kwargs
    )
    expected_heading, expected_omega = _alignment_oracle(
        headings, angular_velocities, targets, **kwargs
    )
    assert actual_heading == pytest.approx(expected_heading)
    assert actual_omega == pytest.approx(expected_omega)


def test_alignment_torque_converges_to_target_over_time() -> None:
    """Repeated damped steps settle the heading onto the target orientation."""
    heading = np.array([0.0], dtype=float)
    omega = np.array([0.0], dtype=float)
    target = np.array([1.2], dtype=float)
    for _ in range(400):
        heading, omega = step_alignment_torque_heading(
            heading,
            omega,
            target,
            dt=0.05,
            k_theta=4.0,
            k_omega=4.0,  # critical damping: k_omega == 2*sqrt(k_theta)
            max_angular_speed=np.pi,
        )
    assert float(heading[0]) == pytest.approx(float(target[0]), abs=1e-3)
    assert float(omega[0]) == pytest.approx(0.0, abs=1e-3)


def test_alignment_torque_respects_angular_speed_cap() -> None:
    """A large error cannot produce an angular speed above the configured cap."""
    heading = np.array([0.0], dtype=float)
    _, next_omega = step_alignment_torque_heading(
        heading,
        np.zeros_like(heading),
        np.array([np.pi], dtype=float),
        dt=1.0,
        k_theta=100.0,
        k_omega=0.0,
        max_angular_speed=0.5,
    )
    assert abs(float(next_omega[0])) == pytest.approx(0.5)


def test_alignment_torque_takes_shortest_signed_turn() -> None:
    """The error uses the wrapped shortest angle, so a near-pi target turns the short way."""
    heading = np.array([3.0], dtype=float)  # close to +pi
    target = np.array([-3.0], dtype=float)  # close to -pi; shortest turn is CCW past +pi
    _, next_omega = step_alignment_torque_heading(
        heading,
        np.zeros_like(heading),
        target,
        dt=0.01,
        k_theta=4.0,
        k_omega=0.0,
        max_angular_speed=np.pi,
    )
    # Shortest signed error (-3 - 3) wraps to +0.283 rad, so the turn is positive (CCW).
    assert float(next_omega[0]) > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0, "k_theta": 1.0, "k_omega": 1.0, "max_angular_speed": 1.0},
        {"dt": 0.1, "k_theta": 0.0, "k_omega": 1.0, "max_angular_speed": 1.0},
        {"dt": 0.1, "k_theta": 1.0, "k_omega": -1.0, "max_angular_speed": 1.0},
        {"dt": 0.1, "k_theta": 1.0, "k_omega": 1.0, "max_angular_speed": 0.0},
    ],
)
def test_alignment_torque_fails_closed_on_invalid_parameters(kwargs: dict) -> None:
    """Non-positive gains / timestep / speed cap raise instead of stepping silently."""
    heading = np.zeros(2, dtype=float)
    with pytest.raises(ValueError):
        step_alignment_torque_heading(heading, heading.copy(), heading.copy(), **kwargs)


def test_alignment_torque_rejects_shape_and_finiteness_violations() -> None:
    """Mismatched shapes and non-finite inputs are rejected."""
    heading = np.zeros(3, dtype=float)
    valid = {"dt": 0.1, "k_theta": 1.0, "k_omega": 1.0, "max_angular_speed": 1.0}
    with pytest.raises(ValueError, match="angular_velocities must have shape"):
        step_alignment_torque_heading(heading, np.zeros(2), heading.copy(), **valid)
    with pytest.raises(ValueError, match="target_headings must have shape"):
        step_alignment_torque_heading(heading, heading.copy(), np.zeros(2), **valid)
    with pytest.raises(ValueError, match="must be finite"):
        step_alignment_torque_heading(
            np.array([np.nan, 0.0, 0.0]), heading.copy(), heading.copy(), **valid
        )


def test_alignment_torque_config_validates_parameters() -> None:
    """The config dataclass fails closed on non-positive gains and speed cap."""
    default = AlignmentTorqueConfig()
    assert default.enabled is False
    assert default.k_omega == pytest.approx(2.0 * np.sqrt(default.k_theta))
    with pytest.raises(ValueError, match="k_theta must be > 0"):
        AlignmentTorqueConfig(k_theta=0.0)
    with pytest.raises(ValueError, match="k_omega must be >= 0"):
        AlignmentTorqueConfig(k_omega=-1.0)
    with pytest.raises(ValueError, match="max_angular_speed_rad_s must be > 0"):
        AlignmentTorqueConfig(max_angular_speed_rad_s=0.0)


def test_alignment_torque_selector_marks_config_enabled() -> None:
    """Selecting the alignment-torque model records active provenance on the config."""
    settings = SimulationSettings(pedestrian_model=HSFM_ALIGNMENT_TORQUE_V1)
    assert settings.pedestrian_model == HSFM_ALIGNMENT_TORQUE_V1
    assert settings.alignment_torque.enabled is True
    # The default model leaves the opt-in config disabled.
    assert SimulationSettings().alignment_torque.enabled is False


def test_scenario_alignment_torque_selector_marks_config_enabled(tmp_path) -> None:
    """Scenario ``simulation_config`` can select the alignment-torque model and params."""
    config = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "pedestrian_model": HSFM_ALIGNMENT_TORQUE_V1,
                "alignment_torque": {"k_theta": 9.0, "k_omega": 6.0},
            }
        },
        scenario_path=tmp_path / "scenario.yaml",
    )
    assert config.sim_config.pedestrian_model == HSFM_ALIGNMENT_TORQUE_V1
    assert config.sim_config.alignment_torque.enabled is True
    assert config.sim_config.alignment_torque.k_theta == pytest.approx(9.0)
    assert config.sim_config.alignment_torque.k_omega == pytest.approx(6.0)


def _make_alignment_torque_sim(
    state: np.ndarray, headings: np.ndarray
) -> tuple[Simulator, _FakePedState]:
    peds = _FakePedState(state)
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_ALIGNMENT_TORQUE_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimulationSettings(
        pedestrian_model=HSFM_ALIGNMENT_TORQUE_V1,
        alignment_torque={"k_theta": 4.0, "k_omega": 4.0, "max_angular_speed_rad_s": np.pi},
        time_per_step_in_secs=0.1,
    )
    sim.ped_headings = headings.copy()
    sim.ped_angular_velocities = np.zeros_like(headings)
    return sim, peds


def test_simulator_alignment_torque_one_step_is_finite_and_does_not_snap() -> None:
    """The alignment-torque selector steps kinematics like HSFM but lags the heading.

    Contrasted with ``hsfm_total_force_v1``, whose heading snaps to the force direction
    (``pi/2`` here), the alignment-torque model only partially rotates in one step.
    """
    state = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    sim, peds = _make_alignment_torque_sim(state, np.array([0.0], dtype=float))

    sim._step_pedestrians(np.array([[0.0, 2.0]], dtype=float), [[0]])

    assert peds.updated_state is state
    assert peds.updated_groups == [[0]]
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(sim.ped_headings))
    # Kinematics match the HSFM total-force velocity update (force points +y).
    assert state[0, 3] == pytest.approx(0.2)
    # Heading turned toward +pi/2 but did NOT snap to it (decoupled from instantaneous force).
    assert 0.0 < float(sim.ped_headings[0]) < np.pi / 2
    assert float(sim.ped_angular_velocities[0]) > 0.0


def test_simulator_alignment_torque_step_is_deterministic() -> None:
    """Identical initial state and config yield identical alignment-torque output."""
    state_a = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    state_b = state_a.copy()
    sim_a, _ = _make_alignment_torque_sim(state_a, np.array([0.0], dtype=float))
    sim_b, _ = _make_alignment_torque_sim(state_b, np.array([0.0], dtype=float))

    sim_a._step_pedestrians(np.array([[1.0, 1.0]], dtype=float), [[0]])
    sim_b._step_pedestrians(np.array([[1.0, 1.0]], dtype=float), [[0]])

    assert state_a == pytest.approx(state_b)
    assert sim_a.ped_headings == pytest.approx(sim_b.ped_headings)
    assert sim_a.ped_angular_velocities == pytest.approx(sim_b.ped_angular_velocities)


def test_simulator_hsfm_total_force_still_snaps_heading() -> None:
    """Regression guard: the Phase 1 HSFM model keeps its instant force-direction heading."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    peds = _FakePedState(state)
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_TOTAL_FORCE_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimulationSettings(pedestrian_model=HSFM_TOTAL_FORCE_V1, time_per_step_in_secs=0.1)
    sim.ped_headings = np.array([0.0], dtype=float)
    sim.ped_angular_velocities = np.zeros(1, dtype=float)

    sim._step_pedestrians(np.array([[0.0, 2.0]], dtype=float), [[0]])

    assert float(sim.ped_headings[0]) == pytest.approx(np.pi / 2)

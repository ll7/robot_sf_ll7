"""Tests for the opt-in HSFM total-force pedestrian-model variant."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ANISOTROPIC_FOV_V1,
    HSFM_TOTAL_FORCE_V1,
    HSFM_TTC_PREDICTIVE_V1,
    SOCIAL_FORCE_DEFAULT,
    anisotropic_fov_total_force,
    anisotropic_fov_weights,
    heading_from_total_force,
    normalize_pedestrian_model,
    step_hsfm_total_force,
)
from robot_sf.sim.sim_config import SimulationSettings
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


def test_heading_from_total_force_uses_combined_force_vector() -> None:
    """Heading follows the total force vector, even when desired force points elsewhere."""
    previous_heading = 0.0
    desired_force = np.array([1.0, 0.0])
    total_force = np.array([0.0, 1.0])

    assert desired_force.tolist() == [1.0, 0.0]
    assert heading_from_total_force(previous_heading, total_force) == pytest.approx(np.pi / 2)


def test_heading_from_total_force_preserves_heading_for_zero_force() -> None:
    """A zero total-force vector keeps the previous heading stable."""
    assert heading_from_total_force(1.23, np.zeros(2)) == pytest.approx(1.23)


def test_anisotropic_fov_weights_attenuate_behind_heading() -> None:
    """The opt-in FoV helper down-weights actors outside the heading cone."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=float)
    headings = np.array([0.0, np.pi, 0.0], dtype=float)

    weights = anisotropic_fov_weights(
        positions,
        headings,
        cone_half_angle_rad=np.pi / 2,
        rear_weight=0.1,
    )

    assert weights[0, 1] == pytest.approx(1.0)
    assert weights[0, 2] == pytest.approx(0.1)
    assert weights[1, 0] == pytest.approx(1.0)
    assert weights[1, 2] == pytest.approx(1.0)


def test_anisotropic_fov_total_force_attenuates_rear_actor_fixture() -> None:
    """Diagnostic fixture: a rear actor reduces the aggregate force for that pedestrian."""
    positions = np.array([[0.0, 0.0], [-1.0, 0.0]], dtype=float)
    headings = np.array([0.0, 0.0], dtype=float)
    total_forces = np.array([[2.0, 0.0], [2.0, 0.0]], dtype=float)

    adjusted = anisotropic_fov_total_force(
        positions,
        headings,
        total_forces,
        cone_half_angle_rad=np.pi / 2,
        rear_weight=0.25,
    )

    assert adjusted[0].tolist() == pytest.approx([0.5, 0.0])
    assert adjusted[1].tolist() == pytest.approx([2.0, 0.0])


def test_step_hsfm_total_force_caps_velocity_and_updates_heading_from_force() -> None:
    """The HSFM variant mirrors PySocialForce kinematics and orients by total force."""
    state = np.array([[0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    force = np.array([[0.0, 10.0]], dtype=float)

    next_state, next_headings = step_hsfm_total_force(
        state,
        force,
        np.array([0.0], dtype=float),
        dt=0.1,
        max_speeds=np.array([1.0], dtype=float),
    )

    capped_component = 1.0 / np.sqrt(2.0)
    assert next_state[0, 2:4] == pytest.approx([capped_component, capped_component])
    assert next_state[0, 0:2] == pytest.approx([0.1 * capped_component, 0.1 * capped_component])
    assert next_headings[0] == pytest.approx(np.pi / 2)


def test_simulator_hsfm_step_preserves_shared_pysf_state_buffer() -> None:
    """Simulator HSFM stepping mutates the PySocialForce state buffer in place."""
    state = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5]], dtype=float)
    peds = _FakePedState(state)
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_TOTAL_FORCE_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimpleNamespace(time_per_step_in_secs=0.1)
    sim.ped_headings = np.array([0.0], dtype=float)

    sim._step_pedestrians(np.array([[1.0, 0.0]], dtype=float), [[0]])

    assert peds.state is state
    assert peds.updated_state is state
    assert peds.updated_groups == [[0]]
    assert state[0, 0] == pytest.approx(0.01)
    assert state[0, 2] == pytest.approx(0.1)
    assert sim.ped_headings[0] == pytest.approx(0.0)


def test_pedestrian_model_selector_normalizes_and_rejects_unknown_values() -> None:
    """SimulationSettings exposes only the supported pedestrian-model variants."""
    assert SimulationSettings().pedestrian_model == SOCIAL_FORCE_DEFAULT
    assert SimulationSettings(pedestrian_model=HSFM_TOTAL_FORCE_V1).pedestrian_model == (
        HSFM_TOTAL_FORCE_V1
    )
    assert SimulationSettings(pedestrian_model=HSFM_TTC_PREDICTIVE_V1).pedestrian_model == (
        HSFM_TTC_PREDICTIVE_V1
    )
    assert SimulationSettings(pedestrian_model=HSFM_ANISOTROPIC_FOV_V1).pedestrian_model == (
        HSFM_ANISOTROPIC_FOV_V1
    )
    assert normalize_pedestrian_model(None) == SOCIAL_FORCE_DEFAULT
    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        SimulationSettings(pedestrian_model="unknown_model")


def test_scenario_simulation_config_can_select_hsfm_total_force(tmp_path) -> None:
    """Scenario ``simulation_config`` can select the opt-in pedestrian model."""
    config = build_robot_config_from_scenario(
        {"simulation_config": {"pedestrian_model": HSFM_TOTAL_FORCE_V1}},
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.sim_config.pedestrian_model == HSFM_TOTAL_FORCE_V1


def test_scenario_anisotropic_fov_selector_marks_config_enabled(tmp_path) -> None:
    """Selecting the FoV model records active anisotropic provenance."""
    config = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "pedestrian_model": HSFM_ANISOTROPIC_FOV_V1,
                "anisotropic_fov": {"cone_half_angle_rad": 1.0, "rear_weight": 0.25},
            }
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.sim_config.pedestrian_model == HSFM_ANISOTROPIC_FOV_V1
    assert config.sim_config.anisotropic_fov.enabled is True
    assert config.sim_config.anisotropic_fov.cone_half_angle_rad == pytest.approx(1.0)
    assert config.sim_config.anisotropic_fov.rear_weight == pytest.approx(0.25)


def test_simulator_hsfm_anisotropic_fov_one_step_smoke_is_finite_and_deterministic() -> None:
    """The FoV selector applies deterministic rear attenuation before HSFM stepping."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.5],
            [-1.0, 0.0, 0.0, 0.0, -10.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    peds = _FakePedState(state)
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_ANISOTROPIC_FOV_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimulationSettings(
        pedestrian_model=HSFM_ANISOTROPIC_FOV_V1,
        anisotropic_fov={"cone_half_angle_rad": np.pi / 2, "rear_weight": 0.25},
        time_per_step_in_secs=0.1,
    )
    sim.ped_headings = np.array([0.0, 0.0], dtype=float)

    sim._step_pedestrians(np.array([[2.0, 0.0], [2.0, 0.0]], dtype=float), [[0], [1]])

    assert peds.updated_groups == [[0], [1]]
    assert peds.updated_state is state
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(sim.ped_headings))
    assert state[0, 2] == pytest.approx(0.05)
    assert state[0, 0] == pytest.approx(0.005)
    assert state[1, 2] == pytest.approx(0.2)
    assert state[1, 0] == pytest.approx(-0.98)

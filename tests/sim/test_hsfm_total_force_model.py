"""Tests for the opt-in HSFM total-force pedestrian-model variant."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_TOTAL_FORCE_V1,
    SOCIAL_FORCE_DEFAULT,
    heading_from_total_force,
    normalize_pedestrian_model,
    step_hsfm_total_force,
)
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


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


def test_pedestrian_model_selector_normalizes_and_rejects_unknown_values() -> None:
    """SimulationSettings exposes only the supported pedestrian-model variants."""
    assert SimulationSettings().pedestrian_model == SOCIAL_FORCE_DEFAULT
    assert SimulationSettings(pedestrian_model=HSFM_TOTAL_FORCE_V1).pedestrian_model == (
        HSFM_TOTAL_FORCE_V1
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

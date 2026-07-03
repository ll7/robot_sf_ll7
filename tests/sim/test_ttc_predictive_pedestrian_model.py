"""Tests for the opt-in HSFM + TTC predictive pedestrian force variant."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_TTC_PREDICTIVE_V1,
    pairwise_time_to_collision,
    ttc_predictive_repulsion,
)
from robot_sf.sim.sim_config import SimulationSettings, TtcPredictiveForceConfig
from robot_sf.sim.simulator import Simulator
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


class _FakePedState:
    def __init__(self, state: np.ndarray) -> None:
        self.state = state
        self.max_speeds = np.full(state.shape[0], 10.0, dtype=float)
        self.updated_state: np.ndarray | None = None
        self.updated_groups: list[list[int]] | None = None

    def update(self, state: np.ndarray, groups: list[list[int]]) -> None:
        self.updated_state = state.copy()
        self.updated_groups = groups


def test_pairwise_time_to_collision_finite_for_closing_and_inf_for_separating() -> None:
    """Closing pedestrians report TTC; separating pedestrians are inactive."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)

    closing = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    separating = -closing

    ttc = pairwise_time_to_collision(positions, closing, radii, horizon_s=5.0)
    assert ttc[0, 1] == pytest.approx(1.6)
    assert ttc[1, 0] == pytest.approx(1.6)

    separating_ttc = pairwise_time_to_collision(positions, separating, radii, horizon_s=5.0)
    assert np.isinf(separating_ttc[0, 1])
    assert np.isinf(separating_ttc[1, 0])


def test_pairwise_time_to_collision_rejects_invalid_inputs() -> None:
    """TTC math fails closed on malformed or non-finite inputs."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)

    with pytest.raises(ValueError, match="positions"):
        pairwise_time_to_collision(np.zeros((2, 3)), velocities, radii, horizon_s=5.0)
    with pytest.raises(ValueError, match="velocities"):
        pairwise_time_to_collision(positions, np.zeros((3, 2)), radii, horizon_s=5.0)
    with pytest.raises(ValueError, match="radii"):
        pairwise_time_to_collision(positions, velocities, np.array([0.2]), horizon_s=5.0)
    with pytest.raises(ValueError, match="finite"):
        pairwise_time_to_collision(
            np.array([[0.0, 0.0], [np.nan, 0.0]]),
            velocities,
            radii,
            horizon_s=5.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        pairwise_time_to_collision(positions, velocities, np.array([0.2, -0.1]), horizon_s=5.0)
    with pytest.raises(ValueError, match="horizon_s"):
        pairwise_time_to_collision(positions, velocities, radii, horizon_s=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        pairwise_time_to_collision(positions, velocities, radii, horizon_s=5.0, epsilon=0.0)


def test_pairwise_time_to_collision_returns_inf_when_paths_miss() -> None:
    """Closing relative motion that misses the summed radii stays inactive."""
    ttc = pairwise_time_to_collision(
        np.array([[0.0, 0.0], [2.0, 1.0]], dtype=float),
        np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float),
        np.array([0.2, 0.2], dtype=float),
        horizon_s=5.0,
    )

    assert np.isinf(ttc[0, 1])
    assert np.isinf(ttc[1, 0])


def test_ttc_predictive_repulsion_grows_as_collision_time_decreases() -> None:
    """Nearer TTC produces stronger bounded repulsion."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)
    slow_closing = np.array([[0.25, 0.0], [-0.25, 0.0]], dtype=float)
    fast_closing = np.array([[0.75, 0.0], [-0.75, 0.0]], dtype=float)

    slow_force = ttc_predictive_repulsion(
        positions,
        slow_closing,
        radii,
        tau0_s=1.0,
        horizon_s=5.0,
        force_scale=2.0,
        max_force=10.0,
    )
    fast_force = ttc_predictive_repulsion(
        positions,
        fast_closing,
        radii,
        tau0_s=1.0,
        horizon_s=5.0,
        force_scale=2.0,
        max_force=10.0,
    )

    assert np.linalg.norm(fast_force[0]) > np.linalg.norm(slow_force[0])
    assert fast_force[0, 0] < 0.0
    assert fast_force[1, 0] > 0.0


def test_ttc_predictive_repulsion_zero_for_no_collision_and_respects_cap() -> None:
    """No-collision actors stay inactive; active force is capped per actor."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)
    separating = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)

    zero_force = ttc_predictive_repulsion(
        positions,
        separating,
        radii,
        tau0_s=1.0,
        horizon_s=5.0,
        force_scale=100.0,
        max_force=0.25,
    )
    assert np.allclose(zero_force, 0.0)

    capped_force = ttc_predictive_repulsion(
        positions,
        -separating,
        radii,
        tau0_s=10.0,
        horizon_s=5.0,
        force_scale=100.0,
        max_force=0.25,
    )
    assert np.max(np.linalg.norm(capped_force, axis=-1)) <= 0.25 + 1e-9


def test_ttc_predictive_repulsion_handles_overlapping_degenerate_directions() -> None:
    """Overlapping actors use velocity fallback or zero vector without non-finite force."""
    positions = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)

    velocity_fallback_force = ttc_predictive_repulsion(
        positions,
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=float),
        radii,
        tau0_s=1.0,
        horizon_s=5.0,
        force_scale=1.0,
        max_force=5.0,
    )
    assert np.all(np.isfinite(velocity_fallback_force))
    assert velocity_fallback_force[0, 0] > 0.0

    zero_force = ttc_predictive_repulsion(
        positions,
        np.zeros((2, 2), dtype=float),
        radii,
        tau0_s=1.0,
        horizon_s=5.0,
        force_scale=1.0,
        max_force=5.0,
    )
    assert np.allclose(zero_force, 0.0)


def test_ttc_predictive_repulsion_rejects_invalid_parameters() -> None:
    """Predictive-force parameters fail closed before simulator integration."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    radii = np.array([0.2, 0.2], dtype=float)

    with pytest.raises(ValueError, match="tau0_s"):
        ttc_predictive_repulsion(
            positions,
            velocities,
            radii,
            tau0_s=0.0,
            horizon_s=5.0,
            force_scale=1.0,
            max_force=5.0,
        )
    with pytest.raises(ValueError, match="force_scale"):
        ttc_predictive_repulsion(
            positions,
            velocities,
            radii,
            tau0_s=1.0,
            horizon_s=5.0,
            force_scale=-1.0,
            max_force=5.0,
        )
    with pytest.raises(ValueError, match="max_force"):
        ttc_predictive_repulsion(
            positions,
            velocities,
            radii,
            tau0_s=1.0,
            horizon_s=5.0,
            force_scale=1.0,
            max_force=0.0,
        )


def test_ttc_config_validates_and_scenario_can_select_predictive_model(tmp_path) -> None:
    """Scenario config can opt into the TTC predictive selector and parameters."""
    with pytest.raises(ValueError, match="tau0_s"):
        TtcPredictiveForceConfig(tau0_s=0.0)
    with pytest.raises(ValueError, match="force_scale"):
        TtcPredictiveForceConfig(force_scale=-0.1)
    with pytest.raises(ValueError, match="Unsupported pedestrian_model"):
        SimulationSettings(pedestrian_model="unknown_model")

    config = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "pedestrian_model": HSFM_TTC_PREDICTIVE_V1,
                "ttc_predictive_force": {"tau0_s": 0.75, "force_scale": 1.5},
            }
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.sim_config.pedestrian_model == HSFM_TTC_PREDICTIVE_V1
    assert config.sim_config.ttc_predictive_force.enabled is True
    assert config.sim_config.ttc_predictive_force.tau0_s == pytest.approx(0.75)
    assert config.sim_config.ttc_predictive_force.force_scale == pytest.approx(1.5)


def test_scenario_predictive_selector_marks_default_ttc_config_enabled(tmp_path) -> None:
    """Selecting the predictive model alone still records active TTC provenance."""
    config = build_robot_config_from_scenario(
        {"simulation_config": {"pedestrian_model": HSFM_TTC_PREDICTIVE_V1}},
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.sim_config.pedestrian_model == HSFM_TTC_PREDICTIVE_V1
    assert config.sim_config.ttc_predictive_force.enabled is True


def test_scenario_predictive_selector_overrides_stale_disabled_ttc_flag(tmp_path) -> None:
    """The selector is the activation surface, so provenance cannot stay disabled."""
    config = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "pedestrian_model": HSFM_TTC_PREDICTIVE_V1,
                "ttc_predictive_force": {"enabled": False},
            }
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    assert config.sim_config.pedestrian_model == HSFM_TTC_PREDICTIVE_V1
    assert config.sim_config.ttc_predictive_force.enabled is True


def test_hsfm_ttc_predictive_one_step_smoke_is_finite_and_deterministic() -> None:
    """The new selector adds a finite deterministic TTC force before HSFM stepping."""
    state = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 10.0, 0.0, 0.5],
            [2.0, 0.0, -0.5, 0.0, -10.0, 0.0, 0.5],
        ],
        dtype=float,
    )
    peds = _FakePedState(state)
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_TTC_PREDICTIVE_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimulationSettings(
        pedestrian_model=HSFM_TTC_PREDICTIVE_V1,
        ttc_predictive_force=TtcPredictiveForceConfig(
            tau0_s=10.0,
            horizon_s=5.0,
            force_scale=1.0,
            max_force=2.0,
        ),
    )
    sim.ped_headings = np.array([0.0, np.pi], dtype=float)

    sim._step_pedestrians(np.zeros((2, 2), dtype=float), [[0], [1]])

    assert peds.updated_groups == [[0], [1]]
    assert peds.updated_state is not None
    assert np.all(np.isfinite(peds.updated_state))
    assert np.all(np.isfinite(sim.ped_headings))
    assert peds.updated_state[0, 0] < 0.05
    assert peds.updated_state[1, 0] > 1.95


def test_hsfm_ttc_predictive_robot_proxy_fails_closed() -> None:
    """Robot-proxy TTC coupling is explicit future work, not silent behavior."""
    peds = _FakePedState(
        np.array(
            [
                [0.0, 0.0, 0.5, 0.0, 10.0, 0.0, 0.5],
                [2.0, 0.0, -0.5, 0.0, -10.0, 0.0, 0.5],
            ],
            dtype=float,
        )
    )
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_TTC_PREDICTIVE_V1
    sim.pysf_sim = SimpleNamespace(peds=peds)
    sim.config = SimulationSettings(
        pedestrian_model=HSFM_TTC_PREDICTIVE_V1,
        ttc_predictive_force=TtcPredictiveForceConfig(include_robot_proxy=True),
    )
    sim.ped_headings = np.array([0.0, np.pi], dtype=float)

    with pytest.raises(RuntimeError, match="robot proxy coupling is not implemented"):
        sim._step_pedestrians(np.zeros((2, 2), dtype=float), [[0], [1]])

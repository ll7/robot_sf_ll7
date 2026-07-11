"""Tests for the opt-in Zanlungo collision-prediction pedestrian force."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pysocialforce.config import SocialForceConfig
from pysocialforce.forces import SocialForce

from robot_sf.sim.pedestrian_model_variants import (
    HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
    SOCIAL_FORCE_DEFAULT,
    pairwise_social_force_contributions,
    step_hsfm_total_force,
    zanlungo_collision_prediction_repulsion,
)
from robot_sf.sim.sim_config import (
    SimulationSettings,
    ZanlungoCollisionPredictionConfig,
)
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


def _force(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    return zanlungo_collision_prediction_repulsion(
        positions,
        velocities,
        interaction_strength=1.13,
        interaction_range_m=0.71,
        anisotropy_lambda=0.29,
        angle_threshold_rad=np.pi / 4,
        max_force=5.0,
    )


def _social_force_kwargs(config: SocialForceConfig) -> dict[str, float | int]:
    return {
        "activation_threshold": config.activation_threshold,
        "n": config.n,
        "n_prime": config.n_prime,
        "lambda_importance": config.lambda_importance,
        "gamma": config.gamma,
        "factor": config.factor,
    }


def test_offset_head_on_paths_produce_opposite_early_lateral_yield_forces() -> None:
    """Slightly offset head-on paths turn apart before their closest approach."""
    positions = np.array([[0.0, 0.0], [2.0, 0.2]], dtype=float)
    velocities = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)

    force = _force(positions, velocities)

    heading_cosine = 2.0 / np.sqrt(4.04)
    anisotropy = 0.29 + (1.0 - 0.29) * (1.0 + heading_cosine) / 2.0
    expected_magnitude = 1.13 * 0.5 / 2.0 * np.exp(-0.2 / 0.71) * anisotropy

    assert force[0, 1] == pytest.approx(-expected_magnitude)
    assert force[1, 1] == pytest.approx(expected_magnitude)
    assert force[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert force[1, 0] == pytest.approx(0.0, abs=1e-12)


def test_common_earliest_time_makes_collision_prediction_non_additive() -> None:
    """A later neighbor is projected at the actor's earlier collision time."""
    two_positions = np.array([[0.0, 0.0], [2.0, 0.2]], dtype=float)
    two_velocities = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
    three_positions = np.vstack((two_positions, [4.0, 1.0]))
    three_velocities = np.vstack((two_velocities, [-0.5, 0.0]))

    two_force = _force(two_positions, two_velocities)
    three_force = _force(three_positions, three_velocities)

    assert two_force[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert three_force[0, 0] < 0.0
    assert not np.allclose(three_force[0], two_force[0])


def test_separating_and_outside_approach_cone_pairs_are_inactive() -> None:
    """Only closing pairs inside the paper's pi/4 approach cone contribute."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    separating = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=float)
    perpendicular = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=float)

    np.testing.assert_array_equal(_force(positions, separating), np.zeros((2, 2)))
    np.testing.assert_array_equal(_force(positions, perpendicular), np.zeros((2, 2)))


def test_exact_centered_collision_uses_finite_deterministic_direction_and_cap() -> None:
    """The paper's zero-distance limit uses current separation without random symmetry breaking."""
    positions = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=float)
    velocities = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=float)

    first = _force(positions, velocities)
    second = _force(positions, velocities)

    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert first[0, 0] < 0.0 < first[1, 0]
    assert np.max(np.linalg.norm(first, axis=-1)) <= 5.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interaction_strength": -1.0}, "interaction_strength"),
        ({"interaction_range_m": 0.0}, "interaction_range_m"),
        ({"anisotropy_lambda": 1.1}, "anisotropy_lambda"),
        ({"angle_threshold_rad": 0.0}, "angle_threshold_rad"),
        ({"max_force": 0.0}, "max_force"),
    ],
)
def test_force_parameters_fail_closed(kwargs: dict[str, float], message: str) -> None:
    """Malformed parameters fail before entering the simulator step."""
    parameters = {
        "interaction_strength": 1.13,
        "interaction_range_m": 0.71,
        "anisotropy_lambda": 0.29,
        "angle_threshold_rad": np.pi / 4,
        "max_force": 5.0,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        zanlungo_collision_prediction_repulsion(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            **parameters,
        )


def test_scenario_selector_enables_validated_config_without_changing_default(tmp_path) -> None:
    """Scenario YAML opts in explicitly while the default model remains unchanged."""
    assert SimulationSettings().pedestrian_model == SOCIAL_FORCE_DEFAULT
    assert SimulationSettings().zanlungo_collision_prediction.enabled is False
    with pytest.raises(ValueError, match="anisotropy_lambda"):
        ZanlungoCollisionPredictionConfig(anisotropy_lambda=-0.1)

    config = build_robot_config_from_scenario(
        {
            "simulation_config": {
                "pedestrian_model": HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
                "zanlungo_collision_prediction": {
                    "interaction_strength": 2.0,
                    "interaction_range_m": 0.8,
                },
            }
        },
        scenario_path=tmp_path / "scenario.yaml",
    )

    variant = config.sim_config.zanlungo_collision_prediction
    assert config.sim_config.pedestrian_model == HSFM_ZANLUNGO_COLLISION_PREDICTION_V1
    assert variant.enabled is True
    assert variant.interaction_strength == pytest.approx(2.0)
    assert variant.interaction_range_m == pytest.approx(0.8)


def test_runtime_replaces_only_existing_pedestrian_social_force() -> None:
    """The selector preserves base forces while replacing the pairwise social aggregate."""
    state = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 10.0, 0.0, 0.5],
            [2.0, 0.2, -0.5, 0.0, -10.0, 0.2, 0.5],
        ],
        dtype=float,
    )
    social_config = SocialForceConfig()
    pairwise_social = pairwise_social_force_contributions(
        state[:, :2], state[:, 2:4], **_social_force_kwargs(social_config)
    )
    preserved_forces = np.array([[0.3, 0.1], [-0.2, -0.1]], dtype=float)
    incoming_total = preserved_forces + pairwise_social.sum(axis=1)
    predicted = _force(state[:, :2], state[:, 2:4])
    expected_state, expected_headings = step_hsfm_total_force(
        state.copy(),
        preserved_forces + predicted,
        np.array([0.0, np.pi]),
        dt=0.1,
        max_speeds=np.full(2, 10.0),
    )

    peds = _FakePedState(state.copy())
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_ZANLUNGO_COLLISION_PREDICTION_V1
    sim.pysf_sim = SimpleNamespace(peds=peds, forces=[SocialForce(social_config, None)])
    sim.config = SimulationSettings(
        pedestrian_model=HSFM_ZANLUNGO_COLLISION_PREDICTION_V1,
        time_per_step_in_secs=0.1,
    )
    sim.ped_headings = np.array([0.0, np.pi])
    sim._step_pedestrians(incoming_total, [[0], [1]])

    np.testing.assert_allclose(peds.updated_state, expected_state, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(sim.ped_headings, expected_headings, rtol=1e-9, atol=1e-9)


def test_runtime_fails_closed_when_social_component_cannot_be_replaced() -> None:
    """Missing baseline social provenance never silently changes replacement semantics."""
    state = np.array(
        [
            [0.0, 0.0, 0.5, 0.0, 10.0, 0.0, 0.5],
            [2.0, 0.2, -0.5, 0.0, -10.0, 0.2, 0.5],
        ]
    )
    sim = object.__new__(Simulator)
    sim.pedestrian_model = HSFM_ZANLUNGO_COLLISION_PREDICTION_V1
    sim.pysf_sim = SimpleNamespace(peds=_FakePedState(state), forces=[])
    sim.config = SimulationSettings(pedestrian_model=HSFM_ZANLUNGO_COLLISION_PREDICTION_V1)
    sim.ped_headings = np.array([0.0, np.pi])

    with pytest.raises(RuntimeError, match="SocialForce component is unavailable"):
        sim._step_pedestrians(np.zeros((2, 2)), [[0], [1]])

"""Tests for per-pedestrian response-law mixtures and multipliers (issue #3574)."""

from __future__ import annotations

import numpy as np

from robot_sf.ped_npc.ped_robot_force import PedRobotForce, PedRobotForceConfig


class DummyPedState:
    """Mock PySocialForce PedState."""

    def __init__(self, positions: np.ndarray, radius: float = 0.3):
        """Initialize mock state."""
        self._pos = positions
        self.agent_radius = radius

    def pos(self) -> np.ndarray:
        """Return positions."""
        return self._pos

    def size(self) -> int:
        """Return count."""
        return self._pos.shape[0]


def test_ped_robot_force_multipliers_default_to_all_ones() -> None:
    """Without a multiplier callback, robot forces are computed normally (multipliers = 1.0)."""
    peds = DummyPedState(np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float))
    config = PedRobotForceConfig(
        is_active=True,
        robot_radius=0.5,
        activation_threshold=5.0,
        force_multiplier=10.0,
    )
    prf = PedRobotForce(config, peds, get_robot_pos=lambda: np.array([0.0, 0.0], dtype=float))
    forces = prf()
    assert forces.shape == (2, 2)
    assert not np.allclose(forces, 0.0)


def test_non_reactive_multiplier_zeroes_force() -> None:
    """Multipliers of 0.0 zero out robot forces for specific pedestrians."""
    peds = DummyPedState(np.array([[1.0, 0.0], [2.0, 0.0]], dtype=float))
    config = PedRobotForceConfig(
        is_active=True,
        robot_radius=0.5,
        activation_threshold=5.0,
        force_multiplier=10.0,
    )
    # Ped 0 is non-reactive (multiplier 0.0), Ped 1 is reactive (multiplier 1.0)
    multipliers = np.array([0.0, 1.0], dtype=float)
    prf = PedRobotForce(
        config,
        peds,
        get_robot_pos=lambda: np.array([0.0, 0.0], dtype=float),
        get_ped_response_multipliers=lambda: multipliers,
    )
    forces = prf()
    assert forces.shape == (2, 2)
    assert np.allclose(forces[0], 0.0)
    assert not np.allclose(forces[1], 0.0)


def test_response_law_allocation_and_determinism() -> None:
    """Seeded response law assignment yields exact, deterministic allocations."""
    from robot_sf.ped_npc.ped_archetypes import allocate_archetype_counts, assign_archetype_labels

    comp = {"reactive": 0.8, "non_reactive": 0.2}
    n = 10
    counts = allocate_archetype_counts(n, comp)
    assert counts["reactive"] == 8
    assert counts["non_reactive"] == 2

    # Deterministic assignment
    laws_1 = assign_archetype_labels(n, comp, seed=3574)
    laws_2 = assign_archetype_labels(n, comp, seed=3574)
    assert np.array_equal(laws_1, laws_2)

    # Reactive is 8, non-reactive is 2
    assert np.count_nonzero(laws_1 == "reactive") == 8
    assert np.count_nonzero(laws_1 == "non_reactive") == 2

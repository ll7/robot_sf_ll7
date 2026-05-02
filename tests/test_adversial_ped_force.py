"""Tests for adversarial pedestrian force computations."""

from __future__ import annotations

import numpy as np

from robot_sf.ped_npc.adversial_ped_force import (
    AdversarialPedForce,
    AdversarialPedForceConfig,
    AdversialPedForce,
    AdversialPedForceConfig,
    adversarial_ped_force,
    adversial_ped_force,
)


class _DummyPedState:
    """Minimal pedestrian-state stub for APF wrapper tests."""

    agent_radius = 0.4

    def __init__(
        self, positions: np.ndarray, velocities: np.ndarray, max_speeds: np.ndarray
    ) -> None:
        self._positions = positions
        self._velocities = velocities
        self.max_speeds = max_speeds

    def size(self) -> int:
        return int(self._positions.shape[0])

    def pos(self) -> np.ndarray:
        return self._positions

    def vel(self) -> np.ndarray:
        return self._velocities


def test_adversarial_ped_force_applies_only_to_targets() -> None:
    """Verify forces are applied only to target indices to avoid unintended drift."""
    out_forces = np.zeros((2, 2), dtype=np.float64)
    # Two pedestrians, only index 0 should receive a force.
    ped_positions = np.array([[-1.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([1.0, 1.0], dtype=np.float64)
    robot_pos = np.array([0.0, 0.0], dtype=np.float64)

    adversarial_ped_force(
        out_forces=out_forces,
        relaxation_time=1.0,
        ped_positions=ped_positions,
        ped_velocities=ped_velocities,
        ped_max_speeds=ped_max_speeds,
        robot_pos=robot_pos,
        robot_orient=0.0,
        offset=1.0,
        threshold=10.0,
        target_ped_idx=np.array([0], dtype=np.int64),
    )

    # With offset=1.0 and zero velocity, desired force is (1, 0) for ped 0 only.
    expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(out_forces, expected, rtol=1e-6, atol=1e-6)


def test_adversarial_ped_force_handles_multiple_targets() -> None:
    """Check multi-target forces match desired velocity computation for stability."""
    out_forces = np.zeros((2, 2), dtype=np.float64)
    # Both pedestrians are targeted, with different max speeds.
    ped_positions = np.array([[-1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([1.0, 2.0], dtype=np.float64)
    robot_pos = np.array([0.0, 0.0], dtype=np.float64)

    adversarial_ped_force(
        out_forces=out_forces,
        relaxation_time=1.0,
        ped_positions=ped_positions,
        ped_velocities=ped_velocities,
        ped_max_speeds=ped_max_speeds,
        robot_pos=robot_pos,
        robot_orient=0.0,
        offset=1.0,
        threshold=10.0,
        target_ped_idx=np.array([0, 1], dtype=np.int64),
    )

    # Compute expected force on ped 1 toward the attraction point.
    attraction_point = np.array([1.0, 0.0], dtype=np.float64)
    direction = attraction_point - ped_positions[1]
    direction = direction / np.linalg.norm(direction)
    expected = np.array(
        [
            [1.0, 0.0],
            direction * ped_max_speeds[1],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(out_forces, expected, rtol=1e-6, atol=1e-6)


def test_adversarial_ped_force_includes_velocity_term_current_formula() -> None:
    """Verify force computation includes velocity damping term in current formula.

    Tests that the adversarial force applies relaxation-time damping to current
    velocity: force = (v_desired - v_current) / tau. This ensures pedestrians
    smoothly accelerate toward attraction points rather than jumping abruptly.
    """
    out_forces = np.zeros((1, 2), dtype=np.float64)
    ped_positions = np.array([[-1.0, 0.0]], dtype=np.float64)
    ped_velocities = np.array([[0.2, 0.0]], dtype=np.float64)
    ped_max_speeds = np.array([1.0], dtype=np.float64)
    robot_pos = np.array([0.0, 0.0], dtype=np.float64)

    adversarial_ped_force(
        out_forces=out_forces,
        relaxation_time=0.5,
        ped_positions=ped_positions,
        ped_velocities=ped_velocities,
        ped_max_speeds=ped_max_speeds,
        robot_pos=robot_pos,
        robot_orient=0.0,
        offset=1.0,
        threshold=10.0,
        target_ped_idx=np.array([0], dtype=np.int64),
    )

    expected = np.array([[1.6, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(out_forces, expected, rtol=1e-6, atol=1e-6)


def test_adversarial_ped_force_invalid_target_indices_noop() -> None:
    """Ignore out-of-range target indices instead of indexing outside the crowd array."""
    ped_state = _DummyPedState(
        positions=np.array([[0.0, 0.0]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        max_speeds=np.array([1.0], dtype=np.float64),
    )
    force = AdversarialPedForce(
        AdversarialPedForceConfig(is_active=True, target_ped_idx=99),
        ped_state,
        get_robot_pose=lambda: ((0.0, 0.0), 0.0),
    )

    np.testing.assert_allclose(force(), np.zeros((1, 2), dtype=np.float64))


def test_adversarial_ped_force_wrapper_initializes_and_copies_last_forces() -> None:
    """Wrapper state should start with a correctly shaped buffer and retain a copied result."""
    ped_state = _DummyPedState(
        positions=np.array([[-1.0, 0.0], [0.0, -2.0]], dtype=np.float64),
        velocities=np.zeros((2, 2), dtype=np.float64),
        max_speeds=np.array([1.0, 1.5], dtype=np.float64),
    )
    force = AdversarialPedForce(
        AdversarialPedForceConfig(
            is_active=True,
            target_ped_idx=[0, 1],
            force_multiplier=2.0,
            offset=1.0,
            activation_threshold=10.0,
        ),
        ped_state,
        get_robot_pose=lambda: ((0.0, 0.0), 0.0),
    )

    assert force.last_forces.shape == (2, 2)

    computed = force()
    np.testing.assert_allclose(force.last_forces, computed)
    computed[0, 0] = -999.0
    assert force.last_forces[0, 0] != -999.0


def test_adversarial_ped_force_wrapper_handles_empty_crowd() -> None:
    """Return an empty force matrix when no pedestrians are present."""
    ped_state = _DummyPedState(
        positions=np.zeros((0, 2), dtype=np.float64),
        velocities=np.zeros((0, 2), dtype=np.float64),
        max_speeds=np.zeros((0,), dtype=np.float64),
    )
    force = AdversarialPedForce(
        AdversarialPedForceConfig(is_active=True, target_ped_idx=-1),
        ped_state,
        get_robot_pose=lambda: ((0.0, 0.0), 0.0),
    )

    result = force()

    assert result.shape == (0, 2)
    assert force.last_forces.shape == (0, 2)


def test_adversarial_ped_force_wrapper_handles_invalid_target_lists() -> None:
    """Return zeros when every requested target index falls outside the current crowd."""
    ped_state = _DummyPedState(
        positions=np.array([[0.0, 0.0]], dtype=np.float64),
        velocities=np.zeros((1, 2), dtype=np.float64),
        max_speeds=np.array([1.0], dtype=np.float64),
    )
    force = AdversarialPedForce(
        AdversarialPedForceConfig(is_active=True, target_ped_idx=[10, 11]),
        ped_state,
        get_robot_pose=lambda: ((0.0, 0.0), 0.0),
    )

    np.testing.assert_allclose(force(), np.zeros((1, 2), dtype=np.float64))
    np.testing.assert_allclose(force.last_forces, np.zeros((1, 2), dtype=np.float64))


def test_adversarial_ped_force_skips_far_and_near_targets() -> None:
    """Do not apply attraction when a target is too near or beyond the threshold."""
    out_forces = np.zeros((2, 2), dtype=np.float64)
    ped_positions = np.array([[0.25, 0.0], [5.0, 0.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([1.0, 1.0], dtype=np.float64)

    adversarial_ped_force(
        out_forces=out_forces,
        relaxation_time=1.0,
        ped_positions=ped_positions,
        ped_velocities=ped_velocities,
        ped_max_speeds=ped_max_speeds,
        robot_pos=np.array([0.0, 0.0], dtype=np.float64),
        robot_orient=0.0,
        offset=1.0,
        threshold=2.0,
        target_ped_idx=np.array([0, 1], dtype=np.int64),
    )

    np.testing.assert_allclose(out_forces, np.zeros((2, 2), dtype=np.float64))


def test_adversarial_ped_force_respects_robot_orientation() -> None:
    """Compute the attraction point using robot orientation, not only x-axis offsets."""
    out_forces = np.zeros((1, 2), dtype=np.float64)
    ped_positions = np.array([[0.0, 0.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([2.0], dtype=np.float64)

    adversarial_ped_force.py_func(
        out_forces=out_forces,
        relaxation_time=1.0,
        ped_positions=ped_positions,
        ped_velocities=ped_velocities,
        ped_max_speeds=ped_max_speeds,
        robot_pos=np.array([0.0, 0.0], dtype=np.float64),
        robot_orient=np.pi / 2,
        offset=2.0,
        threshold=10.0,
        target_ped_idx=np.array([0], dtype=np.int64),
    )

    np.testing.assert_allclose(
        out_forces,
        np.array([[0.0, 2.0]], dtype=np.float64),
        atol=1e-12,
    )


def test_adversarial_ped_force_legacy_aliases_remain_available() -> None:
    """Preserve the old misspelled names while exposing corrected public symbols."""
    assert AdversialPedForce is AdversarialPedForce
    assert AdversialPedForceConfig is AdversarialPedForceConfig
    assert adversial_ped_force is adversarial_ped_force

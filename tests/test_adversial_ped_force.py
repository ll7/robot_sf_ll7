"""Tests for adversarial pedestrian force computations."""

from __future__ import annotations

import numpy as np

from robot_sf.ped_npc.adversial_ped_force import adversial_ped_force


def test_adversial_ped_force_applies_only_to_targets() -> None:
    """Verify forces are applied only to target indices to avoid unintended drift."""
    out_forces = np.zeros((2, 2), dtype=np.float64)
    # Two pedestrians, only index 0 should receive a force.
    ped_positions = np.array([[-1.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([1.0, 1.0], dtype=np.float64)
    robot_pos = np.array([0.0, 0.0], dtype=np.float64)

    adversial_ped_force(
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


def test_adversial_ped_force_handles_multiple_targets() -> None:
    """Check multi-target forces match desired velocity computation for stability."""
    out_forces = np.zeros((2, 2), dtype=np.float64)
    # Both pedestrians are targeted, with different max speeds.
    ped_positions = np.array([[-1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    ped_velocities = np.zeros_like(ped_positions)
    ped_max_speeds = np.array([1.0, 2.0], dtype=np.float64)
    robot_pos = np.array([0.0, 0.0], dtype=np.float64)

    adversial_ped_force(
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


def test_adversial_ped_force_includes_velocity_term_current_formula() -> None:
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

    adversial_ped_force(
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

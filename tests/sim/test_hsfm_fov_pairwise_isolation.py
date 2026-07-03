"""Tests for pairwise-isolated HSFM FoV attenuation and the vectorized TTC weight path.

This slice closes two blockers the maintainer flagged on issue #3481 before
benchmark-scale use of the opt-in HSFM field-of-view models:

1. pairwise isolation of pedestrian-pedestrian repulsion attenuation, so a rear
   neighbor is down-weighted without disturbing an in-cone neighbor's push
   (contrast with the coarse ``np.min`` aggregate mode); and
2. vectorization of the ``O(N^2)`` pairwise time-to-collision (TTC) weight path,
   which must remain numerically identical to the earlier scalar double loop.

Evidence tier: diagnostic/prototype. No default model change and no
calibrated-realism claim is made here.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.sim.pedestrian_model_variants import (
    anisotropic_fov_total_force,
    anisotropic_fov_weights,
    pairwise_fov_attenuated_forces,
    pairwise_time_to_collision,
)


def _reference_scalar_ttc(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    *,
    horizon_s: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Explicit scalar TTC reference mirroring the pre-vectorization double loop."""
    count = positions.shape[0]
    ttc = np.full((count, count), np.inf, dtype=float)
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            relative_position = positions[j] - positions[i]
            relative_velocity = velocities[j] - velocities[i]
            combined_radius = radii[i] + radii[j]
            a = float(np.dot(relative_velocity, relative_velocity))
            b = 2.0 * float(np.dot(relative_position, relative_velocity))
            c = float(np.dot(relative_position, relative_position) - combined_radius**2)
            if c <= 0.0:
                ttc[i, j] = 0.0
                continue
            if a <= epsilon or b >= 0.0:
                continue
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                continue
            root = (-b - float(np.sqrt(discriminant))) / (2.0 * a)
            if epsilon < root <= horizon_s:
                ttc[i, j] = root
    return ttc


def test_pairwise_isolation_preserves_in_cone_neighbor_contribution() -> None:
    """Narrow-passage fixture: rear neighbor is attenuated, front neighbor is not.

    Actor 0 faces +x. Neighbor 1 is directly ahead (inside the cone) and neighbor 2 is
    directly behind (outside the cone). Each neighbor pushes actor 0 with an equal-length
    force. Pairwise isolation attenuates only the rear push; the aggregate ``np.min``
    mode over-attenuates by scaling the whole summed force.
    """
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=float)
    headings = np.array([0.0, np.pi, 0.0], dtype=float)
    rear_weight = 0.25

    # Per-pair contributions on actor 0: front neighbor pushes -x, rear neighbor pushes +x.
    pairwise = np.zeros((3, 3, 2), dtype=float)
    pairwise[0, 1] = np.array([-2.0, 0.0])  # from the in-cone neighbor ahead
    pairwise[0, 2] = np.array([2.0, 0.0])  # from the rear neighbor behind

    isolated = pairwise_fov_attenuated_forces(
        pairwise,
        positions,
        headings,
        cone_half_angle_rad=np.pi / 2,
        rear_weight=rear_weight,
    )

    # Front push kept at full strength; rear push scaled by rear_weight, then summed.
    expected_actor0 = np.array([-2.0, 0.0]) + rear_weight * np.array([2.0, 0.0])
    assert isolated[0].tolist() == pytest.approx(expected_actor0.tolist())

    # The aggregate mode collapses to np.min weight and scales the *summed* force,
    # which differs from (and here over-attenuates) the isolated result.
    aggregate_summed_force = pairwise.sum(axis=1)
    aggregate = anisotropic_fov_total_force(
        positions,
        headings,
        aggregate_summed_force,
        cone_half_angle_rad=np.pi / 2,
        rear_weight=rear_weight,
    )
    assert isolated[0].tolist() != pytest.approx(aggregate[0].tolist())


def test_pairwise_fov_matches_weighted_sum_definition() -> None:
    """The isolated force equals ``sum_j weights[i, j] * pairwise_forces[i, j]``."""
    rng = np.random.default_rng(3481)
    positions = rng.normal(size=(5, 2))
    headings = rng.uniform(-np.pi, np.pi, size=5)
    pairwise = rng.normal(size=(5, 5, 2))
    np.fill_diagonal(pairwise[..., 0], 0.0)
    np.fill_diagonal(pairwise[..., 1], 0.0)

    weights = anisotropic_fov_weights(
        positions, headings, cone_half_angle_rad=np.pi / 3, rear_weight=0.4
    )
    expected = np.einsum("ij,ijk->ik", weights, pairwise)

    isolated = pairwise_fov_attenuated_forces(
        positions=positions,
        pairwise_forces=pairwise,
        headings=headings,
        cone_half_angle_rad=np.pi / 3,
        rear_weight=0.4,
    )
    assert isolated == pytest.approx(expected)


def test_pairwise_fov_no_attenuation_when_all_in_cone() -> None:
    """A full-circle cone leaves every contribution unattenuated (isolation is identity)."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    headings = np.zeros(3, dtype=float)
    pairwise = np.ones((3, 3, 2), dtype=float)
    np.fill_diagonal(pairwise[..., 0], 0.0)
    np.fill_diagonal(pairwise[..., 1], 0.0)

    isolated = pairwise_fov_attenuated_forces(
        pairwise,
        positions,
        headings,
        cone_half_angle_rad=np.pi,  # entire plane in cone -> rear_weight never applied
        rear_weight=0.0,
    )
    assert isolated == pytest.approx(pairwise.sum(axis=1))


def test_pairwise_fov_rejects_bad_shape_and_nonfinite() -> None:
    """Fail closed on mismatched shapes and non-finite force contributions."""
    positions = np.zeros((3, 2), dtype=float)
    headings = np.zeros(3, dtype=float)
    with pytest.raises(ValueError, match="shape"):
        pairwise_fov_attenuated_forces(
            np.zeros((3, 2), dtype=float),
            positions,
            headings,
            cone_half_angle_rad=np.pi / 2,
            rear_weight=0.5,
        )
    bad = np.zeros((3, 3, 2), dtype=float)
    bad[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        pairwise_fov_attenuated_forces(
            bad,
            positions,
            headings,
            cone_half_angle_rad=np.pi / 2,
            rear_weight=0.5,
        )


def test_pairwise_fov_empty_population_returns_empty() -> None:
    """Zero pedestrians produce a well-formed empty force array."""
    isolated = pairwise_fov_attenuated_forces(
        np.zeros((0, 0, 2), dtype=float),
        np.zeros((0, 2), dtype=float),
        np.zeros((0,), dtype=float),
        cone_half_angle_rad=np.pi / 2,
        rear_weight=0.5,
    )
    assert isolated.shape == (0, 2)


def test_vectorized_ttc_matches_scalar_reference_on_random_cloud() -> None:
    """Vectorized TTC weight path is numerically identical to the scalar reference."""
    rng = np.random.default_rng(20250703)
    positions = rng.uniform(-5.0, 5.0, size=(9, 2))
    velocities = rng.uniform(-1.5, 1.5, size=(9, 2))
    radii = rng.uniform(0.2, 0.5, size=9)

    vectorized = pairwise_time_to_collision(positions, velocities, radii, horizon_s=4.0)
    reference = _reference_scalar_ttc(positions, velocities, radii, horizon_s=4.0)
    np.testing.assert_allclose(vectorized, reference, rtol=1e-12, atol=1e-12)


def test_vectorized_ttc_bottleneck_fixture_is_finite_and_bounded() -> None:
    """Bottleneck fixture: two columns converging on a gap yield finite in-horizon TTCs.

    Left- and right-hand pedestrians move toward each other through a narrow gap. The
    vectorized path must match the scalar reference, keep separating/self pairs at
    ``inf``, and report closing pairs within the horizon.
    """
    positions = np.array(
        [
            [-2.0, 0.3],
            [-2.0, -0.3],
            [2.0, 0.3],
            [2.0, -0.3],
        ],
        dtype=float,
    )
    velocities = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    radii = np.full(4, 0.25, dtype=float)

    ttc = pairwise_time_to_collision(positions, velocities, radii, horizon_s=5.0)
    reference = _reference_scalar_ttc(positions, velocities, radii, horizon_s=5.0)
    np.testing.assert_allclose(ttc, reference, rtol=1e-12, atol=1e-12)

    # Self pairs stay inf; the head-on closing pairs across the gap are finite.
    assert np.all(np.isinf(np.diag(ttc)))
    assert np.isfinite(ttc[0, 2])
    assert np.isfinite(ttc[1, 3])
    # Same-side co-moving pedestrians never collide within the horizon.
    assert np.isinf(ttc[0, 1])
    assert np.isinf(ttc[2, 3])
    # All finite entries are strictly within the horizon window.
    finite = ttc[np.isfinite(ttc)]
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 5.0)

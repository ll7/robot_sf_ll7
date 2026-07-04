"""Tests for pairwise-isolated HSFM FoV attenuation and the vectorized TTC weight path.

This slice closes two blockers the maintainer flagged on issue #3481 before
benchmark-scale use of the opt-in HSFM field-of-view models:

1. pairwise isolation of pedestrian-pedestrian repulsion attenuation, so a rear
   neighbor is down-weighted without disturbing an in-cone neighbor's push
   (contrast with the coarse ``np.min`` aggregate mode); and
2. vectorization of the ``O(N^2)`` pairwise time-to-collision (TTC) weight path,
   which must remain numerically identical to the earlier scalar double loop; and
3. vectorization of the ``O(N^2)`` pairwise pedestrian-pedestrian *social-force*
   contribution matrix (:func:`pairwise_social_force_contributions`), which must stay
   pair-by-pair identical to the PySocialForce scalar kernel it replaces before
   benchmark-scale use.

Evidence tier: diagnostic/prototype. No default model change and no
calibrated-realism claim is made here.
"""

from __future__ import annotations

import numpy as np
import pytest
from pysocialforce.config import SocialForceConfig
from pysocialforce.forces import social_force

from robot_sf.sim.pedestrian_model_variants import (
    anisotropic_fov_total_force,
    anisotropic_fov_weights,
    fov_attenuated_total_force,
    pairwise_fov_attenuated_forces,
    pairwise_social_force_contributions,
    pairwise_time_to_collision,
)


def _social_kwargs(config: SocialForceConfig) -> dict:
    """Mirror a ``SocialForceConfig`` as pairwise-contribution keyword arguments."""
    return {
        "activation_threshold": config.activation_threshold,
        "n": config.n,
        "n_prime": config.n_prime,
        "lambda_importance": config.lambda_importance,
        "gamma": config.gamma,
        "factor": config.factor,
    }


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


def test_pairwise_social_contributions_sum_matches_pysf_aggregate() -> None:
    """Summing per-pair contributions reproduces PySocialForce's aggregate social force.

    This is the correctness contract that lets the runtime substitute the per-pair
    decomposition for the aggregate the physics engine already sums into the total force.
    """
    rng = np.random.default_rng(3481)
    positions = rng.uniform(-3.0, 3.0, size=(6, 2))
    velocities = rng.uniform(-1.0, 1.0, size=(6, 2))
    config = SocialForceConfig()

    contributions = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    assert contributions.shape == (6, 6, 2)

    expected_aggregate = (
        social_force(
            positions,
            velocities,
            config.activation_threshold,
            config.n,
            config.n_prime,
            config.lambda_importance,
            config.gamma,
        )
        * config.factor
    )
    np.testing.assert_allclose(contributions.sum(axis=1), expected_aggregate, rtol=1e-9, atol=1e-9)


def test_pairwise_social_contributions_respect_activation_threshold() -> None:
    """Pairs beyond the activation threshold contribute exactly zero, like PySocialForce."""
    positions = np.array([[0.0, 0.0], [0.5, 0.0], [50.0, 0.0]], dtype=float)
    velocities = np.zeros((3, 2), dtype=float)
    config = SocialForceConfig()

    contributions = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    # Distant actor 2 is outside the 20 m activation radius in either direction.
    assert contributions[0, 2].tolist() == [0.0, 0.0]
    assert contributions[2, 0].tolist() == [0.0, 0.0]
    # The diagonal is always zero (no self-interaction).
    assert np.allclose(np.einsum("iik->ik", contributions), 0.0)


def test_pairwise_social_contributions_fail_closed() -> None:
    """Fail closed on bad shapes and non-finite inputs."""
    good = np.zeros((3, 2), dtype=float)
    config = SocialForceConfig()
    with pytest.raises(ValueError, match="positions"):
        pairwise_social_force_contributions(
            np.zeros((3, 3), dtype=float), good, **_social_kwargs(config)
        )
    with pytest.raises(ValueError, match="finite"):
        bad = good.copy()
        bad[0, 0] = np.nan
        pairwise_social_force_contributions(bad, good, **_social_kwargs(config))
    with pytest.raises(ValueError, match="activation_threshold"):
        pairwise_social_force_contributions(
            good, good, **{**_social_kwargs(config), "activation_threshold": -1.0}
        )


def test_fov_attenuated_total_force_full_cone_recovers_total() -> None:
    """A full-plane cone leaves the total force unchanged (attenuation is identity)."""
    rng = np.random.default_rng(11)
    positions = rng.normal(size=(4, 2))
    velocities = rng.normal(size=(4, 2))
    headings = rng.uniform(-np.pi, np.pi, size=4)
    total = rng.normal(size=(4, 2))
    config = SocialForceConfig()
    pairwise = pairwise_social_force_contributions(positions, velocities, **_social_kwargs(config))

    result = fov_attenuated_total_force(
        total,
        pairwise,
        positions,
        headings,
        cone_half_angle_rad=np.pi,  # every neighbor in cone -> no attenuation
        rear_weight=0.0,
    )
    np.testing.assert_allclose(result, total, rtol=1e-9, atol=1e-9)


def test_fov_attenuated_total_force_only_rescales_social_component() -> None:
    """Only the ped-ped social term is re-weighted; the rest of the total is preserved.

    The result must equal ``total - social_aggregate + per_pair_attenuated_social`` and,
    equivalently, differ from the input total by exactly the change in the social term.
    """
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=float)
    headings = np.array([0.0, np.pi, 0.0], dtype=float)
    total = np.array([[3.0, 1.0], [-2.0, 0.5], [0.0, -1.0]], dtype=float)
    rear_weight = 0.25
    cone_half_angle_rad = np.pi / 2

    pairwise = np.zeros((3, 3, 2), dtype=float)
    pairwise[0, 1] = np.array([-2.0, 0.0])  # in-cone neighbor ahead of actor 0
    pairwise[0, 2] = np.array([2.0, 0.0])  # rear neighbor behind actor 0

    weights = anisotropic_fov_weights(
        positions, headings, cone_half_angle_rad=cone_half_angle_rad, rear_weight=rear_weight
    )
    expected_attenuated = np.einsum("ij,ijk->ik", weights, pairwise)
    expected = total - pairwise.sum(axis=1) + expected_attenuated

    result = fov_attenuated_total_force(
        total,
        pairwise,
        positions,
        headings,
        cone_half_angle_rad=cone_half_angle_rad,
        rear_weight=rear_weight,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    # Non-interacting actors (rows 1 and 2 have no pairwise contributions here) keep their
    # full total force untouched by the FoV re-weighting.
    np.testing.assert_allclose(result[1], total[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result[2], total[2], rtol=1e-12, atol=1e-12)


def test_fov_attenuated_total_force_rejects_bad_shapes() -> None:
    """Fail closed on mismatched total/pairwise shapes and non-finite totals."""
    positions = np.zeros((3, 2), dtype=float)
    headings = np.zeros(3, dtype=float)
    pairwise = np.zeros((3, 3, 2), dtype=float)
    with pytest.raises(ValueError, match="total_forces must have shape"):
        fov_attenuated_total_force(
            np.zeros((2, 2), dtype=float),
            pairwise,
            positions,
            headings,
            cone_half_angle_rad=np.pi / 2,
            rear_weight=0.5,
        )
    with pytest.raises(ValueError, match="pairwise_social_forces must have shape"):
        fov_attenuated_total_force(
            np.zeros((3, 2), dtype=float),
            np.zeros((3, 2), dtype=float),
            positions,
            headings,
            cone_half_angle_rad=np.pi / 2,
            rear_weight=0.5,
        )
    with pytest.raises(ValueError, match="total_forces must be finite"):
        bad_total = np.zeros((3, 2), dtype=float)
        bad_total[0, 0] = np.inf
        fov_attenuated_total_force(
            bad_total,
            pairwise,
            positions,
            headings,
            cone_half_angle_rad=np.pi / 2,
            rear_weight=0.5,
        )


def _reference_scalar_social_contributions(
    positions: np.ndarray,
    velocities: np.ndarray,
    config: SocialForceConfig,
) -> np.ndarray:
    """Per-pair reference via PySocialForce's scalar ``social_force_ped_ped`` kernel.

    This mirrors the pre-vectorization ``O(N^2)`` double loop the maintainer named on
    issue #3481, so the closed-form NumPy implementation can be pinned pair-by-pair
    (not just on the aggregate sum) to the exact kernel it replaces.
    """
    from pysocialforce.forces import social_force_ped_ped

    count = positions.shape[0]
    reference = np.zeros((count, count, 2), dtype=float)
    threshold_sq = float(config.activation_threshold) ** 2
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            pos_diff = positions[i] - positions[j]
            if float(pos_diff[0] ** 2 + pos_diff[1] ** 2) > threshold_sq:
                continue
            vel_diff = velocities[j] - velocities[i]
            force_x, force_y = social_force_ped_ped(
                (float(pos_diff[0]), float(pos_diff[1])),
                (float(vel_diff[0]), float(vel_diff[1])),
                config.n,
                config.n_prime,
                config.lambda_importance,
                config.gamma,
            )
            reference[i, j, 0] = float(config.factor) * float(force_x)
            reference[i, j, 1] = float(config.factor) * float(force_y)
    return reference


@pytest.mark.parametrize("count", [2, 5, 17, 40])
def test_vectorized_social_contributions_match_scalar_kernel(count: int) -> None:
    """Vectorized per-pair contributions equal the scalar njit kernel loop.

    Pins the whole ``(N, N, 2)`` matrix (every pair, not just the neighbor sum) to the
    exact PySocialForce kernel the O(N^2) loop used, across several population sizes and
    a mix of interacting and out-of-range pairs. This is the equivalence contract that
    keeps the vectorization behavior-preserving (issue #3481).
    """
    rng = np.random.default_rng(20260704 + count)
    positions = rng.uniform(-6.0, 6.0, size=(count, 2))
    velocities = rng.uniform(-1.5, 1.5, size=(count, 2))
    config = SocialForceConfig()

    vectorized = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    reference = _reference_scalar_social_contributions(positions, velocities, config)

    assert vectorized.shape == (count, count, 2)
    # Tolerance far tighter than the aggregate 1e-9 contract; only fastmath-vs-IEEE
    # rounding separates the two paths.
    np.testing.assert_allclose(vectorized, reference, rtol=1e-9, atol=1e-11)


def test_vectorized_social_contributions_handle_coincident_pairs() -> None:
    """Coincident actors (zero position diff) stay finite and match the scalar kernel.

    The scalar kernel's ``norm_vec`` maps the zero position difference to a zero unit
    direction, and ``arctan2(0, 0) == 0``; the vectorized path must reproduce that
    degenerate handling (via masked division) instead of producing NaN from 0/0. With a
    zero separation but non-zero relative velocity the interaction vector is still
    ``lambda_importance * vel_diff``, so the force is finite and generally non-zero — the
    contract is finiteness and pair-by-pair equivalence, not a zero force.
    """
    positions = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
    velocities = np.array([[0.2, 0.0], [-0.2, 0.0], [0.0, 0.1]], dtype=float)
    config = SocialForceConfig()

    contributions = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    assert np.all(np.isfinite(contributions))
    reference = _reference_scalar_social_contributions(positions, velocities, config)
    np.testing.assert_allclose(contributions, reference, rtol=1e-9, atol=1e-11)


def test_vectorized_social_contributions_fully_degenerate_pair_is_zero() -> None:
    """Actors sharing both position and velocity produce an exactly zero force.

    Here the position diff and the interaction vector are both zero, so ``norm_vec``
    returns a zero direction and the force collapses to zero — the pure zero-vector path
    the masked division must handle without a NaN.
    """
    positions = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]], dtype=float)
    velocities = np.array([[0.5, -0.3], [0.5, -0.3], [0.0, 0.0]], dtype=float)
    config = SocialForceConfig()

    contributions = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    assert np.all(np.isfinite(contributions))
    assert np.array_equal(contributions[0, 1], np.zeros(2))
    assert np.array_equal(contributions[1, 0], np.zeros(2))


def test_vectorized_social_contributions_scale_to_large_population() -> None:
    """The vectorized seam runs at benchmark-scale N and stays sum-consistent.

    A performance-oriented smoke: a size that is impractical for a per-pair Python loop
    completes as a single NumPy evaluation and still reproduces the aggregate social
    force PySocialForce sums, confirming the O(N^2) loop removal did not change results.
    """
    rng = np.random.default_rng(4352)
    positions = rng.uniform(-15.0, 15.0, size=(256, 2))
    velocities = rng.uniform(-1.0, 1.0, size=(256, 2))
    config = SocialForceConfig()

    contributions = pairwise_social_force_contributions(
        positions, velocities, **_social_kwargs(config)
    )
    assert contributions.shape == (256, 256, 2)
    assert np.all(np.isfinite(contributions))
    assert np.allclose(np.einsum("iik->ik", contributions), 0.0)

    expected_aggregate = (
        social_force(
            positions,
            velocities,
            config.activation_threshold,
            config.n,
            config.n_prime,
            config.lambda_importance,
            config.gamma,
        )
        * config.factor
    )
    np.testing.assert_allclose(contributions.sum(axis=1), expected_aggregate, rtol=1e-9, atol=1e-9)

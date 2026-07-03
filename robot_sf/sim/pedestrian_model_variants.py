"""Opt-in pedestrian dynamics variants behind the simulator interface."""

from __future__ import annotations

from math import atan2

import numpy as np

SOCIAL_FORCE_DEFAULT = "social_force_default"
HSFM_TOTAL_FORCE_V1 = "hsfm_total_force_v1"
HSFM_TTC_PREDICTIVE_V1 = "hsfm_ttc_predictive_v1"
HSFM_ANISOTROPIC_FOV_V1 = "hsfm_anisotropic_fov_v1"
SUPPORTED_PEDESTRIAN_MODELS = frozenset(
    {
        SOCIAL_FORCE_DEFAULT,
        HSFM_TOTAL_FORCE_V1,
        HSFM_TTC_PREDICTIVE_V1,
        HSFM_ANISOTROPIC_FOV_V1,
    }
)

PYSF_POSITION_SLICE = slice(0, 2)
PYSF_VELOCITY_SLICE = slice(2, 4)


def normalize_pedestrian_model(value: str | None) -> str:
    """Return a supported pedestrian-model key or raise a clear error."""
    if value is None:
        return SOCIAL_FORCE_DEFAULT
    normalized = str(value).strip()
    if normalized in SUPPORTED_PEDESTRIAN_MODELS:
        return normalized
    supported = ", ".join(sorted(SUPPORTED_PEDESTRIAN_MODELS))
    raise ValueError(f"Unsupported pedestrian_model {value!r}. Supported values: {supported}.")


def heading_from_total_force(
    previous_heading: float,
    total_force_xy: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Derive heading from the combined force vector, preserving heading at zero force.

    Returns:
        Heading angle in radians.
    """
    total_force = np.asarray(total_force_xy, dtype=float)
    if total_force.shape != (2,):
        raise ValueError("total_force_xy must have shape (2,)")
    if float(np.linalg.norm(total_force)) <= epsilon:
        return float(previous_heading)
    return float(atan2(float(total_force[1]), float(total_force[0])))


def _validate_ttc_inputs(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    *,
    horizon_s: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_array = np.asarray(positions, dtype=float)
    velocity_array = np.asarray(velocities, dtype=float)
    radius_array = np.asarray(radii, dtype=float)

    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if velocity_array.shape != position_array.shape:
        raise ValueError("velocities must have shape (N, 2) matching positions")
    if radius_array.shape != (position_array.shape[0],):
        raise ValueError("radii must have shape (N,) matching positions")
    if not (
        np.all(np.isfinite(position_array))
        and np.all(np.isfinite(velocity_array))
        and np.all(np.isfinite(radius_array))
    ):
        raise ValueError("positions, velocities, and radii must be finite")
    if np.any(radius_array < 0):
        raise ValueError("radii must be non-negative")
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("horizon_s must be finite and > 0")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0")

    return position_array, velocity_array, radius_array


def pairwise_time_to_collision(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    *,
    horizon_s: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Return pairwise TTC in seconds, or ``inf`` when no collision is within horizon.

    Vectorized ``O(N^2)`` weight path: the pairwise quadratic collision condition
    ``||p_ij + v_ij t|| <= r_i + r_j`` is solved with NumPy broadcasting instead of a
    Python double loop, so the same weights scale to benchmark-size pedestrian counts.
    The masking reproduces the earlier scalar branches exactly:

    - already-overlapping pairs (``c <= 0``) return ``0.0``;
    - non-closing pairs (``a <= epsilon`` or ``b >= 0``) and misses (``discriminant < 0``)
      stay ``inf``;
    - a positive first root inside ``(epsilon, horizon_s]`` is the reported TTC.
    """

    position_array, velocity_array, radius_array = _validate_ttc_inputs(
        positions,
        velocities,
        radii,
        horizon_s=horizon_s,
        epsilon=epsilon,
    )
    pedestrian_count = position_array.shape[0]
    ttc = np.full((pedestrian_count, pedestrian_count), np.inf, dtype=float)
    if pedestrian_count == 0:
        return ttc

    # relative_position[i, j] = p_j - p_i and relative_velocity[i, j] = v_j - v_i.
    relative_position = position_array[np.newaxis, :, :] - position_array[:, np.newaxis, :]
    relative_velocity = velocity_array[np.newaxis, :, :] - velocity_array[:, np.newaxis, :]
    combined_radius = radius_array[:, np.newaxis] + radius_array[np.newaxis, :]

    a = np.sum(relative_velocity * relative_velocity, axis=-1)
    b = 2.0 * np.sum(relative_position * relative_velocity, axis=-1)
    c = np.sum(relative_position * relative_position, axis=-1) - combined_radius**2

    off_diagonal = ~np.eye(pedestrian_count, dtype=bool)
    already_overlapping = off_diagonal & (c <= 0.0)
    ttc[already_overlapping] = 0.0

    # Only closing pairs that are not yet overlapping can produce a future collision root.
    closing = off_diagonal & ~already_overlapping & (a > epsilon) & (b < 0.0)
    discriminant = b * b - 4.0 * a * c
    closing &= discriminant >= 0.0

    # Guard the sqrt/divide so masked-out entries never emit NaN warnings.
    safe_discriminant = np.where(closing, discriminant, 0.0)
    safe_a = np.where(closing, a, 1.0)
    root = (-b - np.sqrt(safe_discriminant)) / (2.0 * safe_a)
    within_horizon = closing & (root > epsilon) & (root <= horizon_s)
    ttc[within_horizon] = root[within_horizon]

    return ttc


def _repulsion_direction(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    away = -relative_position
    norm = float(np.linalg.norm(away))
    if norm > epsilon:
        return away / norm

    away_from_closing_velocity = -relative_velocity
    velocity_norm = float(np.linalg.norm(away_from_closing_velocity))
    if velocity_norm > epsilon:
        return away_from_closing_velocity / velocity_norm
    return np.zeros(2, dtype=float)


def _outside_fov_cone(
    forward_vector: np.ndarray,
    offset: np.ndarray,
    *,
    cone_half_angle_rad: float,
    epsilon: float,
) -> bool:
    """Return whether an offset falls outside the heading cone."""
    distance = float(np.linalg.norm(offset))
    if distance <= epsilon:
        return False
    direction = offset / distance
    cos_angle = float(np.clip(np.dot(forward_vector, direction), -1.0, 1.0))
    return float(np.arccos(cos_angle)) > cone_half_angle_rad


def _validate_anisotropic_fov_inputs(
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    cone_half_angle_rad: float,
    rear_weight: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite FoV input arrays or raise a clear validation error."""
    position_array = np.asarray(positions, dtype=float)
    heading_array = np.asarray(headings, dtype=float)
    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if heading_array.shape != (position_array.shape[0],):
        raise ValueError("headings must have shape (N,) matching positions")
    if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(heading_array)):
        raise ValueError("positions and headings must be finite")
    if not np.isfinite(cone_half_angle_rad) or not 0.0 <= cone_half_angle_rad <= np.pi:
        raise ValueError("cone_half_angle_rad must be finite and within [0, pi]")
    if not np.isfinite(rear_weight) or not 0.0 <= rear_weight <= 1.0:
        raise ValueError("rear_weight must be finite and within [0, 1]")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0")
    return position_array, heading_array


def anisotropic_fov_weights(
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    cone_half_angle_rad: float,
    rear_weight: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Return per-pair FoV attenuation weights for actors outside each heading cone."""
    position_array, heading_array = _validate_anisotropic_fov_inputs(
        positions,
        headings,
        cone_half_angle_rad=cone_half_angle_rad,
        rear_weight=rear_weight,
        epsilon=epsilon,
    )

    pedestrian_count = position_array.shape[0]
    weights = np.ones((pedestrian_count, pedestrian_count), dtype=float)
    if pedestrian_count <= 1:
        return weights
    forward_vectors = np.column_stack((np.cos(heading_array), np.sin(heading_array)))

    offsets = position_array[None, :, :] - position_array[:, None, :]
    distances = np.linalg.norm(offsets, axis=2)
    valid = distances > epsilon
    directions = np.zeros_like(offsets, dtype=float)
    directions[valid] = offsets[valid] / distances[valid, None]
    cos_angles = np.einsum("ik,ijk->ij", forward_vectors, directions)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    outside = valid & (np.arccos(cos_angles) > cone_half_angle_rad)
    weights[outside] = float(rear_weight)
    np.fill_diagonal(weights, 1.0)

    return weights


def anisotropic_fov_total_force(
    positions: np.ndarray,
    headings: np.ndarray,
    total_forces: np.ndarray,
    *,
    cone_half_angle_rad: float,
    rear_weight: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Apply diagnostic FoV attenuation to aggregated forces behind each heading.

    This is the coarse *aggregate* attenuation mode: the per-pair weight matrix is
    collapsed to a single ``np.min`` factor per actor and applied to that actor's already
    summed force. It cannot isolate a rear neighbor from an in-cone neighbor, because the
    pairwise decomposition is lost once the forces are summed — a pedestrian with one
    neighbor ahead and one behind has its whole force scaled by ``rear_weight``.

    When per-pair pedestrian-pedestrian force contributions are available, prefer
    :func:`pairwise_fov_attenuated_forces`, which attenuates each neighbor's contribution
    independently before summing.

    Returns:
        Force array with one attenuation factor per pedestrian row.
    """
    force_array = np.asarray(total_forces, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    if force_array.shape != position_array.shape:
        raise ValueError("total_forces must have shape (N, 2) matching positions")
    weights = anisotropic_fov_weights(
        position_array,
        headings,
        cone_half_angle_rad=cone_half_angle_rad,
        rear_weight=rear_weight,
        epsilon=epsilon,
    )
    if position_array.shape[0] == 0:
        return force_array.copy()
    per_actor_weight = np.min(weights, axis=1)
    return force_array * np.expand_dims(per_actor_weight, axis=-1)


def pairwise_fov_attenuated_forces(
    pairwise_forces: np.ndarray,
    positions: np.ndarray,
    headings: np.ndarray,
    *,
    cone_half_angle_rad: float,
    rear_weight: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Attenuate each pedestrian-pedestrian force contribution by its own FoV weight.

    This is the *pairwise-isolated* attenuation mode. Given the per-pair repulsion
    contributions ``pairwise_forces[i, j]`` (the force neighbor ``j`` exerts on actor
    ``i``), each contribution is scaled by its own ``anisotropic_fov_weights[i, j]``
    factor before summing over neighbors. Unlike :func:`anisotropic_fov_total_force`,
    a rear neighbor is down-weighted without disturbing an in-cone neighbor's push, so
    the FoV attenuation no longer bleeds across unrelated interactions.

    The layer stays independent of simulator objects so it is unit-testable without a
    full environment. It remains diagnostic/prototype: no default model changes and no
    calibrated-realism claim.

    Args:
        pairwise_forces: Per-pair force contributions with shape ``(N, N, 2)``; entry
            ``[i, j]`` is the force neighbor ``j`` exerts on actor ``i`` (diagonal is
            typically zero and is preserved unaffected because self-weights are ``1``).
        positions: Pedestrian positions with shape ``(N, 2)``.
        headings: Per-pedestrian heading angles with shape ``(N,)``.
        cone_half_angle_rad: Half-angle of the forward field-of-view cone in radians.
        rear_weight: Attenuation factor in ``[0, 1]`` applied outside the cone.
        epsilon: Small positive tolerance for degenerate zero-distance pairs.

    Returns:
        Per-actor attenuated force with shape ``(N, 2)``.
    """
    force_array = np.asarray(pairwise_forces, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    pedestrian_count = position_array.shape[0]
    if force_array.shape != (pedestrian_count, pedestrian_count, 2):
        raise ValueError("pairwise_forces must have shape (N, N, 2) matching positions")
    if not np.all(np.isfinite(force_array)):
        raise ValueError("pairwise_forces must be finite")
    if pedestrian_count == 0:
        return np.zeros((0, 2), dtype=float)
    weights = anisotropic_fov_weights(
        position_array,
        headings,
        cone_half_angle_rad=cone_half_angle_rad,
        rear_weight=rear_weight,
        epsilon=epsilon,
    )
    attenuated = force_array * weights[:, :, np.newaxis]
    return attenuated.sum(axis=1)


def ttc_predictive_repulsion(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    *,
    tau0_s: float,
    horizon_s: float,
    force_scale: float,
    max_force: float,
    epsilon: float = 1e-9,
) -> np.ndarray:
    """Compute bounded TTC-scaled pedestrian-pedestrian predictive repulsion.

    Returns:
        Per-pedestrian predictive force array with shape ``(N, 2)``.
    """

    if not np.isfinite(tau0_s) or tau0_s <= 0:
        raise ValueError("tau0_s must be finite and > 0")
    if not np.isfinite(force_scale) or force_scale < 0:
        raise ValueError("force_scale must be finite and >= 0")
    if not np.isfinite(max_force) or max_force <= 0:
        raise ValueError("max_force must be finite and > 0")

    position_array, velocity_array, radius_array = _validate_ttc_inputs(
        positions,
        velocities,
        radii,
        horizon_s=horizon_s,
        epsilon=epsilon,
    )
    ttc = pairwise_time_to_collision(
        position_array,
        velocity_array,
        radius_array,
        horizon_s=horizon_s,
        epsilon=epsilon,
    )
    repulsion = np.zeros_like(position_array, dtype=float)

    for i in range(position_array.shape[0]):
        for j in range(position_array.shape[0]):
            if i == j or not np.isfinite(ttc[i, j]):
                continue

            relative_position = position_array[j] - position_array[i]
            relative_velocity = velocity_array[j] - velocity_array[i]
            direction = _repulsion_direction(
                relative_position,
                relative_velocity,
                epsilon=epsilon,
            )
            repulsion[i] += direction * float(force_scale) * float(np.exp(-ttc[i, j] / tau0_s))

    norms = np.linalg.norm(repulsion, axis=-1)
    factors = np.ones_like(norms, dtype=float)
    np.divide(max_force, norms, out=factors, where=norms > max_force)
    np.minimum(factors, 1.0, out=factors)
    return repulsion * np.expand_dims(factors, axis=-1)


def step_hsfm_total_force(
    state: np.ndarray,
    total_forces: np.ndarray,
    headings: np.ndarray,
    *,
    dt: float,
    max_speeds: np.ndarray,
    epsilon: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Step PySocialForce state while orienting pedestrians by total force.

    Returns:
        Updated PySocialForce state and per-pedestrian headings.
    """
    next_state = np.asarray(state, dtype=float).copy()
    force_array = np.asarray(total_forces, dtype=float)
    previous_headings = np.asarray(headings, dtype=float)
    speed_caps = np.asarray(max_speeds, dtype=float)

    pedestrian_count = next_state.shape[0]
    if force_array.shape != (pedestrian_count, 2):
        raise ValueError("total_forces must have shape (N, 2) matching state rows")
    if previous_headings.shape != (pedestrian_count,):
        raise ValueError("headings must have shape (N,) matching state rows")
    if speed_caps.shape != (pedestrian_count,):
        raise ValueError("max_speeds must have shape (N,) matching state rows")

    desired_velocity = next_state[:, PYSF_VELOCITY_SLICE] + float(dt) * force_array
    desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
    factors = np.ones_like(desired_speeds, dtype=float)
    np.divide(speed_caps, desired_speeds, out=factors, where=desired_speeds > epsilon)
    np.minimum(factors, 1.0, out=factors)
    capped_velocity = desired_velocity * np.expand_dims(factors, axis=-1)

    next_state[:, PYSF_POSITION_SLICE] += capped_velocity * float(dt)
    next_state[:, PYSF_VELOCITY_SLICE] = capped_velocity
    force_norms = np.linalg.norm(force_array, axis=-1)
    force_headings = np.arctan2(force_array[:, 1], force_array[:, 0])
    next_headings = np.where(force_norms <= epsilon, previous_headings, force_headings)
    return next_state, next_headings

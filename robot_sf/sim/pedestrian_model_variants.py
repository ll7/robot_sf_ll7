"""Opt-in pedestrian dynamics variants behind the simulator interface."""

from __future__ import annotations

from math import atan2

import numpy as np

SOCIAL_FORCE_DEFAULT = "social_force_default"
HSFM_TOTAL_FORCE_V1 = "hsfm_total_force_v1"
HSFM_TTC_PREDICTIVE_V1 = "hsfm_ttc_predictive_v1"
SUPPORTED_PEDESTRIAN_MODELS = frozenset(
    {SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1, HSFM_TTC_PREDICTIVE_V1}
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
    """Return pairwise TTC in seconds, or ``inf`` when no collision is within horizon."""

    position_array, velocity_array, radius_array = _validate_ttc_inputs(
        positions,
        velocities,
        radii,
        horizon_s=horizon_s,
        epsilon=epsilon,
    )
    pedestrian_count = position_array.shape[0]
    ttc = np.full((pedestrian_count, pedestrian_count), np.inf, dtype=float)

    for i in range(pedestrian_count):
        for j in range(pedestrian_count):
            if i == j:
                continue

            relative_position = position_array[j] - position_array[i]
            relative_velocity = velocity_array[j] - velocity_array[i]
            combined_radius = radius_array[i] + radius_array[j]

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

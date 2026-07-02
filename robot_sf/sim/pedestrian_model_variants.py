"""Opt-in pedestrian dynamics variants behind the simulator interface."""

from __future__ import annotations

from math import atan2

import numpy as np

SOCIAL_FORCE_DEFAULT = "social_force_default"
HSFM_TOTAL_FORCE_V1 = "hsfm_total_force_v1"
SUPPORTED_PEDESTRIAN_MODELS = frozenset({SOCIAL_FORCE_DEFAULT, HSFM_TOTAL_FORCE_V1})

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
    next_headings = np.asarray(
        [
            heading_from_total_force(previous_heading, total_force, epsilon=epsilon)
            for previous_heading, total_force in zip(previous_headings, force_array, strict=True)
        ],
        dtype=float,
    )
    return next_state, next_headings

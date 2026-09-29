"""Versioned geometric clearance calculations for planner adapters."""

from __future__ import annotations

from typing import Literal

import numpy as np

ClearanceModel = Literal["center_v1", "surface_v2"]
CENTER_CLEARANCE_V1: ClearanceModel = "center_v1"
SURFACE_CLEARANCE_V2: ClearanceModel = "surface_v2"


def validate_clearance_model(model: str) -> ClearanceModel:
    """Return a supported clearance model or reject an unknown model name."""
    if model not in {CENTER_CLEARANCE_V1, SURFACE_CLEARANCE_V2}:
        raise ValueError(f"clearance_model must be 'center_v1' or 'surface_v2', got {model!r}")
    return model  # type: ignore[return-value]


def validate_surface_clearance_radii(
    model: str,
    *,
    robot_radius: float,
    pedestrian_radius: float,
) -> None:
    """Require declared positive body radii when surface clearance is selected."""
    if validate_clearance_model(model) != SURFACE_CLEARANCE_V2:
        return
    for name, value in (
        ("robot_radius", robot_radius),
        ("pedestrian_radius", pedestrian_radius),
    ):
        radius = float(value)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{name} must be finite and positive for surface_v2")


def pedestrian_clearance(
    center_distance: float | np.ndarray,
    *,
    model: str,
    robot_radius: float,
    pedestrian_radius: float,
) -> float | np.ndarray:
    """Return center distance or signed surface separation, in metres."""
    selected = validate_clearance_model(model)
    distance = np.asarray(center_distance, dtype=float)
    if selected == SURFACE_CLEARANCE_V2:
        distance = distance - float(robot_radius) - float(pedestrian_radius)
    if distance.ndim == 0:
        return float(distance)
    return distance


def occupied_cell_clearance(
    row_offsets: np.ndarray,
    column_offsets: np.ndarray,
    *,
    resolution: float,
    model: str,
    robot_radius: float,
    point_offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> float:
    """Return distance to occupied cell centers or signed body-to-cell clearance.

    In ``surface_v2``, occupied cells are treated as axis-aligned square regions
    with the supplied resolution. The returned value is the shortest distance
    from the robot center to a cell square, less the robot radius.
    """
    selected = validate_clearance_model(model)
    resolution_m = float(resolution)
    robot_radius_m = float(robot_radius)
    if not np.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError("resolution must be finite and positive")
    if not np.isfinite(robot_radius_m) or robot_radius_m < 0.0:
        raise ValueError("robot_radius must be finite and non-negative")

    rows = np.asarray(row_offsets, dtype=float)
    columns = np.asarray(column_offsets, dtype=float)
    if rows.shape != columns.shape:
        raise ValueError("row_offsets and column_offsets must have matching shapes")
    if rows.size == 0:
        return float("inf")

    if selected == CENTER_CLEARANCE_V1:
        # Retain the historical expression byte-for-byte for v1 callers.
        distance = np.sqrt(rows**2 + columns**2) * resolution_m
    else:
        offset_x, offset_y = point_offset_xy_m
        row_gap = np.maximum(np.abs(rows * resolution_m - offset_y) - resolution_m / 2.0, 0.0)
        column_gap = np.maximum(np.abs(columns * resolution_m - offset_x) - resolution_m / 2.0, 0.0)
        distance = np.hypot(row_gap, column_gap) - robot_radius_m
    return float(np.min(distance))


def surface_search_radius_cells(robot_radius: float, margin: float, resolution: float) -> int:
    """Cover every square that can intersect the body plus a safety margin.

    Returns:
        int: Inclusive grid-cell search radius.
    """
    return max(1, int(np.ceil((robot_radius + margin) / resolution)) + 1)


def time_to_circle_contact(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    combined_radius: float,
) -> float:
    """Return first non-negative time at which two moving circles touch.

    The relative vectors are in metres and metres per second. ``inf`` means
    that constant relative velocity does not produce contact.
    """
    position = np.asarray(relative_position, dtype=float).reshape(-1)
    velocity = np.asarray(relative_velocity, dtype=float).reshape(-1)
    radius = float(combined_radius)
    if position.size != 2 or velocity.size != 2:
        raise ValueError("relative_position and relative_velocity must each have two values")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("relative position and velocity must be finite")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("combined_radius must be finite and non-negative")

    a = float(np.dot(velocity, velocity))
    c = float(np.dot(position, position) - radius**2)
    if c <= 0.0:
        return 0.0
    if a <= 1e-12:
        return float("inf")
    b = 2.0 * float(np.dot(position, velocity))
    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0.0:
        return float("inf")
    contact_time = (-b - float(np.sqrt(discriminant))) / (2.0 * a)
    return float(contact_time) if contact_time >= 0.0 else float("inf")


__all__ = [
    "CENTER_CLEARANCE_V1",
    "SURFACE_CLEARANCE_V2",
    "ClearanceModel",
    "occupied_cell_clearance",
    "pedestrian_clearance",
    "surface_search_radius_cells",
    "time_to_circle_contact",
    "validate_clearance_model",
    "validate_surface_clearance_radii",
]

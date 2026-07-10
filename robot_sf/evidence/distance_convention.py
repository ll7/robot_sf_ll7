"""Explicit distance-convention vocabulary for exported series and metrics.

Issue #5141: distance quantities cross the codebase in at least three
incompatible conventions, and exports historically did not state which one they
carry. That ambiguity already caused a trace bundle misreading (1.37 m of
center-to-center distance read as surface clearance, flipping a collision
attribution).

This module is the single source of truth for the convention enum and the
metadata field name. Export scripts attach ``distance_convention`` to their
series/metric metadata, and the evidence-writer lint
(``scripts/ci/pr_contract_check.py`` rule 4) fails distance-like series that
omit it.

Conventions:

- ``center_center``: Euclidean distance between two point centers (e.g. robot
  center to pedestrian center, or the runtime collision trigger = radius sum).
  The trace export ``min_robot_ped_distance_m`` column is center-to-center.
- ``surface_clearance``: center-to-center distance minus the summed footprint
  radii (clearance = center_distance - robot_radius - ped_radius). The
  benchmark pedestrian safety metric ``surface_clearance_m`` uses this; a
  collision is ``clearance < 0``.
- ``center_segment``: shortest distance from a point center to a wall/obstacle
  segment (e.g. robot center to obstacle point/segment in wall-collision
  tests).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

#: Canonical metadata field name carried by series/metric export payloads.
DISTANCE_CONVENTION_FIELD: str = "distance_convention"

#: Filename substrings that mark an exported file as a distance-like series.
#: The evidence-writer lint requires these to carry the convention field.
DISTANCE_LIKE_FILENAME_TOKENS: tuple[str, ...] = ("distance", "clearance")

#: Column-header substrings that mark a CSV column as distance-like.
DISTANCE_LIKE_COLUMN_TOKENS: tuple[str, ...] = (
    "distance",
    "clearance",
    "_dist",
    "min_clearance",
)


class DistanceConvention(StrEnum):
    """The three distance conventions used by exported series and metrics.

    Inherits ``str`` so values serialize directly into JSON metadata and
    compare equal to their plain string value (``center_center`` etc.).
    """

    CENTER_CENTER = "center_center"
    SURFACE_CLEARANCE = "surface_clearance"
    CENTER_SEGMENT = "center_segment"


#: Human-readable description of each convention, used by docs and lint messages.
CONVENTION_DESCRIPTIONS: dict[DistanceConvention, str] = {
    DistanceConvention.CENTER_CENTER: (
        "center-to-center Euclidean distance between two point centers "
        "(e.g. robot-to-pedestrian center distance; the runtime collision "
        "trigger is the radius sum). Surface footprint is NOT subtracted."
    ),
    DistanceConvention.SURFACE_CLEARANCE: (
        "surface clearance = center-to-center distance minus the summed "
        "footprint radii (clearance = center_distance - robot_radius - "
        "ped_radius). Collisions are clearance < 0."
    ),
    DistanceConvention.CENTER_SEGMENT: (
        "shortest distance from a point center to a wall/obstacle segment "
        "(e.g. robot center to obstacle point/segment in wall-collision tests)."
    ),
}


def validate_distance_convention(value: Any) -> DistanceConvention:
    """Return the :class:`DistanceConvention` for ``value`` or raise ``ValueError``.

    Accepts a :class:`DistanceConvention`, or its string value
    (``center_center`` / ``surface_clearance`` / ``center_segment``).

    Raises:
        ValueError: If ``value`` is not a recognized convention.
    """
    if isinstance(value, DistanceConvention):
        return value
    if isinstance(value, str):
        try:
            return DistanceConvention(value)
        except ValueError as exc:
            valid = ", ".join(c.value for c in DistanceConvention)
            raise ValueError(
                f"Unknown {DISTANCE_CONVENTION_FIELD}={value!r}; expected one of: {valid}"
            ) from exc
    raise ValueError(
        f"{DISTANCE_CONVENTION_FIELD} must be a string or DistanceConvention, "
        f"got {type(value).__name__}"
    )


def require_distance_convention(metadata: dict[str, Any], series_name: str) -> DistanceConvention:
    """Assert that ``metadata`` carries a valid ``distance_convention`` field.

    Used by export scripts to fail closed at write time when a distance-like
    series omits the field, and by the evidence-writer lint.

    Args:
        metadata: The series/metric export metadata mapping.
        series_name: Human-readable series identifier for the error message
            (e.g. ``"min_distance_series.csv"``).

    Returns:
        The validated :class:`DistanceConvention`.

    Raises:
        ValueError: If the field is missing or not a recognized convention.
    """
    if DISTANCE_CONVENTION_FIELD not in metadata:
        raise ValueError(
            f"distance-like series {series_name!r} is missing the required "
            f"{DISTANCE_CONVENTION_FIELD!r} metadata field; set one of: "
            + ", ".join(c.value for c in DistanceConvention)
        )
    return validate_distance_convention(metadata[DISTANCE_CONVENTION_FIELD])


def describe_distance_convention(convention: DistanceConvention | str) -> str:
    """Return the human-readable description for a convention value."""
    resolved = validate_distance_convention(convention)
    return CONVENTION_DESCRIPTIONS[resolved]


def is_distance_like_filename(name: str) -> bool:
    """Return True if a filename looks like a distance-like series export.

    Matches filenames such as ``min_distance_series.csv`` or
    ``robot_clearance_series.csv``.
    """
    lowered = name.lower()
    return any(token in lowered for token in DISTANCE_LIKE_FILENAME_TOKENS)


def has_distance_like_columns(header_line: str) -> bool:
    """Return True if a CSV header line declares a distance-like column."""
    lowered = header_line.lower()
    return any(token in lowered for token in DISTANCE_LIKE_COLUMN_TOKENS)


__all__ = [
    "CONVENTION_DESCRIPTIONS",
    "DISTANCE_CONVENTION_FIELD",
    "DISTANCE_LIKE_COLUMN_TOKENS",
    "DISTANCE_LIKE_FILENAME_TOKENS",
    "DistanceConvention",
    "describe_distance_convention",
    "has_distance_like_columns",
    "is_distance_like_filename",
    "require_distance_convention",
    "validate_distance_convention",
]

"""Analytic surface distances for an explicitly selected capsule body.

The pose is the body centre, and the axis follows its heading. These are
instantaneous distances, not swept collision or contact-resolution mechanics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import cos, dist, isfinite, sin
from numbers import Real
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

    from robot_sf.common.types import Line2D, RobotPose, Vec2D


def point_segment_distance(point: Vec2D, segment: Line2D) -> float:
    """Return Euclidean distance to a closed segment, including a point segment."""
    start, end = segment
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return dist(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    return dist(point, (start[0] + t * dx, start[1] + t * dy))


def segment_segment_distance(first: Line2D, second: Line2D) -> float:
    """Return the minimum distance between closed segments, including crossings."""
    a, b = first
    c, d = second
    ux, uy = b[0] - a[0], b[1] - a[1]
    vx, vy = d[0] - c[0], d[1] - c[1]
    wx, wy = c[0] - a[0], c[1] - a[1]
    cross = ux * vy - uy * vx
    if cross != 0.0:
        t = (wx * vy - wy * vx) / cross
        s = (wx * uy - wy * ux) / cross
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
            return 0.0
    return min(
        point_segment_distance(a, second),
        point_segment_distance(b, second),
        point_segment_distance(c, first),
        point_segment_distance(d, first),
    )


@dataclass(frozen=True)
class CapsuleFootprint:
    """A heading-aligned capsule with total end-to-end length and full width."""

    length: float
    width: float

    def __post_init__(self) -> None:
        """Reject invalid dimensions instead of falling back to a disc."""
        for name, value in (("length", self.length), ("width", self.width)):
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"capsule.{name} must be finite and positive")
            if value <= 0:
                raise ValueError(f"capsule.{name} must be finite and positive")
        if self.length < self.width:
            raise ValueError("capsule.length must be at least capsule.width")

    @property
    def radius(self) -> float:
        """Return the circular end-cap radius."""
        return self.width / 2.0

    @property
    def half_segment(self) -> float:
        """Return half the centre-segment length."""
        return (self.length - self.width) / 2.0

    @property
    def covering_radius(self) -> float:
        """Return the radius of the smallest centred covering disc."""
        return self.length / 2.0

    def axis(self, pose: RobotPose) -> Line2D:
        """Return the centre segment at the live body-centre pose."""
        (x, y), heading = pose
        dx, dy = self.half_segment * cos(heading), self.half_segment * sin(heading)
        return ((x - dx, y - dy), (x + dx, y + dy))

    def circle_gap(self, pose: RobotPose, centre: Vec2D, radius: float) -> float:
        """Return signed capsule-to-disc surface gap (contact is zero)."""
        return point_segment_distance(centre, self.axis(pose)) - self.radius - radius

    def segment_gap(self, pose: RobotPose, segment: Line2D) -> float:
        """Return signed capsule-to-wall surface gap."""
        return segment_segment_distance(self.axis(pose), segment) - self.radius

    def capsule_gap(self, pose: RobotPose, other: CapsuleFootprint, other_pose: RobotPose) -> float:
        """Return signed surface gap between two independently oriented capsules."""
        return (
            segment_segment_distance(self.axis(pose), other.axis(other_pose))
            - self.radius
            - other.radius
        )


def resolve_capsule(value: object | None, covering_radius: float) -> CapsuleFootprint | None:
    """Validate an opt-in selector and its conservative radius for disc consumers.

    Returns:
        Validated capsule, or ``None`` for the historical disc path.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        if set(value) != {"length", "width"}:
            raise ValueError("capsule requires exactly length and width")
        dimensions = cast("Mapping[str, Any]", value)
        value = CapsuleFootprint(length=dimensions["length"], width=dimensions["width"])
    if not isinstance(value, CapsuleFootprint):
        raise ValueError("capsule must be a CapsuleFootprint or a length/width mapping")
    if (
        isinstance(covering_radius, bool)
        or not isinstance(covering_radius, Real)
        or not isfinite(covering_radius)
        or covering_radius < value.covering_radius
    ):
        raise ValueError("radius must cover the capsule: radius >= capsule.length / 2")
    return value

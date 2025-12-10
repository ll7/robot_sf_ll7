"""Route utilities for global navigation."""

from dataclasses import dataclass
from math import dist

from robot_sf.common.types import Rect, Vec2D


@dataclass
class GlobalRoute:
    """Global route from a spawn zone to a goal zone.

    Attributes:
        spawn_id: Identifier of the spawn zone.
        goal_id: Identifier of the goal zone.
        waypoints: Ordered list of 2D waypoints defining the path.
        spawn_zone: Polygon describing the spawn area.
        goal_zone: Polygon describing the goal area.
    """

    spawn_id: int
    goal_id: int
    waypoints: list[Vec2D]
    spawn_zone: Rect
    goal_zone: Rect

    def __post_init__(self):
        """Validate spawn/goal identifiers and waypoint list."""

        if self.spawn_id < 0:
            raise ValueError("Spawn id needs to be an integer >= 0!")
        if self.goal_id < 0:
            raise ValueError("Goal id needs to be an integer >= 0!")
        if len(self.waypoints) < 1:
            raise ValueError(f"Route {self.spawn_id} -> {self.goal_id} contains no waypoints!")

    @property
    def sections(self) -> list[tuple[Vec2D, Vec2D]]:
        """List consecutive waypoint pairs.

        Returns:
            list[tuple[Vec2D, Vec2D]]: Start/end points for each segment.
        """

        return (
            []
            if len(self.waypoints) < 2
            else list(zip(self.waypoints[:-1], self.waypoints[1:], strict=False))
        )

    @property
    def section_lengths(self) -> list[float]:
        """Compute length of each segment."""

        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> list[float]:
        """Cumulative offsets for each segment start.

        Returns:
            list[float]: Distance from route start to each segment start.
        """

        lengths = self.section_lengths
        offsets = []
        temp_offset = 0.0
        for section_length in lengths:
            offsets.append(temp_offset)
            temp_offset += section_length
        return offsets

    @property
    def total_length(self) -> float:
        """Total path length."""

        return 0 if len(self.waypoints) < 2 else sum(self.section_lengths)

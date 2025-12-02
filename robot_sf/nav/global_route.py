"""Module global_route auto-generated docstring."""

from dataclasses import dataclass
from math import dist

from robot_sf.common.types import Rect, Vec2D


@dataclass
class GlobalRoute:
    """
    A class to represent a global route.

    Attributes
    ----------
    spawn_id : int
        The id of the spawn point.
    goal_id : int
        The id of the goal point.
    waypoints : List[Vec2D]
        The waypoints of the route.
    spawn_zone : Rect
        The spawn zone of the route.
    goal_zone : Rect
        The goal zone of the route.

    Methods
    -------
    __post_init__():
        Validates the spawn_id, goal_id, and waypoints.
    sections():
        Returns the sections of the route.
    section_lengths():
        Returns the lengths of the sections.
    section_offsets():
        Returns the offsets of the sections.
    total_length():
        Returns the total length of the route.
    """

    spawn_id: int
    goal_id: int
    waypoints: list[Vec2D]
    spawn_zone: Rect
    goal_zone: Rect

    def __post_init__(self):
        """
        Validates the spawn_id, goal_id, and waypoints.
        Raises a ValueError if spawn_id or goal_id is less than 0 or if waypoints is empty.
        """

        if self.spawn_id < 0:
            raise ValueError("Spawn id needs to be an integer >= 0!")
        if self.goal_id < 0:
            raise ValueError("Goal id needs to be an integer >= 0!")
        if len(self.waypoints) < 1:
            raise ValueError(f"Route {self.spawn_id} -> {self.goal_id} contains no waypoints!")

    @property
    def sections(self) -> list[tuple[Vec2D, Vec2D]]:
        """
        Returns the sections of the route as a list of tuples, where each tuple
        contains two Vec2D objects
        representing the start and end points of the section.
        """

        return (
            []
            if len(self.waypoints) < 2
            else list(zip(self.waypoints[:-1], self.waypoints[1:], strict=False))
        )

    @property
    def section_lengths(self) -> list[float]:
        """
        Returns the lengths of the sections as a list of floats.
        """

        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> list[float]:
        """
        Returns the offsets of the sections as a list of floats.
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
        """
        Returns the total length of the route as a float.
        """

        return 0 if len(self.waypoints) < 2 else sum(self.section_lengths)

"""TODO docstring. Document this module."""

from dataclasses import dataclass, field
from math import dist

import numpy as np

Vec2D = tuple[float, float]
Line2D = tuple[float, float, float, float]
Circle = tuple[Vec2D, float]
Rect = tuple[Vec2D, Vec2D, Vec2D]
Zone = tuple[Vec2D, Vec2D, Vec2D]  # rect ABC with sides |A B|, |B C| and diagonal |A C|


def sample_zone(zone: Zone, num_samples: int) -> list[Vec2D]:
    """
    Sample points within a given zone.

    Args:
        zone (Zone): The zone defined by three points.
        num_samples (int): The number of points to sample.

    Returns:
        List[Vec2D]: A list of sampled points within the zone.
    """
    a, b, c = zone
    a, b, c = np.array(a), np.array(b), np.array(c)
    vec_ba, vec_bc = a - b, c - b
    rel_width = np.random.uniform(0, 1, (num_samples, 1))
    rel_height = np.random.uniform(0, 1, (num_samples, 1))
    points = b + rel_width * vec_ba + rel_height * vec_bc
    return [(x, y) for x, y in points]


def sample_circle(circle: Circle, num_samples: int) -> list[Vec2D]:
    """
    Sample points within a given circle.

    Args:
        circle (Circle): The circle defined by center point and radius.
        num_samples (int): The number of points to sample.

    Returns:
        List[Vec2D]: A list of sampled points within the zone.
    """
    center, radius = circle
    rot = np.random.uniform(0, np.pi * 2, (num_samples, 1))
    radius = np.random.uniform(0, radius, (num_samples, 1))
    rel_x, rel_y = np.cos(rot) * radius, np.sin(rot) * radius
    points = np.concatenate((rel_x, rel_y), axis=1) + np.array([center])
    return [(x, y) for x, y in points]


@dataclass
class Obstacle:
    """
    Represents an obstacle in the map.

    Attributes:
        vertices (List[Vec2D]): The vertices of the obstacle.
        lines (List[Line2D]): The lines formed by connecting the vertices.
        vertices_np (np.ndarray): The vertices as a NumPy array.
    """

    vertices: list[Vec2D]
    lines: list[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

    def __post_init__(self):
        """TODO docstring. Document this function."""
        if not self.vertices:
            raise ValueError("No vertices specified for obstacle!")

        self.vertices_np = np.array(self.vertices)
        edges = list(zip(self.vertices[:-1], self.vertices[1:], strict=False)) + [
            (self.vertices[-1], self.vertices[0])
        ]
        # remove fake lines that are just points
        edges = list(filter(lambda edge: edge[0] != edge[1], edges))
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines


@dataclass
class GlobalRoute:
    """
    Represents a global route from a spawn point to a goal point in a map.
    """

    waypoints: list[Vec2D]
    spawn_radius: float = 5.0

    def __post_init__(self):
        """
        Initializes the GlobalRoute object.

        Raises:

            ValueError: If the route contains no waypoints.
        """

        if len(self.waypoints) < 1:
            raise ValueError("Route contains no waypoints!")

    @property
    def spawn_circle(self) -> Circle:
        """
        Returns the spawn circle of the route.

        Returns:
            Circle: The spawn circle of the route.
        """
        return (self.waypoints[0], self.spawn_radius)

    @property
    def sections(self) -> list[tuple[Vec2D, Vec2D]]:
        """
        Returns a list of sections along the route.

        Returns:
            List[Tuple[Vec2D, Vec2D]]: The list of sections, where each section is represented by a tuple of two waypoints.
        """
        return (
            []
            if len(self.waypoints) < 2
            else list(zip(self.waypoints[:-1], self.waypoints[1:], strict=False))
        )

    @property
    def section_lengths(self) -> list[float]:
        """
        Returns a list of lengths of each section along the route.

        Returns:
            List[float]: The list of section lengths.
        """
        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> list[float]:
        """
        Returns a list of offsets for each section along the route.

        Returns:
            List[float]: The list of section offsets.
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
        Returns the total length of the route.

        Returns:
            float: The total length of the route.
        """
        return 0 if len(self.waypoints) < 2 else sum(self.section_lengths)


@dataclass
class MapDefinition:
    """
    Represents the definition of a map.

    Attributes:
        obstacles (List[Obstacle]): A list of obstacles in the map.
        routes (List[GlobalRoute]): A list of global routes in the map.
        crowded_zones (List[Zone]): A list of crowded zones in the map.
    """

    obstacles: list[Obstacle]
    routes: list[GlobalRoute]
    crowded_zones: list[Zone]

import os
import json
import random
from math import sqrt, dist
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass, field

import numpy as np

Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class Obstacle:
    """
    A class to represent an obstacle.

    Attributes
    ----------
    vertices : List[Vec2D]
        The vertices of the obstacle.
    lines : List[Line2D]
        The lines that make up the obstacle. This is calculated in the post-init method.
    vertices_np : np.ndarray
        The vertices of the obstacle as a numpy array. This is calculated in the post-init method.

    Methods
    -------
    __post_init__():
        Validates and processes the vertices to create the lines and vertices_np attributes.
    """

    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)
    vertices_np: np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Validates and processes the vertices to create the lines and vertices_np attributes.
        Raises a ValueError if no vertices are specified.
        """

        if not self.vertices:
            raise ValueError('No vertices specified for obstacle!')

        # Convert vertices to numpy array
        self.vertices_np = np.array(self.vertices)

        # Create edges from vertices
        edges = list(zip(self.vertices[:-1], self.vertices[1:])) \
            + [(self.vertices[-1], self.vertices[0])]

        # Remove fake lines that are just points
        edges = list(filter(lambda l: l[0] != l[1], edges))

        # Create lines from edges
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

        if not self.vertices:
            print('WARNING: obstacle is just a single point that cannot collide!')


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
    waypoints: List[Vec2D]
    spawn_zone: Rect
    goal_zone: Rect

    def __post_init__(self):
        """
        Validates the spawn_id, goal_id, and waypoints.
        Raises a ValueError if spawn_id or goal_id is less than 0 or if waypoints is empty.
        """

        if self.spawn_id < 0:
            raise ValueError('Spawn id needs to be an integer >= 0!')
        if self.goal_id < 0:
            raise ValueError('Goal id needs to be an integer >= 0!')
        if len(self.waypoints) < 1:
            raise ValueError(f'Route {self.spawn_id} -> {self.goal_id} contains no waypoints!')

    @property
    def sections(self) -> List[Tuple[Vec2D, Vec2D]]:
        """
        Returns the sections of the route as a list of tuples, where each tuple
        contains two Vec2D objects
        representing the start and end points of the section.
        """

        return [] if len(self.waypoints) < 2 else list(zip(self.waypoints[:-1], self.waypoints[1:]))

    @property
    def section_lengths(self) -> List[float]:
        """
        Returns the lengths of the sections as a list of floats.
        """

        return [dist(p1, p2) for p1, p2 in self.sections]

    @property
    def section_offsets(self) -> List[float]:
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


@dataclass
class MapDefinition:
    width: float
    height: float
    obstacles: List[Obstacle]
    robot_spawn_zones: List[Rect]
    ped_spawn_zones: List[Rect]
    robot_goal_zones: List[Rect]
    bounds: List[Line2D]
    robot_routes: List[GlobalRoute]
    ped_goal_zones: List[Rect]
    ped_crowded_zones: List[Rect]
    ped_routes: List[GlobalRoute]
    obstacles_pysf: List[Line2D] = field(init=False)
    robot_routes_by_spawn_id: Dict[int, List[GlobalRoute]] = field(init=False)

    def __post_init__(self):
        obstacle_lines = [line for o in self.obstacles for line in o.lines]
        self.obstacles_pysf = obstacle_lines + self.bounds

        self.robot_routes_by_spawn_id = dict()
        for route in self.robot_routes:
            if route.spawn_id in self.robot_routes_by_spawn_id:
                self.robot_routes_by_spawn_id[route.spawn_id].append(route)
            else:
                self.robot_routes_by_spawn_id[route.spawn_id] = [route]

        if self.width < 0 or self.height < 0:
            raise ValueError("Map width and height mustn't be zero or negative!")
        if not self.robot_spawn_zones or not self.robot_goal_zones:
            raise ValueError("Spawn and goal zones mustn't be empty!")
        if len(self.bounds) != 4:
            raise ValueError("Invalid bounds! Expected exactly 4 bounds!")

    @property
    def num_start_pos(self) -> int:
        return len(set([r.spawn_id for r in self.robot_routes]))

    @property
    def max_target_dist(self) -> float:
        return sqrt(2) * (max(self.width, self.height) * 2)

    def find_route(self, spawn_id: int, goal_id: int) -> Union[GlobalRoute, None]:
        return next(filter(lambda r:
            r.goal_id == goal_id and r.spawn_id == spawn_id, self.robot_routes), None)


@dataclass
class MapDefinitionPool:
    maps_folder: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")
    map_defs: List[MapDefinition] = field(default_factory=list)

    def __post_init__(self):
        if not self.map_defs:
            if not os.path.exists(self.maps_folder) or not os.path.isdir(self.maps_folder):
                raise ValueError(f"Map directory '{self.maps_folder}' does not exist!")

            def load_json(path: str) -> dict:
                with open(path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            map_files = [os.path.join(self.maps_folder, f) for f in os.listdir(self.maps_folder)]
            self.map_defs = [serialize_map(load_json(f)) for f in map_files]

        if not self.map_defs:
            raise ValueError('Map pool is empty! Please specify some maps!')

    def choose_random_map(self) -> MapDefinition:
        return random.choice(self.map_defs)


def serialize_map(map_structure: dict) -> MapDefinition:
    (min_x, max_x), (min_y, max_y) = map_structure['x_margin'], map_structure['y_margin']
    width, height = max_x - min_x, max_y - min_y

    def norm_pos(pos: Vec2D) -> Vec2D:
        return (pos[0] - min_x, pos[1] - min_y)

    obstacles = [Obstacle([norm_pos(p) for p in vertices])
                 for vertices in map_structure['obstacles']]

    def norm_zone(rect: Rect) -> Rect:
        return (norm_pos(rect[0]), norm_pos(rect[1]), norm_pos(rect[2]))

    robot_goal_zones = [norm_zone(z) for z in map_structure['robot_goal_zones']]
    robot_spawn_zones = [norm_zone(z) for z in map_structure['robot_spawn_zones']]
    ped_goal_zones = [norm_zone(z) for z in map_structure['ped_goal_zones']]
    ped_spawn_zones = [norm_zone(z) for z in map_structure['ped_spawn_zones']]
    ped_crowded_zones = [norm_zone(z) for z in map_structure['ped_crowded_zones']]

    robot_routes = [GlobalRoute(o['spawn_id'], o['goal_id'], [norm_pos(p) for p in o['waypoints']],
                                robot_spawn_zones[o['spawn_id']], robot_goal_zones[o['goal_id']])
                    for o in map_structure['robot_routes']]
    ped_routes = [GlobalRoute(o['spawn_id'], o['goal_id'], [norm_pos(p) for p in o['waypoints']],
                              ped_spawn_zones[o['spawn_id']], ped_goal_zones[o['goal_id']])
                  for o in map_structure['ped_routes']]

    def reverse_route(route: GlobalRoute) -> GlobalRoute:
        return GlobalRoute(
            route.goal_id, route.spawn_id, list(reversed(route.waypoints)),
            route.goal_zone, route.spawn_zone)

    # info: this works because spawn and goal zones are the same
    rev_robot_routes = [reverse_route(r) for r in robot_routes]
    robot_routes = robot_routes + rev_robot_routes

    map_bounds = [
        (0, width, 0, 0),           # bottom
        (0, width, height, height), # top
        (0, 0, 0, height),          # left
        (width, width, 0, height)]  # right

    return MapDefinition(
        width, height, obstacles, robot_spawn_zones,
        ped_spawn_zones, robot_goal_zones, map_bounds, robot_routes,
        ped_goal_zones, ped_crowded_zones, ped_routes)

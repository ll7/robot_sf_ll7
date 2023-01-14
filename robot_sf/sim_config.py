import os
import json
import random
from typing import List, Tuple
from dataclasses import dataclass, field

# import numpy as np
# from shapely.geometry import Polygon


Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class Obstacle:
    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)

    def __post_init__(self):
        edges = list(zip(self.vertices[:-1], self.vertices[1:]))
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

    # @property
    # def as_polygon(self) -> Polygon:
    #     return Polygon(np.array(self.vertices))


@dataclass
class GlobalRoute:
    spawn_id: int
    goal_id: int
    waypoints: List[Vec2D]


@dataclass
class MapDefinition:
    width: float
    height: float
    obstacles: List[Obstacle]
    robot_spawn_zones: List[Rect]
    ped_spawn_zones: List[Rect]
    goal_zones: List[Rect]
    bounds: List[Line2D]
    robot_routes: List[GlobalRoute]
    obstacles_pysf: List[Line2D] = field(init=False)

    def __post_init__(self):
        obstacle_lines = [line for o in self.obstacles for line in o.lines]
        self.obstacles_pysf = obstacle_lines + self.bounds


@dataclass
class MapDefinitionPool:
    maps_folder: str = os.path.join(os.path.dirname(__file__), "maps")
    map_defs: List[MapDefinition] = field(default_factory=list)

    def __post_init__(self):
        if not self.map_defs:
            def load_json(path: str) -> dict:
                with open(path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            map_files = [os.path.join(self.maps_folder, f) for f in os.listdir(self.maps_folder)]
            self.map_defs = [serialize_map(load_json(f)) for f in map_files]

    def choose_random_map(self) -> MapDefinition:
        return random.choice(self.map_defs)


def serialize_map(map_structure: dict) -> MapDefinition:
    (min_x, max_x), (min_y, max_y) = map_structure['x_margin'], map_structure['y_margin']
    width, height = max_x - min_x, max_y - min_y

    def norm_pos(pos: Vec2D) -> Vec2D:
        return (pos[0] - min_x, pos[1] - min_y)

    obstacles = [Obstacle([norm_pos(p) for p in map_structure['Obstacles'][k]['Vertex']])
                 for k in map_structure['Obstacles']]

    robot_routes = [GlobalRoute(o['spawn_id'], o['goal_id'], [norm_pos(p) for p in o['waypoints']])
                    for o in map_structure['robot_routes']]

    map_bounds = [
        (0, width, 0, 0),           # bottom
        (0, width, height, height), # top
        (0, 0, 0, height),          # left
        (width, width, 0, height)]  # right

    def norm_zone(rect: Rect) -> Rect:
        return (norm_pos(rect[0]), norm_pos(rect[1]), norm_pos(rect[2]))

    goal_zones = [norm_zone(z) for z in map_structure['goal_zones']]
    robot_spawn_zones = [norm_zone(z) for z in map_structure['robot_spawn_zones']]
    ped_spawn_zones = [norm_zone(z) for z in map_structure['ped_spawn_zones']]

    return MapDefinition(
        width, height, obstacles, robot_spawn_zones,
        ped_spawn_zones, goal_zones, map_bounds, robot_routes)

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
class MapDefinition:
    box_size: float # TODO: think of computing the box size here ...
    obstacles: List[Obstacle]
    spawn_zones: List[Rect]
    goal_zones: List[Rect]
    bounds: List[Line2D]
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
    x_span = map_structure['x_margin'][1] - map_structure['x_margin'][0]
    y_span = map_structure['y_margin'][1] - map_structure['y_margin'][0]
    min_x, min_y = map_structure['x_margin'][0], map_structure['y_margin'][0]
    norm_span = max(x_span, y_span)
    box_size = 20.0 # TODO: remove this hard-coded scale, pysf forces are calibrated for 1m scale

    def norm_coords(point: Vec2D) -> Vec2D:
        return ((point[0] - min_x) / norm_span * 2 * box_size - box_size,
                (point[1] - min_y) / norm_span * 2 * box_size - box_size)

    obstacle_vertices = [[norm_coords(p) for p in map_structure['Obstacles'][k]['Vertex']]
                         for k in map_structure['Obstacles']]
    obstacles = [Obstacle(v) for v in obstacle_vertices]

    map_bounds = [
        (-box_size,  box_size, -box_size, -box_size), # bottom
        (-box_size,  box_size,  box_size,  box_size), # top
        (-box_size, -box_size, -box_size,  box_size), # left
        ( box_size,  box_size, -box_size,  box_size)] # right

    if 'GoalZones' in map_structure and 'SpawnZones' in map_structure:
        goal_zones = map_structure['GoalZones']
        spawn_zones = map_structure['SpawnZones']
    else:
        # TODO: remove this fallback logic for maps without explicit spawn / goal zones
        box_rect = ((-box_size, box_size), (-box_size, -box_size), (box_size, -box_size))
        spawn_zones, goal_zones = [box_rect], [box_rect]

    return MapDefinition(box_size, obstacles, spawn_zones, goal_zones, map_bounds)

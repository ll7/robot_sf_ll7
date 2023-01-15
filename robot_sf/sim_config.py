import os
import json
import random
from typing import List, Tuple
from dataclasses import dataclass, field


Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class Obstacle:
    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)

    def __post_init__(self):
        if not self.vertices:
            raise ValueError('No vertices specified for obstacle!')

        edges = list(zip(self.vertices[:-1], self.vertices[1:])) \
            + [(self.vertices[-1], self.vertices[0])]
        edges = list(filter(lambda l: l[0] != l[1], edges)) # remove fake lines that are just points
        lines = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = lines

        if not self.vertices:
            print('WARNING: obstacle is just a single point that cannot collide!')


@dataclass
class GlobalRoute:
    spawn_id: int
    goal_id: int
    waypoints: List[Vec2D]

    def __post_init__(self):
        if self.spawn_id < 0:
            raise ValueError('Spawn id needs to be an integer >= 0!')
        if self.goal_id < 0:
            raise ValueError('Goal id needs to be an integer >= 0!')
        if not self.waypoints:
            raise ValueError(f'Route {self.spawn_id} -> {self.goal_id} contains no waypoints!')


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

        if self.width < 0 or self.width < 0:
            raise ValueError("Map height and width mustn't be zero or negative!")
        if not self.robot_spawn_zones or not self.ped_spawn_zones or not self.goal_zones:
            raise ValueError("Spawn and goal zones mustn't be empty!")
        if len(self.bounds) != 4:
            raise ValueError("Invalid bounds! Expected exactly 4 bounds!")

        def route_exists_once(spawn_id: int, goal_id: int) -> bool:
            return len(list(filter(
                lambda r: r.goal_id == goal_id and r.spawn_id == spawn_id,
                self.robot_routes))) == 1

        num_spawns, num_goals = len(self.robot_spawn_zones), len(self.goal_zones)
        spawn_goal_perms = [(s, g) for s in range(num_spawns) for g in range(num_goals)]
        missing_routes = list(filter(lambda perm: route_exists_once(perm[0], perm[1]), spawn_goal_perms))
        if len(missing_routes) > 0:
            missing = ', '.join([f'{s} -> {g}' for s, g in missing_routes])
            raise ValueError((f'Missing or ambiguous routes {missing}! Please ensure that every ',
                'spawn zone is connected to every goal zone by exactly one route!'))


@dataclass
class MapDefinitionPool:
    maps_folder: str = os.path.join(os.path.dirname(__file__), "maps")
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

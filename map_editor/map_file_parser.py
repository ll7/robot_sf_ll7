import json
from typing import Union, Tuple, List, Set
from dataclasses import dataclass, field

import numpy as np


Range2D = Tuple[float, float] # (low, high)
Vec2D = Tuple[float, float]
Rect = Tuple[Vec2D, Vec2D, Vec2D]


@dataclass
class GlobalRoute:
    spawn_id: int
    goal_id: int
    waypoints: List[Vec2D]


@dataclass
class VisualizableMapConfig:
    x_margin: Range2D
    y_margin: Range2D
    obstacles: np.ndarray
    goal_zones: List[Rect] = field(default_factory=list)
    robot_spawn_zones: List[Rect] = field(default_factory=list)
    ped_spawn_zones: List[Rect] = field(default_factory=list)
    robot_routes: List[GlobalRoute] = field(default_factory=list)


MAP_VERSION_V0 = 'v0'
MAP_VERSION_V1 = 'v1'


def determine_mapfile_version(text: str) -> Union[str, None]:
    def intersect_sets(s1: Set, s2: Set) -> Set:
        return {k for k in s1 if k in s2}

    def contains_all(s1: Set, comp: Set) -> bool:
        return intersect_sets(s1, comp) == comp

    try:
        map_data: dict = json.loads(text)

        required_toplevel_keys = { 'x_margin', 'y_margin' }
        if not contains_all(set(map_data.keys()), required_toplevel_keys):
            return None

        obsolete_v0_toplevel_keys = { 'Obstacles', 'NumberObstacles', 'Created' }
        if any([k for k in obsolete_v0_toplevel_keys if k in map_data]):
            return MAP_VERSION_V0

        required_v1_keys = { 'obstacles', 'ped_spawn_zones', 'robot_spawn_zones', 'goal_zones', 'robot_routes' }
        if not contains_all(set(map_data.keys()), required_v1_keys):
            return MAP_VERSION_V0

        return MAP_VERSION_V1
    except:
        return None


def parse_mapfile_text_v0(text: str) -> Union[VisualizableMapConfig, None]:
    try:
        map_data = json.loads(text)

        all_lines = list()
        for obstacle in map_data['obstacles']:
            vertices: List[Vec2D] = map_data['obstacles'][obstacle]['Vertex']
            edges = list(zip(vertices[:-1], vertices[1:])) \
                + [(vertices[-1], vertices[0])]
            for (s_x, s_y), (e_x, e_y) in edges:
                line = [s_x, e_x, s_y, e_y]
                all_lines.append(line)
        obstacles = np.array(all_lines)

        x_margin = map_data['x_margin']
        x_margin = (x_margin[0], x_margin[1])

        y_margin = map_data['y_margin']
        y_margin = (y_margin[0], y_margin[1])

        return VisualizableMapConfig(x_margin, y_margin, obstacles)
    except:
        return None


def parse_mapfile_text_v1(text: str) -> Union[VisualizableMapConfig, None]:
    try:
        map_data = json.loads(text)

        all_lines = list()
        for vertices in map_data['obstacles']:
            vertices: List[Vec2D]
            edges = list(zip(vertices[:-1], vertices[1:])) \
                + [(vertices[-1], vertices[0])]
            for (s_x, s_y), (e_x, e_y) in edges:
                line = [s_x, e_x, s_y, e_y]
                all_lines.append(line)
        obstacles = np.array(all_lines)

        routes = [GlobalRoute(o['spawn_id'], o['goal_id'], o['waypoints'])
                  for o in map_data['robot_routes']]

        x_margin = map_data['x_margin']
        x_margin = (x_margin[0], x_margin[1])

        y_margin = map_data['y_margin']
        y_margin = (y_margin[0], y_margin[1])

        goal_zones = map_data['goal_zones']
        robot_spawn_zones = map_data['robot_spawn_zones']
        ped_spawn_zones = map_data['ped_spawn_zones']

        return VisualizableMapConfig(
            x_margin, y_margin, obstacles, goal_zones,
            robot_spawn_zones, ped_spawn_zones, routes)
    except:
        return None


parsers_by_version = {
    MAP_VERSION_V0: parse_mapfile_text_v0,
    MAP_VERSION_V1: parse_mapfile_text_v1,
}


def parse_mapfile_text(text: str) -> Union[VisualizableMapConfig, None]:
    version = determine_mapfile_version(text)
    print(f'file version is {version}')
    return parsers_by_version[version](text) if version else None

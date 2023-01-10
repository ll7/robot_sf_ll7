import os
import toml
import json
import random
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon
from natsort import natsorted

from robot_sf.ped_robot_force import PedRobotForceConfig


Vec2D = Tuple[float, float]
Line2D = Tuple[float, float, float, float]


@dataclass
class Obstacle:
    vertices: List[Vec2D]
    lines: List[Line2D] = field(init=False)

    def __post_init__(self):
        edges = list(zip(self.vertices[:-1], self.vertices[1:])) \
            + [[self.vertices[-1], self.vertices[0]]]
        edges = [(p1[0], p2[0], p1[1], p2[1]) for p1, p2 in edges]
        self.lines = edges

    @property
    def as_polygon(self) -> Polygon:
        return Polygon(np.array(self.vertices))


def load_map_name(data: dict) -> str:
    map_names = natsorted(list(data['map_files'].values()))

    if data['simulator']['flags']['random_map']:
        map_id = random.choice(list(range(len(map_names))))
    else:
        try:
            map_id = int(data['simulator']['custom']['map_number'])
        except:
            map_id = 0

    return map_names[map_id]


def load_randomly_init_map(data: dict, maps_config_path: str) -> Tuple[float, List[Line2D]]:
    map_name = load_map_name(data)
    path_to_map = os.path.join(maps_config_path, map_name)

    with open(path_to_map, 'r', encoding='utf-8') as file:
        map_structure = json.load(file)

    # TODO: extend maps to provide spawn zones / goal zones

    # TODO: support another shape that's called a path (lines between points without area)

    x_span = map_structure['x_margin'][1] - map_structure['x_margin'][0]
    y_span = map_structure['y_margin'][1] - map_structure['y_margin'][0]
    min_x, min_y = map_structure['x_margin'][0], map_structure['y_margin'][0]
    norm_span = max(x_span, y_span)
    box_size = 20 # TODO: remove this hard-coded scale, pysf forces are calibrated for 1m scale

    def norm_coords(point: Vec2D) -> Vec2D:
        return ((point[0] - min_x) / norm_span * 2 * box_size - box_size,
                (point[1] - min_y) / norm_span * 2 * box_size - box_size)

    for key in map_structure['Obstacles'].keys():
        vertices = map_structure['Obstacles'][key]['Vertex']
        norm_vertices = [norm_coords(p) for p in vertices]
        map_structure['Obstacles'][key]['Vertex'] = norm_vertices

    obstacle = []
    for key in map_structure['Obstacles'].keys():
        vertices = map_structure['Obstacles'][key]['Vertex']
        valid_edges = list(zip(vertices[:-1], vertices[1:])) + [[vertices[-1], vertices[0]]]
        valid_edges = [[p1[0], p2[0], p1[1], p2[1]] for p1, p2 in valid_edges]
        obstacle += valid_edges

    # append map bounds as obstacles
    obstacle.append([-box_size, box_size, -box_size, -box_size]) # bottom
    obstacle.append([-box_size, box_size, box_size, box_size])   # top
    obstacle.append([-box_size, -box_size, -box_size, box_size]) # left
    obstacle.append([box_size, box_size, -box_size, box_size])   # right

    return box_size, obstacle


def load_config(config_filepath: str) -> Tuple[float, PedRobotForceConfig, List[Line2D]]:
    data = toml.load(config_filepath)
    config_path = os.path.join(os.path.dirname(__file__), 'maps')
    box_size, obstacles = load_randomly_init_map(data, config_path)

    robot_config = PedRobotForceConfig(
        data['simulator']['flags']['activate_ped_robot_force'],
        data['simulator']['robot']['robot_radius'],
        data['simulator']['robot']['activation_threshold'],
        data['simulator']['robot']['force_multiplier'])

    return box_size, robot_config, obstacles

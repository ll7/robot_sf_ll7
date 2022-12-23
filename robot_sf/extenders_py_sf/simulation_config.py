import os
import toml
import json
import random
from math import ceil, floor
from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon
from natsort import natsorted


Vec2D = Tuple[float, float]


@dataclass
class SimulatorConfigFlags:
    dynamic_grouping: bool
    enable_max_steps_stop: bool
    enable_topology_on_stopped: bool
    enable_rendezvous_centroid: bool
    update_peds: bool
    activate_ped_robot_force: bool


@dataclass
class ActionPool:
    actions: List[str]
    probs_in_percent: List[float]


@dataclass
class NewPedsParams:
    max_single_peds: int
    max_grp_size: int
    group_width_max: float
    group_width_min: float
    average_speed: float
    max_standalone_grouping: int
    max_nsteps_ped_stopped: int
    max_nteps_group_stopped: int
    max_stopping_pedestrians: int
    max_unfreezing_pedestrians: int
    max_group_splitting: int=5


@dataclass
class RobotForceConfig:
    robot_radius: float
    activation_threshold: int
    force_multiplier: float


@dataclass
class SimulationConfiguration:
    box_size: float
    peds_sparsity: int
    ped_generation_action_pool: ActionPool
    group_action_pool: ActionPool
    stopping_action_pool: ActionPool
    flags: SimulatorConfigFlags
    robot_force_config: RobotForceConfig
    new_peds_params: NewPedsParams
    obstacle_avoidance_params: List
    obstacles_lolol: List


def load_polygon(vertices: List[Tuple[float, float]]) -> Polygon:
    return Polygon(np.array(vertices))


def load_toml(config_file: str) -> dict:
    if not config_file:
        dirname = os.path.dirname(__file__)
        parent = os.path.split(dirname)[0]
        filename = os.path.join(parent, "utils", "config", "map_config.toml")
        config_file = filename
    return toml.load(config_file)


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


def fill_state(coordinate_a: int, coordinate_b: Union[int, np.ndarray],
               origin: bool, box_size: float):
    distance = box_size * 1.1 if origin else box_size * 1.6
    if isinstance(coordinate_b, np.ndarray):
        return build_coordinates_array(coordinate_a, coordinate_b, len(coordinate_b), distance)
    else:
        return build_coordinates_scalar(coordinate_a, coordinate_b, distance)


def build_coordinates_array(coordinate_a, coordinate_b, dim, distance):
    # spawn pedestrians at the edges
    if coordinate_a == 0:
        return np.concatenate((-distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]) , axis=1)
    elif coordinate_a == 1:
        return np.concatenate((coordinate_b[:, np.newaxis], -distance * np.ones([dim, 1])), axis=1)
    elif coordinate_a == 2:
        return np.concatenate((distance * np.ones([dim, 1]), coordinate_b[:, np.newaxis]), axis=1)
    elif coordinate_a == 3:
        return np.concatenate((coordinate_b[:, np.newaxis], distance * np.ones([dim, 1])), axis=1)
    raise ValueError(f'unknown coordinate {coordinate_a}, needs to be within [0, 3]')


def build_coordinates_scalar(coordinate_a, coordinate_b, distance):
    if coordinate_a == 0:
        return np.array([-distance, coordinate_b])
    elif coordinate_a == 1:
        return np.array([coordinate_b, -distance])
    elif coordinate_a == 2:
        return np.array([distance, coordinate_b])
    elif coordinate_a == 3:
        return np.array([coordinate_b, distance])
    else:
        raise ValueError()


def pick_random_velocity_and_goal(box_size: float):
    destination_a: int = random.choice([0, 1, 2, 3])
    low, high = -(floor(box_size) + 1), ceil(box_size) + 1
    destination_b = random.randint(low, high)
    goal = fill_state(destination_a, destination_b, False, box_size)

    speed = 0.5
    angle = random.uniform(-np.pi, np.pi)
    velocity = [speed * np.cos(angle), speed * np.sin(angle)]

    return velocity, goal


def init_random_ped_states(n_peds: int, box_size: float, max_initial_groups: int,
                           max_peds_per_group: int) -> Tuple[np.ndarray, List[List[int]]]:

    index_list: List[int] = np.arange(n_peds).tolist()
    state = np.zeros((n_peds, 6))
    groups = []
    grouped_peds = []
    available_peds = index_list.copy()

    for _ in range(max_initial_groups):
        max_n = min(len(available_peds), max_peds_per_group)
        group = random.sample(available_peds, max_n)
        groups.append(group)
        grouped_peds += group
        available_peds = [ped for ped in available_peds if ped not in group]

        # generate group target for grouped peds
        if group:
            velocity, goal = pick_random_velocity_and_goal(box_size)
            state[group, 4:6] = goal
            state[group, 2:4] = velocity

    ungrouped_peds = [ped for ped in index_list if ped not in grouped_peds]
    for ped in ungrouped_peds:
        velocity, goal = pick_random_velocity_and_goal(box_size)
        state[ped, 4:6] = goal
        state[ped, 2:4] = velocity

    return state, groups


def load_randomly_init_map(data: dict, maps_config_path: str, difficulty: int) \
        -> Tuple[float, List, List, np.ndarray, List]:
    map_name = load_map_name(data)
    path_to_map = os.path.join(maps_config_path, map_name)

    with open(path_to_map, 'r', encoding='utf-8') as file:
        map_structure = json.load(file)

    x_span = map_structure['x_margin'][1] - map_structure['x_margin'][0]
    y_span = map_structure['y_margin'][1] - map_structure['y_margin'][0]
    min_x, min_y = map_structure['x_margin'][0], map_structure['y_margin'][0]
    norm_span = max(x_span, y_span)
    box_size = 20

    def norm_coords(point: Vec2D) -> Vec2D:
        return ((point[0] - min_x) / norm_span * 2 * box_size - box_size,
                (point[1] - min_y) / norm_span * 2 * box_size - box_size)

    for key in map_structure['Obstacles'].keys():
        vertices = map_structure['Obstacles'][key]['Vertex']
        norm_vertices = [norm_coords(p) for p in vertices]
        map_structure['Obstacles'][key]['Vertex'] = norm_vertices

    obstacle = []
    obstacles_lolol = []
    for key in map_structure['Obstacles'].keys():
        vertices = map_structure['Obstacles'][key]['Vertex']
        valid_edges = list(zip(vertices[:-1], vertices[1:])) + [[vertices[-1], vertices[0]]]
        valid_edges = [[p1[0], p2[0], p1[1], p2[1]] for p1, p2 in valid_edges]
        obstacle += valid_edges
        obstacles_lolol.append(valid_edges)

    # append map bounds as obstacles
    obstacle.append([-box_size, box_size, -box_size, -box_size]) # bottom
    obstacle.append([-box_size, box_size, box_size, box_size])   # top
    obstacle.append([-box_size, -box_size, -box_size, box_size]) # left
    obstacle.append([box_size, box_size, -box_size, box_size])   # right

    if data['simulator']['flags']['random_initial_population']:
        n_peds = data['simulator']['custom']['random_population']['max_initial_peds'][difficulty]
        max_initial_groups = data['simulator']['custom']['random_population']['max_initial_groups']
        max_peds_per_group = data['simulator']['custom']['random_population']['max_peds_per_group']
        state, groups = init_random_ped_states(
            n_peds, box_size, max_initial_groups, max_peds_per_group)
    else:
        # use hard-coded settings from config file
        state = np.array(data['simulator']['default']['states'])
        groups = data['simulator']['default']['groups']
        n_peds = len(state)

    # TODO: figure out why this code is required to init pedestrian positions
    for i in range(n_peds):
        state[i, :2] = np.random.uniform(-box_size, box_size, (1,2))
        for _, obstacle_name in enumerate(map_structure['Obstacles']):
            obs_recreated = load_polygon(map_structure['Obstacles'][obstacle_name]['Vertex'])
            vert = np.array(map_structure['Obstacles'][obstacle_name]['Vertex'])
            radius = max(np.linalg.norm(vert - obs_recreated.centroid.coords, axis = 1)) +1

            if np.linalg.norm(state[i, :2] - obs_recreated.centroid.coords) < radius:
                state[i, :2] = np.random.uniform(-box_size, box_size, (1,2))
                break

    return box_size, obstacle, obstacles_lolol, state, groups


def load_config(path_to_filename: str=None, difficulty: int=0) \
        -> Tuple[SimulationConfiguration, np.ndarray, List, List]:
    # TODO: don't allow the argument to be None, set this explicitly in calling scope

    data = load_toml(path_to_filename)
    maps_config_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))) , 'utils', 'maps')
    box_size, obstacle, obstacles_lolol, state, groups = \
        load_randomly_init_map(data, maps_config_path, difficulty)

    robot_force_config = RobotForceConfig(
        data['simulator']['robot']['robot_radius'],
        data['simulator']['robot']['activation_threshold'],
        data['simulator']['robot']['force_multiplier'])

    peds_sparsity = data['simulator']['custom']['ped_sparsity']
    ped_generation_action_pool = ActionPool(
        data['simulator']['generation']['actions'],
        data['simulator']['generation']['probabilities'])
    group_action_pool = ActionPool(
        data['simulator']['group_actions']['actions'],
        data['simulator']['group_actions']['probabilities'])
    stopping_action_pool = ActionPool(
        data['simulator']['stopping']['actions'],
        data['simulator']['stopping']['probabilities'])

    flags = SimulatorConfigFlags(
        data['simulator']['flags']['allow_dynamic_grouping'],
        data['simulator']['flags']['allow_max_steps_stop'],
        data['simulator']['flags']['topology_operations_on_stopped'],
        data['simulator']['flags']['rendezvous_centroid'],
        data['simulator']['flags']['update_peds'],
        data['simulator']['flags']['activate_ped_robot_force'])

    group_action_params = data['simulator']['group_actions']['parameters']
    stopping_params = data['simulator']['stopping']['parameters']
    new_peds_params = NewPedsParams(
        max_single_peds = data['simulator']['generation']['parameters']['max_single_peds'],
        max_grp_size = data['simulator']['generation']['parameters']['max_group_size'],
        group_width_max = data['simulator']['generation']['parameters']['max_group_size'],
        group_width_min = data['simulator']['generation']['parameters']['group_width_min'],
        average_speed = data['simulator']['generation']['parameters']['average_speed'],
        max_standalone_grouping = group_action_params['max_standalone_grouping'],
        max_nsteps_ped_stopped = stopping_params['max_nsteps_ped_stopped'],
        max_nteps_group_stopped = stopping_params['max_nsteps_group_stopped'],
        max_stopping_pedestrians = stopping_params['max_stopping_pedestrians'],
        max_unfreezing_pedestrians = stopping_params['max_unfreezing_pedestrians'])

    # angles to evaluate
    angles_neg = np.array([-1, -.8, -.6, -.5, -.4, -.3, -.25, -.2, -.15, -.1, -.05])
    angles_pos = np.sort(-angles_neg)
    angles = np.multiply(np.pi, np.concatenate((angles_neg, angles_pos)))

    #obstacles points to get the segments (commented part refers to in-function implementation)
    p_0 = np.empty((0, 2))
    p_1 = np.empty((0, 2))
    ##get obstacles on the scenes in p0,p1 format (do be moved to class attributes)
    for obi in obstacle:
        p_0 = np.append(p_0, np.array([obi[0], obi[2]])[np.newaxis, :], axis=0)
        p_1 = np.append(p_1, np.array([obi[1], obi[3]])[np.newaxis, :], axis=0)

    # obstacle_avoidance_params
    view_distance = 15
    forgetting_factor = 0.8
    obstacle_avoidance_params = [angles, p_0, p_1, view_distance, forgetting_factor]

    config = SimulationConfiguration(
        box_size, peds_sparsity, ped_generation_action_pool,
        group_action_pool, stopping_action_pool, flags, robot_force_config,
        new_peds_params, obstacle_avoidance_params, obstacles_lolol)

    return config, state, groups, obstacle

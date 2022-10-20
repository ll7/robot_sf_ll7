import os
import toml
import json
import random
from typing import List
from dataclasses import dataclass

import numpy as np
from natsort import natsorted

from robot_sf.utils.utilities import fill_state
from robot_sf.utils.poly import load_polygon, random_polygon, PolygonCreationSettings


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
    probs_in_percent: List[int]


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


def load_toml(path_to_filename: str) -> dict:
    # TODO: remove branching / error handling, just do the damn thing ...
    if path_to_filename is None:
        try:
            dirname = os.path.dirname(__file__)
            parent = os.path.split(dirname)[0]
            filename = os.path.join(parent, "utils", "config", "map_config.toml")
            data: dict = toml.load(filename)
        except Exception:
            raise ValueError("Cannot load valid config toml file")
    else:
        if not isinstance(path_to_filename, str):
            raise ValueError("invalid input filename")
        else:
            try:
                data: dict = toml.load(path_to_filename)
            except Exception:
                raise ValueError("Cannot load valid config toml file at specified path:" + path_to_filename)
    return data


def load_map_name(data: dict) -> str:
    if not data['simulator']['flags']['random_map']:
        n = data['simulator']['custom']['map_number']
        try:
            map_name = natsorted(list(data['map_files'].values()))[n]
        except:
            map_name = natsorted(list(data['map_files'].values()))[0]
            raise Warning('Invalid map number: Using map 0')
    else:
        map_name = random.choice(list(data['map_files'].values()))
    return map_name


def load_randomly_init_map(data: dict, maps_config_path: str, difficulty: int):
    map_name = load_map_name(data)
    path_to_map = os.path.join(maps_config_path, map_name)

    with open(path_to_map, 'r') as f:
        map_structure = json.load(f)

    box_size = max((map_structure['x_margin'][1]), (map_structure['y_margin'][1]))
    #print(path_to_map)
    #Start creating obstacles, peds ecc...
    obstacle = []
    obstacles_lolol = []

    for key in map_structure['Obstacles'].keys():
        tmp = map_structure['Obstacles'][key]['Edges'].copy()
        valid_edges = []
        for n, sub_list in enumerate(tmp):
            #if not sub_list[0]==sub_list[1] and not sub_list[2]==sub_list[3]:
            valid_edges.append(sub_list)
        obstacle += valid_edges
        obstacles_lolol.append(valid_edges)

    if not data['simulator']['flags']['random_initial_population']:
        state = np.array(data['simulator']['default']['states'])
        groups = data['simulator']['default']['groups']
        
        grouped_peds = []
        
        for group in groups:
            grouped_peds += group
        n_peds = len(state)
        
        
    else:
        n_peds = data['simulator']['custom']['random_population']['max_initial_peds'][difficulty]
        index_list = np.arange(n_peds).tolist() #Available index to group
        
        #initialize state matrix
        state = np.zeros((n_peds, 6))
        groups = []
                            
        
        #Initialize groups
        grouped_peds = []
        available_peds = index_list.copy()
        
                        
        for i in range(data['simulator']['custom']['random_population']['max_initial_groups']):
            max_n = min(len(available_peds), data['simulator']['custom']['random_population']['max_peds_per_group'])
            
            group = random.sample(available_peds, max_n)
            groups.append(group)
            grouped_peds += group
            
            available_peds = [ped for ped in available_peds if ped not in group]
            
            
            #generate group target for grouped peds
            if group:
                group_destination_a = random.choice([0,1,2,3])
                group_destination_b = random.randint(-(box_size +1),box_size +1)*np.ones((len(group),))
                
                #Initial speed
                dot_x_0 = 0.5 #Module
                
                #random angle
                angle = random.uniform(-np.pi, np.pi)
                dot_x_x = dot_x_0*np.cos(angle)
                dot_x_y = dot_x_0*np.sin(angle)

            #Based on group origin and group destination compute the new states of the 
            #new added pedestrians
                destination_states = fill_state(group_destination_a, group_destination_b, False, box_size)
                state[group, 4:6] = destination_states
                state[group, 2:4] = [dot_x_x, dot_x_y]
        
        #Check for state validity
        obs_recreated = random_polygon(PolygonCreationSettings(5, irregularity=0, spikeness=0))
    for i in range(n_peds):
        #Check if initial position is valid
        state[i, :2] = np.random.uniform(-box_size, box_size, (1,2))

        while True:
            for iter_num, ob in enumerate(map_structure['Obstacles'].keys()):
                #Compute safety radius for each obstacle
                obs_recreated = load_polygon(map_structure['Obstacles'][ob]['Vertex'])
                vert = np.array(map_structure['Obstacles'][ob]['Vertex'])
                radius = max(np.linalg.norm(vert - obs_recreated.centroid.coords, axis = 1)) +1

                if np.linalg.norm(state[i, :2] - obs_recreated.centroid.coords) < radius:
                    #Generate new point, break and restart the for loop check
                    state[i, :2] = np.random.uniform(-box_size, box_size, (1,2))
                    break

                #Break the endless loop and let i-index increase
            if iter_num == len(map_structure['Obstacles'].keys())-1:
                break

        #Generate target
        if data['simulator']['flags']['random_initial_population']:
            #print("Generate target")
            if i not in grouped_peds:
                destination_a = random.choice([0,1,2,3])
                destination_b = random.randint(-(box_size +1),box_size +1)
                
                destination_state = fill_state(destination_a, destination_b, False, box_size)
                state[i, 4:6] = destination_state
                
                dot_x_0 = 0.5
                angle = random.uniform(-np.pi, np.pi)
                
                dot_x_x = dot_x_0*np.cos(angle)
                dot_x_y = dot_x_0*np.sin(angle)
                
                state[i,2:4] = [dot_x_x, dot_x_y]

    return box_size, obstacle, obstacles_lolol, state, groups


def load_config(path_to_filename: str=None, difficulty: int=0):
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

    new_peds_params = NewPedsParams(
        max_single_peds = data['simulator']['generation']['parameters']['max_single_peds'],
        max_grp_size = data['simulator']['generation']['parameters']['max_group_size'],
        group_width_max = data['simulator']['generation']['parameters']['max_group_size'],
        group_width_min = data['simulator']['generation']['parameters']['group_width_min'],
        average_speed = data['simulator']['generation']['parameters']['average_speed'],
        max_standalone_grouping = data['simulator']['group_actions']['parameters']['max_standalone_grouping'],
        max_nsteps_ped_stopped = data['simulator']['stopping']['parameters']['max_nsteps_ped_stopped'],
        max_nteps_group_stopped = data['simulator']['stopping']['parameters']['max_nsteps_group_stopped'],
        max_stopping_pedestrians = data['simulator']['stopping']['parameters']['max_stopping_pedestrians'],
        max_unfreezing_pedestrians = data['simulator']['stopping']['parameters']['max_unfreezing_pedestrians'])

    # angles to evaluate
    angles_neg = np.array([-1, -.8, -.6, -.5, -.4, -.3, -.25, -.2, -.15, -.1, -.05])
    angles_pos = np.sort(-angles_neg)
    angles = np.multiply(np.pi, np.concatenate((angles_neg, angles_pos)))

    #obstacles points to get the segments (commented part refers to in-function implementation)
    p0 = np.empty((0, 2))
    p1 = np.empty((0, 2))
    ##get obstacles on the scenes in p0,p1 format (do be moved to class attributes)
    for obi in obstacle:
        p0 = np.append(p0, np.array([obi[0], obi[2]])[np.newaxis, :], axis=0)
        p1 = np.append(p1, np.array([obi[1], obi[3]])[np.newaxis, :], axis=0)

    # obstacle_avoidance_params
    view_distance = 15
    forgetting_factor = 0.8
    obstacle_avoidance_params = [angles, p0, p1, view_distance, forgetting_factor]

    config = SimulationConfiguration(
        box_size, peds_sparsity, ped_generation_action_pool,
        group_action_pool, stopping_action_pool, flags, robot_force_config,
        new_peds_params, obstacle_avoidance_params, obstacles_lolol)

    return config, state, groups, obstacle
